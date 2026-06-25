#!/usr/bin/env python3
"""
H15 hierarchical figure tokens — STANDALONE + FIGURE-AWARE SELECTION (option 1, notes/h15_hierarchical_findings.md).

The first figure run (train_motif_figure.py) gave only a modest sweep lift and early-stopped at epoch 1. Three
compounding reasons, each fixed here:
  A) REDUNDANCY — it sat on top of the continuous motif, which already encodes the section figure mix, so the
     figure embedding got ~no gradient. FIX: train figure STANDALONE (warm gen_motif_hr; NEVER feed the
     continuous motif), so the figure token is the SOLE section-figure signal and MUST be learned.
  B) FIGURE-BLIND SELECTION — early-stop watched val CE, blind to CONTROL. FIX: select/early-stop on a
     generation-time CONTROL metric (set figure=sweep, measure realized sweep-section-fraction lift vs
     figure=null), with a quality guard. (experiment-design Rule 2: select under deployment conditions.)
  C) TOO FEW EPOCHS — a consequence of A+B; raise the cap and let control plateau.

PROOF stays leak-free: training self-conditions on the real section figure (mildly leaky), but SELECTION is the
leak-free user-set control eval, so we never certify on the leaky quantity.

Usage:
    python experiments/generation_typed/train_motif_figure_standalone.py --data_dir data/ --audio_dir data/ \
        --epochs 30 --warmup_freeze 2 --batch_size 8 --section 64 --patience 5
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator, NUM_FIGURE_CLASSES
from src.generation.typed import NUM_TYPES, NUM_PANELS, panels_to_pattern, pair_holds
from src.generation.evaluation import onset_density_metrics
from src.generation.motif_codebook import figure_token, figure_token_schedule, FIGURE_CLASSES
sys.path.insert(0, str(PROJECT_ROOT / "experiments/generation_typed"))
from train_motif_local import focal_bce, focal_ce

HR_CKPT = "checkpoints/gen_motif_hr/best_val.pt"   # warm-start = Track-A base (NO continuous local motif used)
SWEEP = FIGURE_CLASSES.index("sweep/staircase")    # = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5, help='early stop on the (noisy) CONTROL metric')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_train_len', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_freeze', type=int, default=2)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--section', type=int, default=64)
    p.add_argument('--checkpoint_dir', default='checkpoints/gen_motif_figure_solo')
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--radar_drop', type=float, default=0.15)
    p.add_argument('--figure_drop', type=float, default=0.15)
    p.add_argument('--ctrl_songs', type=int, default=8, help='songs for the per-epoch control selection metric')
    p.add_argument('--ctrl_len', type=int, default=512)
    p.add_argument('--f1_floor', type=float, default=0.62, help='quality guard: do not checkpoint below this onset_F1')
    return p.parse_args()


def collect_typed(ds, cap, section):
    out = []
    for i in range(len(ds)):
        sample = ds[i]; meta = ds.valid_samples[i]
        nd = next((n for n in meta['chart'].note_data
                   if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        typed_full = ds.parser.convert_to_tensor_typed(meta['chart'], nd)
        figs = figure_token_schedule(typed_full, section)
        T = min(int(sample['mask'].sum().item()), cap, typed_full.shape[0])
        out.append({'typed': typed_full[:T].astype(np.int64),
                    'audio': sample['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(meta['difficulty_class']),
                    'radar': meta['groove_radar'].to_vector().astype(np.float32),
                    'figure': figs[:T]})
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
    S = args.section

    chart_files = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(chart_files, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        max_seq_len = yaml.safe_load(f)['classifier']['max_sequence_length']
    ext42 = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                     use_metric_phase=True, use_highres_onset=True))
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=max_seq_len,
                                          feature_extractor=ext42, cache_dir=args.cache_dir)
    print("Warming caches (cache/samples_v3)..."); train_ds.warm_cache(show_progress=True); val_ds.warm_cache(show_progress=True)
    print(f"Collecting typed samples + figure tokens (section={S}, NO continuous motif)...")
    train = collect_typed(train_ds, args.max_train_len, S)
    val = collect_typed(val_ds, args.max_train_len, S)
    print(f"train={len(train)} val={len(val)}")
    audio_dim = train[0]['audio'].shape[1]; assert audio_dim == 42
    ctrl = val[:args.ctrl_songs]                                              # held-out control-selection songs
    ctrl_density = float(np.mean([(s['typed'] != 0).any(1).mean() for s in ctrl]))

    try:
        import mlflow; mlflow.set_experiment("stepmania-chart-generator"); mlflow_on = True
    except ImportError:
        mlflow_on = False

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    ck = torch.load(HR_CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
    print(f"warm-started from gen_motif_hr; fresh params: {missing} (figure only); unexpected: {len(unexpected)}")
    assert all('figure' in k for k in missing) and not unexpected, "warm-start mismatch beyond figure params"
    model.freeze_audio_encoder(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batches(samples, bs, shuffle):
        idx = np.arange(len(samples))
        if shuffle: rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [samples[j] for j in idx[i:i + bs]]

    def to_tensors(batch):
        T = max(len(s['typed']) for s in batch); B = len(batch)
        audio = torch.zeros(B, T, audio_dim); states = torch.zeros(B, T, NUM_PANELS, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool); diff = torch.zeros(B, dtype=torch.long)
        radar = torch.zeros(B, 5); figure = torch.zeros(B, T, dtype=torch.long)
        for b, s in enumerate(batch):
            t = len(s['typed'])
            audio[b, :t] = torch.from_numpy(s['audio']); states[b, :t] = torch.from_numpy(s['typed'])
            mask[b, :t] = True; diff[b] = s['difficulty']
            radar[b] = torch.from_numpy(s['radar']); figure[b, :t] = torch.from_numpy(s['figure'])
        active = (states != 0)
        pat = torch.from_numpy(panels_to_pattern(active.numpy())).clamp(min=0)
        typ = (states - 1).clamp(min=0)
        return (audio.to(device), states.to(device), mask.to(device), diff.to(device), pat.to(device),
                typ.to(device), active.to(device), radar.to(device), figure.to(device))

    def losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active):
        onset_t = (states != 0).any(-1).float()
        o = focal_bce(ol[mask], onset_t[mask], args.focal_gamma)
        sel = mask & (onset_t > 0.5)
        p = focal_ce(pat_l[sel], pat_t[sel], args.focal_gamma) if sel.any() else torch.zeros((), device=device)
        act = active & mask.unsqueeze(-1)
        t = focal_ce(typ_l[act], typ_t[act], args.focal_gamma) if act.any() else torch.zeros((), device=device)
        return o, p, t

    def sweep_frac(ch):
        T = ch.shape[0]; cnt = Counter(figure_token(ch[i:i + S]) for i in range(0, T, S))
        tot = sum(cnt.values())
        return cnt[SWEEP] / tot if tot else 0.0

    @torch.no_grad()
    def control_metric():
        """leak-free: set figure=sweep vs null, measure realized sweep-section-fraction lift + quality."""
        model.eval()
        lifts, f1s, fr_set, fr_base = [], [], [], []
        for s in ctrl:
            L = min(s['typed'].shape[0], args.ctrl_len)
            audio = torch.from_numpy(s['audio'][:L]).unsqueeze(0).to(device)
            diff = torch.tensor([s['difficulty']], device=device)
            radar = torch.from_numpy(s['radar']).unsqueeze(0).to(device)
            tau = float(np.quantile(torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff, radar=radar)).cpu().numpy(),
                                    1 - ctrl_density))
            def g(fig_tok):
                fig = None if fig_tok is None else torch.full((1, L), fig_tok, dtype=torch.long, device=device)
                out = model.generate(audio, diff, lengths=torch.tensor([L], device=device), onset_threshold=tau,
                                     radar=radar, figure=fig, guidance_scale=1.0, type_sample=True,
                                     type_temperature=0.4, hold_aware=True, pattern_sample=True,
                                     pattern_temperature=1.0, max_jack_run=1)[0].cpu().numpy()
                return pair_holds(out[:L])
            ch_s = g(SWEEP); ch_0 = g(None)
            fr_set.append(sweep_frac(ch_s)); fr_base.append(sweep_frac(ch_0))
            f1s.append(onset_density_metrics((ch_s != 0).astype(np.float32),
                                             reference=(s['typed'][:L] != 0).astype(np.float32))['onset_f1'])
        return float(np.mean(fr_set) - np.mean(fr_base)), float(np.mean(fr_set)), float(np.mean(fr_base)), float(np.mean(f1s))

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_lift = -1e9; best_path = Path(args.checkpoint_dir) / "best_val.pt"; no_improve = 0
    if mlflow_on:
        mlflow.start_run(run_name="gen-motif-figure-solo")
        mlflow.log_params({'epochs': args.epochs, 'patience': args.patience, 'warm': 'gen_motif_hr',
                           'mode': 'standalone-figure', 'select': 'sweep-control', 'section': S})

    for epoch in range(args.epochs):
        if epoch == args.warmup_freeze:
            model.freeze_audio_encoder(False); print(f"  [epoch {epoch+1}] unfroze audio encoder")
        model.train(); tr = [0.0, 0.0, 0.0]; nb = 0
        for batch in batches(train, args.batch_size, True):
            audio, states, mask, diff, pat_t, typ_t, active, radar, figure = to_tensors(batch)
            optimizer.zero_grad()
            cr = None if rng.random() < args.radar_drop else radar
            cfg_ = None if rng.random() < args.figure_drop else figure
            ol, pat_l, typ_l = model(audio, states, diff, mask, radar=cr, figure=cfg_)  # NO motif, style OFF
            o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
            (o + p + t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tr[0] += o.item(); tr[1] += p.item(); tr[2] += t.item(); nb += 1
        lift, fr_set, fr_base, f1 = control_metric()
        ok = lift > best_lift and f1 >= args.f1_floor
        print(f"  epoch {epoch+1}/{args.epochs}  train_pat={tr[1]/max(1,nb):.3f}  "
              f"CONTROL sweep_lift={lift:+.3f} (set {fr_set:.2f} / base {fr_base:.2f})  onset_F1={f1:.2f}"
              + ("  *" if ok else ""), flush=True)
        if mlflow_on: mlflow.log_metrics({'sweep_lift': lift, 'sweep_set': fr_set, 'sweep_base': fr_base, 'ctrl_f1': f1}, step=epoch)
        if ok:
            best_lift = lift; no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'sweep_lift': lift,
                        'ctrl_f1': f1, 'section': S}, best_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stopping: no control improvement for {args.patience} epochs (best lift {best_lift:+.3f})"); break

    print(f"\nbest sweep_lift={best_lift:+.3f} -> {best_path}")
    if mlflow_on:
        mlflow.log_metric('best_sweep_lift', best_lift); mlflow.end_run()
    print("Eval: eval_figure_control.py --ckpt checkpoints/gen_motif_figure_solo/best_val.pt --highres")


if __name__ == '__main__':
    main()
