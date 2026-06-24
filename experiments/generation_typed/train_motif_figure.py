#!/usr/bin/env python3
"""
H15 HIERARCHICAL pick-then-realize (minimal) — DISCRETE per-section FIGURE-token conditioning.

ONE new variable vs gen_motif_local2 (continuous per-section motif): a discrete per-section FIGURE token
(jack/sweep/trill/candle/jump/step/sparse) embedded and added to the DECODER conditioning. The continuous
radar-orthogonal knobs ENTANGLE figures — knob-0 'jack<->sweep' is really a jack detector, with sweep mushed
into step/candle (diag_figure_labels.py) — so the continuous lever can't isolate SWEEP. A discrete figure label
cleanly separates it. "Pick" = the section's figure class; "realize" = the pattern head conditioned on its
learned embedding. We warm-start gen_motif_local2 and KEEP the continuous motif (so candle/trill still work),
adding figure as the only new signal; eval isolates figure (figure=token, motif=null).

STYLE OFF (Track A). Figure feeds ONLY the decoder (not onset), like motif. patience-3 early stopping.

Usage:
    python experiments/generation_typed/train_motif_figure.py --data_dir data/ --audio_dir data/ \
        --epochs 20 --warmup_freeze 2 --batch_size 8 --section 64 --patience 3
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator, MOTIF_DIM, NUM_FIGURE_CLASSES
from src.generation.typed import NUM_TYPES, NUM_PANELS, panels_to_pattern
from src.generation.motif_codebook import MotifBasis, figure_token_schedule
sys.path.insert(0, str(PROJECT_ROOT / "experiments/generation_typed"))
from train_motif_local import local_motif_targets, focal_bce, focal_ce

LOCAL2_CKPT = "checkpoints/gen_motif_local2/best_val.pt"   # warm-start = continuous per-section motif (Track A)
MOTIF_BASIS = PROJECT_ROOT / "cache/motif_basis.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_train_len', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_freeze', type=int, default=2)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--section', type=int, default=64)
    p.add_argument('--checkpoint_dir', default='checkpoints/gen_motif_figure')
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--radar_drop', type=float, default=0.15)
    p.add_argument('--motif_drop', type=float, default=0.30, help='drop continuous motif more (figure is the focus)')
    p.add_argument('--figure_drop', type=float, default=0.15)
    return p.parse_args()


def collect_typed(ds, cap, basis, section):
    out = []
    for i in range(len(ds)):
        sample = ds[i]; meta = ds.valid_samples[i]
        nd = next((n for n in meta['chart'].note_data
                   if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        typed_full = ds.parser.convert_to_tensor_typed(meta['chart'], nd)
        radar = meta['groove_radar'].to_vector().astype(np.float32)
        sched = local_motif_targets(typed_full, radar, basis, section)         # (T,K) continuous
        figs = figure_token_schedule(typed_full, section)                      # (T,) discrete tokens
        T = min(int(sample['mask'].sum().item()), cap, typed_full.shape[0])
        out.append({'typed': typed_full[:T].astype(np.int64),
                    'audio': sample['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(meta['difficulty_class']), 'radar': radar,
                    'motif': sched[:T], 'figure': figs[:T]})
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
    basis = MotifBasis.load(MOTIF_BASIS)
    assert basis.K == MOTIF_DIM

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
    print(f"Collecting typed samples + motif schedules + figure tokens (section={args.section})...")
    train = collect_typed(train_ds, args.max_train_len, basis, args.section)
    val = collect_typed(val_ds, args.max_train_len, basis, args.section)
    print(f"train={len(train)} val={len(val)}")
    audio_dim = train[0]['audio'].shape[1]
    assert audio_dim == 42

    try:
        import mlflow; mlflow.set_experiment("stepmania-chart-generator"); mlflow_on = True
    except ImportError:
        mlflow_on = False

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    ck = torch.load(LOCAL2_CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
    print(f"warm-started from gen_motif_local2; fresh params: {missing} (figure only); unexpected: {len(unexpected)}")
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
        radar = torch.zeros(B, 5); motif = torch.zeros(B, T, MOTIF_DIM); figure = torch.zeros(B, T, dtype=torch.long)
        for b, s in enumerate(batch):
            t = len(s['typed'])
            audio[b, :t] = torch.from_numpy(s['audio']); states[b, :t] = torch.from_numpy(s['typed'])
            mask[b, :t] = True; diff[b] = s['difficulty']
            radar[b] = torch.from_numpy(s['radar']); motif[b, :t] = torch.from_numpy(s['motif'])
            figure[b, :t] = torch.from_numpy(s['figure'])
        active = (states != 0)
        pat = torch.from_numpy(panels_to_pattern(active.numpy())).clamp(min=0)
        typ = (states - 1).clamp(min=0)
        return (audio.to(device), states.to(device), mask.to(device), diff.to(device), pat.to(device),
                typ.to(device), active.to(device), radar.to(device), motif.to(device), figure.to(device))

    def losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active):
        onset_t = (states != 0).any(-1).float()
        o = focal_bce(ol[mask], onset_t[mask], args.focal_gamma)
        sel = mask & (onset_t > 0.5)
        p = focal_ce(pat_l[sel], pat_t[sel], args.focal_gamma) if sel.any() else torch.zeros((), device=device)
        act = active & mask.unsqueeze(-1)
        t = focal_ce(typ_l[act], typ_t[act], args.focal_gamma) if act.any() else torch.zeros((), device=device)
        return o, p, t

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best = float('inf'); best_path = Path(args.checkpoint_dir) / "best_val.pt"; no_improve = 0
    if mlflow_on:
        mlflow.start_run(run_name="gen-motif-figure")
        mlflow.log_params({'epochs': args.epochs, 'patience': args.patience, 'lr': args.lr,
                           'warm': 'gen_motif_local2', 'section': args.section, 'style': 'OFF',
                           'radar_drop': args.radar_drop, 'motif_drop': args.motif_drop, 'figure_drop': args.figure_drop})

    for epoch in range(args.epochs):
        if epoch == args.warmup_freeze:
            model.freeze_audio_encoder(False); print(f"  [epoch {epoch+1}] unfroze audio encoder")
        model.train(); tr = [0.0, 0.0, 0.0]; nb = 0
        for batch in batches(train, args.batch_size, True):
            audio, states, mask, diff, pat_t, typ_t, active, radar, motif, figure = to_tensors(batch)
            optimizer.zero_grad()
            cr = None if rng.random() < args.radar_drop else radar
            cm = None if rng.random() < args.motif_drop else motif
            cfg_ = None if rng.random() < args.figure_drop else figure
            ol, pat_l, typ_l = model(audio, states, diff, mask, radar=cr, motif=cm, figure=cfg_)  # style OFF
            o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
            (o + p + t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tr[0] += o.item(); tr[1] += p.item(); tr[2] += t.item(); nb += 1
        model.eval(); v = [0.0, 0.0, 0.0]; vnb = 0
        with torch.no_grad():
            for batch in batches(val, args.batch_size, False):
                audio, states, mask, diff, pat_t, typ_t, active, radar, motif, figure = to_tensors(batch)
                ol, pat_l, typ_l = model(audio, states, diff, mask, radar=radar, motif=motif, figure=figure)
                o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
                v[0] += o.item(); v[1] += p.item(); v[2] += t.item(); vnb += 1
        v = [x / max(1, vnb) for x in v]; vt = sum(v)
        improved = vt < best
        print(f"  epoch {epoch+1}/{args.epochs}  train(o={tr[0]/max(1,nb):.3f} pat={tr[1]/max(1,nb):.3f} typ={tr[2]/max(1,nb):.3f})  "
              f"val(o={v[0]:.3f} pat={v[1]:.3f} typ={v[2]:.3f} tot={vt:.3f})" + ("  *" if improved else ""), flush=True)
        if mlflow_on: mlflow.log_metrics({'val_onset': v[0], 'val_pattern': v[1], 'val_type': v[2], 'val_total': vt}, step=epoch)
        if improved:
            best = vt; no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_total': vt,
                        'section': args.section}, best_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stopping: no val improvement for {args.patience} epochs (best {best:.4f})"); break

    print(f"\nbest val_total={best:.4f} -> {best_path}")
    if mlflow_on:
        mlflow.log_metric('best_val_total', best); mlflow.end_run()
    print("Eval: eval_figure_control.py --ckpt checkpoints/gen_motif_figure/best_val.pt --highres "
          "(set figure=sweep token; does the generated chart's sweep-section fraction rise?)")


if __name__ == '__main__':
    main()
