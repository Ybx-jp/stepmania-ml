#!/usr/bin/env python3
"""
Train H15 MOTIF-vocabulary conditioning on top of the layered typed generator (Phase 2).

The pattern head learns to render a style's CHARACTERISTIC FIGURES from a 12-d radar-orthogonal motif-knob
vector (src/generation/motif_codebook.py::MotifBasis), the residual the radar can't reach (notes/
h15_motif_findings.md). We WARM-START from gen_style — the exact radar+style+CFG checkpoint behind the OH
WORLD g3.5 breakthrough — load it whole (strict=False; only motif_proj/null_motif are fresh, and motif_proj is
zero-init so step 0 is an identity no-op), and fine-tune with INDEPENDENT classifier-free-guidance dropout on
radar / style / motif so each knob can be amplified or used alone. Self-conditioned (each chart on its own
motif vector); the radar-orthogonal motif vector + the autoregressive head's exposure to real figures teach
the mapping.

Usage:
    python experiments/generation_typed/train_motif.py --data_dir data/ --audio_dir data/ \
        --epochs 12 --warmup_freeze 2 --batch_size 8
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
from src.generation.typed_model import LayeredTypedChartGenerator, MOTIF_DIM
from src.generation.typed import NUM_TYPES, NUM_PANELS, pair_holds, panels_to_pattern
from src.generation.evaluation import onset_density_metrics
from src.generation.motif_codebook import MotifBasis

STYLE_CKPT = "checkpoints/gen_style/best_val.pt"          # warm-start = the OH WORLD radar+style+CFG model
MOTIF_BASIS = PROJECT_ROOT / "cache/motif_basis.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_train_len', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_freeze', type=int, default=2)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--eval_songs', type=int, default=48)
    p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--checkpoint_dir', default='checkpoints/gen_motif')
    p.add_argument('--radar_drop', type=float, default=0.15)
    p.add_argument('--style_drop', type=float, default=0.30, help='drop style more — the breakthrough used radar+motif, not reference')
    p.add_argument('--motif_drop', type=float, default=0.15)
    return p.parse_args()


def collect_typed(ds, cap, basis):
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
        motif = basis.encode_chart(typed_full, radar).astype(np.float32)   # global motif descriptor (full chart)
        T = min(int(sample['mask'].sum().item()), cap, typed_full.shape[0])
        out.append({'typed': typed_full[:T].astype(np.int64),
                    'audio': sample['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(meta['difficulty_class']), 'radar': radar, 'motif': motif})
    return out


def focal_bce(logits, targets, gamma):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits); p_t = p * targets + (1 - p) * (1 - targets)
    return ((1 - p_t) ** gamma * bce).mean()


def focal_ce(logits, targets, gamma):
    logp = nn.functional.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (-((1 - logp_t.exp()) ** gamma) * logp_t).mean()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
    basis = MotifBasis.load(MOTIF_BASIS)
    assert basis.K == MOTIF_DIM, f"basis K={basis.K} != model MOTIF_DIM={MOTIF_DIM}"

    chart_files = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(chart_files, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        max_seq_len = yaml.safe_load(f)['classifier']['max_sequence_length']
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=max_seq_len,
                                          cache_dir='cache/samples')   # 23-dim features (matches gen_style)
    print("Warming caches..."); train_ds.warm_cache(show_progress=True); val_ds.warm_cache(show_progress=True)
    print("Collecting typed samples + motif vectors...")
    train = collect_typed(train_ds, args.max_train_len, basis)
    val = collect_typed(val_ds, args.max_train_len, basis)
    print(f"train={len(train)} val={len(val)}")
    audio_dim = train[0]['audio'].shape[1]
    target_density = float(np.mean([(s['typed'] != 0).any(1).mean() for s in val]))

    try:
        import mlflow; mlflow.set_experiment("stepmania-chart-generator"); mlflow_on = True
    except ImportError:
        mlflow_on = False

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    ck = torch.load(STYLE_CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
    print(f"warm-started from gen_style; fresh params: {missing} (motif only); unexpected: {len(unexpected)}")
    assert all('motif' in k for k in missing) and not unexpected, "warm-start mismatch beyond motif params"
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
        radar = torch.zeros(B, 5); motif = torch.zeros(B, MOTIF_DIM)
        for b, s in enumerate(batch):
            t = len(s['typed'])
            audio[b, :t] = torch.from_numpy(s['audio']); states[b, :t] = torch.from_numpy(s['typed'])
            mask[b, :t] = True; diff[b] = s['difficulty']
            radar[b] = torch.from_numpy(s['radar']); motif[b] = torch.from_numpy(s['motif'])
        active = (states != 0)
        pat = torch.from_numpy(panels_to_pattern(active.numpy())).clamp(min=0)
        typ = (states - 1).clamp(min=0)
        return (audio.to(device), states.to(device), mask.to(device), diff.to(device),
                pat.to(device), typ.to(device), active.to(device), radar.to(device), motif.to(device))

    def losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active):
        onset_t = (states != 0).any(-1).float()
        o = focal_bce(ol[mask], onset_t[mask], args.focal_gamma)
        sel = mask & (onset_t > 0.5)
        p = focal_ce(pat_l[sel], pat_t[sel], args.focal_gamma) if sel.any() else torch.zeros((), device=device)
        act = active & mask.unsqueeze(-1)
        t = focal_ce(typ_l[act], typ_t[act], args.focal_gamma) if act.any() else torch.zeros((), device=device)
        return o, p, t

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best = float('inf'); best_path = Path(args.checkpoint_dir) / "best_val.pt"
    if mlflow_on:
        mlflow.start_run(run_name="gen-motif")
        mlflow.log_params({'epochs': args.epochs, 'lr': args.lr, 'warm': 'gen_style',
                           'radar_drop': args.radar_drop, 'style_drop': args.style_drop, 'motif_drop': args.motif_drop})

    for epoch in range(args.epochs):
        if epoch == args.warmup_freeze:
            model.freeze_audio_encoder(False); print(f"  [epoch {epoch+1}] unfroze audio encoder")
        model.train(); tr = [0.0, 0.0, 0.0]; nb = 0
        for batch in batches(train, args.batch_size, True):
            audio, states, mask, diff, pat_t, typ_t, active, radar, motif = to_tensors(batch)
            optimizer.zero_grad()
            cond_radar = None if rng.random() < args.radar_drop else radar
            ref = None if rng.random() < args.style_drop else states
            ref_mask = None if ref is None else mask
            cond_motif = None if rng.random() < args.motif_drop else motif
            ol, pat_l, typ_l = model(audio, states, diff, mask, radar=cond_radar,
                                     reference=ref, reference_mask=ref_mask, motif=cond_motif)
            o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
            (o + p + t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tr[0] += o.item(); tr[1] += p.item(); tr[2] += t.item(); nb += 1
        model.eval(); v = [0.0, 0.0, 0.0]; vnb = 0
        with torch.no_grad():
            for batch in batches(val, args.batch_size, False):
                audio, states, mask, diff, pat_t, typ_t, active, radar, motif = to_tensors(batch)
                ol, pat_l, typ_l = model(audio, states, diff, mask, radar=radar,
                                         reference=states, reference_mask=mask, motif=motif)
                o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
                v[0] += o.item(); v[1] += p.item(); v[2] += t.item(); vnb += 1
        v = [x / max(1, vnb) for x in v]; vt = sum(v)
        print(f"  epoch {epoch+1}/{args.epochs}  train(o={tr[0]/max(1,nb):.3f} pat={tr[1]/max(1,nb):.3f} typ={tr[2]/max(1,nb):.3f})  "
              f"val(o={v[0]:.3f} pat={v[1]:.3f} typ={v[2]:.3f} tot={vt:.3f})" + ("  *" if vt < best else ""), flush=True)
        if mlflow_on: mlflow.log_metrics({'val_onset': v[0], 'val_pattern': v[1], 'val_type': v[2], 'val_total': vt}, step=epoch)
        if vt < best:
            best = vt
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_total': vt}, best_path)

    print(f"\nbest val_total={best:.4f} -> {best_path}")
    if mlflow_on:
        mlflow.log_metric('best_val_total', best); mlflow.end_run()
    print("Run eval_motif.py for the motif-knob steerability test (does pushing a knob change realized figures?).")


if __name__ == '__main__':
    main()
