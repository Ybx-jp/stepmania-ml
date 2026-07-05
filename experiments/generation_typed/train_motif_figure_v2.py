#!/usr/bin/env python3
"""data-layer-v2 retrain (phase 4): the deployed gen_motif_full_fixed re-trained on the 48th grid.

A version bump of train_motif_figure.py — SAME architecture/heads/conditioning, only the DATA GRID changes
(timesteps_per_beat 4 -> 12, the finer 48th subdivision that resolves the triplet tax; notes/data_layer_v2_scope.md).
Surgical differences from train_motif_figure.py:
  * feature extractor `highres_v2` (timesteps_per_beat=12) + StepManiaParser.for_v2() + cache/samples_v3_48th
  * max_sequence_length 5400 (1440 at the 48th grid = only ~30s/song) + model max_len 5504 (pos-encoding cap)
  * WARM-START from the deployed gen_motif_full_fixed (a fine-tune to the finer grid, not from scratch), with the
    pos_encoding `pe` buffer FILTERED (2048 vs 5504 shape mismatch; the sinusoidal buffer is rebuilt fresh = correct)
  * bf16 AMP (the fit check: bf16 @ 3x context ~= fp32 @ 1x — mandatory for the throughput)
Onset loss is unchanged focal_bce; the 48th onset target is ~3x sparser (occupancy 20% vs 61%) so WATCH val_onset
and retune focal_gamma/add pos_weight only if onset recall collapses (one change at a time — don't pre-tune blind).

Usage (after the 48th cache is built, cache/samples_v3_48th):
    OMP_NUM_THREADS=4 python experiments/generation_typed/train_motif_figure_v2.py --data_dir data --audio_dir data \
        --epochs 20 --warmup_freeze 2 --batch_size 8 --section 64 --patience 3
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.stepmania_parser import StepManiaParser
from src.generation.typed_model import LayeredTypedChartGenerator, MOTIF_DIM
from src.generation.typed import NUM_PANELS, panels_to_pattern
from src.generation.motif_codebook import MotifBasis, figure_token_schedule
from src.generation.decode_harness import make_feature_extractor
sys.path.insert(0, str(PROJECT_ROOT / "experiments/generation_typed"))
from train_motif_local import local_motif_targets, focal_bce, focal_ce
from train_motif_figure import collect_typed  # grid-agnostic: reads ds.parser (for_v2) + sample audio

WARM_CKPT = "checkpoints/gen_motif_full_fixed/best_val.pt"   # the DEPLOYED model — fine-tune it to the finer grid
MOTIF_BASIS = PROJECT_ROOT / "cache/motif_basis.npz"
V2_MSL = 5400          # 130s @ 200BPM * 12/beat; the cache is keyed at this length
V2_MAX_LEN = 5504      # model positional-encoding capacity (>= V2_MSL)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=4, help='FITTED for the 12GB 3060 at T=3072 (B4=7.2GB; '
                   'B6=10.8GB tight; B8 OOMs). The training-shaped O(T^2) decoder attention is the limiter.')
    p.add_argument('--max_train_len', type=int, default=3072, help='per-sample training cap = 256 beats at the '
                   '48th grid = v1 gen_motif_figure coverage (its 1024 frames @ 4/beat). NOT 4608 — same MUSICAL '
                   'span, quarters the attention memory, keeps the retrain ~5min/epoch.')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_freeze', type=int, default=2)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--section', type=int, default=64)
    p.add_argument('--checkpoint_dir', default='checkpoints/gen_motif_v2_48th')
    p.add_argument('--cache_dir', default='cache/samples_v3_48th')
    p.add_argument('--radar_drop', type=float, default=0.15)
    p.add_argument('--motif_drop', type=float, default=0.30)
    p.add_argument('--figure_drop', type=float, default=0.15)
    p.add_argument('--amp', dest='amp', action='store_true', default=True, help='bf16 mixed precision (default ON)')
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--warm_ckpt', default=WARM_CKPT, help='warm-start source ("" = train from scratch)')
    return p.parse_args()


def load_warm_start(model, ckpt_path):
    """Load the deployed checkpoint, FILTERING the pos_encoding `pe` buffer (shape mismatch 2048 vs 5504; the
    sinusoidal buffer is rebuilt fresh at the larger size = correct). Returns cleanly or asserts on a real mismatch."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    dropped = [k for k in sd if k.endswith('pos_encoding.pe')]
    for k in dropped:
        sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # the only 'missing' keys should be the pe buffers we dropped (the model has its own fresh ones)
    non_pe_missing = [k for k in missing if not k.endswith('pos_encoding.pe')]
    print(f"warm-start {ckpt_path}: dropped pe={dropped}; fresh(non-pe)={non_pe_missing}; unexpected={unexpected}")
    assert not non_pe_missing and not unexpected, "warm-start mismatch beyond the pos_encoding buffers"


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
    basis = MotifBasis.load(MOTIF_BASIS); assert basis.K == MOTIF_DIM

    chart_files = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(chart_files, random_state=args.seed)

    spec = make_feature_extractor("highres_v2")   # timesteps_per_beat=12, 42-dim
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=V2_MSL,
                                          feature_extractor=spec.extractor, cache_dir=args.cache_dir,
                                          parser=StepManiaParser.for_v2())   # 48th grid, round quantization
    print("Warming caches (cache/samples_v3_48th)..."); train_ds.warm_cache(show_progress=True); val_ds.warm_cache(show_progress=True)
    print(f"Collecting typed samples (48th grid, section={args.section})...")
    train = collect_typed(train_ds, args.max_train_len, basis, args.section)
    val = collect_typed(val_ds, args.max_train_len, basis, args.section)
    print(f"train={len(train)} val={len(val)}")
    audio_dim = train[0]['audio'].shape[1]; assert audio_dim == 42

    try:
        import mlflow; mlflow.set_experiment("stepmania-chart-generator"); mlflow_on = True
    except ImportError:
        mlflow_on = False

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2,
                                       max_len=V2_MAX_LEN).to(device)
    if args.warm_ckpt:
        load_warm_start(model, args.warm_ckpt)
    else:
        print("training v2 from SCRATCH (no warm-start)")
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
        mlflow.start_run(run_name="gen-motif-v2-48th")
        mlflow.log_params({'grid': '48th', 'timesteps_per_beat': 12, 'epochs': args.epochs, 'patience': args.patience,
                           'lr': args.lr, 'warm': args.warm_ckpt or 'scratch', 'amp': args.amp,
                           'max_train_len': args.max_train_len, 'batch_size': args.batch_size, 'section': args.section})

    amp_ctx = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp and device.type == 'cuda'))
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
            with amp_ctx():
                ol, pat_l, typ_l = model(audio, states, diff, mask, radar=cr, motif=cm, figure=cfg_)
                o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
            (o + p + t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tr[0] += o.item(); tr[1] += p.item(); tr[2] += t.item(); nb += 1
        model.eval(); v = [0.0, 0.0, 0.0]; vnb = 0
        with torch.no_grad():
            for batch in batches(val, args.batch_size, False):
                audio, states, mask, diff, pat_t, typ_t, active, radar, motif, figure = to_tensors(batch)
                with amp_ctx():
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
                        'grid': '48th', 'timesteps_per_beat': 12, 'max_len': V2_MAX_LEN, 'section': args.section}, best_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stopping: no val improvement for {args.patience} epochs (best {best:.4f})"); break

    print(f"\nbest val_total={best:.4f} -> {best_path}")
    if mlflow_on:
        mlflow.log_metric('best_val_total', best); mlflow.end_run()
    print("Next: export a triplet song with the v2 model + for_v2 features and PLAY it (phase 6, the binding gate).")


if __name__ == '__main__':
    main()
