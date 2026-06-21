#!/usr/bin/env python3
"""
Choreography spot-check battery — geometric properties of the ARROW patterns (the which-arrow axis, H1,
which our timing-heavy metrics miss). Starter battery, each validated against structure-destroying nulls.

(1) SPACE-TIME MOVEMENT (velocity). Pad as 2D points L(-1,0) D(0,-1) U(0,1) R(1,0); each note frame ->
    centroid of active panels. For consecutive notes: dist = |c_i - c_{i+1}|, ioi = frame gap (16ths),
    velocity = dist/ioi. Pure distance is timing-blind (L,R on-beat == L,R with a pause == L on/off-beat
    in distance) — velocity separates them. The musicality signal is corr(dist, ioi): real charts give
    big moves more time (ergonomic coupling); random doesn't.
    Nulls: ioi-shuffled (permute gaps vs distances -> kills the coupling); panel-shuffled (random panels).

(2) PANEL TRANSITION MATRIX + SYMMETRY. P(next panel | cur) over single-note transitions (4x4). Real is
    ~L<->R symmetric and has ergonomic structure. Report symmetry score + KL(gen||real).
    Null: uniform/panel-shuffled.

Run on REAL vs nulls vs gen_stage1 (plain recommended decode).
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
COORDS = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]], dtype=np.float64)  # L, D, U, R
LR_SWAP = [3, 1, 2, 0]  # L<->R panel permutation


def note_starts(typed):
    t = np.asarray(typed); return (t == 1) | (t == 2) | (t == 4)  # (T,4)


def movement(ns):
    """Return (distances, iois) over consecutive note frames; centroid of active panels per frame."""
    rows = np.where(ns.any(1))[0]
    cents = [COORDS[ns[r]].mean(0) for r in rows]
    if len(rows) < 3:
        return np.array([]), np.array([])
    d = np.array([np.linalg.norm(cents[i + 1] - cents[i]) for i in range(len(rows) - 1)])
    g = np.diff(rows).astype(np.float64)
    return d, g


def vel_stats(d, g):
    if len(d) < 3:
        return dict(vel=np.nan, corr=np.nan)
    v = d / np.maximum(g, 1e-9)
    corr = np.corrcoef(d, g)[0, 1] if d.std() > 1e-9 and g.std() > 1e-9 else 0.0
    return dict(vel=float(v.mean()), corr=float(corr))


def transition_matrix(ns):
    """4x4 P(next|cur) over consecutive SINGLE-note frames."""
    rows = np.where(ns.any(1))[0]
    seq = [np.where(ns[r])[0] for r in rows]
    M = np.zeros((4, 4))
    prev = None
    for ps in seq:
        if len(ps) != 1:
            prev = None; continue
        p = ps[0]
        if prev is not None:
            M[prev, p] += 1
        prev = p
    return M


def sym_score(P):
    Pm = P[np.ix_(LR_SWAP, LR_SWAP)]  # mirror rows AND cols
    return 1.0 - 0.5 * np.abs(P - Pm).sum() / max(P.sum() * 0 + 1.0, 1.0) if P.sum() else np.nan


def kl(p, q):
    p = p + 1e-9; q = q + 1e-9; p /= p.sum(); q /= q.sum()
    return float((p * np.log(p / q)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=40); ap.add_argument('--max_len', type=int, default=1024)
    args = ap.parse_args()
    set_seed(42); rng = np.random.default_rng(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()

    SRC = ['REAL', 'REAL ioi-shuffled', 'REAL panel-shuffled', 'gen_stage1']
    mov = {s: [] for s in SRC}; tmats = {s: np.zeros((4, 4)) for s in SRC}
    seen, used = set(), 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 128: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        ns_r = note_starts(orig)
        if ns_r.any(1).sum() < 8: continue
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - float((orig != 0).any(1).mean())))
            g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                             type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                             pattern_temperature=0.7, no_jump_during_hold=True)[0].cpu().numpy()
        ns_g = note_starts(pair_holds(g))

        # movement
        d, gg = movement(ns_r); mov['REAL'].append(vel_stats(d, gg))
        if len(d) >= 3:  # ioi-shuffled null: permute gaps vs distances -> kills coupling
            mov['REAL ioi-shuffled'].append(vel_stats(d, rng.permutation(gg)))
        # panel-shuffled null: random panel per note frame (single notes)
        ns_ps = np.zeros_like(ns_r); rows = np.where(ns_r.any(1))[0]
        for r in rows: ns_ps[r, rng.integers(4)] = True
        d2, g2 = movement(ns_ps); mov['REAL panel-shuffled'].append(vel_stats(d2, g2))
        dg, ggg = movement(ns_g); mov['gen_stage1'].append(vel_stats(dg, ggg))
        # transition matrices
        tmats['REAL'] += transition_matrix(ns_r)
        tmats['REAL panel-shuffled'] += transition_matrix(ns_ps)
        tmats['gen_stage1'] += transition_matrix(ns_g)
        seen.add(meta['chart_file']); used += 1

    print(f"\n=== Choreography battery ({used} songs) ===\n")
    print("(1) SPACE-TIME MOVEMENT — velocity (dist/ioi) + corr(dist,ioi) [ergonomic coupling]")
    print(f"{'source':<22} {'mean_vel':>9} {'corr(d,ioi)':>12}")
    print("-" * 46)
    for s in SRC:
        ms = [m for m in mov[s] if not np.isnan(m['vel'])]
        if not ms: continue
        print(f"{s:<22} {np.mean([m['vel'] for m in ms]):>9.3f} {np.nanmean([m['corr'] for m in ms]):>12.3f}")
    print("\n  corr(d,ioi): REAL > 0 (big moves get more time) ; ioi-shuffled ~0 validates the coupling.\n")
    print("(2) PANEL TRANSITION MATRIX — L<->R symmetry + KL to REAL (single-note transitions)")
    Preal = tmats['REAL'] / max(tmats['REAL'].sum(), 1)
    print(f"{'source':<22} {'symmetry':>9} {'KL(.||REAL)':>12}")
    print("-" * 46)
    for s in ['REAL', 'gen_stage1', 'REAL panel-shuffled']:
        P = tmats[s] / max(tmats[s].sum(), 1)
        sy = 1.0 - 0.5 * np.abs(P - P[np.ix_(LR_SWAP, LR_SWAP)]).sum()
        print(f"{s:<22} {sy:>9.3f} {kl(P.flatten(), Preal.flatten()):>12.3f}")
    print("\n  REAL transition matrix (rows=from L,D,U,R -> cols=to):")
    for r in range(4):
        print("   " + "  ".join(f"{Preal[r,c]:.2f}" for c in range(4)))


if __name__ == '__main__':
    main()
