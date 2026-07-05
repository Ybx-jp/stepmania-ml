#!/usr/bin/env python3
"""M1a-v2 PROBE: does the more-expressive 48th grid (data-layer-v2) LIFT the audio->placement AUC curve?
(2026-07-05; lineages seq-onset-arc.md [the ~0.65 audio cap] x meter-grid-arc.md [the 48th-grid build])

CONTEXT — why this probe exists. The seq-onset arc established a WALL: fine note placement is a chart-sequence
PRIOR, not in the audio. Its floor number is the AUDIO onset head's 16th-localization AUC, capped ~0.65 across
FOUR independent measurements (0.649 06-22 / 0.656 06-28 C0 / 0.624 M1a frozen-h) vs a note-context CEILING ~0.89.
BUT every one of those was measured on the HARD-4/4 DUPLE-16th grid (t%4), where triplet/compound content is
FLOORED to the nearest 16th (the confirmed triplet tax, meter-grid-arc.md). HYPOTHESIS (the user's): part of that
0.65 cap is a GRID ARTIFACT — the audio->placement map was scored against a target that MIS-QUANTIZED triplets, so
audio was penalized for "misplacing" notes the grid itself had displaced. On the data-layer-v2 48th grid (12/beat,
StepManiaParser.for_v2 + highres_v2 beat-synchronous features) triplets resolve EXACTLY -> the target is faithful,
so audio may predict placement better. This is a CROSS-ARC test: seq-onset's cap x data-layer-v2's grid fix.

THIS IS A NEW FILE — the v1 probe_seqcontext_frozenh.py + its numbers (audio 0.624 / both_real 0.892) are the
REFERENCE and are left UNTOUCHED. Reuses the v1 probe's arm nets (HRead/HReadConv/Probe) so the controls are
byte-identical; only the DATA GRID + the fine-subdivision AUC bands change.

FAIR-COMPARISON DISCIPLINE (experiment-design skill):
  * Raw AUC is NOT comparable across grids (different base rate / frame population). The grid-robust readout is the
    ANCHORED BRACKET: audio floor vs note-context ceiling ON THE SAME GRID, reported as the fraction of the
    chance->ceiling gap the audio reaches. A higher fraction on v2 = the grid genuinely lifted audio's reach.
  * The audio + both_real arms are FROM-SCRATCH probes (no decoder h) -> checkpoint-INDEPENDENT: the audio-vs-
    ceiling bracket isolates (grid + beat-sync), NOT the v2 retrain. frozen_h (v2 ckpt) is the build-sizing bonus.
  * Fine-AUC is split into THREE position bands on the 48th grid (phase_band_positions(12)):
      duple16  = t%12 in {3,9}   -> the DIRECT analog of v1's {1,3} 16th-AUC (apples-to-apples vs 0.624/0.892)
      triplet  = t%12 in {4,8}   -> the NEW positions the 16th grid could not represent (the crux of the fix)
      offbeat  = t%12 != 0       -> the aggregate
  * Stratify (Rule 12): the grid fix should lift audio MOST on triplet-heavy songs, ~nothing on duple-only songs.
    A median split on per-song triplet occupancy tests that the effect is triplet-LOCALIZED (causal), not global.
BOUNDARY (Rule 9/10): like v1 M1a this settles REPRESENTATION (is placement in audio / in h), NOT drift. h is
teacher-forced on REAL notes = the upper bound a frozen readout could see; gen-time drift is a separate gate.

  python experiments/generation_typed/probe_seqcontext_frozenh_v2.py --ckpt checkpoints/gen_motif_v2_48th_cont/best_val.pt
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.stepmania_parser import StepManiaParser
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.decode_harness import make_feature_extractor
from src.generation.decode_defaults import phase_band_positions
# Reuse the EXACT v1 arm nets + helpers (grid-agnostic) so the controls are identical (fair comparison).
from probe_seqcontext_frozenh import (HRead, HReadConv, extract_typed, load_or_extract,
                                      precompute_h, batches, AD, NP, DMODEL)
from diag_seqcontext_probe import Probe, auc

SUBDIV = 12                       # the data-layer-v2 48th grid (timesteps_per_beat)
V2_MSL = 5400                     # matches train_motif_figure_v2.V2_MSL (cache is keyed at this length)
V2_MAX_LEN = 5504                 # model positional-encoding capacity for the v2 checkpoint


def fine_masks(T, subdiv=SUBDIV):
    """The three within-beat position bands on the `subdiv`-per-beat grid (canonical vocabulary via
    phase_band_positions). duple16 = the direct v1-{1,3} analog; triplet = the NEW positions the 16th grid
    floored; offbeat = everything off the strong beat. Returns dict name -> boolean (T,) frame mask."""
    t = np.arange(T)
    _e8, (s16a, s16b) = phase_band_positions(subdiv)       # subdiv=12 -> (6, (3, 9))
    ph = t % subdiv
    return {
        "duple16": (ph == s16a) | (ph == s16b),            # {3,9}
        "triplet": (ph == subdiv // 3) | (ph == 2 * subdiv // 3),  # {4,8} = the 8th-triplet positions
        "offbeat": ph != 0,
    }


def song_triplet_occupancy(song, subdiv=SUBDIV):
    """Fraction of a song's NOTE-bearing frames that land on a triplet subdivision (t%subdiv in {2,4,8,10}) — the
    positions the 16th grid could not represent. The stratification covariate (Rule 12)."""
    typed = song["typed"][: song["T"]]
    onset = (typed != 0).any(-1)
    if onset.sum() == 0:
        return 0.0
    ph = np.arange(song["T"]) % subdiv
    trip = np.isin(ph, [2, 4, 8, 10])
    return float((onset & trip).sum()) / float(onset.sum())


def train_eval_v2(kind, train, val, device, epochs, bs, lr, pw):
    """Train an arm (audio / both / frozen_h / frozen_h_conv), return (onset_auc, {band: fine_auc}) plus the
    per-frame (P, Y) on val so the caller can compute stratified AUCs. Mirrors v1 train_eval but with the
    parameterized subdiv bands instead of the hard-coded t%4 16th mask."""
    set_seed(42)
    if kind == "frozen_h":
        m = HRead(DMODEL).to(device); fwd = lambda X, Np, H: m(H)
    elif kind == "frozen_h_conv":
        m = HReadConv(DMODEL).to(device); fwd = lambda X, Np, H: m(H)
    else:
        m = Probe(kind).to(device); fwd = lambda X, Np, H: m(X, Np)
    opt = torch.optim.Adam(m.parameters(), lr=lr); rng = np.random.default_rng(0)
    for _ in range(epochs):
        m.train()
        for X, Np, H, Y, M in batches(train, bs, rng, True, device):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(X, Np, H)[M], Y[M], pos_weight=pw)
            loss.backward(); opt.step()
    m.eval(); ps, ys, phs = [], [], []
    with torch.no_grad():
        for X, Np, H, Y, M in batches(val, bs, rng, False, device):
            p = torch.sigmoid(fwd(X, Np, H)).cpu().numpy(); B, T = Y.shape
            ph = (np.arange(T) % SUBDIV)[None].repeat(B, 0); mm = M.cpu().numpy()
            ps.append(p[mm]); ys.append(Y.cpu().numpy()[mm]); phs.append(ph[mm])
    P = np.concatenate(ps); Yv = np.concatenate(ys); PH = np.concatenate(phs)
    _e8, (s16a, s16b) = phase_band_positions(SUBDIV)
    bands = {
        "duple16": np.isin(PH, [s16a, s16b]),
        "triplet": np.isin(PH, [SUBDIV // 3, 2 * SUBDIV // 3]),
        "offbeat": PH != 0,
    }
    fine = {name: auc(P[msk], Yv[msk]) for name, msk in bands.items()}
    return auc(P, Yv), fine, (P, Yv, PH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_v2_48th_cont/best_val.pt",
                    help="the v2 (48th-grid) checkpoint for the frozen_h arm; audio/both arms are ckpt-independent")
    ap.add_argument("--max_train", type=int, default=800)
    ap.add_argument("--max_val", type=int, default=400)
    ap.add_argument("--max_len", type=int, default=3072, help="256 beats @ 48th = the v1 1024@16th musical span")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--precompute_bs", type=int, default=2, help="teacher-forced h precompute batch; O(T^2) decoder "
                    "attention at the 48th grid T=3072 OOMs the 12GB 3060 above ~2 (v2 train itself used B4 w/ grad)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reparse", action="store_true", help="ignore the npz caches and re-extract from data/")
    args = ap.parse_args()
    set_seed(42); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cache = PROJECT_ROOT / "cache/seqctx_frozenh_v2_train.npz"
    val_cache = PROJECT_ROOT / "cache/seqctx_frozenh_v2_val.npz"
    if args.reparse:
        for c in (train_cache, val_cache):
            if c.exists():
                c.unlink()

    def make_datasets():
        cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
        tf, vf, _ = create_data_splits(cf, random_state=42)
        spec = make_feature_extractor("highres_v2")           # timesteps_per_beat=12, beat_sync, 42-dim
        # cache_dir=None: re-parse fresh (avoid the index-cache footgun [[dataset-cache-footgun]]); the probe
        # keeps its OWN identity-safe npz caches below.
        return create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                               max_sequence_length=V2_MSL, feature_extractor=spec.extractor,
                               cache_dir=None, parser=StepManiaParser.for_v2())

    if train_cache.exists() and val_cache.exists():
        print("both v2 feature caches present -> skipping dataset re-parse", flush=True)
        tr_ds = va_ds = None
    else:
        tr_ds, va_ds, _ = make_datasets()

    # frozen_h arm needs the v2 decoder h -> build the model at the v2 positional-encoding capacity.
    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=DMODEL, num_layers=4, onset_layers=2,
                                       max_len=V2_MAX_LEN).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False); model.eval()
    print(f"v2 checkpoint: {args.ckpt}", flush=True)

    print("extracting REAL train (all difficulties, typed states, 48th grid)...", flush=True)
    train = load_or_extract(tr_ds, args.max_train, args.max_len, train_cache, hard_only=False)
    print("extracting REAL val (Hard only, typed states, 48th grid)...", flush=True)
    val = load_or_extract(va_ds, args.max_val, args.max_len, val_cache, hard_only=True)
    print("computing frozen v2-decoder h (teacher-forced) for train+val...", flush=True)
    precompute_h(model, train, device, args.max_len, bs=args.precompute_bs)
    precompute_h(model, val, device, args.max_len, bs=args.precompute_bs)

    posrate = np.mean([(s["typed"] != 0).any(-1).mean() for s in train])
    pw = torch.tensor((1 - posrate) / posrate, device=device)
    trip = np.array([song_triplet_occupancy(s) for s in val])
    med = float(np.median(trip))
    print(f"\ntrain={len(train)} | eval={len(val)} (Hard) | onset-rate {posrate:.3f} "
          f"| val triplet-occupancy: median={med:.3f} range [{trip.min():.3f},{trip.max():.3f}]\n", flush=True)

    print(f"  {'predictor':<14} {'onset-AUC':>9} {'duple16':>9} {'triplet':>9} {'offbeat':>9}", flush=True)
    res = {}
    for kind in ["audio", "both", "frozen_h", "frozen_h_conv"]:
        oa, fine, pyph = train_eval_v2(kind, train, val, device, args.epochs, args.bs, args.lr, pw)
        res[kind] = dict(onset=oa, fine=fine, pyph=pyph)
        label = "both_real" if kind == "both" else kind
        print(f"  {label:<14} {oa:>9.3f} {fine['duple16']:>9.3f} {fine['triplet']:>9.3f} {fine['offbeat']:>9.3f}",
              flush=True)

    a, cr = res["audio"]["fine"], res["both"]["fine"]
    print("\n  === BRACKET (grid-robust): audio reach as % of the chance(0.5)->note-context-ceiling gap ===", flush=True)
    print("  v1 REFERENCE (16th grid, probe_seqcontext_frozenh.py): audio 0.624 / both_real 0.892 "
          "-> audio reaches (0.624-0.5)/(0.892-0.5) = 32% of the placement gap.", flush=True)
    for band in ["duple16", "triplet", "offbeat"]:
        gap = max(cr[band] - 0.5, 1e-6); frac = 100 * (a[band] - 0.5) / gap
        ceilclear = cr[band] - a[band]
        flag = "" if ceilclear > 0.05 else "  !! ceiling did not clear audio -> UNDERPOWERED, do not interpret"
        print(f"    {band:<9} audio={a[band]:.3f}  ceiling={cr[band]:.3f}  -> audio reaches {frac:.0f}% of gap{flag}",
              flush=True)
    print("  READ: duple16 % > v1's 32% => the finer grid lifted audio's reach even on the OLD positions;", flush=True)
    print("        triplet ceiling >> audio => triplets are (like 16ths) a chart PRIOR; triplet audio-reach is the", flush=True)
    print("        NEW datum the 16th grid could not measure at all.", flush=True)

    # Stratified (Rule 12): does the lift concentrate on triplet-heavy songs?
    # batches(val, shuffle=False) flattens (B,T)->frames in val song order (row-major, mask-selected s['T'] frames
    # per song), IDENTICALLY for every arm -> a frame->song map reconstructs the strata on the stored (P,Y,PH).
    print("\n  === STRATIFIED by per-song triplet occupancy (median split) — is the lift triplet-LOCALIZED? ===",
          flush=True)
    Pa, Ya, PHa = res["audio"]["pyph"]; Pc, Yc, _ = res["both"]["pyph"]
    frame_song = np.concatenate([np.full(s["T"], j, int) for j, s in enumerate(val)])
    assert frame_song.shape[0] == Pa.shape[0] == Pc.shape[0], "frame->song map misaligned with flattened frames"
    hi = trip > med
    print(f"    triplet-heavy songs: {int(hi.sum())} | duple songs: {int((~hi).sum())} (median occ {med:.3f})",
          flush=True)
    print(f"    {'stratum':<14} {'band':<9} {'audio':>7} {'ceiling':>8} {'gap%':>6}", flush=True)
    for lbl, sel in [("triplet-heavy", hi), ("duple", ~hi)]:
        fsel = sel[frame_song]
        for band, pos in [("duple16", [SUBDIV // 4, 3 * SUBDIV // 4]),
                          ("triplet", [SUBDIV // 3, 2 * SUBDIV // 3])]:
            msk = fsel & np.isin(PHa, pos)
            if msk.sum() < 50 or Ya[msk].sum() < 5 or Ya[msk].sum() == msk.sum():
                print(f"    {lbl:<14} {band:<9} {'n/a (too few positives to score)':>0}", flush=True)
                continue
            aud = auc(Pa[msk], Ya[msk]); cei = auc(Pc[msk], Yc[msk])
            gap = max(cei - 0.5, 1e-6)
            print(f"    {lbl:<14} {band:<9} {aud:>7.3f} {cei:>8.3f} {100 * (aud - 0.5) / gap:>5.0f}%", flush=True)
    print("    READ: if audio 'triplet' reach is HIGHER in triplet-heavy than duple songs, the grid fix is doing", flush=True)
    print("          real work exactly where triplets live (causal, triplet-localized) — not a global shift.", flush=True)

    print("\n  BOUNDARY: REPRESENTATION only (is placement in audio / in the v2 h), NOT drift. Reference v1 numbers "
          "left untouched in probe_seqcontext_frozenh.py / notes/onset_frozenh_findings.md.", flush=True)


if __name__ == "__main__":
    main()
