#!/usr/bin/env python3
"""M1b-4 PLACEMENT QUALITY: does the SS onset head place 16ths in the RIGHT spots when it FREE-RUNS? (2026-06-29)
Lineage seq-onset-arc.md. M1b-3 (`probe_seqonset_ss.py`) broke the DRIFT wall: the note-dropout-SS head free-runs at
real DENSITY + real RUN-LENGTH (tau≈0.56). But density/run is STABILITY, not musicality (onset_ss_findings.md §Boundary,
exp-design Rule 9). This probe asks the binding QUALITY question: at the operating point, does the head's free-run
ranking put onsets on the musically-right 16th frames — toward the M1a teacher-forced ceiling (16th-AUC 0.892) or only
the audio-only floor (~0.62)?

METRIC = 16th-AUC: among 16th-OFFBEAT frames (t%4∈{1,3}), does the predicted onset PROBABILITY rank the frames that
carry a REAL onset above those that don't (the EXACT M1a metric, `diag_seqcontext_probe.auc`, pooled over val frames).
AUC is threshold-free; for FREE-run the per-frame logit is the head's score along the REALIZED tau-thresholded
trajectory (logit at t conditioned on the head's OWN notes fed for <t) — i.e. quality UNDER drift, not teacher-forced.

THREE ARMS, SAME saved SS head (cache/seqonset_ss_head.pt), SAME Hard-val songs, predictions POOLED:
  FLOOR    : the DEPLOYED audio-only onset head (`model.onset_logits`, native) — the head this would REPLACE   ~0.62
  CEILING  : the SS head on TEACHER-FORCED h (real-note context) — POSITIVE CONTROL (Rule 11), must reproduce M1a ~0.89
  FREE     : the SS head FREE-RUNNING at tau≈0.56 (own-note h, the same incremental decode as generate())  <- MEASUREMENT
plus, at the operating tau, the REALIZED-binary precision/recall/F1 on 16th frames (grounds the AUC in the artifact,
exp-design Rule 8) and a first/second-HALF AUC split (does ranking degrade late in the rollout = a drift tell).

READ:
  FREE 16th-AUC ≈ CEILING (recovers most of the ceiling−floor gap) -> placement QUALITY survives drift -> the win is
    real; proceed to generate() wiring + by-ear (the binding gate).
  FREE 16th-AUC ≈ FLOOR -> the head free-runs at real density but places onsets no better than audio-only -> the
    density/run stability is HOLLOW for quality; the chart-context advantage did NOT survive drift -> more SS /
    audio-anchor, or BANK the drift-break and fall to the nearest-shippable.
CONTROL GATE (Rule 11): if CEILING does not clear FLOOR by >0.05, the run is underpowered / mis-set -> DO NOT interpret
FREE (raise --n_eval / re-extract). BOUNDARY: governors OFF, tap-only greedy pattern (the pure onset-from-h regime, =
the M1b/M1b-3 gate); this is a RANKING probe, NOT a playtest — by-ear stays the binding gate (exp-design Rule 8).

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqonset_placement.py --load_head
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from pathlib import Path
import numpy as np, torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from probe_seqcontext_frozenh import AD, DMODEL, CKPT, HReadConv, precompute_h, load_or_extract
from probe_seqonset_rollout import rollout, _runlen, train_head
from diag_seqcontext_probe import auc


def _i16(T):
    """16th-OFFBEAT frame mask on a 16th grid: t%4∈{1,3} (quarter=0, 8th=2). Matches M1a/conditioning-mechanics §6."""
    t = np.arange(T)
    return (t % 4 == 1) | (t % 4 == 3)


def _prf(pred, real):
    """precision/recall/F1 of a BOOLEAN prediction vs BOOLEAN truth (realized-onset placement at the operating tau)."""
    tp = float((pred & real).sum()); pp = float(pred.sum()); rp = float(real.sum())
    p = tp / pp if pp else 0.0; r = tp / rp if rp else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


@torch.no_grad()
def floor_probs(model, audio_np, diff, T, device):
    """The DEPLOYED audio-only onset head (native mode, radar/style/motif=None — matches gen_c0 / the TF decode_hidden
    cond). This is the placement FLOOR: the head the seq-onset head would replace. Returns sigmoid probs (T,)."""
    audio_t = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    memory = model.encode_audio(audio_t)
    ol = model.onset_logits(memory, torch.tensor([diff], device=device))   # (1,T) native audio-only
    return torch.sigmoid(ol)[0, :T].cpu().numpy()


def pooled_auc(probs, onset, masks):
    """Pool per-song (prob, real-onset, 16th-mask) and return (onset-AUC over all frames, 16th-AUC). Mirrors M1a:
    concatenate then one AUC, so the number is directly comparable to the 0.624 floor / 0.892 ceiling."""
    P = np.concatenate(probs); Y = np.concatenate(onset); I = np.concatenate(masks).astype(bool)
    return auc(P, Y), auc(P[I], Y[I])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load_head', action='store_true', help='load cache/seqonset_ss_head.pt (the M1b-3 SS head)')
    ap.add_argument('--cap', type=int, default=512); ap.add_argument('--n_eval', type=int, default=12)
    ap.add_argument('--tau', type=float, default=0.56, help='the M1b-3 operating point (free_d≈real 0.27)')
    ap.add_argument('--robust_taus', type=str, default='0.50,0.55', help='extra plateau taus for a ranking-robustness check')
    ap.add_argument('--max_train', type=int, default=800, help='train songs for the pure-TF ceiling head')
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vc = PROJECT_ROOT / "cache/seqctx_frozenh_val.npz"
    assert vc.exists(), "run probe_seqcontext_frozenh.py first (builds the val cache)"
    head_path = PROJECT_ROOT / "cache/seqonset_ss_head.pt"
    assert head_path.exists(), "run probe_seqonset_ss.py first (trains + saves the SS head)"
    val = load_or_extract(None, 400, args.cap, vc, hard_only=True)[:args.n_eval]

    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=DMODEL, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    set_seed(42); head = HReadConv(DMODEL).to(device)
    head.load_state_dict(torch.load(head_path, map_location=device)); head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    print(f"loaded SS head from {head_path} | {len(val)} Hard val songs (cap {args.cap})\n", flush=True)

    # TRUE representation CEILING: a PURE teacher-forced conv head (NO scheduled-sampling dropout) = the M1a 0.892
    # arm, measured on THIS set. The SS head's TF pass is NOT this ceiling (SS trades TF accuracy for drift-
    # robustness) — so without this arm the bracket is invalid (the control failure on the first run). Cached.
    tc_path = PROJECT_ROOT / "cache/seqonset_tfceiling_head.pt"
    set_seed(42); head_tc = HReadConv(DMODEL).to(device)
    if tc_path.exists():
        head_tc.load_state_dict(torch.load(tc_path, map_location=device)); head_tc.eval()
        print(f"loaded pure-TF ceiling head from {tc_path}", flush=True)
    else:
        tcache = PROJECT_ROOT / "cache/seqctx_frozenh_train.npz"; assert tcache.exists(), "train cache missing"
        train = load_or_extract(None, args.max_train, args.cap, tcache, hard_only=False)
        precompute_h(model, train, device, args.cap)
        posrate = np.mean([(s['typed'] != 0).any(-1).mean() for s in train])
        pw = torch.tensor((1 - posrate) / posrate, device=device)
        print(f"training pure-TF ceiling head ({len(train)} songs, {args.epochs} ep)...", flush=True)
        head_tc = train_head(model, train, device, args.epochs, args.bs, args.lr, pw)
        torch.save(head_tc.state_dict(), tc_path); print(f"saved -> {tc_path}", flush=True)
    for p in head_tc.parameters():
        p.requires_grad_(False)

    precompute_h(model, val, device, args.cap)                              # teacher-forced (real-note) h for the CEILING arms

    # accumulators (pooled over songs)
    fl_p, tc_p, ss_p, fr_p, ons, m16 = [], [], [], [], [], []
    fr_half1_p, fr_half1_y, fr_half2_p, fr_half2_y = [], [], [], []
    bin_pred16, bin_real16, free_d, real_d = [], [], [], []
    print(f"  per-song free-run @ tau={args.tau} (greedy pattern, governors OFF — the M1b-3 regime):", flush=True)
    print(f"  {'song':>4} {'T':>5} {'real_d':>7} {'free_d':>7} {'free_run':>8}", flush=True)
    for i, s in enumerate(val):
        T = min(s['T'], args.cap)
        real_on = (s['typed'][:T] != 0).any(-1)
        i16 = _i16(T)
        h_t = torch.from_numpy(s['h'][:T]).unsqueeze(0).to(device)
        # FLOOR: deployed audio-only onset head
        fl_p.append(floor_probs(model, s['audio'][:T], s['diff'], T, device))
        # CEILING (control): PURE-TF conv head on real-note h (the M1a 0.892 representation upper bound)
        tc_p.append(torch.sigmoid(head_tc(h_t)[0]).cpu().numpy())
        # SS head on teacher-forced h (secondary: how much TF accuracy SS training sacrificed)
        ss_p.append(torch.sigmoid(head(h_t)[0]).cpu().numpy())
        # MEASUREMENT: SS head FREE-RUN at the operating tau; capture logits + realized binary
        on, lg = rollout(model, head, s['audio'][:T], s['diff'], T, args.tau, device, collect_logits=True)
        fp = 1.0 / (1.0 + np.exp(-lg))                                      # sigmoid(free-run logit)
        fr_p.append(fp); ons.append(real_on); m16.append(i16)
        # halves split (does ranking degrade late = a drift tell)
        half = T // 2; h1, h2 = i16.copy(), i16.copy(); h1[half:] = False; h2[:half] = False
        fr_half1_p.append(fp[h1]); fr_half1_y.append(real_on[h1])
        fr_half2_p.append(fp[h2]); fr_half2_y.append(real_on[h2])
        # realized-binary placement on 16th frames at the operating tau
        bin_pred16.append(on[i16]); bin_real16.append(real_on[i16])
        free_d.append(float(on.mean())); real_d.append(float(real_on.mean()))
        print(f"  {i:>4} {T:>5} {float(real_on.mean()):>7.3f} {float(on.mean()):>7.3f} {_runlen(on):>8.2f}", flush=True)

    fl_on, fl_16 = pooled_auc(fl_p, ons, m16)
    tc_on, tc_16 = pooled_auc(tc_p, ons, m16)
    ss_on, ss_16 = pooled_auc(ss_p, ons, m16)
    fr_on, fr_16 = pooled_auc(fr_p, ons, m16)
    rd, fd = float(np.mean(real_d)), float(np.mean(free_d))

    print(f"\n  16th-AUC (pooled, M1a reference: audio-probe floor 0.624 / conv ceiling 0.892):", flush=True)
    print(f"  {'arm':<30} {'onset-AUC':>10} {'16th-AUC':>10}", flush=True)
    print(f"  {'FLOOR (deployed audio head)':<30} {fl_on:>10.3f} {fl_16:>10.3f}", flush=True)
    print(f"  {'CEILING (pure-TF conv, real h)':<30} {tc_on:>10.3f} {tc_16:>10.3f}   <- POSITIVE CONTROL (must >> floor)", flush=True)
    print(f"  {'SS head, TF h (sacrifice ref)':<30} {ss_on:>10.3f} {ss_16:>10.3f}   (secondary: SS TF vs pure-TF ceiling)", flush=True)
    print(f"  {'FREE-RUN (SS head, own h)':<30} {fr_on:>10.3f} {fr_16:>10.3f}   <- MEASUREMENT  (tau={args.tau}, free_d {fd:.3f} vs real {rd:.3f})", flush=True)

    # CONTROL GATE (Rule 11): the metric must be able to MOVE (ceiling clears floor) before FREE is interpretable
    gap = tc_16 - fl_16
    if gap <= 0.05:
        print(f"\n  !! CONTROL FAILED: CEILING 16th-AUC {tc_16:.3f} did not clear FLOOR {fl_16:.3f} by >0.05 -> the metric", flush=True)
        print(f"     can't separate placement here (underpowered / mis-set). DO NOT interpret FREE-run. Raise --n_eval.", flush=True)
        return
    rec = (fr_16 - fl_16) / gap
    print(f"\n  CONTROL ok: CEILING {tc_16:.3f} >> FLOOR {fl_16:.3f} (gap {gap:.3f}). FREE-run recovers {100 * rec:.0f}% of the", flush=True)
    print(f"  ceiling-floor placement gap under its OWN-context drift.   (SS-TF sacrifice: {tc_16 - ss_16:+.3f} vs pure-TF)", flush=True)

    # realized-binary placement at the operating tau (grounds the AUC: of the 16ths it FIRES, are they real?)
    pr, rc, f1 = _prf(np.concatenate(bin_pred16), np.concatenate(bin_real16))
    print(f"  realized 16th placement @ tau={args.tau}: precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}", flush=True)

    # halves split — late-rollout ranking drift
    a1 = auc(np.concatenate(fr_half1_p), np.concatenate(fr_half1_y))
    a2 = auc(np.concatenate(fr_half2_p), np.concatenate(fr_half2_y))
    print(f"  FREE 16th-AUC halves: first {a1:.3f}  second {a2:.3f}  (Δ {a2 - a1:+.3f}; <0 = ranking degrades late = drift tell)", flush=True)

    # ranking robustness across the plateau (logits depend on the realized trajectory -> on tau)
    extra = [float(x) for x in args.robust_taus.split(',') if x.strip()]
    if extra:
        print(f"\n  RANKING ROBUSTNESS (free-run 16th-AUC at other plateau taus):", flush=True)
        for tau in extra:
            ps, ys, ms = [], [], []
            for s in val:
                T = min(s['T'], args.cap)
                _on, lg = rollout(model, head, s['audio'][:T], s['diff'], T, tau, device, collect_logits=True)
                ps.append(1.0 / (1.0 + np.exp(-lg))); ys.append((s['typed'][:T] != 0).any(-1)); ms.append(_i16(T))
            _o, a16 = pooled_auc(ps, ys, ms)
            print(f"    tau={tau:.2f}: 16th-AUC {a16:.3f}", flush=True)

    # VERDICT
    print(f"\n  VERDICT:", flush=True)
    if rec >= 0.7:
        print(f"  PLACEMENT QUALITY SURVIVES DRIFT (recovered {100*rec:.0f}% ≥70%, FREE {fr_16:.3f} ≈ ceiling {tc_16:.3f}).", flush=True)
        print(f"  The free-run head places 16ths in the right spots, not just at the right density -> the win is real ->", flush=True)
        print(f"  PROCEED: wire into generate() (opt-in, compose with the stamina gate) + by-ear (the binding gate).", flush=True)
    elif rec <= 0.2:
        print(f"  PLACEMENT HOLLOW (recovered {100*rec:.0f}% ≤20%, FREE {fr_16:.3f} ≈ floor {fl_16:.3f}).", flush=True)
        print(f"  The head free-runs at real density but places onsets no better than audio-only -> the M1b-3 stability", flush=True)
        print(f"  did NOT carry placement quality under drift -> escalate (more SS / audio-anchor) or BANK + nearest-shippable.", flush=True)
    else:
        print(f"  PARTIAL (recovered {100*rec:.0f}%). Free-run keeps SOME chart-context placement but loses some to drift.", flush=True)
        print(f"  By-ear (Rule 8) decides whether the partial gain is audible; weigh vs the nearest-shippable fallback.", flush=True)
    print(f"\n  BOUNDARY: RANKING under drift, governors OFF, greedy tap-only. NOT a playtest — by-ear is the binding gate.", flush=True)


if __name__ == '__main__':
    main()
