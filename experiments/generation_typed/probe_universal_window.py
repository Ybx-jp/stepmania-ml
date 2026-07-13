#!/usr/bin/env python3
"""Universal sub-train-length window probe (the RIGHT population, no AR decode).

Question (user's OPEN priority, HANDOFF 2026-07-11): short songs (T<=5400) get NO onset windowing today
(`onset_window=V2_MSL=5400`, fires only when T>window). But the v2 TRAINING length distribution tops out at
5128 with median 3120 and only ~31%/13% of songs reaching abs-PE position 3500/4000 -- so any song longer
than ~3500 already sits its END in the UNDER-TRAINED absolute-PE tail of the onset encoder (non-causal, reads
pos_encoding over ABS positions 0..T). Hypothesis: this causes broad short-song END-degeneration, and a
UNIVERSAL window (< the degeneration onset, so it fires on these songs) applied to ALL songs -- each tiled at
a well-trained LOCAL PE, song-end centered by the hangover pad -- fixes it.

This is the CHEAP DECISIVE test on the CORRECT population (exp-design Rule 5/6/11): the predecessor's
onset_window_sweep tested smaller-W on ONE long song's MIDDLES (wrong population). Here we read CACHED VAL
samples (already the deployed 42-dim highres_v2 features, frame-aligned to the REAL human chart = ground
truth), and measure tail (last 20%) vs body onset quality, single-pass (today's default) vs windowed.

Metrics per song-region, vs the human chart (Rule 5 ground truth):
  - onset AUC (p_onset ranks note-present frames above absent) -- ranking quality, peak-compression handled by...
  - p95 of p_onset -- peak compression ("mean holds, p95 regresses toward mean" is the documented tail failure)
  - recall@tau -- of real note frames, fraction that FIRE under the per-arm global tau (the DEPLOYED decision)
  - backbone Herfindahl -- phase-concentration of FIRED frames vs the HUMAN chart's, same region (smear check)

Bands: UNDER-TRAINED tail (len in [3800,5128]) = where the effect should live; CONTROL (len<3000) = tail still
well-trained, the window must be ~inert (Rule 4/11 -- don't 'fix' what isn't broken; confirm dynamic range).
"""
import warnings, os, sys, glob, argparse
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch
ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from scripts.generate import V2_CHECKPOINT, V2_CTX
from src.generation.decode_harness import load_generator, compute_tau, apply_phase_calib, MODEL_ARCH

SUBDIV = 12
CALIB = (0.0, 1.0)                       # the 16th-unlock; canonical
TRIPLET = {2, 4, 8, 10}; JITTER = {1, 5, 7, 11}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# arms: (label, window, hangover). window=None = today's single-pass default.
ARMS = [
    ("single-pass", None, 0),
    ("W3000",       3000, 1500),
    ("W3600",       3600, 1800),
    ("W4320",       4320, 2160),
]


def rank_auc(scores, labels):
    """Mann-Whitney AUC: P(score[pos] > score[neg]). No sklearn dep."""
    labels = labels.astype(bool)
    npos = labels.sum(); nneg = (~labels).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def herfindahl(phases, n):
    if len(phases) == 0:
        return np.nan
    h = np.array([(phases == k).sum() for k in range(n)], float) / len(phases)
    return (h ** 2).sum()


def process(model, sample, arms):
    L = int(sample['length'])
    audio = sample['audio'][:L].unsqueeze(0).to(device)          # (1,L,42) deployed features
    diff = sample['difficulty'].view(1).to(device)
    chart = sample['chart'][:L].numpy()                          # (L,4) real human chart
    present = ((chart == 1) | (chart == 2) | (chart == 4)).any(1)   # tap/hold-head/roll-head
    density = present.mean()
    if density < 1e-4:
        return None
    with torch.no_grad():
        memory = model.encode_audio(audio)
    pos = np.arange(L) / max(L - 1, 1)
    body_m = pos < 0.8; tail_m = pos >= 0.8
    hum_tail_H = herfindahl(np.nonzero(present & tail_m)[0] % SUBDIV, SUBDIV)
    out = {'L': L, 'density': density, 'hum_tail_H': hum_tail_H}
    for label, W, hang in arms:
        with torch.no_grad():
            ol = model.onset_logits(memory, diff, window=W, tail_hangover=hang,
                                    hop_frac=0.5, hangover_reflect=False)[0]
            ol = apply_phase_calib(ol, CALIB, SUBDIV)
            p = torch.sigmoid(ol).cpu().numpy()
        tau = compute_tau(p, density)                            # per-arm global tau, real-density (deployed fallback)
        fired = p >= tau
        rec = {}
        for rlabel, m in [('body', body_m), ('tail', tail_m)]:
            rec[f'auc_{rlabel}'] = rank_auc(p[m], present[m])
            rec[f'p95_{rlabel}'] = np.percentile(p[m], 95)
            pres_m = present & m
            rec[f'recall_{rlabel}'] = fired[pres_m].mean() if pres_m.any() else np.nan
            rec[f'H_{rlabel}'] = herfindahl(np.nonzero(fired & m)[0] % SUBDIV, SUBDIV)
        out[label] = rec
    return out


def summarize(results, arms, band_name):
    print(f"\n{'='*100}\nBAND: {band_name}  (n={len(results)} songs)\n{'='*100}")
    hum = np.nanmean([r['hum_tail_H'] for r in results])
    print(f"human-chart TAIL backbone Herfindahl (reference): {hum:.3f}   (higher=peaked backbone, 1/12=0.083 uniform smear)\n")
    hdr = (f"{'arm':12s} | {'AUC body':>8s} {'AUC tail':>8s} {'dAUC':>6s} | {'p95 body':>8s} {'p95 tail':>8s} "
           f"{'dp95':>6s} | {'rec body':>8s} {'rec tail':>8s} | {'H body':>6s} {'H tail':>6s}")
    print(hdr); print('-' * len(hdr))
    for label, _, _ in arms:
        def mean(k): return np.nanmean([r[label][k] for r in results])
        ab, at = mean('auc_body'), mean('auc_tail')
        pb, pt = mean('p95_body'), mean('p95_tail')
        rb, rt = mean('recall_body'), mean('recall_tail')
        hb, ht = mean('H_body'), mean('H_tail')
        print(f"{label:12s} | {ab:8.3f} {at:8.3f} {at-ab:+6.3f} | {pb:8.3f} {pt:8.3f} {pt-pb:+6.3f} | "
              f"{rb:8.3f} {rt:8.3f} | {hb:6.3f} {ht:6.3f}")
    print("\n  dAUC/dp95 = tail MINUS body (negative = tail degraded). recall = frac of REAL notes that fire under tau.")
    print("  Read: does single-pass show tail<body (degeneration)? Does a window CLOSE the tail gap toward body?")
    print(f"        Does the windowed tail Herfindahl move toward the human {hum:.3f}?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60, help='songs per band')
    args = ap.parse_args()

    print("Loading v2 generator...")
    model = load_generator(str(ROOT / V2_CHECKPOINT), 42, device, arch=dict(MODEL_ARCH, max_len=V2_CTX))

    files = sorted(glob.glob('cache/samples_v3_48th/val/*.pt'))
    bands = {'UNDER-TRAINED tail (len 3800-5128)': (3800, 5128),
             'TRANSITION (len 3500-3800)': (3500, 3800),
             'CONTROL well-trained tail (len <3000)': (0, 3000)}
    buckets = {k: [] for k in bands}
    rng = np.random.default_rng(42)
    rng.shuffle(files)
    for f in files:
        if all(len(v) >= args.n for v in buckets.values()):
            break
        d = torch.load(f, map_location='cpu', weights_only=False)
        L = int(d['sample']['length'])
        for k, (lo, hi) in bands.items():
            if lo <= L < hi and len(buckets[k]) < args.n:
                buckets[k].append(d['sample'])
                break
    for k in bands:
        res = [r for s in buckets[k] if (r := process(model, s, ARMS)) is not None]
        summarize(res, ARMS, k)


if __name__ == '__main__':
    main()
