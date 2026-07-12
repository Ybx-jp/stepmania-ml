#!/usr/bin/env python3
"""Does harm_calib --harm_quiet_feat perc FILL the empty MIDDLES (the melodic/perc-sparse under-placement holes)?
Offline onset->tau (no AR decode). Baseline = deployed + hangover (the shipped Lick HANGOVER). Harm arms add the
sparse-harm-in-quiet onset offset (density-preserving: tau recomputed WITH it, as the exporter does). We track the
big baseline gaps and whether they fill, per gate (perc vs total) and gain."""
import warnings, os, sys
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch
ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from scripts.generate import build_stub_chart, DIFFICULTY_METER, V2_CHECKPOINT, V2_CTX, _sparse_harm_offset
from src.generation.decode_harness import make_feature_extractor, load_generator, compute_tau, apply_phase_calib, MODEL_ARCH

SR = 22050; BPM = 128.0; SUBDIV = 12; HOP = int(SR * 60 / (BPM * SUBDIV))
DUR = 420.0; ANCHOR = 0.056; GEN_DENSITY = 0.107; CALIB = (0.0, 1.0)
W, HF, HANG = 5400, 0.50, 2700          # deployed + hangover (the shipped Lick HANGOVER)
AUD = "/home/ybx/sm-personal/Yb's Home Cooked/Lick the Rainbow/Mord Fustang - Lick The Rainbow [Electro House _ Plasmapool].ogg"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fspec = make_feature_extractor("highres_v2")
stub = build_stub_chart(AUD, BPM, DUR, HOP, subdiv=SUBDIV, offset=ANCHOR)
audio_tensor = fspec.extractor.extract_from_chart(AUD, stub).get_aligned_features()
T = audio_tensor.shape[0]
model = load_generator(str(ROOT / V2_CHECKPOINT), fspec.audio_dim, device, arch=dict(MODEL_ARCH, max_len=V2_CTX))
audio = torch.from_numpy(audio_tensor[:T].astype(np.float32)).unsqueeze(0).to(device)
diff = torch.tensor([list(DIFFICULTY_METER).index("Hard")], device=device)
with torch.no_grad():
    memory = model.encode_audio(audio)
    ol_base = apply_phase_calib(model.onset_logits(memory, diff, window=W, tail_hangover=HANG,
                                                   hop_frac=HF, hangover_reflect=True)[0], CALIB, SUBDIV)


def fired(extra_offset=None):
    ol = ol_base if extra_offset is None else ol_base + torch.tensor(extra_offset, device=device)
    p = torch.sigmoid(ol).cpu().numpy()
    tau = compute_tau(p, GEN_DENSITY)
    return p >= tau


def gap_runs(fm, minlen=48):
    idx = np.nonzero(fm)[0]; runs = []; prev = -1
    for i in idx:
        if i - prev - 1 >= minlen:
            runs.append((prev + 1, i - prev - 1))
        prev = i
    return idx, runs


energy = audio_tensor[:T, 0].astype(np.float64)
base_fm = fired()
base_idx, base_runs = gap_runs(base_fm)
base_runs.sort(key=lambda r: -r[1])
print(f"Lick | T={T} | density {GEN_DENSITY} | baseline = deployed+hangover ({len(base_idx)} notes)\n")

# For each harm arm: total notes (should ~hold), maxgap, #gaps>=1meas/3meas, and the fill of the baseline top-5 holes
print(f"{'arm':22s} {'notes':>5s} {'maxgap':>6s} {'≥1meas':>6s} {'≥3meas':>6s} | fill of baseline top-5 holes (notes added inside)")
print("-" * 118)
def summarize(name, fm):
    idx, runs = gap_runs(fm); runs2 = sorted(runs, key=lambda r: -r[1])
    maxg = runs2[0][1] if runs2 else 0
    big = sum(1 for _, L in runs if L >= 48); huge = sum(1 for _, L in runs if L >= 144)
    fired_set = set(idx.tolist())
    fills = []
    for s, L in base_runs[:5]:
        added = sum(1 for f in range(s, s + L) if f in fired_set)
        fills.append(f"p{int(s/T*100)}%:{added:+d}")
    print(f"{name:22s} {len(idx):>5d} {maxg:>6d} {big:>6d} {huge:>6d} | {'  '.join(fills)}")

summarize("baseline (no harm)", base_fm)
for gate in ["perc", "total"]:
    for gain in [5.0, 10.0, 20.0]:
        off = _sparse_harm_offset(audio_tensor[:T], gain, 40.0, gate)
        summarize(f"harm {gate} g{gain:g}", fired(off))
# LOCAL TAU: allocate density per-region (sliding-window quantile) so onset-poor sections aren't starved by the ONE
# global quantile. Risk = the Rule-13 global-quota anti-pattern (forces notes into genuinely-silent sections). Also
# test an ENERGY-GATED local tau (only lift where audio energy is decent -> don't fill true silence).
print("\n--- LOCAL TAU arms (per-region density; the leading empty-middles candidate) ---")
p_base = torch.sigmoid(ol_base).cpu().numpy()
def local_tau_fire(win_frames, energy_gate=None):
    # sliding-window quantile threshold; fire p>=local tau. energy_gate: skip (keep empty) frames below this z-energy.
    fm = np.zeros(T, dtype=bool)
    half = win_frames // 2
    for s in range(0, T, half or 1):
        e = min(s + (half or 1), T)
        seg = p_base[max(0, s - half):min(T, e + half)]
        thr = np.quantile(seg, 1 - GEN_DENSITY)
        fm[s:e] = p_base[s:e] >= thr
    if energy_gate is not None:
        fm &= (energy >= energy_gate)
    return fm
for win in [192, 384]:      # 4-measure, 8-measure local windows
    summarize(f"local-tau w{win//48}meas", local_tau_fire(win))
summarize("local-tau w4meas E>p20", local_tau_fire(192, np.percentile(energy, 20)))

print("\nbaseline top-5 holes (pos%, measures, mean-energy):")
for s, L in base_runs[:5]:
    print(f"  p{int(s/T*100)}%  {L/48:.1f} meas  E={energy[s:s+L].mean():+.2f}")
print("(a harm arm that FILLS the perc-sparse holes = large + fills at those pos%; watch it doesn't DRAIN elsewhere,"
      " i.e. maxgap/≥3meas shouldn't grow — density-preserving means fills here come from somewhere.)")
