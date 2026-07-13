#!/usr/bin/env python3
"""OFFLINE onset-window sweep (no AR decode): does MORE OVERLAP (smaller hop) / SMALLER W / SILENCE-pad fix the
'long empty middle sections with scattered notes' the user heard on long songs? The empty middles are an ONSET->tau
phenomenon (global-tau starves low-confidence frames where the onset head is weak at high local-PE positions), so
we can test it BEFORE the slow pattern decode. Metric: firing-density uniformity + longest empty gap, per config."""
import warnings, os, sys
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch
ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from scripts.generate import build_stub_chart, DIFFICULTY_METER, V2_CHECKPOINT, V2_CTX
from src.generation.decode_harness import make_feature_extractor, load_generator, compute_tau, apply_phase_calib, MODEL_ARCH

SR = 22050; BPM = 128.0; SUBDIV = 12; HOP = int(SR * 60 / (BPM * SUBDIV))
DUR = 420.0; ANCHOR = 0.056; GEN_DENSITY = 0.107; CALIB = (0.0, 1.0)
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
print(f"Lick | T={T} frames | hop={HOP} | density target {GEN_DENSITY} | 1 beat=12fr, 1 measure=48fr\n")


def fired(W, hop_frac, hangover, reflect):
    with torch.no_grad():
        ol = model.onset_logits(memory, diff, window=W, tail_hangover=hangover,
                                hop_frac=hop_frac, hangover_reflect=reflect)[0]
        ol = apply_phase_calib(ol, CALIB, SUBDIV)
        p = torch.sigmoid(ol).cpu().numpy()
    tau = compute_tau(p, GEN_DENSITY)
    return p >= tau


def stats(fm):
    idx = np.nonzero(fm)[0]
    n = len(idx)
    # empty-gap run lengths (consecutive non-fired frames between fired notes)
    gaps = np.diff(idx) - 1 if n > 1 else np.array([0])
    maxgap = int(gaps.max()) if len(gaps) else 0
    big = int((gaps >= 48).sum())        # gaps >= 1 measure
    huge = int((gaps >= 144).sum())      # gaps >= 3 measures (a "long empty section")
    # per-decile density (notes/frame) to see empty middles vs busy ends
    NB = 10
    b = np.clip((np.arange(T) / max(T - 1, 1) * NB).astype(int), 0, NB - 1)
    nb = np.clip((idx / max(T - 1, 1) * NB).astype(int), 0, NB - 1)
    dens = [(nb == k).sum() / max((b == k).sum(), 1) for k in range(NB)]
    return n, maxgap, big, huge, dens


CONFIGS = [
    ("DEPLOYED   W5400 h.50 hang2700 refl", 5400, 0.50, 2700, True),
    ("+overlap   W5400 h.25 hang2700 refl", 5400, 0.25, 2700, True),
    ("+overlap+  W5400 h.125 hang2700 refl", 5400, 0.125, 2700, True),
    ("smallerW   W4096 h.50 hang2048 refl", 4096, 0.50, 2048, True),
    ("both       W4096 h.25 hang2048 refl", 4096, 0.25, 2048, True),
    ("both+silen W4096 h.25 hang2048 SILEN", 4096, 0.25, 2048, False),
    ("smaller3600 W3600 h.25 hang1800 refl", 3600, 0.25, 1800, True),
]
print(f"{'config':40s} {'notes':>5s} {'maxgap':>6s} {'≥1meas':>6s} {'≥3meas':>6s} | density-by-decile (x1000)")
print("-" * 118)
for name, W, hf, hg, refl in CONFIGS:
    fm = fired(W, hf, hg, refl)
    n, mg, big, huge, dens = stats(fm)
    dstr = " ".join(f"{d*1000:3.0f}" for d in dens)
    print(f"{name:40s} {n:>5d} {mg:>6d} {big:>6d} {huge:>6d} | {dstr}")
print("\nmaxgap/≥3meas = the 'long empty sections'; a UNIFORM density-by-decile (no low bins) = consistent backbone.")
print("Want: fewer/shorter empty gaps + flatter deciles. (silence pad should also thin the very-end wind-down.)")

# LOCALIZE the biggest gaps under DEPLOYED + are they LOW-ENERGY (correct sparse) or NORMAL-energy (onset defect)?
print("\n--- top empty gaps under DEPLOYED (W5400 h.50) — energy check: is the hole a quiet passage or a defect? ---")
energy = audio_tensor[:T, 0].astype(np.float64)      # dim-0 total energy
perc = audio_tensor[:T, 35].astype(np.float64)       # dim-35 perc onset
e_lo, e_hi = np.percentile(energy, 10), np.percentile(energy, 90)
song_e = energy.mean()
fm = fired(5400, 0.50, 2700, True)
idx = np.nonzero(fm)[0]
runs = []  # (start, length) of empty runs
prev = -1
for i in idx:
    if i - prev - 1 >= 48:      # >= 1 measure empty
        runs.append((prev + 1, i - prev - 1))
    prev = i
runs.sort(key=lambda r: -r[1])
print(f"song mean energy {song_e:.3f} (p10 {e_lo:.3f} / p90 {e_hi:.3f})")
print(f"{'pos%':>5s} {'len(fr)':>7s} {'measures':>8s} {'gap_energy':>10s} {'vs song':>8s} {'gap_perc':>8s}")
for s, L in runs[:8]:
    ge = energy[s:s + L].mean(); gp = perc[s:s + L].mean()
    rel = "LOW" if ge < e_lo else ("norm" if ge < e_hi else "HIGH")
    print(f"{s/T*100:5.0f} {L:7d} {L/48:8.1f} {ge:10.3f} {rel:>8s} {gp:8.3f}")
print("(LOW-energy gaps = correct sparse passages / melodic-under-placement; norm/HIGH-energy gaps = a real onset defect.)")
