#!/usr/bin/env python3
"""
H16 — is CFG guidance "HARMONIC" (non-monotonic; discrete g are coherent nodes, intermediates degrade), or is
the playtest non-monotonicity (OH WORLD glitch: g3.5 great, g4 1/8-biased+structure-ruined, g4.5 recovers)
just SAMPLING NOISE? (experiment-design rule 11: separate the claimed signal from noise BEFORE believing it.)

Sweeps g finely on the SAME song (OH WORLD) at 3 seeds each, measuring:
  - rhythm shares quarter/8th/16th of placed notes -> does the 1/8-bias spike NON-monotonically (peak at g4)?
  - motif-repetition (recurring 1-beat panel windows) -> a crude structure/symmetry proxy (good charts recur).
Reads:
  a reproducible non-monotonic feature (8th-bias peak / repetition dip at g4) that EXCEEDS the across-seed
    std -> harmonic structure is real -> worth mapping the nodes.
  the feature is within seed std / monotonic -> NOT harmonic; g3.5>g4 was a lucky draw; quality is a noisy
    monotonic-ish trend and the lever is elsewhere (H15 motifs).

  python experiments/generation_typed/diag_harmonic_guidance.py
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.playtest_export import enforce_playability
from src.generation.radar_manifold import RadarManifold

CKPT = "checkpoints/gen_style/best_val.pt"; MANIFOLD = PROJECT_ROOT / "cache/radar_manifold.npz"
STYLE = "chaos=high,air=low,stream=mod"            # glitch tech (the breakthrough style)
GUIDANCES = [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
SEEDS = [42, 1, 7]


def to_binary(t): t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def rhythm_shares(typed):
    on = to_binary(typed).any(1); n = max(int(on.sum()), 1); t = np.arange(len(on))
    return (100 * on[t % 4 == 0].sum() / n, 100 * on[t % 4 == 2].sum() / n,
            100 * on[(t % 4 == 1) | (t % 4 == 3)].sum() / n)   # quarter%, 8th%, 16th%


def repetition(typed, win=16):
    """fraction of non-empty 1-beat (16-frame) panel windows that RECUR (>=2x) -> structure/motif proxy."""
    b = to_binary(typed); code = (b * np.array([1, 2, 4, 8])).sum(1).astype(int)
    wins = [tuple(code[i:i + win]) for i in range(0, len(code) - win, win)]
    ne = [w for w in wins if any(w)]
    if not ne: return 0.0
    c = Counter(wins)
    return 100 * sum(1 for w in ne if c[w] >= 2) / len(ne)


@torch.no_grad()
def gen(model, A, diff, R, g, dens, device):
    memory = model.encode_audio(A); ol = model.onset_logits(memory, diff, radar=R)[0]
    ol_u = model.onset_logits(memory, diff, radar=None)[0]; ol = ol_u + g * (ol - ol_u)
    p = torch.sigmoid(ol).cpu().numpy(); tau = float(np.quantile(p, 1 - dens))
    kw = dict(onset_threshold=tau, type_sample=True, type_temperature=0.4, pattern_sample=True,
              pattern_temperature=0.7, radar=R, guidance_scale=g, max_jack_run=1)
    enforce_playability(kw)
    return pair_holds(model.generate(A, diff, lengths=torch.tensor([A.shape[1]], device=device), **kw)[0].cpu().numpy())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--song', default='OH WORLD'); ap.add_argument('--max_len', type=int, default=1024)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ds = StepManiaDataset(chart_files=vf[:200], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=None, cache_dir='cache/samples')
    idx = next((i for i in range(len(ds.valid_samples))
                if args.song.lower() in (ds.valid_samples[i]['chart'].title or '').lower()
                and ds.valid_samples[i]['difficulty_class'] == 3), None)
    if idx is None:
        print(f"'{args.song}' (Hard) not found"); return
    meta = ds.valid_samples[idx]; s = ds[idx]; T = min(int(s['mask'].sum().item()), args.max_len)
    a = s['audio'][:T, :23].numpy().astype(np.float32); A = torch.from_numpy(a).unsqueeze(0).to(device)
    diff = torch.tensor([3], device=device)
    model = LayeredTypedChartGenerator(audio_dim=23, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()
    mani = RadarManifold.load(MANIFOLD)
    tvec, info = mani.build_target(STYLE, 3); R = torch.from_numpy(tvec).unsqueeze(0).to(device); dens = info['density']

    print(f"\n=== H16 harmonic-guidance sweep: {meta['chart'].title} (glitch, {len(SEEDS)} seeds) ===")
    print(f"{'g':>5} | {'quarter%':>16} | {'8th%':>16} | {'16th%':>16} | {'repetition%':>16}")
    print(f"{'':>5} | {'mean±std':>16} | {'mean±std (1/8 bias)':>16} | {'mean±std':>16} | {'mean±std (struct)':>16}")
    print("-" * 92)
    rows = {}
    for g in GUIDANCES:
        q, e, x, rep = [], [], [], []
        for sd in SEEDS:
            set_seed(sd)
            typed = gen(model, A, diff, R, g, dens, device)
            sh = rhythm_shares(typed); q.append(sh[0]); e.append(sh[1]); x.append(sh[2]); rep.append(repetition(typed))
        rows[g] = (np.mean(e), np.std(e), np.mean(rep), np.std(rep))
        f = lambda v: f"{np.mean(v):5.1f}±{np.std(v):4.1f}"
        print(f"{g:>5.2f} | {f(q):>16} | {f(e):>16} | {f(x):>16} | {f(rep):>16}")
    # is the 1/8-bias / repetition non-monotonic BEYOND seed noise?
    e_means = np.array([rows[g][0] for g in GUIDANCES]); e_stds = np.array([rows[g][1] for g in GUIDANCES])
    rep_means = np.array([rows[g][2] for g in GUIDANCES]); rep_stds = np.array([rows[g][3] for g in GUIDANCES])
    print(f"\n  8th% range {e_means.min():.1f}-{e_means.max():.1f} (peak @ g={GUIDANCES[e_means.argmax()]}); "
          f"typical seed std {np.median(e_stds):.1f}")
    print(f"  repetition range {rep_means.min():.1f}-{rep_means.max():.1f} (min @ g={GUIDANCES[rep_means.argmin()]}); "
          f"typical seed std {np.median(rep_stds):.1f}")
    print("  HARMONIC if a non-monotonic feature (8th-peak / repetition-dip near g4) EXCEEDS the seed std; "
          "else it's sampling noise.")


if __name__ == '__main__':
    main()
