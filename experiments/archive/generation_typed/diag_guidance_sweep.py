#!/usr/bin/env python3
"""
H14 guidance-sweep diagnostic — is there a "decision boundary" where manifold style conditioning visibly
DIVERGES the chart, and where does it break (go OOD)? Playtest said conditioning is coherent but too weak at
g=1.5 (hold_ballad ~ glitch_tech on the same song). This sweeps CFG guidance and measures, per (song, style):
the generation PROXIES the radar dims summarize (density=stream/voltage, jump-rate=air, hold-rate=freeze,
off-beat-16th-rate=chaos) + the taste critic P(real) as the OOD/breakage detector.

Reads:
  STYLE SEPARATION (mean pairwise distance between the 3 styles' proxy vectors) rises with guidance -> the
    conditioning is biting harder; the "decision boundary" is where it jumps.
  taste critic P(real) stays flat then COLLAPSES at high g -> that's the OOD cliff (conditioning forced the
    chart off the realistic manifold). The usable band is high-separation BEFORE the critic collapses.

  python experiments/generation_typed/diag_guidance_sweep.py --songs 3
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
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.playtest_export import enforce_playability
from src.generation.radar_manifold import RadarManifold, DIMS
from src.models import LateFusionClassifier
from src.data.song_selection import select_by_groove

CKPT = "checkpoints/gen_style/best_val.pt"; CRITIC = "checkpoints/realism_critic/best_val.pt"
MANIFOLD = PROJECT_ROOT / "cache/radar_manifold.npz"
STYLES = {"glitch_tech": "chaos=high,air=low,stream=mod",
          "hold_ballad": "freeze=high,stream=low,chaos=low",
          "stream_machine": "stream=high,chaos=low,air=low"}
GUIDANCES = [1.0, 1.5, 2.0, 3.0, 5.0]
PROXY = ["density", "jump", "hold", "off16"]   # stream/volt, air, freeze, chaos


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def proxies(typed):
    b = to_binary(typed); on = b.any(1); T = len(on); n = max(int(on.sum()), 1)
    nact = b.sum(1); t = np.arange(T)
    return np.array([on.mean(),                                          # density
                     (nact >= 2).sum() / n,                             # jump rate (air)
                     (((typed == 2) | (typed == 4)).any(1)).sum() / n,  # hold-open rate (freeze)
                     (on & ((t % 4 == 1) | (t % 4 == 3))).sum() / n])   # off-beat 16th rate (chaos)


@torch.no_grad()
def critic_pr(critic, a23, chart_bin, device):
    a = torch.from_numpy(a23).unsqueeze(0).to(device); c = torch.from_numpy(chart_bin).unsqueeze(0).to(device)
    lo = critic(a, c, torch.ones(1, a.shape[1], device=device))
    if isinstance(lo, dict): lo = lo['logits']
    return float(torch.softmax(lo, 1)[0, 1])


@torch.no_grad()
def gen_chart(model, A, diff, R, g, dens, device):
    memory = model.encode_audio(A)
    ol = model.onset_logits(memory, diff, radar=R)[0]
    if g != 1.0:                                          # CFG on the onset head (matches the exporter)
        ol_u = model.onset_logits(memory, diff, radar=None)[0]
        ol = ol_u + g * (ol - ol_u)
    p = torch.sigmoid(ol).cpu().numpy()
    tau = float(np.quantile(p, 1 - dens)) if dens > 0 else 0.5
    kw = dict(onset_threshold=tau, type_sample=True, type_temperature=0.4, pattern_sample=True,
              pattern_temperature=0.7, radar=R, guidance_scale=g, max_jack_run=1)
    enforce_playability(kw)                               # MANDATORY pad-playability
    gen = model.generate(A, diff, lengths=torch.tensor([A.shape[1]], device=device), **kw)[0].cpu().numpy()
    return pair_holds(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--songs', type=int, default=3); ap.add_argument('--max_len', type=int, default=768)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ds = StepManiaDataset(chart_files=vf[:args.songs * 40], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=None, cache_dir='cache/samples')
    order = select_by_groove(ds, n=args.songs, by='rich', difficulty=3)
    model = LayeredTypedChartGenerator(audio_dim=23, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()
    ckc = torch.load(CRITIC, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ckc['config']).to(device); critic.load_state_dict(ckc['model_state_dict']); critic.eval()
    mani = RadarManifold.load(MANIFOLD)

    # results[g][style] = list of proxy vecs (per song); crit[g][style] = list of P(real)
    res = {g: {s: [] for s in STYLES} for g in GUIDANCES}
    crit = {g: {s: [] for s in STYLES} for g in GUIDANCES}
    titles = []
    for i in order[:args.songs]:
        meta = ds.valid_samples[i]; s = ds[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        a = s['audio'][:T, :23].numpy().astype(np.float32); A = torch.from_numpy(a).unsqueeze(0).to(device)
        diff = torch.tensor([meta['difficulty_class']], device=device); dcl = meta['difficulty_class']
        titles.append((meta['chart'].title or "?")[:22])
        for sname, spec in STYLES.items():
            tvec, info = mani.build_target(spec, dcl)
            R = torch.from_numpy(tvec).unsqueeze(0).to(device); dens = info['density']
            for g in GUIDANCES:
                typed = gen_chart(model, A, diff, R, g, dens, device)
                res[g][sname].append(proxies(typed))
                crit[g][sname].append(critic_pr(critic, a, to_binary(typed), device))

    print(f"\n=== H14 guidance sweep ({len(titles)} Hard songs: {', '.join(titles)}) ===")
    print(f"{'g':>4} | {'style-separation':>16} | {'critic P(real) mean':>19} | per-style on-axis proxy")
    print("-" * 96)
    base_sep = None
    for g in GUIDANCES:
        # style separation = mean over songs of mean pairwise L2 between the 3 styles' proxy vectors
        seps = []
        for k in range(len(titles)):
            vs = [res[g][s][k] for s in STYLES]
            seps.append(np.mean([np.linalg.norm(vs[a] - vs[b]) for a in range(3) for b in range(a + 1, 3)]))
        sep = float(np.mean(seps)); base_sep = base_sep or sep
        cmean = np.mean([np.mean(crit[g][s]) for s in STYLES])
        # on-axis: glitch->off16, hold->hold, stream->density (the dim each style is meant to push)
        gl = np.mean([v[3] for v in res[g]['glitch_tech']])
        hb = np.mean([v[2] for v in res[g]['hold_ballad']])
        sm = np.mean([v[0] for v in res[g]['stream_machine']])
        print(f"{g:>4.1f} | {sep:>10.4f} ({sep/base_sep:>3.1f}x) | {cmean:>19.3f} | "
              f"glitch off16={gl:.2f}  hold hold={hb:.2f}  stream dens={sm:.2f}")
    print("\n  separation rising = conditioning biting harder; critic P(real) falling = approaching the OOD cliff.")
    print("  per-style on-axis: does each style push ITS dim as g rises (glitch->off16, hold->hold, stream->density)?")


if __name__ == '__main__':
    main()
