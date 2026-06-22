#!/usr/bin/env python3
"""
Does the model know WHICH songs deserve chaos? (see notes/phase_aware_threshold_findings.md)

The flat phase-alloc quota was smearing: it forces the SAME 16th share on every song, so a calm song gets
spurious 16ths and a chaotic song is capped below what it deserves. The per-song VARIATION is the chaos
signal; a quota destroys it. The right knob is a per-phase calibration OFFSET (fix the model's systematic
16th under-confidence once), then a single per-song threshold -> the 16th COUNT floats with the audio.

But that only works if the model's 16th onset confidence actually discriminates songs. This is the cheap
offline test (no generation -- onsets are decided up front, so just the onset posterior):

Per song:
  real16   = real chart's 16th note fraction (the song's true "chaos level")
  pon16    = mean onset prob over 16th frames (does the posterior run hotter on 16th-heavy songs?)
  gen16[m] = 16th fraction of the onset mask under method m:
               global  : p_on > tau                      (single threshold, per-song density d)
               alloc   : flat phase quota (real shares)   (the smearing baseline -- ~constant)
               calib   : (p_on logit + per-phase offset) > tau   (offset fit so AGGREGATE 16th share=target;
                         per song the count floats with confidence)

Report Spearman(real16, .) for pon16 and each method's gen16, plus the spread (std) of gen16.
  calib corr > 0 with real spread  -> the model discriminates; calibrated threshold delivers variable
    chaos to the songs that deserve it (THEN a playtest is meaningful).
  calib corr ~ 0 / alloc corr ~ 0  -> the model can't tell which songs deserve chaos; decode can't fix it,
    the 16th signal must come from the feature/objective (back to the model).

  python experiments/generation_typed/diag_song_chaos.py --num_songs 60 --target16 0.041
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

MODELS = {'gen_highres_v4': ("checkpoints/gen_highres_v4/best_val.pt", 42)}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0 or len(a) < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def is16_mask(T):
    t = np.arange(T); return (t % 4 == 1) | (t % 4 == 3)


def frac16(onset):
    t = np.arange(len(onset)); n = max(int(onset.sum()), 1)
    return onset[(t % 4 == 1) | (t % 4 == 3)].sum() / n


def alloc_mask(p, tau, shares):
    T = len(p); t = np.arange(T)
    bands = [t % 4 == 0, t % 4 == 2, (t % 4 == 1) | (t % 4 == 3)]
    N = int((p > tau).sum()); onset = np.zeros(T, bool)
    for share, band in zip(shares, bands):
        idx = np.where(band)[0]
        nk = min(int(round(N * share)), len(idx))
        if nk > 0:
            onset[idx[np.argsort(p[idx])[-nk:]]] = True
    return onset


def calib_mask(p, d, b16, b8=0.0):
    """add a per-phase LOGIT offset (16th += b16, 8th += b8), then a single per-song threshold at density d.
    Count floats with the song's confidence -- not a quota."""
    T = len(p); t = np.arange(T)
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    logit = logit + np.where(t % 4 == 2, b8, 0.0) + np.where((t % 4 == 1) | (t % 4 == 3), b16, 0.0)
    pc = 1 / (1 + np.exp(-logit))
    tau = np.quantile(pc, 1 - d)
    return pc > tau


def adaptive_mask(p, d, tau16):
    """PER-SONG ADAPTIVE 16th volume. A 16th is placed only where raw p_on clears an ABSOLUTE cross-song
    bar tau16 -- so the 16th COUNT floats with the audio (chaotic song -> many, calm song -> zero). The rest
    of the per-song budget N = round(d*T) is filled with the top-p_on quarter/8th frames, so total density
    (=difficulty) is preserved. tau16 is calibrated once across songs to hit the aggregate 16th share."""
    T = len(p); t = np.arange(T); N = int(round(d * T))
    is16 = (t % 4 == 1) | (t % 4 == 3)
    onset = np.zeros(T, bool)
    s16 = np.where(is16 & (p > tau16))[0]            # earned 16ths (absolute bar)
    if len(s16) > N:                                  # cap at the budget (keep most confident)
        s16 = s16[np.argsort(p[s16])[-N:]]
    onset[s16] = True
    rem = N - len(s16)
    oidx = np.where(~is16)[0]                          # fill remainder from quarter+8th
    if rem > 0 and len(oidx) > 0:
        onset[oidx[np.argsort(p[oidx])[-min(rem, len(oidx)):]]] = True
    return onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=60)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--min_difficulty', type=int, default=2)
    ap.add_argument('--target16', type=float, default=0.041, help='aggregate 16th share to calibrate to')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')
    name, (ckpt, ad) = next(iter(MODELS.items()))
    m = LayeredTypedChartGenerator(audio_dim=ad, d_model=128, num_layers=4, onset_layers=2).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict']); m.eval()

    songs = []  # (real16, density, p_on array)
    real_at_16 = []  # per-frame within-16th: (p_on, is_real_note) over 16th frames, all songs -> discrimination AUC
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs:
            break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty:
            continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 256:
            continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        note_r = (orig != 0).any(1)
        if note_r.sum() < 32:
            continue
        diff = torch.tensor([meta['difficulty_class']], device=device)
        audio = sample['audio'][:T, :ad].unsqueeze(0).to(device)
        with torch.no_grad():
            p_on = torch.sigmoid(m.onset_logits(m.encode_audio(audio), diff))[0].cpu().numpy()
        songs.append((frac16(note_r), float(note_r.mean()), p_on))
        s16 = is16_mask(T)
        real_at_16.append((p_on[s16], note_r[s16]))
        used += 1

    # within-song discrimination: at 16th frames, does p_on separate real-16th-note from not? (pooled AUC)
    pa = np.concatenate([x[0] for x in real_at_16]); ya = np.concatenate([x[1] for x in real_at_16]).astype(int)
    auc = float('nan')
    if ya.sum() > 0 and ya.sum() < len(ya):
        order = np.argsort(pa); rank = np.empty_like(order, float); rank[order] = np.arange(len(pa))
        n1 = ya.sum(); n0 = len(ya) - n1
        auc = (rank[ya == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

    real16 = np.array([s[0] for s in songs])
    pon16 = np.array([s[2][is16_mask(len(s[2]))].mean() for s in songs])

    # calibrate b16 so aggregate realized 16th share == target (bisection; monotone in b16)
    def agg16(b16):
        num = den = 0
        for r16, d, p in songs:
            mk = calib_mask(p, d, b16)
            num += mk[is16_mask(len(p))].sum(); den += mk.sum()
        return num / max(den, 1)
    lo, hi = 0.0, 6.0
    for _ in range(22):
        mid = (lo + hi) / 2
        if agg16(mid) < args.target16:
            lo = mid
        else:
            hi = mid
    b16 = (lo + hi) / 2

    # calibrate the absolute 16th bar tau16 for adaptive so aggregate 16th share == target (monotone: lower
    # tau16 -> more 16ths clear -> higher share)
    def agg16_adaptive(tau16):
        num = den = 0
        for r16, d, p in songs:
            mk = adaptive_mask(p, d, tau16)
            num += mk[is16_mask(len(p))].sum(); den += mk.sum()
        return num / max(den, 1)
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if agg16_adaptive(mid) > args.target16:   # bar too low -> too many 16ths -> raise it
            lo = mid
        else:
            hi = mid
    tau16 = (lo + hi) / 2

    shares = np.array([0.707, 0.252, 0.041]); shares /= shares.sum()
    gen = {'global': [], 'alloc': [], 'calib': [], 'adaptive': []}
    for r16, d, p in songs:
        tau = np.quantile(p, 1 - d)
        gen['global'].append(frac16(p > tau))
        gen['alloc'].append(frac16(alloc_mask(p, tau, shares)))
        gen['calib'].append(frac16(calib_mask(p, d, b16)))
        gen['adaptive'].append(frac16(adaptive_mask(p, d, tau16)))

    print(f"\n=== Does the model know which songs deserve chaos? ({used} songs) ===")
    print(f"  within-16th-band discrimination (pooled AUC, p_on vs real-16th-note): {auc:.3f}  (0.5=chance)")
    print(f"  per-song real 16th-rate: mean {real16.mean()*100:.1f}%  std {real16.std()*100:.1f}%  "
          f"range [{real16.min()*100:.1f}, {real16.max()*100:.1f}]%")
    print(f"  calib logit offset b16 = {b16:.2f};  adaptive absolute 16th bar tau16 = {tau16:.3f}")
    print(f"\n  signal     Spearman(real16, .)   mean16%   std16%   range16%      (real std 6.6, range[0,24])")
    print(f"  {'raw pon16':<12} {spearman(real16, pon16):>10.3f}")
    for mth in ('global', 'alloc', 'calib', 'adaptive'):
        g = np.array(gen[mth])
        print(f"  {mth:<12} {spearman(real16, g):>10.3f}        {g.mean()*100:>6.1f}   {g.std()*100:>6.1f}   "
              f"[{g.min()*100:.1f}, {g.max()*100:.1f}]")
    print("\n  adaptive: want HIGH corr + std/range approaching real (variable chaos to songs that earn it).")
    print("  alloc std~0 => quota = smearing (the flaw). If adaptive corr<=global, the absolute bar adds")
    print("  volume without losing the model's song-discrimination => the chaos knob.")


if __name__ == '__main__':
    main()
