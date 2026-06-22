#!/usr/bin/env python3
"""
Chaos as a PERIODIC GROOVE (decode probe; chaos_mechanism_plan.md). The per-frame chaos gate felt
arbitrary because syncopation is a repeated rhythmic figure, not per-frame placement (H10), and the
non-causal onset head can't produce periodicity. Here we IMPOSE periodicity at decode: per section, pick
the top-K off-beat within-measure slots by the section's AGGREGATE audio onset (so the groove is
audio-grounded, not arbitrary) and fire them in EVERY measure -> a repeated, grounded off-beat groove.
On-beat backbone = normal decode. Question: does imposed periodicity FEEL musical? If yes, "produce
periodic grooves" is the right training target.

Reports periodicity at lags 4/8/16 (beat/half-measure/measure), density, taste P(real); exports groove vs
baseline for A/B playtest.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, re, shutil, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT / 'experiments/generation_typed'))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.data.song_selection import select_by_groove
from src.models import LateFusionClassifier
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.sm_writer import charts_to_sm
from diag_transitions_freerun import foote_boundaries, SSM_DIMS

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"; CRITIC = "checkpoints/realism_critic/best_val.pt"
OFFBEAT = [j for j in range(16) if j % 4 != 0]
DECODE = dict(type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
              pattern_temperature=0.7, no_jump_during_hold=True, no_cross_during_hold=True)


def safe_name(s): return (re.sub(r'[^\w\- ]+', '', (s or 'x').strip(), flags=re.UNICODE).strip() or 'x')[:60]
def to_binary(t): t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def autocorr(x, lag):
    x = x.astype(np.float64) - x.mean(); d = (x * x).sum()
    return 0.0 if (d < 1e-9 or lag >= len(x)) else float((x[lag:] * x[:-lag]).sum() / d)


def ac_off(typed, T, lag):
    sig = (np.asarray(typed)[:T] != 0).any(1).astype(np.float64); t = np.arange(T)
    sig = sig.copy(); sig[t % 4 == 0] = 0.0
    return autocorr(sig, lag)


def groove_mask(p_on, T, bnds, tau, accents):
    t = np.arange(T)
    mask = (p_on > tau) & (t % 4 == 0)               # on-beat backbone = normal decode on-beats
    segs = [0] + sorted(int(b) for b in bnds) + [T]
    for a, b in zip(segs, segs[1:]):
        if b - a < 16: continue
        sal = {j: (np.mean([p_on[f] for f in range(a, b) if f % 16 == j]) if any(f % 16 == j for f in range(a, b)) else -1)
               for j in OFFBEAT}
        top = set(sorted(OFFBEAT, key=lambda j: -sal[j])[:accents])  # most audio-salient off-beat slots
        for f in range(a, b):
            if f % 16 in top:
                mask[f] = True                          # fire that slot EVERY measure -> periodic groove
    return mask


@torch.no_grad()
def p_real(critic, a23, cb, device):
    a = torch.from_numpy(a23).unsqueeze(0).to(device); c = torch.from_numpy(cb).unsqueeze(0).to(device)
    lg = critic(a, c, torch.ones(1, a.shape[1], device=device)); lg = lg['logits'] if isinstance(lg, dict) else lg
    return float(torch.softmax(lg, 1)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=6); ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--accents', type=int, default=3, help='off-beat groove slots fired per measure')
    ap.add_argument('--out_dir', default='outputs/chaos_groove'); ap.add_argument('--install', action='store_true')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 40], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()
    ck = torch.load(CRITIC, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()
    order = select_by_groove(ds, by='rich', difficulty=3)

    out = Path(args.out_dir)
    for tg in ('groove', 'baseline'): (out / tg).mkdir(parents=True, exist_ok=True)
    rows = []; used = 0
    for i in order:
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]; sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 256: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        bnds = foote_boundaries(sample['audio'][:T, SSM_DIMS].numpy())
        if (orig != 0).any(1).sum() < 32: continue
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        a23 = sample['audio'][:T, :23].numpy().astype(np.float32); rd = float((orig != 0).any(1).mean())
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - rd))
            set_seed(42)
            g0 = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau, **DECODE)[0].cpu().numpy()
            mask = groove_mask(p_on, T, bnds, tau, args.accents)
            set_seed(42)
            g1 = gen.generate(audio, diff, lengths=torch.tensor([T], device=device),
                              onset_override=torch.from_numpy(mask).unsqueeze(0).to(device), **DECODE)[0].cpu().numpy()
        charts = {'baseline': pair_holds(g0), 'groove': pair_holds(g1)}
        title = meta['chart'].title or Path(meta['chart_file']).stem
        rows.append((safe_name(title)[:20], orig, charts, a23, T))
        bpm = float(meta['chart'].bpm); music = os.path.basename(meta['audio_file']); dn = DIFFICULTY_NAMES[meta['difficulty_class']]
        for tg in ('baseline', 'groove'):
            folder = out / tg / f"{used:02d}_{safe_name(title)}"; folder.mkdir(parents=True, exist_ok=True)
            if os.path.exists(meta['audio_file']):
                try: shutil.copy2(meta['audio_file'], folder / music)
                except Exception: pass
            sm = charts_to_sm(charts=[{"chart": charts[tg], "difficulty_name": "Challenge", "difficulty_value": nd.difficulty_value, "author": f"chaos-{tg}"},
                                      {"chart": orig, "difficulty_name": dn, "difficulty_value": nd.difficulty_value, "author": "original"}],
                              bpm=bpm, title=f"{title} ({tg})", artist=meta['chart'].artist or "", music=music, offset=float(meta['chart'].offset), typed=True)
            (folder / "chart.sm").write_text(sm, encoding="utf-8")
        used += 1

    print(f"\n=== Chaos periodic-groove decode ({used} songs, accents/measure={args.accents}) ===")
    print(f"{'song':<20} {'src':>14} {'baseline':>22} {'groove':>22}")
    print(f"{'':<20} {'ac4/ac8/ac16':>14} {'ac4/ac8/ac16 dens P':>22} {'ac4/ac8/ac16 dens P':>22}")
    print("-" * 82)
    def acs(typed, T): return "/".join(f"{ac_off(typed,T,l):.2f}" for l in (4, 8, 16))
    for name, orig, charts, a23, T in rows:
        def cell(c): return f"{acs(c,T)} {(c!=0).any(1).mean():.2f} {p_real(critic,a23,to_binary(c),device):.2f}"
        print(f"{name:<20} {acs(orig,T):>14} {cell(charts['baseline']):>22} {cell(charts['groove']):>22}")
    print("-" * 82)
    print("ac@8/16 up in groove vs baseline = periodicity imposed. PLAYTEST groove vs baseline: musical or mechanical?")
    if args.install:
        from src.utils.sm_install import install_to_stepmania
        for d in install_to_stepmania(args.out_dir): print("installed:", d)


if __name__ == '__main__':
    main()
