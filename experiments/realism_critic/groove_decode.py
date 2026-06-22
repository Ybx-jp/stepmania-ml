#!/usr/bin/env python3
"""
Phase-aware off-beat-budget decode (the cheap groove fix; see groove_periodicity_findings.md).

The default global onset threshold under-places off-beats (on-beat p_on >> off-beat p_on), so the model
decodes more on-beat-biased than real (93% vs 88%) and grooves less (ac_off 0.11 vs real 0.19). But the
groove IS in p_on: given an adequate off-beat BUDGET and selecting top off-beats by p_on, ac_off hits
real level (0.318 vs 0.300). This builds that decode and exports normal-vs-budget for an A/B playtest.

  normal : onset = (p_on > global_tau)   [what export does]
  budget : on-beat = top n_on on-beat by p_on ; off-beat = top n_off off-beat by p_on
           (n_on, n_off matched to the real chart -> same density AND syncopation amount; only placement
            differs). Fed to generate(onset_override=...) so pattern/type/hold come from the model.

Reports ac_off (groove) + taste-critic P(real) for both, vs real. Exports outputs/groove_decode/{budget,normal}.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, re, shutil, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.models import LateFusionClassifier
from src.generation.typed import pair_holds
from src.generation.sm_writer import charts_to_sm

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
CRITIC = "checkpoints/realism_critic/best_val.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/'); p.add_argument('--audio_dir', default='data/')
    p.add_argument('--num_songs', type=int, default=8); p.add_argument('--max_len', type=int, default=1024)
    p.add_argument('--offbeat_scale', type=float, default=1.0,
                   help='multiply the real off-beat budget (1.0=match real; >1 = more syncopation dial)')
    p.add_argument('--out_dir', default='outputs/groove_decode'); p.add_argument('--install', action='store_true')
    return p.parse_args()


def safe_name(s):
    s = re.sub(r'[^\w\- ]+', '', (s or 'untitled').strip(), flags=re.UNICODE).strip()
    return (s or 'untitled')[:60]


def autocorr(x, lag):
    x = x.astype(np.float64) - x.mean(); d = (x * x).sum()
    return 0.0 if (d < 1e-9 or lag >= len(x)) else float((x[lag:] * x[:-lag]).sum() / d)


def ac_off(typed, T):
    sig = (np.asarray(typed)[:T] != 0).any(1).astype(np.float64)
    t = np.arange(T); sig = sig.copy(); sig[t % 4 == 0] = 0.0
    return autocorr(sig, 4)


def budget_mask(p_on, T, n_on, n_off):
    t = np.arange(T); on_idx = np.where(t % 4 == 0)[0]; off_idx = np.where(t % 4 != 0)[0]
    m = np.zeros(T, bool)
    if n_on > 0 and len(on_idx): m[on_idx[np.argsort(p_on[on_idx])[::-1][:n_on]]] = True
    if n_off > 0 and len(off_idx): m[off_idx[np.argsort(p_on[off_idx])[::-1][:n_off]]] = True
    return m


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


@torch.no_grad()
def p_real(critic, a23, cb, device):
    a = torch.from_numpy(a23).unsqueeze(0).to(device); c = torch.from_numpy(cb).unsqueeze(0).to(device)
    lg = critic(a, c, torch.ones(1, a.shape[1], device=device)); lg = lg['logits'] if isinstance(lg, dict) else lg
    return float(torch.softmax(lg, 1)[0, 1])


def main():
    args = parse_args(); set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 6], audio_dir=args.audio_dir,
                          max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()
    ck = torch.load(CRITIC, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()

    out = Path(args.out_dir)
    for tag in ('budget', 'normal'): (out / tag).mkdir(parents=True, exist_ok=True)
    rows = []; seen, used = set(), 0
    decode_kw = dict(type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                     pattern_temperature=0.7, no_jump_during_hold=True)
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
        onset_r = (orig != 0).any(1); t = np.arange(T)
        n_on = int(onset_r[t % 4 == 0].sum()); n_off = int(round(onset_r[t % 4 != 0].sum() * args.offbeat_scale))
        real_density = float(onset_r.mean())
        if onset_r.sum() < 16: continue
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        a23 = sample['audio'][:T, :23].numpy().astype(np.float32)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
        tau = float(np.quantile(p_on, 1 - real_density)) if real_density > 0 else 0.5
        masks = {'normal': (p_on > tau), 'budget': budget_mask(p_on, T, n_on, n_off)}
        res = {}
        for tag, mask in masks.items():
            ov = torch.from_numpy(mask).unsqueeze(0).to(device)
            with torch.no_grad():
                g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_override=ov, **decode_kw)[0].cpu().numpy()
            g = pair_holds(g)
            res[tag] = dict(g=g, ac=ac_off(g, T), pr=p_real(critic, a23, to_binary(g), device),
                            ob=100 * float((g != 0).any(1)[t % 4 == 0].sum()) / max(int((g != 0).any(1).sum()), 1))
        title = meta['chart'].title or Path(meta['chart_file']).stem
        rows.append((safe_name(title)[:22], ac_off(orig, T), p_real(critic, a23, to_binary(orig), device), res))

        bpm = float(meta['chart'].bpm); music = os.path.basename(meta['audio_file']); dname = DIFFICULTY_NAMES[meta['difficulty_class']]
        for tag in ('budget', 'normal'):
            folder = out / tag / f"{used:02d}_{safe_name(title)}"; folder.mkdir(parents=True, exist_ok=True)
            if os.path.exists(meta['audio_file']):
                try: shutil.copy2(meta['audio_file'], folder / music)
                except Exception: pass
            sm = charts_to_sm(charts=[
                {"chart": res[tag]['g'], "difficulty_name": "Challenge", "difficulty_value": nd.difficulty_value, "author": f"groove-{tag}"},
                {"chart": orig, "difficulty_name": dname, "difficulty_value": nd.difficulty_value, "author": "original"},
            ], bpm=bpm, title=f"{title} ({tag})", artist=meta['chart'].artist or "", music=music, offset=float(meta['chart'].offset), typed=True)
            (folder / "chart.sm").write_text(sm, encoding="utf-8")
        seen.add(meta['chart_file']); used += 1

    print(f"\n=== Groove decode: normal vs budget ({used} songs, offbeat_scale={args.offbeat_scale}) ===")
    print(f"{'song':<22} {'real_ac':>7} {'norm_ac':>7} {'budg_ac':>7} | {'real_P':>6} {'norm_P':>6} {'budg_P':>6} | {'norm_ob':>7} {'budg_ob':>7}")
    print("-" * 96)
    for name, rac, rpr, res in rows:
        print(f"{name:<22} {rac:>7.3f} {res['normal']['ac']:>7.3f} {res['budget']['ac']:>7.3f} | "
              f"{rpr:>6.3f} {res['normal']['pr']:>6.3f} {res['budget']['pr']:>6.3f} | "
              f"{res['normal']['ob']:>6.0f}% {res['budget']['ob']:>6.0f}%")
    print("-" * 96)
    mac = lambda k: np.mean([r[3][k]['ac'] for r in rows]); mpr = lambda k: np.mean([r[3][k]['pr'] for r in rows])
    print(f"{'MEAN':<22} {np.mean([r[1] for r in rows]):>7.3f} {mac('normal'):>7.3f} {mac('budget'):>7.3f} | "
          f"{np.mean([r[2] for r in rows]):>6.3f} {mpr('normal'):>6.3f} {mpr('budget'):>6.3f}")
    print("\nbudget ac_off toward real = groove recovered offline. Playtest budget vs normal: does it FEEL groovier?")
    if args.install:
        from src.utils.sm_install import install_to_stepmania
        for d in install_to_stepmania(args.out_dir): print("installed:", d)


if __name__ == '__main__':
    main()
