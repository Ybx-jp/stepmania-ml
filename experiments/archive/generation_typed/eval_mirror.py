#!/usr/bin/env python3
"""
Mirror-augmentation A/B: does L<->R mirror (train_stage1.py --mirror, ~2x data) improve over gen_stage1?
Same decode, same val songs. Reports the metrics mirror should move — panel balance (L/D/U/R), jack rate
— plus quality guards (onset_F1, density) and the taste-critic P(real) (our validated musicality signal).
Real-chart panel/jack shown as the target.
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
from src.models import LateFusionClassifier
from src.generation.typed import pair_holds
from src.generation.evaluation import onset_density_metrics

PANELS = ['L', 'D', 'U', 'R']


def note_starts(typed):
    t = np.asarray(typed)
    return (t == 1) | (t == 2) | (t == 4)  # (T,4) bool: a note begins on this panel


def panel_pct(ns):
    c = ns.sum(0).astype(float); tot = c.sum()
    return (100 * c / tot) if tot else np.zeros(4)


def jack_rate(ns):
    """Among single-panel note frames, fraction whose panel repeats the previous single's panel."""
    rows = np.where(ns.any(1))[0]
    prev, same, n = None, 0, 0
    for r in rows:
        ps = np.where(ns[r])[0]
        if len(ps) != 1:
            prev = None; continue
        p = ps[0]
        if prev is not None:
            n += 1; same += (p == prev)
        prev = p
    return (same / n) if n else float('nan')


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


@torch.no_grad()
def p_real(critic, a23, chart_bin, device):
    a = torch.from_numpy(a23).unsqueeze(0).to(device); c = torch.from_numpy(chart_bin).unsqueeze(0).to(device)
    m = torch.ones(1, a.shape[1], device=device)
    lg = critic(a, c, m); lg = lg['logits'] if isinstance(lg, dict) else lg
    return float(torch.softmax(lg, 1)[0, 1])


def load(ckpt, device):
    m = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict']); m.eval(); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mirror_ckpt', default='checkpoints/gen_stage1_mirror/best_val.pt')
    ap.add_argument('--base_ckpt', default='checkpoints/gen_stage1/best_val.pt')
    ap.add_argument('--num_songs', type=int, default=32); ap.add_argument('--max_len', type=int, default=1024)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    models = {'gen_stage1 (base)': load(args.base_ckpt, device), 'gen_stage1_mirror': load(args.mirror_ckpt, device)}
    ck = torch.load('checkpoints/realism_critic/best_val.pt', map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()

    agg = {k: {'panel': [], 'jack': [], 'f1': [], 'dens': [], 'preal': []} for k in models}
    real = {'panel': [], 'jack': []}
    seen, used = set(), 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 128: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        real_density = float((orig != 0).any(1).mean())
        a23 = sample['audio'][:T, :23].numpy().astype(np.float32)
        ns_r = note_starts(orig); real['panel'].append(panel_pct(ns_r)); real['jack'].append(jack_rate(ns_r))
        for name, m in models.items():
            with torch.no_grad():
                p_on = torch.sigmoid(m.onset_logits(m.encode_audio(audio), diff))[0].cpu().numpy()
                tau = float(np.quantile(p_on, 1 - real_density)) if real_density > 0 else 0.5
                g = m.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                               type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                               pattern_temperature=0.7, no_jump_during_hold=True)[0].cpu().numpy()
            g = pair_holds(g); ns = note_starts(g)
            agg[name]['panel'].append(panel_pct(ns)); agg[name]['jack'].append(jack_rate(ns))
            mtr = onset_density_metrics((g != 0).astype(np.float32), reference=(orig != 0).astype(np.float32))
            agg[name]['f1'].append(mtr['onset_f1']); agg[name]['dens'].append((g != 0).any(1).mean())
            agg[name]['preal'].append(p_real(critic, a23, to_binary(g), device))
        seen.add(meta['chart_file']); used += 1

    print(f"\n=== Mirror A/B ({used} songs, recommended decode) ===\n")
    print(f"{'model':<20} {'L':>5} {'D':>5} {'U':>5} {'R':>5} {'|L-R|':>6} {'jack':>6} {'onset_F1':>9} {'dens':>6} {'P(real)':>8}")
    print("-" * 92)
    rp = np.mean(real['panel'], 0)
    print(f"{'REAL charts':<20} " + " ".join(f"{v:>4.0f}%" for v in rp) +
          f" {abs(rp[0]-rp[3]):>5.1f}% {100*np.nanmean(real['jack']):>5.1f}% {'—':>9} {'—':>6} {'—':>8}")
    for name in models:
        p = np.mean(agg[name]['panel'], 0)
        print(f"{name:<20} " + " ".join(f"{v:>4.0f}%" for v in p) +
              f" {abs(p[0]-p[3]):>5.1f}% {100*np.nanmean(agg[name]['jack']):>5.1f}% "
              f"{np.mean(agg[name]['f1']):>9.3f} {np.mean(agg[name]['dens']):>6.3f} {np.mean(agg[name]['preal']):>8.3f}")
    print("-" * 92)
    print("Mirror should: shrink |L-R| asymmetry, move panel%/jack toward REAL, hold onset_F1/density,")
    print("and (the real prize) lift P(real) if ~2x data improved generalization.")


if __name__ == '__main__':
    main()
