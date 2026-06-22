#!/usr/bin/env python3
"""
Is the generator's near-random panel choreography (transition-matrix KL ~ random; see
choreography_metrics_findings.md) a DECODE artifact of pattern_temperature? Sweep pattern_temperature and
measure transition-matrix KL(gen||real) + L<->R symmetry + panel balance (to catch the always-Left greedy
collapse at low temp). If a temp lowers KL toward 0 (real-like opposite-panel structure) WITHOUT collapsing
panel balance, it's another decode fix.
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
from src.generation.typed import pair_holds

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
LR_SWAP = [3, 1, 2, 0]
TEMPS = [0.3, 0.5, 0.7, 0.9, 1.2]


def note_starts(t): t = np.asarray(t); return (t == 1) | (t == 2) | (t == 4)


def trans(ns):
    rows = np.where(ns.any(1))[0]; M = np.zeros((4, 4)); prev = None
    for r in rows:
        ps = np.where(ns[r])[0]
        if len(ps) != 1: prev = None; continue
        if prev is not None: M[prev, ps[0]] += 1
        prev = ps[0]
    return M


def kl(p, q):
    p = p.flatten() + 1e-9; q = q.flatten() + 1e-9; p /= p.sum(); q /= q.sum()
    return float((p * np.log(p / q)).sum())


def panel_pct(ns):
    c = ns.sum(0).astype(float); return 100 * c / max(c.sum(), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--num_songs', type=int, default=24)
    ap.add_argument('--max_len', type=int, default=1024); args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()

    Mreal = np.zeros((4, 4)); Mgen = {t: np.zeros((4, 4)) for t in TEMPS}; pan = {t: np.zeros(4) for t in TEMPS}
    samples = []
    seen, used = set(), 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen: continue
        sm = ds[i]; T = min(int(sm['mask'].sum().item()), args.max_len)
        if T < 128: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        if note_starts(orig).any(1).sum() < 8: continue
        Mreal += trans(note_starts(orig))
        samples.append((sm['audio'][:T], meta['difficulty_class'], float((orig != 0).any(1).mean()), T))
        seen.add(meta['chart_file']); used += 1

    for audio_t, dcl, dens, T in samples:
        audio = audio_t.unsqueeze(0).to(device); diff = torch.tensor([dcl], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - dens))
            for tp in TEMPS:
                set_seed(42)
                g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                                 type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                                 pattern_temperature=tp, no_jump_during_hold=True)[0].cpu().numpy()
                ns = note_starts(pair_holds(g)); Mgen[tp] += trans(ns); pan[tp] += ns.sum(0)

    Preal = Mreal / max(Mreal.sum(), 1)
    real_pan = 100 * Mreal.sum(0) / max(Mreal.sum(), 1)  # transitions INTO each panel ~ panel usage
    print(f"\n=== Transition structure vs pattern_temperature ({used} songs) ===")
    print(f"REAL: KL=0.000  symmetry={1-0.5*np.abs(Preal-Preal[np.ix_(LR_SWAP,LR_SWAP)]).sum():.3f}  "
          f"panel L/D/U/R={'/'.join(f'{v:.0f}' for v in real_pan)}%")
    print(f"\n{'pat_temp':<9} {'KL(gen||real)':>14} {'symmetry':>9} {'panel L/D/U/R %':>22}")
    print("-" * 58)
    for tp in TEMPS:
        P = Mgen[tp] / max(Mgen[tp].sum(), 1)
        sy = 1 - 0.5 * np.abs(P - P[np.ix_(LR_SWAP, LR_SWAP)]).sum()
        pp = 100 * pan[tp] / max(pan[tp].sum(), 1)
        print(f"{tp:<9} {kl(P, Preal):>14.3f} {sy:>9.3f} {'/'.join(f'{v:.0f}' for v in pp):>22}%")
    print("-" * 58)
    print("Lower KL toward 0 = real-like opposite-panel structure. Watch panel% for always-Left collapse")
    print("at low temp (L >> 25%). The sweet spot (low KL, balanced panels) would be a decode fix.")


if __name__ == '__main__':
    main()
