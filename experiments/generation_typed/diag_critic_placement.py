#!/usr/bin/env python3
"""
OBJECTIVE gate (Phase-3 design, notes/phase3_generative_design.md): can the taste critic supervise PLACEMENT?
If it can't tell good placement from bad AT MATCHED amount, critic-as-distribution-objective is dead.

For N chaotic-Hard songs, critic P(real) of:
  REAL        : the real chart (good placement)
  v4-gen      : v4 generated at the song's real density (awkward placement, matched amount)
  shuf16      : REAL chart with its 16th notes REDISTRIBUTED to random 16th frames (placement broken, amount
                + backbone + panels preserved) -- a direct "bad placement, everything else equal" probe
Reads:
  REAL > v4-gen AND REAL > shuf16 (clear margin) -> critic SEES placement quality -> can supervise it.
  REAL ~ v4-gen / shuf16 -> critic blind to placement at fixed amount -> critic-as-objective is out.

  python experiments/generation_typed/diag_critic_placement.py --num_songs 40
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
from src.models import LateFusionClassifier

V4 = "checkpoints/gen_highres_v4/best_val.pt"; CRITIC = "checkpoints/realism_critic/best_val.pt"


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


@torch.no_grad()
def score(critic, a23, chart, device):
    a = torch.from_numpy(a23).unsqueeze(0).to(device); c = torch.from_numpy(chart).unsqueeze(0).to(device)
    m = torch.ones(1, a.shape[1], device=device)
    lo = critic(a, c, m)
    if isinstance(lo, dict): lo = lo['logits']
    return float(torch.softmax(lo, 1)[0, 1])


def shuffle16(binary, rng):
    """redistribute 16th-phase NOTE rows to random 16th frames (break placement; keep amount+panels+backbone)."""
    T = binary.shape[0]; t = np.arange(T); i16 = np.where((t % 4 == 1) | (t % 4 == 3))[0]
    out = binary.copy()
    note_rows = [binary[k].copy() for k in i16 if binary[k].any()]
    out[i16] = 0.0
    if note_rows:
        tgt = rng.choice(i16, size=len(note_rows), replace=False)
        for r, k in zip(note_rows, tgt): out[k] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=40); ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--min16', type=float, default=0.05); ap.add_argument('--min_difficulty', type=int, default=2)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rng = np.random.default_rng(42)
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 6], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')
    ck = torch.load(CRITIC, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()
    v4 = LayeredTypedChartGenerator(audio_dim=42, d_model=128, num_layers=4, onset_layers=2).to(device)
    v4.load_state_dict(torch.load(V4, map_location=device)['model_state_dict']); v4.eval()

    real_s, gen_s, shuf_s = [], [], []; used = 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 256: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        note = (typed != 0).any(1); n = int(note.sum())
        if n < 32: continue
        tt = np.arange(T); s16 = (note & ((tt % 4 == 1) | (tt % 4 == 3))).sum() / max(n, 1)
        if s16 < args.min16: continue
        a23 = sample['audio'][:T, :23].numpy().astype(np.float32); d = float(note.mean())
        rb = to_binary(typed)
        # v4 gen at real density
        diff = torch.tensor([meta['difficulty_class']], device=device)
        rad = torch.from_numpy(meta['groove_radar'].to_vector().astype(np.float32)).unsqueeze(0).to(device)
        A = sample['audio'][:T, :42].unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(v4.onset_logits(v4.encode_audio(A), diff, radar=rad))[0].cpu().numpy()
            tau = float(np.quantile(p, 1 - d))
            g = v4.generate(A, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                            type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                            pattern_temperature=0.7, no_jump_during_hold=True, radar=rad)[0].cpu().numpy()
        gb = to_binary(pair_holds(g))
        real_s.append(score(critic, a23, rb, device)); gen_s.append(score(critic, a23, gb, device))
        shuf_s.append(score(critic, a23, shuffle16(rb, rng), device)); used += 1

    print(f"\n=== OBJECTIVE gate: does the taste critic SEE placement? ({used} chaotic-Hard songs) ===")
    print(f"  critic P(real), matched density:")
    print(f"    REAL chart            {np.mean(real_s):.3f}")
    print(f"    v4-gen (awkward)      {np.mean(gen_s):.3f}   (REAL - gen = {np.mean(real_s)-np.mean(gen_s):+.3f})")
    print(f"    shuf16 (placement broken, amount/backbone/panels kept)  {np.mean(shuf_s):.3f}   "
          f"(REAL - shuf = {np.mean(real_s)-np.mean(shuf_s):+.3f})")
    real_gt_gen = np.mean(np.array(real_s) > np.array(gen_s)); real_gt_shuf = np.mean(np.array(real_s) > np.array(shuf_s))
    print(f"  per-song REAL>gen {real_gt_gen*100:.0f}%   REAL>shuf16 {real_gt_shuf*100:.0f}%")
    print(f"\n  REAL clearly > gen AND > shuf16 -> critic SEES placement -> can supervise it (Phase-3 objective).")
    print(f"  REAL ~ gen/shuf16 -> critic blind to placement at fixed amount -> critic-as-objective is OUT.")


if __name__ == '__main__':
    main()
