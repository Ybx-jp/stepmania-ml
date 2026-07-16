"""critic-v3 validation gates (2026-07-13) — ground the training numbers in the artifact.

(1) LENGTH gate: does the tail-corruption response hold on the LONG songs (>2304 frames) the
    old graded_v2 critic TRUNCATED (and so scored blind)? Stratified by length.
(2) LOCALITY gate: inject a localized defect in the FIRST / MIDDLE / LAST third; does the score
    drop concentrate in the windows overlapping that third (localization), not elsewhere?
"""
import warnings, sys, glob
from pathlib import Path
import numpy as np, torch
ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.stepmania_parser import StepManiaParser
from src.generation.decode_harness import make_feature_extractor
from experiments.realism_critic.windowed_critic import WindowedLocalCritic
from experiments.realism_critic.train_critic_v3 import collect, corrupt_jitter_typed, to_t, V2_MSL

set_seed(42)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
_, val_files, _ = create_data_splits(cf, random_state=42)
spec = make_feature_extractor("highres_v2")
_, val_ds, _ = create_datasets(train_files=[], val_files=val_files, test_files=[], audio_dir="data",
                               max_sequence_length=V2_MSL, feature_extractor=spec.extractor,
                               cache_dir="cache/samples_v3_48th", parser=StepManiaParser.for_v2())
val = collect(val_ds, 200, V2_MSL)

ck = torch.load("checkpoints/realism_critic_v3/best_val.pt", map_location=dev)
model = WindowedLocalCritic(audio_dim=42, scales=ck['scales'], softmin_beta=ck['softmin_beta']).to(dev)
model.load_state_dict(ck['state_dict']); model.eval()
rng = np.random.default_rng(0)


@torch.no_grad()
def region_drop(song, lo, hi):
    """Corrupt only rows in [lo,hi) frac of the song; return (drop in overlapping windows, drop elsewhere)."""
    typed = song['typed']; T = typed.shape[0]; a = int(T * lo); b = int(T * hi)
    corr = corrupt_jitter_typed(typed, 1.0, rng, rows_filter=lambda t: a <= t < b)
    at = torch.from_numpy(song['audio']).float().to(dev)[None]
    ch = torch.stack([to_t(typed, dev), to_t(corr, dev)])
    _, allm, layout = model(at.expand(2, -1, -1), ch, return_windows=True)
    d = (allm[0] - allm[1]).cpu().numpy()                      # real - corrupt, per window
    overlap = np.array([(s < b and e > a) for (s, e, W) in layout])
    return (float(d[overlap].mean()) if overlap.any() else np.nan,
            float(d[~overlap].mean()) if (~overlap).any() else np.nan)


# (1) LENGTH gate — tail(last third) corruption, stratified by length
bins = {'short <2304': [], 'mid 2304-3600': [], 'long >3600': []}
for s in val:
    T = s['typed'].shape[0]
    k = 'short <2304' if T < 2304 else ('mid 2304-3600' if T < 3600 else 'long >3600')
    ov, off = region_drop(s, 2 / 3, 1.0)
    bins[k].append(ov)
print("=== (1) LENGTH gate: tail-corruption drop in tail windows, by song length ===")
print(f"{'bin':16s} {'n':>4s} {'tail_win_drop':>14s}   (>0 = critic sees tail quality; the old critic was BLIND here for >2304)")
for k, v in bins.items():
    v = [x for x in v if not np.isnan(x)]
    print(f"{k:16s} {len(v):>4d} {np.mean(v) if v else float('nan'):>14.2f}")

# (2) LOCALITY gate — where does the drop land when the defect is in the first/middle/last third?
print("\n=== (2) LOCALITY gate: inject defect in one third -> drop should concentrate in overlapping windows ===")
print(f"{'defect region':16s} {'overlap_drop':>13s} {'elsewhere_drop':>15s}   (localized iff overlap >> elsewhere)")
for name, (lo, hi) in [('first third', (0, 1 / 3)), ('middle third', (1 / 3, 2 / 3)), ('last third', (2 / 3, 1.0))]:
    ovs, offs = zip(*[region_drop(s, lo, hi) for s in val])
    ovs = [x for x in ovs if not np.isnan(x)]; offs = [x for x in offs if not np.isnan(x)]
    print(f"{name:16s} {np.mean(ovs):>13.2f} {np.mean(offs):>15.2f}")
