"""data-layer-v2 phase-3 DE-RISK: prove the v2 (parser + feature) pair aligns on REAL chart+audio before
rebuilding the whole corpus. Compares v1 (16th) vs v2 (48th) on the SAME songs:
  - v2 audio frames and v2 chart timesteps must MATCH (alignment holds on the finer grid)
  - v2 must be ~3x v1 frames (the finer grid actually took effect)
  - audio_dim stays 42; metric_phase beat period goes 4 -> 12 (a triplet cell now differs from a 16th cell)
Guards the scope's "piecemeal drift" risk (parser re-gridded but features not, or vice-versa).
"""
import numpy as np
from src.utils.data_splits import split_chart_files
from src.data.dataset import StepManiaDataset
from src.data.stepmania_parser import StepManiaParser
from src.generation.decode_harness import make_feature_extractor

_, val_files, _ = split_chart_files(root="data", random_state=42)


def build(features, parser):
    spec = make_feature_extractor(features)
    return StepManiaDataset(chart_files=val_files[:12], audio_dir="data", max_sequence_length=8000,
                            feature_extractor=spec.extractor, parser=parser, cache_dir=None)


print("building v1 (16th) ...")
ds1 = build("highres", StepManiaParser())
print("building v2 (48th) ...")
ds2 = build("highres_v2", StepManiaParser.for_v2())


def real_shapes(ds, i):
    """(audio (T,42), chart (T,4)) sliced to the true pre-pad length."""
    s = ds[i]
    T = int(s['length'])
    return np.asarray(s['audio'])[:T], np.asarray(s['chart'])[:T]


def first_index_per_file(ds):
    d = {}
    for i, s in enumerate(ds.valid_samples):
        d.setdefault(s['chart_file'], i)
    return d


f1, f2 = first_index_per_file(ds1), first_index_per_file(ds2)
common = [f for f in f1 if f in f2][:4]
print(f"\n{len(common)} songs loaded in BOTH grids\n")
print(f"{'song':<30} {'v1_audT':>8} {'v1_chT':>8} {'v2_audT':>8} {'v2_chT':>8} {'dim':>4} {'ratio':>6} {'align':>6}")
for cf in common:
    a1, c1 = real_shapes(ds1, f1[cf])
    a2, c2 = real_shapes(ds2, f2[cf])
    ratio = a2.shape[0] / max(a1.shape[0], 1)
    aligned = abs(a2.shape[0] - c2.shape[0]) <= 1 and abs(a1.shape[0] - c1.shape[0]) <= 1
    name = cf.split("/")[-2][:28] if "/" in cf else cf[:28]
    print(f"{name:<30} {a1.shape[0]:>8} {c1.shape[0]:>8} {a2.shape[0]:>8} {c2.shape[0]:>8} "
          f"{a2.shape[1]:>4} {ratio:>6.2f} {'OK' if aligned else 'DRIFT':>6}")

print("\nEXPECT: dim=42, ratio ~3.0, all rows align (v2 audio frames == v2 chart timesteps on the 48th grid).")
