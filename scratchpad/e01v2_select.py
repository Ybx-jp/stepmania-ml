"""E0.1-v2 binned song selector (2026-07-13, rev2) — BPM x LENGTH factorial.

3 BPM bins (slow/mid/fast) x 3 LENGTH bins (short/med/long) x 3 songs = 27 songs.
Diverse, sane Hard songs (NOT axis-maxed reference charts). Since conditioning is now a FIXED
MODERATE --style (NOT --match_radar), the song's own radar doesn't drive generation, so selection
optimizes for DIVERSE AUDIO (bpm x length) + avoids only the rich-maxed 'gibberish' pathology.

Length bins = terciles of the Hard-candidate pool (so cells populate). Prints an auditable table
+ a --song_filter string.
"""
import warnings, sys, glob
from pathlib import Path
import numpy as np
ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.stepmania_parser import StepManiaParser
from src.generation.decode_harness import make_feature_extractor

set_seed(42)
HARD = 3
BPM_BINS = [('slow', 0, 130), ('mid', 130, 175), ('fast', 175, 1e9)]
PER_CELL = 3

cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
_, val_files, _ = create_data_splits(cf, random_state=42)
spec = make_feature_extractor("highres_v2")
_, val_ds, _ = create_datasets(train_files=[], val_files=val_files, test_files=[], audio_dir="data",
                               max_sequence_length=5400, feature_extractor=spec.extractor,
                               cache_dir="cache/samples_v3_48th", parser=StepManiaParser.for_v2())

# 1) Hard candidates, MODERATE radar (no maxed dim -> avoids the 'gibberish' the user rejected), deduped by song
cand = []
seen_title = {}
for i, m in enumerate(val_ds.valid_samples):
    if m.get('difficulty_class') != HARD or 'groove_radar' not in m:
        continue
    r = np.asarray(m['groove_radar'].to_vector())
    if not (0.20 <= r[:4].max() <= 0.78):    # MODERATE: no maxed intensity dim, not dead
        continue
    if r[4] > 0.5:                           # exclude extreme-chaos (broken axis)
        continue
    bpm = float(getattr(m['chart'], 'bpm', 0) or 0)
    if not (60 <= bpm <= 320):
        continue
    if ',' in m['chart_file']:          # comma breaks the exporter's --song_filter split
        continue
    title = m['chart'].title or Path(m['chart_file']).stem
    if title.lower() in seen_title:          # dedup: one entry per song
        continue
    seen_title[title.lower()] = 1
    cand.append({'i': i, 'bpm': bpm, 'r': r, 'title': title})
print(f"moderate-Hard candidates (deduped): {len(cand)}")

# 2) length (cache-hit) for all candidates
for c in cand:
    c['T'] = int(val_ds[c['i']]['mask'].sum().item())

# 3) fill the 3x3 grid: length terciles computed WITHIN each BPM bin (so fast/short populates)
picked = []
for bname, blo, bhi in BPM_BINS:
    binsongs = sorted([c for c in cand if blo <= c['bpm'] < bhi], key=lambda c: c['T'])
    if len(binsongs) < PER_CELL * 3:
        print(f"  [!] bpm bin {bname}: only {len(binsongs)} moderate songs (<{PER_CELL*3})")
    q33, q67 = np.percentile([c['T'] for c in binsongs], [33, 67]) if binsongs else (0, 0)
    for lname, sub in (('short', [c for c in binsongs if c['T'] < q33]),
                       ('med',   [c for c in binsongs if q33 <= c['T'] < q67]),
                       ('long',  [c for c in binsongs if c['T'] >= q67])):
        if len(sub) <= PER_CELL:
            take = sub
            if len(sub) < PER_CELL:
                print(f"  [!] cell {bname}/{lname}: only {len(sub)} songs")
        else:
            idx = np.linspace(0, len(sub) - 1, PER_CELL).round().astype(int)   # spread within the cell
            take = [sub[j] for j in idx]
        for c in take:
            c['bin'] = f"{bname}/{lname}"
        picked += take

print(f"\n{'cell':11s} {'title':26s} {'bpm':>6s} {'len':>6s} {'strm':>5s} {'air':>5s} {'frz':>5s} {'chao':>5s}")
for c in picked:
    r = c['r']
    print(f"{c['bin']:11s} {c['title'][:26]:26s} {c['bpm']:>6.1f} {c['T']:>6d} "
          f"{r[0]:>5.2f} {r[2]:>5.2f} {r[3]:>5.2f} {r[4]:>5.2f}")

# emit UNIQUE chart_file paths for --song_filter (a full .sm path can't over-match, unlike short titles)
paths = [val_ds.valid_samples[c['i']]['chart_file'] for c in picked]
print(f"\nn songs picked = {len(picked)}")
print(f"paths with commas (would break --song_filter split): {sum(',' in p for p in paths)}")
Path("outputs/e01v2_paths.txt").write_text(",".join(paths))
print("wrote outputs/e01v2_paths.txt (comma-joined chart_file paths for --song_filter)")
