#!/usr/bin/env python3
"""DATA gate (notes/phase3_generative_design.md): are there multiple human chartings of the same song
(distribution-supervision data)? Group chart files by #TITLE, count titles appearing in >=2 distinct packs.
Caveat: cross-pack same-title may be COPIES; verify-differ is a follow-up. Fast (regex on file head)."""
import glob, re, os
from collections import defaultdict, Counter
files = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
def tag(p, name):
    try: txt = open(p, encoding='utf-8', errors='ignore').read(4000)
    except Exception: return None
    m = re.search(rf'#{name}:([^;]*);', txt); return m.group(1).strip() if m else None
tf = defaultdict(set)
for f in files:
    t = tag(f, 'TITLE')
    if not t: continue
    tf[re.sub(r'\s+', ' ', t.lower()).strip()].add((os.path.dirname(os.path.dirname(f)), f))
multi = {k: v for k, v in tf.items() if len({p for p, _ in v}) >= 2}
print(f"chart files {len(files)}; distinct titles {len(tf)}; multi-pack titles {len(multi)} "
      f"({100*len(multi)/max(len(tf),1):.1f}%)")
print("packs-per-multi-title:", dict(sorted(Counter(len({p for p,_ in v}) for v in multi.values()).items())))
