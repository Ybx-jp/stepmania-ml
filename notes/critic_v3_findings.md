# critic-v3 (WindowedLocalCritic) — build + first-train findings

*2026-07-13, taste-critic arc. The FULL rebuild the user chose after catching the graded_v2 critic's
`max_len=2304` tail-truncation flaw. Closes 3 of 4 captured gaps in one from-scratch model; the preference
objective (E2) rides on top later once E0.1/playtest labels exist. Code: `experiments/realism_critic/
windowed_critic.py` (model) + `train_critic_v3.py` (trainer) + `eval_critic_v3_gates.py` (gates).
Checkpoint: `checkpoints/realism_critic_v3/best_val.pt` (audio_dim=42, 48th grid, scales
[(2304,1152),(1152,576),(576,288)], softmin β=1.0, val jitter-mono 0.985, tail_drop +5.29).*

## What it is
- **(iv) length/locality:** soft-min over MULTI-SCALE overlapping windows (tail always covered by `start=T-W`).
  Length-agnostic (no 2304 truncation); soft-min = worst-region-dominates, a FIXED (non-gameable) aggregation
  for the E4 optimization ladder. Exposes per-window scores = locality.
- **(ii) audio:** full 42-dim highres_v2 (graded_v2 sliced `audio[:, :23]`).
- **(i) chart:** TYPED per-panel symbols {0 none,1 tap,2 hold-head,3 tail,4 roll-head} via a shared Embedding
  (graded_v2 binarized → presence-blind to hold-type / defect #3).
- Objective (pre-labels): graded corruption LADDER (jitter/panel/shift, rank+anchor) on the soft-min song score
  + a LOCALITY term (a tail-only jitter must drop TAIL windows, not body). From scratch (warm-start broken by the
  42-dim + typed change). Reuses the O(T) Conv1DBackbone.

## Results — first train (1200 train / 200 val, early-stop ep7, best ep4)
### ✅ Strong wins (the reason v3 exists)
- **R1 jitter (the f48 sub-16th axis):** mono **0.98**, ladder means well-spread `[+2.44 … −6.14]`. Grades the
  sub-16th placement the 16th critic couldn't see — R1 re-cleared on the new arch.
- **Length fix (gap iv) — DECISIVE, `eval_critic_v3_gates.py`:** tail-corruption drop by length =
  **short 5.13 / mid 5.68 / long >3600 5.46** (n=20/115/65). Uniform across lengths → on the >2304 songs the old
  critic TRUNCATED (180/200 val songs exceed 2304!), v3 responds to tail quality identically. The structural flaw
  is fixed AND measured.
- **Locality (gap i, locality half) — CLEAN:** inject a defect in the first/middle/last third → drop concentrates
  in the OVERLAPPING windows (4.76 / 4.93 / 5.56) with **0.00 leakage** elsewhere. Perfect "where it's bad"
  concentration — the signal the old single-global-score critic structurally could not have.

### ⚠️ Soft spots (reported, not glossed — [[claim-precision]])
- **Panel scramble: mono only 0.56** (peaked ep6). Arrow-CONFIGURATION is the taste cue the interpretability arc
  identified (whole-chart panel-scramble dropped the OLD critic's margin +9.85) → v3 under-grades it.
- **Shift: mono COLLAPSED 0.88 → 0.01** across training. Small global temporal shifts (1–6 frames) become nearly
  invisible.
- **Root hypothesis — within-window MEAN-pool is ORDER-DESTROYING.** Mean over a 2304-frame window is ~invariant
  to (a) a small global roll (shift) and (b) which-panel arrangement (panel), because both are ORDER/arrangement
  signals; jitter survives because it moves notes ON→OFF the 16th grid, a per-frame PHASE change the conv sees and
  mean-pool preserves. So the same mean-pool that bought length-agnosticism + non-gameability TRADED AWAY
  order-sensitivity. v3.1 lever: within-window mean+MAX pool (still fixed) or an order-aware within-window pool;
  or add whole-chart panel-scramble as an explicit anchor. Finer scales (576/288) already preserve more order.

## Verdict + next
The two gaps v3 was BUILT to close (length/locality + f48-visible input) are closed and measured; the two soft
spots are an aggregation trade-off, not a data/harness bug. **Whether panel-weakness MATTERS is an EAR question:**
E1.2-redux (correlate v3's ranking with the user's E0.1 taste ranking) is the arbiter — if v3 tracks the ear, the
soft spots are moot for SELECTION; if not, the order-aware within-window pool is the first fix. Do NOT conclude
"panel-blind = bad critic" before the ear test (the whole arc's discipline: the ear is ground truth).
Pending gates not yet run: 42-vs-23 audio ablation (does the full audio actually discriminate?); E1.2-redux (needs
the E0.1 rankings). See lineage `taste-critic-arc.md`, memory [[taste-critic-transfer]].
