# Constraint-relaxation roadmap

*2026-06-20.* The project began with Phase-1 simplifications (see `docs/development_status.md` "Phase 1
Scope (Enforced)" and `docs/chart_representation.md` "Excluded Patterns" / "Future Considerations").
Some have since been relaxed; this note tracks **what's still constrained, the decision rule for relaxing,
and where each one sits** so we don't relax preemptively or forget what's on the table.

## Decision rule

**Relax a constraint when it becomes the binding limiter on the quality you're chasing, or when it gates
a use case you actually want — not preemptively.** Each relaxation adds representation complexity that
dilutes the core learning signal and multiplies debugging confounds. Lock musicality on the clean
single-BPM 16th grid first (cheap, fast iteration), *then* expand the data layer once the modeling
approach is proven. Right now the binding limiter is musicality (H4 syncopation, H5 structure), which is
independent of every constraint below — except hands (see below), which turns out to bias the data.

## Status

| constraint | status | notes |
|---|---|---|
| 4-panel `dance-single` only | enforced | doubles/other modes out of scope |
| holds | **RELAXED** | typed model generates hold-head/tail; hold-aware decode |
| rolls | representable, untrained | `roll_head` is already one of the 4 types; just rare in data |
| multi-difficulty | **RELAXED** | difficulty conditioning works |
| **fixed BPM** (no tempo changes/stops) | enforced | foundational — see below |
| **max 2 simultaneous** (no hands/quads) | enforced **in data**, NOT in model | see below — biases "Hard" |
| 16th-note resolution (no triplets/swing) | enforced | pairs with variable BPM as data-layer v2 |
| mines | excluded | new symbol/channel |

## Per-constraint placement

### Hands / quads (max-2) — RE-EXAMINE NOW (relevant to the live Hard-tameness thread)
**The model is not the limiter — the data layer is.** The typed pattern vocabulary is 15-way
(`NUM_PATTERNS = 2^4 - 1`), so it *already represents* 3-panel hands and 4-panel quads. But
`StepManiaParser.validate_pattern_quality` rejects any difficulty whose tensor has a single frame with
>2 simultaneous panels, and it's called per-difficulty in `process_chart` (the dataset's path). Since
hands/brackets are a hallmark of high-difficulty DDR, this **disproportionately excludes real Hard
charts** → the dataset's "Hard" is the tame, hands-free subset → the model never sees hands and can't
produce the intensity real 11s have. This compounds H4 (on-grid tameness) as a *second*, independent
cause of "generated Hard feels too tame for an 11."
- **Quantified** (`diag_hands_filter.py`, 3501 files parsed raw):

  | difficulty | #diffs | rejected(>2) | rej% | has-quad% |
  |---|---|---|---|---|
  | Beginner | 1724 | 546 | 31.7% | 29.9% |
  | Easy | 1461 | 50 | 3.4% | 2.1% |
  | Medium | 1888 | 251 | 13.3% | 5.0% |
  | **Hard** | 4613 | 2560 | **55.5%** | **41.5%** |

  **Over half of all real Hard difficulties are excluded**, vs 3.4% of Easy — a massive difficulty-biased
  cull. The dataset's "Hard" is the tame ~44% that avoid 3+ simultaneous panels.
- **Nuance:** the filter uses `convert_to_tensor` (binary *occupancy*), so a frame counts 3+ if 3 panels
  are simultaneously OCCUPIED — which includes overlapping **holds**, not just simultaneous taps. (That's
  why even Beginner shows ~30% rejected / 8% "hands-frame": sustained multi-panel hold sections, not
  literal hands.) So the guard is doubly over-aggressive: it discards both real hands AND legit
  multi-hold sections. Either way it removes >half of Hard. If we want to keep some control, the right
  refinement is to count simultaneous *taps* (symbol-1) rather than occupancy — but the typed pipeline
  already models holds + 15-way patterns, so simply removing the guard is defensible.
- **It's a stale Phase-1 guard the typed model outgrew:** `NUM_PATTERNS=15` represents hands/quads and
  the hold-aware decode handles overlapping holds. The max-2 check predates both.
- **Fix (cheap, no representation change):** stop rejecting >2; either keep hands as-is (model already
  supports 15-way patterns) or cap-by-difficulty. Then rebuild the cache and retrain. The only real cost
  is re-extraction + retrain, and it directly targets Hard quality.
- **Caveat:** hands raise pad-playability stakes (foot legality), so pair with the crossover/jump-in-hold
  decode constraints when generating.

### Fixed BPM — DEFER (heavy, foundational; do as a deliberate "data layer v2")
Not a toggle. The whole pipeline rests on `hop_length = sr·60/(bpm·4)` → "one audio frame = one 16th
note." The high-res onset feature (H4 fix), `metric_phase`, and the onset head all assume this fixed
grid. Variable BPM/stops/warps means a **beat-synchronous re-gridding** of audio — major data-layer
surgery. **Right time:** when you want to *substantially expand the dataset* (many real songs have BPM
changes/stops, so excluding them biases + shrinks the corpus) or *target BPM-variable packs*, and after
musicality has plateaued so data quantity becomes the limiter. Doing it now would destabilize the exact
grid the current H4 work depends on.

### 16th resolution (triplets / swing) — DEFER with variable BPM
Finer grid (e.g. 48ths for triplets) = more frames + a grid change; same surgery class. Bundle into
data-layer v2. Until then, triplet/swing songs are mis-quantized or excluded.

### Rolls — RELAX OPPORTUNISTICALLY (cheap)
`roll_head` already exists in the type vocabulary, so rolls are nearly free to start training properly
(they're just rare in the data). Worth a look once holds are solid; low value, low cost.

### Mines — LOW PRIORITY
A new symbol/channel. Only worth it if targeting gimmick/expert charts where mines are part of the
difficulty. Defer indefinitely unless a use case demands it.

## Suggested sequencing
1. **Now:** H4 high-res-onset retrain (in flight) + quantify the hands filter (this note).
2. **Near-term (if hands rejection is high):** drop the max-2 data filter, rebuild cache, retrain — a
   cheap, direct lever on Hard tameness, complementary to H4.
3. **After musicality plateaus / for dataset expansion:** data-layer v2 = variable BPM + finer
   resolution (beat-synchronous re-gridding), the big foundational step.
4. **Opportunistic:** rolls; (mines only on demand).
