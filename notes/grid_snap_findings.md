# Low-difficulty verification + the 16th-grid SNAP (v2 48th grid)

**Question (user, 2026-07-06):** the v2 48th-grid deploy candidate was only ever by-ear-validated on **Hard**
(the triplet set). Can it generate the **lower difficulties** (Beginner/Easy/Medium) coherently?

## Verdict
**Yes — with one characterized, now-fixed decode defect.** No degeneration at any low/mid difficulty; density
tracks the source chart tightly; critic reads Beginner/Easy-appropriate (never railed-Hard); the SPARSE majority of
low-diff songs are ~100% on the 16th grid, matching the human originals. The defect is confined to the BUSIER
low-diff songs and is a pure-48th placement jitter (below).

## Method
Canonical v2 palette (`--checkpoint checkpoints/gen_motif_v2_48th_cont/best_val.pt --features highres_v2`), one
variable swept = difficulty (`--difficulty_select Beginner|Easy|Medium`, seed 42, N=10 each). No `--style`, no
conditioning (clean audio+difficulty), so the density target falls back to each song's own source chart and any
incoherence is the model's, not a knob's. Offline metrics: gen-vs-source density; critic name; and the phase-share
breakdown of the GENERATED vs the embedded ORIGINAL chart on the 48th grid (`phase_breakdown.py`):
- **16th grid** = phases `{0,3,6,9}` (t%12)   **triplet family** = `{2,4,8,10}`   **pure-48th** = `{1,5,7,11}`

## The defect (measured)
Human low/mid charts are ~**0% pure-48th at EVERY difficulty**; v2 sprinkles pure-48th jitter on the busy songs:

| Difficulty | GEN 16th/trip/**48th** % | ORIG 16th/trip/**48th** % |
|---|---|---|
| Beginner | 95 / 1 / **4** | 99 / 1 / **0** |
| Easy     | 93 / 1 / **6** | 100 / 0 / **0** |
| Medium   | 97 / 0 / **3** | 99 / 1 / **0** |

Per-song the excess is on the BUSY charts (See Me Now 20%, SUPER SUMMER DIVE 18%, Gengaozo 18%, 永遠 23%, Deja loin
10%, TS Terminal Strike 13%); the sparse songs are already 100% on-grid. It fades with difficulty (6→3%) but never
flips. Real Medium is STILL 0% pure-48th → the gating boundary is NOT crossed at Medium.

## Attribution (experiment-design — the important part)
- **Hypothesis 1 (WRONG, refuted by A/B): the 16th-unlock (`onset_phase_calib=0,1.0`) causes it.** Regenerating the
  identical Easy set with the unlock OFF (`--onset_phase_calib "0,0"`) did NOT reduce off-grid (See Me Now 20→27%,
  SUPER SUMMER 18→21%) — it slightly ROSE. Note-count decomposition: the unlock moves ON-16th-grid density (turning
  it off removes ~50 on-grid 16ths from See Me Now); the pure-48th COUNT is independent of it (95→117; Deja loin
  22→22 identical). The unlock is **exonerated**. (Caught the wrong fix before shipping it — exp-design Rules 7–9.)
- **True cause: the 48th grid's double edge.** Beat-synced 12-subdivision cells let the audio-only onset head follow
  SUB-16th audio salience (real transients don't land exactly on 16ths). This is the SAME capability that lets v2
  express triplets (the v2 win); it just also admits 48th "in-between" notes on busy DUPLE songs where a human snaps
  to 16ths. The governor/footspeed-floor gate SPACING/JUMPS, not isolated grid POSITION, so they don't catch it.
- **48th noise ≠ triplet expression** (the decomposition that set the fix design): the excess is on `{1,5,7,11}`
  (pure-48th), NOT the triplet family `{2,4,8,10}` (gen triplet share ~1% ≈ real). Triplets are human-legit even at
  Medium (君のハート ORIGINAL is 6% triplet) and the gen actually UNDER-places them → a separate axis (the `b_trip`
  band), not this defect.

## The fix: `grid_snap_offset` (decode-only, no retrain)
`src/generation/decode_defaults.grid_snap_offset(T, subdiv, keep_triplets)` → a (T,) onset-logit offset of `-30`
(≈ hard veto, sigmoid~0) on the vetoed phases, `0` on the kept phases. Kept = the 16th grid `{0, subdiv//4,
subdiv//2, 3·subdiv//4}`, plus the triplet family when `keep_triplets=True`. Rides the exporter's `harm_off_t` slot →
single-sourced into BOTH tau (`conditioned_p_onset extra_offset=`) and decode (`generate onset_logit_offset=`), so it
can't decouple (conditioning-mechanics §6). **v1 (subdiv=4) no-op BY CONSTRUCTION** (kept phases = all 4).

**Confirmed (identical seed-42 sets, one variable):**
- Off-grid **6.6% → 0.0%** on every Easy song; matches the human originals.
- Density **preserved and improved** — note counts move TOWARD the original (See Me Now 464→532 vs orig 530); no
  over-densification (all snap counts ≤ original).
- **Inert on already-clean songs** (byte-identical: KIM 110→110, B4U 175→175…) — touches only the defective songs.
- **keep-triplets mode** (Medium): drives pure-48th → 0 while PRESERVING triplets (See Me Now keeps its 1%), so it
  composes with `--auto_b_trip` (b_trip boosts triplet frames the snap leaves open).

## Ship wiring (2026-07-06, wired to the canonical default per user directive; BY-EAR PENDING)
Exporter defaults flipped (`check_export_defaults.py` 21 → 25 ✓; canonical block in `notes/HANDOFF.md`):
- **`--grid_snap auto`** (choices auto/off/all): auto = keep-triplets 48th-veto for difficulty ≤ Medium
  (`diff_idx≤2`), OFF at Hard (fast 48th runs legit + the v2 win). No per-difficulty threshold needed — the human
  reference (0% pure-48th) is identical across Beginner/Easy/Medium.
- **`--grid_snap_keep_triplets` default True** (`--no-…` = full snap).
- **`--auto_b_trip` default True** (was opt-in), v2-only (detector skipped on v1).

**Boundary NOT tested: Hard.** Hard is the validated v2 region where fast 48th runs / drills plausibly become legit
— left on canonical (snap off) by the auto gate. Confirm the Hard boundary by ear if extending the snap upward.

## By-ear artifacts (installed to ~/sm-generated)
Canonical: `v2_low_beginner`, `v2_low_easy`, `v2_low_medium`. Fix: `v2_low_easy_snap` (full), `v2_low_medium_snap`
(keep-triplets). A/B the busy songs (See Me Now, SUPER SUMMER DIVE, Gengaozo, Deja loin); sparse songs are identical
between groups.

## Tooling
`grid_snap_offset` (decode_defaults), `--grid_snap`/`--grid_snap_keep_triplets` (exporter); analysis
`scratchpad/phase_breakdown.py` (16th/triplet/48th split, gen vs original). Cross-refs: `conditioning-mechanics` §6
(phase levers, single-source), `generation-defaults` (the canonical defaults + guard), the meter-grid arc
([[meter-4-4-grid]]) — the 48th grid is what made this defect BOTH possible (triplets) and measurable.
