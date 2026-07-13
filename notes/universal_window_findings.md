# Universal sub-train-length window — short-song END-degeneration (taste-critic arc, defect #1 sibling)

**2026-07-12, branch `explore/taste-critic-quality-resolution`.** Closes the HANDOFF/lineage OPEN priority: the
UNIVERSAL sub-train-length window. Companion to `notes/playtest_log.md` (the 2026-07-11 hangover / tail-collapse
chain), lineage `taste-critic-arc.md`; mechanism in `conditioning-mechanics §6` (onset sliding-window + hangover).

## The premise, measured (exp-design Rule 5/6 — the cheapest decisive fact FIRST)
The onset head is **non-causal** and reads `pos_encoding(memory)` over **ABSOLUTE** positions 0..T. So how well a
position behaves depends on how many TRAINING songs had real content there. Measured from `cache/samples_v3_48th/
train` (`length` field, N=4547):
- **median 3120, p75 3648, p80 3797, p90 4122, p95 4378, MAX 5128** — no training song ever filled the 5400 buffer.
- abs-PE exposure collapses well before 5400: only **31% / 13% / 6%** of songs reach position **3500 / 4000 / 4320**.

Yet today `generate.py` pins `onset_window = V2_MSL = 5400` and windowing fires only when `T > window`. So **every
song longer than ~3500 already sits its END in the under-trained abs-PE tail** and never gets windowed. The user's
instinct was mechanistically correct.

## The onset probe — the RIGHT population, no decode (`probe_universal_window.py`, n=60/band)
Reads CACHED VAL samples (already the deployed 42-dim highres_v2 features, frame-aligned to the REAL human chart =
Rule-5 ground truth). Measures tail (last 20%) vs body onset quality, single-pass (today) vs windowed, per band.
**Fixes the predecessor's wrong-population error** (`onset_window_sweep` tested smaller-W on ONE long song's
MIDDLES — exp-design Rule 5/11): here the population is SHORT val songs whose END lands in the under-trained tail.

| band (train-len) | arm | tail AUC−body | tail recall (real notes firing) | tail Herfindahl (backbone) |
|---|---|---|---|---|
| **under-trained 3800–5128** | single-pass | −0.130 | **0.705 → 0.301** | 0.610 → **0.342** (smear) |
| | **W3600** | −0.024 | 0.692 → **0.630** | → **0.607** (human=0.600) |
| | **W3000** | −0.023 | 0.650 → **0.601** | → **0.610** |
| | W4320 | −0.101 | 0.702 → 0.404 | → 0.405 (~no-op) |
| **transition 3500–3800** | single-pass | −0.098 | **0.729 → 0.335** | 0.672 → 0.479 |
| | W3600 | −0.031 | 0.700 → **0.567** | → 0.615 (human 0.646) |
| **CONTROL <3000** | ALL arms | −0.000 | 0.719 → 0.753 (no degen) | byte-identical (no-op) |

**Conclusions (decisive, n=60):**
1. **Short-song end-degeneration is REAL and severe** — under the global tau, single-pass fires only ~30% of the
   real TAIL notes on the under-trained band, and the tail backbone smears toward uniform (Herfindahl 0.61→0.34).
2. **A universal window FIXES it** — W3000/W3600 restore tail recall (~0.60–0.63) and the tail backbone Herfindahl
   to **essentially exactly the human reference** (0.607–0.610 vs 0.600). Body AUC/Herfindahl preserved; the body
   recall dip is REDISTRIBUTION (fixed density → the tau budget reallocates from the over-served body to the
   starved tail), not damage.
3. **Specificity is clean** — the CONTROL band (<3000, tail still well-trained) shows NO single-pass degeneration
   AND every window arm is BYTE-IDENTICAL there (song shorter than W → no-op). Rule-4/11: the fix can't hurt
   already-fine songs, and the effect has dynamic range.
4. **W4320 barely fires** (only on T>4320) → it's a near-no-op on the very bands that degrade. This is the sharpest
   MECHANISTIC proof it's an abs-PE windowing effect: the window must be SMALLER than the degeneration onset (~3500)
   to fire — exactly what the PE-exposure story predicts.

**Recommended default = W ≈ 3600** (= p75 train length: 75% of training songs fit fully in one window; the rest
tile). Restores the tail while disturbing the body least of the fixing arms. W3000 = the more aggressive alternative
(fires on more songs, marginally more body disturbance). auto-hangover = W//2 (song-end centers in a window;
silence pad, the physically-correct future).

## Wiring (off by default; canonical BYTE-IDENTICAL, `tools/check_export_defaults.py` still 25 ✓)
Single-sourced through the existing `decode_harness.conditioned_p_onset(window=, tail_hangover=)` (tau) +
`generate(onset_window=, onset_tail_hangover=)` (decode) so the two can't drift (conditioning-mechanics §6).
- `export_typed_samples.py --onset_window W` (default 0 = off = single-pass = canonical) `--onset_tail_hangover
  auto|N` + `--ab_onset_window` (Challenge = WINDOWED fix / Edit = single-pass, each its OWN tau, shared RNG =
  the by-ear gate).
- `generate.py --onset_window auto|N` (default `auto` = V2_MSL/trained-ctx = current behavior, byte-identical;
  N e.g. 3600 = the universal window). Symmetric with the exporter so the DEFAULT flip to ~3600 (both CLIs) is a
  one-line change per CLI once the ear confirms — do NOT flip the default before the by-ear A/B lands.

## Decoded-chart check — CONFIRMED (the onset fix survives the AR decode; `probe_universal_window_decoded.py`)
`--ab_onset_window` export (W3600) on the 3 longest under-trained val songs, Hard; each `.sm` = Challenge
(WINDOWED fix) / Edit (single-pass, its OWN recomputed tau) / human original. Tail = last 20% of the DECODED chart:

| song (len) | arm | tail quarter% | tail jitter% | tail Herf (human ref) | tail notes |
|---|---|---|---|---|---|
| Chocolate Smile (4992) | **windowed** | **69** | **0** | **0.528** (hum 0.549) | 75 |
| | single-pass | 8 (collapse) | 12 | 0.303 | 74 |
| DOMINION (4973) | **windowed** | **33** | **0** | **0.302** (hum 0.393) | **227** |
| | single-pass | 4 (collapse) | 4 | 0.260 | 136 |
| Luckgakist (4800) | **windowed** | **44** | **0** | 0.307 (hum 0.503) | **173** |
| | single-pass | 31 | 4 | 0.284 | 108 |

**The single-pass tail quarter-backbone COLLAPSES to 4–8% on 2/3 songs** (a spine is normally ~30–70% quarters);
the notes that fire scatter onto off-beats + pure-48th JITTER cells (4–12%). **Windowed restores the quarter
backbone AND drives tail jitter to exactly 0 on ALL three**, tail Herfindahl closest to human on all three, and
recovers the DEAD TAIL (DOMINION 227 vs 136 notes). The onset fix propagates cleanly through the pattern/type
heads (the AR decoder was OOD on the degenerate onset stream; now fed an in-distribution one).

## BY-EAR — ✅ PASSED (2026-07-12, `notes/playtest_log.md`) → DEFAULT FLIPPED
A/B (`~/sm-generated/universal_window_ab/`, Challenge=windowed W3600 / Edit=single-pass / human) on the 3 long
songs: **windowed WON on all three** — DOMINION "was great!" (user names the single-pass defect precisely:
"degenerate towards the end"), GUMI/チョコレートスマイル "windowed was better", "&"/ラクガキスト windowed "fine"
(both arms bland = per-song CONDITIONING, not the window — user's own read). **DEFAULT FLIPPED: `UNIVERSAL_ONSET_
WINDOW=3600` in `decode_defaults.py`; both CLIs default v2 → 3600** (exporter `--onset_window` default 3600 gated
subdiv!=4; generate.py `--onset_window auto`→3600 for v2 + `--onset_tail_hangover auto`). `check_export_defaults.py`
now 27 ✓ (added `onset_window`/`onset_tail_hangover`). v1 + short-fit songs = byte-identical no-op. Disable via
`--onset_window 0`.

**RNG note (user asked "same rng?"):** yes — but shared RNG only keeps arms identical UP TO their first divergence,
and the window diverges them at the ONSET level + shifts the GLOBAL tau (single-pass tau is set over a p-dist whose
tail is flattened → different firing from the INTRO on), so the two charts legitimately differ globally. Not a bug.

## New residual lead — [H-winddown] (SEPARATE from this fix, pre-existing)
DOMINION windowed "did not wind down properly, 1-2 measures after silence still streaming" — but single-pass "did
sorta the same", so NEITHER arm tapers → the wind-down weakness PRE-DATES windowing (windowing just makes the
over-run a coherent stream vs single-pass's disjointed one). Candidate mechanism: the window RESTORES the tail
`p_onset` peaks → the stamina BREATHE arc (energy = smoothed p_onset) reads higher tail energy → thins the outro
LESS. Cheap probe queued (`playtest_log.md` action list); do NOT build blind. Related: `stamina_breathe_floor`
abrupt-ending history (cond-mech §8c).
