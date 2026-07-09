# Sliding-window onset — the "dead tail" on long BYO songs (2026-07-08)

Follow-on to `byo_audio_alignment_findings.md` (FM3, the PE-extend truncation fix) and `byo_offset_detection_findings.md`.
The Toulouse offset A/B surfaced a NEW defect: with the grid correct (anchor-to-beat), the chart still had **dead
spots** — worst in the last ~40s. Root-caused, fixed with a sliding-window onset head, offline-validated. **BY-EAR
GATE PENDING** (does the density redistribution feel right).

## Symptom → decomposition (energy overlay)
`toulouse_anchor_beat` (v2, 128 BPM, 134 measures ≈ 251s) played "on grid but choreography kinda bad, dead spots
where it should be charting notes." Overlaying an audio RMS/onset-strength envelope against per-8-measure note
density split the "dead spots" into TWO different bugs with OPPOSITE audio signatures:
- **Mid-song m56–63 (~105s):** RMS **0.128** = the lowest band in the song (a genuine breakdown). Thinning there is
  mostly musically CORRECT; the deployed path's full 15-s void is mild breathe-governor over-rest. IN-context. Not
  a length bug. (Minor; a `stamina_breathe_floor` nudge if anything.)
- **OOD tail m112–134 (210–251s):** the outro is the **LOUDEST, most onset-dense section of the whole song**
  (onset-strength 1.31→2.03, the song's peak) — yet the chart thinned/died. High energy → neither the audio nor the
  breathe governor can explain it. The only special thing: it is past the **5400-frame / 112.5-measure trained
  context** (`V2_MSL`). This is the **PE-extrapolation ceiling**, and it is the headline defect.

## Attribution (deployment-matched probes; HARNESS→DATA→MODEL)
The dead tail is NOT what the earlier FM3 scoping probe implied ("graceful extrapolation"). Corrections:
1. **NOT the onset encoder's mean p collapsing.** Full-song onset `p` in the tail (0.196) ≈ in-context (0.195).
   (My FIRST probe measured the MEAN — an experiment-design Rule 1 miss: the mean was blind to the real effect.)
2. **NOT stamina.** Baseline vs `--stamina_ceiling 0`: the tail (m112+) is BYTE-IDENTICAL (38/16/36 both). Stamina
   only revived two mid-song dips.
3. **NOT holds** swallowing notes. The tail has ZERO hold-heads; it genuinely fires few onsets (9–19 onset-rows/band
   vs 34–93 mid-song).
4. **IT IS the onset path via PEAK COMPRESSION under the global tau.** Deployment-matched probe (conditioned `p`,
   real tau, density 0.107), full-PE vs windowed local-PE:
   - tail p95 (95th pct of p): full **0.570–0.599** vs windowed **0.714–0.749**; the abs-PE OOD flattens the tail
     onset-probability PEAKS toward the mean.
   - because density is a **global quantile** (tau = 0.572), those flattened tail peaks fall BELOW tau → almost
     nothing fires. Restoring the peaks (local PE) clears tau again.
   - **tail onsets: full = 44 → windowed = 127 (2.9×)**, restoring the tail to mid-song levels. The note budget is a
     fixed quantile, so windowing REDISTRIBUTES it from the over-served intro into the starved outro (in-ctx 643→560,
     tail 44→127; total 687 both).

## The fix — sliding-window ONSET (decoder keeps extrapolating)
The `AudioEncoder` is a Conv1D with **no positional encoding** (translation-equivariant) → `memory = encode_audio()`
is length-safe; the ONLY OOD components are the two `pos_encoding` sites. Fix = run the **non-causal onset encoder**
over IN-DISTRIBUTION local-PE windows; leave the **causal choreography decoder** on the extended absolute PE (its
panel-entropy extrapolates gracefully at 2× context, per FM3, and it can't change the onset count anyway). The two
compose.

- `LayeredTypedChartGenerator.onset_logits(..., window=W)` (`typed_model.py`): when `T > W`, tile the song with
  `W`-frame windows at hop `W//2`, run the onset encoder on each `memory[:, s:e]` slice (`pos_encoding` then adds
  `pe[:, :L]` = LOCAL positions 0..L-1, in-distribution), triangular-blend the overlaps on LOGITS → (B,T). `window=None`
  or `T <= W` → the plain single pass, **BYTE-IDENTICAL** (short songs / v1 / the exporter untouched).
- **Single-sourced** into BOTH tau (`decode_harness.conditioned_p_onset(window=)`) and decode
  (`generate(onset_window=)`) so they can't drift — the §3 tau-coupling holds by construction.
- `scripts/generate.py`: `onset_window = V2_MSL (5400)` on v2, the checkpoint PE size on v1; passed to both the tau
  call and `gen_kwargs`. A song that fits is a no-op. The PE EXTENSION (FM3) stays — it now serves only the decoder.
- Blend on LOGITS (not p) so the caller's phase-calib / CFG / sigmoid / tau all apply downstream identically.

## Validation
- **End-to-end (the artifact, Rule 8):** Toulouse tail m112–134: **90 notes w/ a 5-dead-measure hole → 143 notes,
  0 dead measures**, now tracking the climax. Density preserved (0.107). Redistribution pulls budget from the intro/
  mid (m8–47, m88–103 thin ~15–30%) into the tail — expected (per-window local tau competition), every band still
  healthy (0 dead outside the real breakdown).
- **Contract (unit tests, `tests/test_generation.py`):** `test_onset_window_noop_when_song_fits` (byte-identical when
  the song fits) + `test_onset_window_engages_for_long_song` (engages, finite, differs, generate() threads it). 38/38
  pass. `tools/check_export_defaults.py` still 25 ✓ (exporter defaults untouched).

## Open / by-ear
- **BY-EAR GATE:** does the intro-thins / outro-fills REDISTRIBUTION feel right? (Offline the dead tail is fixed; the
  rebalance is a musical-taste call.) Deliverable for the user: `~/sm-generated/toulouse_win_anchor` (anchor-beat
  audio + sliding window) vs the old `toulouse_anchor_beat` (on-grid, dead tail).
- **Decoder windowing NOT done** (deliberate): the choreography decoder still extrapolates absolute PE past context.
  Entropy stays ~1.0, so tail choreography QUALITY is assessed graceful; if by-ear the tail patterns read off,
  windowing the decoder (KV-flush at seams + overlap re-warm via the existing `boundary_reset`) is the next lever —
  harder + riskier, not evidence-required yet.
- The `min_onset_gap` / `no_fast_jump` / footspeed floor all still apply on the windowed onsets (they run after).

Related: `byo_audio_alignment_findings.md` (FM3), `byo_offset_detection_findings.md`, `notes/playtest_log.md`
(2026-07-08), `conditioning-mechanics §6`, `generation-defaults §0`, memory [[byo-audio-bpm-footgun]],
lineage `experiment_lineage/byo-audio-alignment-arc.md` Ch.3.
