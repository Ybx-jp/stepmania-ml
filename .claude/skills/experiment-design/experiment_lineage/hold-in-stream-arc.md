# Lineage: hold-in-stream defect → hold_stream_penalty + footswitch=False (2026-07-02)

**Thread:** localize the fast-song pattern/type-head quality defect WITHIN the head, then fix it at decode time.
**Spun off** [`quality-feature-attribution-arc.md`](quality-feature-attribution-arc.md) (which pinned the defect to
the pattern/type head but did NOT split which-panel vs tap/hold/tail, and pessimistically called it "not a decode
knob"). **Seed:** user by-ear — "the model does a HOLD with a JACK sequence where a human charts a STREAM."
**Primary notes:** `notes/hold_in_stream_findings.md`, `notes/playtest_log.md` (2026-07-02). **Outcome: TWO shipped
decode defaults, no retrain.**

## Hypothesis chain (believed → learned)
1. **"The hold-SPAN machinery degrades on fast songs" (hold-burst / tail-run-long / hold-rate).** → **REFUTED**
   (`probe_bpm_hold_decomp.py`, paired n=90). All three hold-span metrics at the BPM noise floor.
2. **"Then it's a positional HOLD-IN-STREAM substitution."** → **CONFIRMED** (`probe_stream_holdjack.py`, n=40).
3. **"A density-gated hold-head penalty fixes it."** → **CONFIRMED + shipped** (`probe_holdstream_fix.py` + playtests).
4. **"The residual brutal voltage is intrinsic jacks."** → **OVERTURNED by the footswitch diagnostic** — it's
   dominantly a FOOTSWITCH strategy; forbidding footswitch (a new knob) fixed it and played "sooooo much better."

## Probes + verdicts
| probe | question | verdict |
|---|---|---|
| `probe_bpm_hold_decomp.py` | hold-span metrics vs BPM (denoised, paired) | **NULL @ n=90.** tail-length +0.49 = pooled-vs-paired artifact (paired −0.07); holdrate +0.31@n40 → +0.09@n90 |
| `probe_stream_holdjack.py` | do gen holds+jacks land where REAL streams? | **YES root+chain:** holds-in-streams 18% vs real ~0%; hold→jack +11pp (p=0.008); jacking NOT positionally elevated (in=out=0.259) |
| `probe_holdstream_fix.py` | does `hold_stream_penalty` cut stream-holds w/o side effects? | dense-frame holds −0.086 (p<0.001); sparse holds preserved (+0.001 n.s.); hold_burst ↓ not ↑ |
| footswitch A/B (exporter `--ab_footswitch`) | footswitch-dependent vs intrinsic same-panel runs | forbidding footswitch collapses runs 81–85% (HSL/japa1) → voltage is a FOOTSWITCH strategy |

## Attribution corrections (what would have made a conclusion wrong)
- **Pooled-vs-PAIRED baseline (the big one).** `probe_bpm_hold_decomp` first reported tail-run-long r=+0.49 vs BPM —
  but it subtracted a POOLED real constant while real hold-length ALSO rises with BPM. The correct paired
  (gen − song's-own-real) excess is −0.07 (p=0.67). A pooled reference is right for a distance-to-MANIFOLD question
  (the choreography arc) but WRONG for a "does THIS song's gen deviate from THIS song's real more as X rises" slope.
- **Small-n boundary lead.** Holdrate looked significant at n=40 (+0.31, p=.026) and REGRESSED to +0.09 at n=90. A
  marginal p≈.03 at n=40 is where a true small effect and a fishing artifact are indistinguishable → confirm at
  higher n before building. (Same lesson the quality arc taught with ICC/denoising.)
- **Global rate hid a POSITIONAL defect.** The hold-RATE-vs-BPM null (probe 1) did NOT mean holds are fine — the
  defect is positional (holds in stream sections), which a chart-global rate averages away. Needed a co-occurrence
  metric aligned to real streams (probe 2). Metric must see the property AT ITS RESOLUTION (Rule 1).
- **hold_burst was the wrong lens for THIS defect.** hold_burst counts free-foot CROSSES (dist≥1.4) during a hold; a
  hold+JACK is dist-0 → invisible to it. The user's ear caught what the existing battery couldn't.
- **v1 fix "too blunt" (density is a PROXY).** floor 0.25 cut expressive holds (HSL 39→1). Grounding the floor on the
  density-at-holds distribution (expressive ≤0.5, japa1 grind 0.69) → floor 0.45. Still a proxy; the free-foot-overload
  gate is the robust successor.
- **Sampling-noise confound in the A/B.** Independent gen per arm made OH WORLD Edit 13→25 holds (noise, not the
  knob). Fixed with a SHARED-RNG A/B (common random numbers): restore the RNG before the Edit arm → arms byte-identical
  until the knob first bites. (Rule 11: isolate the variable — sampling was a second uncontrolled variable.)
- **The footswitch reframe.** Would have wrongly tuned the 16th-jack PENALTY to kill "intrinsic" voltage; the
  footswitch knob showed 81–85% of it is footswitch-STRATEGY, not intrinsic — a different lever entirely.

## Current state / open fork
- **SHIPPED (canonical decode defaults, 2026-07-02):** `hold_stream_penalty=8, hold_stream_floor=0.45,
  hold_stream_win=16` + `footswitch=False`. Wired in `decode_defaults.CANONICAL_DECODE`, both entry points
  (`export_typed_samples.py`, `scripts/generate.py`), the HANDOFF canonical block (validated), generation-defaults §1.
- **OPEN:** (1) FREE-FOOT-OVERLOAD gate = robust successor to the density-proxy hold gate (user's next lever); (2)
  16th-jack penalty on the intrinsic residue, tastefully; (3) GRADED footswitch policy vs the hard ban.

## Cross-arc links
- **Depends on / spun off** [`quality-feature-attribution-arc.md`](quality-feature-attribution-arc.md) — it pinned
  the pattern/type head as the fast-song locus; this arc localized WHERE inside it and DECODE-fixed it.
- **Corroborates** the jack-heaviness thread (`notes/jack_heaviness_findings.md`, `[[jack-heaviness]]`) — the
  footswitch finding reframes the pattern head's same-panel heaviness / "voltage" as footswitch strategy.
- **Uses** `conditioning-mechanics` §7 (`hold_stream_penalty`) + §8b (`footswitch`), `foot_fatigue_design.md`
  (the fs_add mechanism), `choreography_metrics_findings.md` (hold_burst = the CROSS metric this is distinct from).
