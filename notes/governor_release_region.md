# Governor "region of good settings" — the release map (2026-06-26)

The release gate (per [[geometry_feasible_region]]): characterize the good region to its BOUNDARIES + a recommended
CENTER, not one lucky point. This maps the decode-time GOVERNOR stack (the fatigue / stamina / arc work, Stages
1–3) for `gen_motif_full_fixed`. "Good" = playable (code-enforced) AND the jack/jump RUN-LENGTH distribution matches
real Hard charts (calib_foot_fatigue.py) AND density held AND the qualitative levers (relief, arc) do their job
without breaking. Each range is backed by a diag; rerun them to re-walk a boundary.

## RECOMMENDED RELEASE CENTER (the default config)
```
# per-note FOOT governor (always on; supersedes the old standalone jack_penalty=1.5 — stamina needs fatigue_on)
fatigue_penalty = 2.0      jack_penalty = 0
jack_weight = 1.0  travel_weight = 0.6  fatigue_tau = 2.0  footswitch_pen = 4.0  fatigue_free = 6–12  fatigue_cap = 30
# per-region STAMINA (opt-in; a relief valve — near-no-op until the chart is over its workload ceiling)
stamina_ceiling = None (off) by default;  ~25 when you want relief;  ~50 = gentlest "tasteful edit"
stamina_tau = 8  stamina_scale = 15  stamina_max_bump = 0.45
# STAGE-3 ARC (opt-in; needs stamina_ceiling)
stamina_breathe = 0 (off);  ~1.2 for a clear arc (up to 1.8)   stamina_breathe_floor = 0.4   stamina_breathe_win = 96
# MANDATORY playability (code-enforced, never off)
hold_aware=True  no_jump_during_hold=True  no_cross_during_hold=True  max_jack_run=2  pattern_temperature≈0.7
```

## THE MAP — per-knob good range + boundaries
| knob | center | GOOD range | low boundary (failure) | high boundary (failure) | diag |
|---|---|---|---|---|---|
| **fatigue_penalty** | 2.0 | **1.5 – 3.0** | <1.5 under-governs (jack≥4 >1.2% vs real 0.8%; maxJackRun→6) | >3 over-governs (jack≥4 <0.6%, stricter than humans) | calib_foot_fatigue.py |
| fatigue_free | 6–12 | 6 – 12 | low → downward pull, jump-STARVES | ≥18 ≈ governor silent | design note CORRECTIONS |
| fatigue_cap | 30 | ~30 | too low → forbids playable footing | high → no hard backstop | conditioning-mechanics §8b |
| **stamina_ceiling** | 25 (when on) | **15 – 50** | <~10 dents the REST windows too (→ a global cut, not a governor) | ≥~200 ≡ OFF (no-op) | diag_stamina.py |
| stamina_tau | 8 beats | 6 – 16 | too short → reacts to single bursts (noise) | too long → never recovers | (Stage-2 build) |
| **stamina_breathe** | 1.2 | **1.2 – 1.8** | 0 = flat (DULLS the model's own arc by thinning climaxes) | >~2 = over-swingy; needs the floor | diag_stamina_arc.py |
| stamina_breathe_floor | 0.4 | 0.3 – 0.5 | <~0.2 → low-energy OUTRO emptied = abrupt early ending | →1.0 cancels the arc | diag_stamina_arc.py + tail check |
| max_jack_run | 2 | 2 (fixed) | 1 = strict-real but rigid | ≥3 / None = brutal fast jacks (H13) | MANDATORY |

## VALIDATED EVIDENCE (the numbers behind the ranges)
- **fatigue_penalty (per-note, vs REAL rich-Hard; real: maxJackRun 3.5, jack≥4 0.8%, dens 0.385):** OFF jackRun 6.2 /
  jack≥4 2.1%; λ=2 → 3.9 / **0.7%** (matches real), density held 0.320 across the whole sweep. Good 1.5–3. (Read the
  JACK metrics, NOT the calib "dist→real" — it's dominated by the jump% gap the governor shouldn't close: the model
  under-jumps 6% vs real 31% for SEPARATE reasons; conditioning-mechanics §8c.)
- **stamina_ceiling (per-region relief, paired peak/rest density):** ceiling 25 → top-decile dense 4-measure windows
  thin 14% (peakΔ −0.039) while moderate windows hold (restΔ −0.002) = ~20:1 selectivity. ceiling 40 = pure-peak but
  weak; 8 starts denting rest (3:1); 200 ≡ OFF byte-identical (no-op invariant). PLAYTEST: under chaos over-crank
  (density 0.400), g50 = "tasteful edit, not a rewrite" (stays Hard), g25 = relief toward natural density, g12 = over.
- **stamina_breathe (the arc, corr(density,energy) / climax-verse Δ):** OFF 0.898/0.180; flat ceiling DULLS to
  0.876/0.131 (thins climaxes); breathe 1.2 → 0.918/0.185, 1.8 → 0.920/0.200 (PAST baseline) at held density. floor
  0.4 fixes the abrupt-ending bug (HSL tail last-10% density 0.000→0.094, back to OFF baseline) at small arc cost.

## REGIMES (when the optional levers matter)
- **Normal generation:** the per-note `fatigue_penalty=2` is the whole story (jacks human-like, density held). Stamina
  is a near-no-op (the chart isn't over its ceiling — playtest: g25 imperceptible on a normal japa1). Ship stamina
  OFF by default.
- **Cranked conditioning** (high chaos/density, e.g. `--style chaos=q0.99 g3.0` → density 0.400): the chart goes over
  its workload ceiling, so stamina (relief) + breathe (arc) earn their keep — g50/g25 + breathe 1.2 = a paced,
  playable over-cranked chart. This is the validated "safety-valve + pacing" use.

## OPEN / NOT a release blocker
- **H-onset-perc-bias** (the onset head under-places on melodic-only sections — the HSL piano-solo feel): a
  feature/retrain thread, not a governor knob. Parked (notes/playtest_log.md).
- **model under-jumps** (6% vs real 31%): a separate density/air thread; do NOT tune the governor to close it.
