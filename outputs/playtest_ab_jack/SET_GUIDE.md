# Jack-governor A/B — foot-exertion penalty OFF vs ON (~/sm-generated/ab_jack_*)

Same songs (rich Hard, seed 42: the IN BETWEEN-type jack-heavy set), same clean model
(`gen_motif_full_fixed`), same hard cap (max_jack_run=2). The ONLY difference is the soft foot-exertion governor:

- **`ab_jack_OFF`** = `--jack_penalty 0` (governor off — long 8th-jack streams allowed).
- **`ab_jack_ON`**  = `--jack_penalty 1.5` (gentle governor — escalating BPM-aware penalty on extending a
  same-panel run).

A/B the SAME song across the two folders.

## What changed (offline, on the actual installed charts)
| | OFF | ON |
|---|---|---|
| same-panel jack runs ≥2 | 330 | 155 |
| runs of length ≥4 | 7.0% | 1.3% |
| **longest jack stream** | **14 notes** | **4 notes** |
| **density (notes/frame)** | **0.346** | **0.346** |

The OFF set literally has a 14-note one-foot stream; ON caps the worst at 4. **Density is identical** — the
governor re-routes long jacks to foot ALTERNATION, it does NOT delete notes. So ON should feel like the same
amount of stuff, danced with two feet instead of one foot hammering.

## The question for your ears
- Does ON feel **more natural / less mechanically jacky** — especially on IN BETWEEN (where you flagged "soooo
  many jacks, long jack streams")?
- Critically: does ON feel like it LOST anything (holes, less intense)? It shouldn't — density is preserved.
  If it does, that tells us the re-routing isn't musical and λ needs lowering (or jack_free_rate tuning).
- Is λ=1.5 about right, too gentle (still feels jacky), or too strong (alternation feels busy/unmusical)?

## Decision it gates
λ=1.5 is the proposed DEFAULT for all future playtests (mandatory cap already relaxed 1→2 to allow justified
2-note 16th jacks). If it feels right → it stays the default. Both sets are pad-playable (cap 2, no jump/cross
during hold). Separate from the `ab_trill_*` A/B (trill-honesty of the retrained model) — that one's still pending too.
