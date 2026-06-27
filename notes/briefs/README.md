# Narrative briefs — the README claims-audit scaffold

**What this is.** One brief per experiment arc, each connecting that arc's story with **direct quotes** from
its source `notes/*.md`, and ending with an **"audit hooks"** table: the exact claims a README line would
cite, traced to verbatim evidence, with a **verb-precision** column (measured vs vouched vs mapped — see the
`claim-precision` discipline: a number being right doesn't make its verb right).

**Why it exists.** The first-pass README audit (`readme-0.1.0-audit.md`, now untracked/gitignored) was
distrusted — claims kept surfacing after it was declared done. These briefs replace it. The method: read
every note in an arc *in full*, grouped by arc, and condense it into a quotable brief. **The re-audit
reconciles `README.md` against these briefs — one brief = one checklist unit.** Source notes stay flat in
`notes/`; this is an overlay (no `see notes/X` pointer was broken).

**How to run the re-audit (next session, fresh context).** For each brief, open its "audit hooks" table and
check every README claim that touches that arc against the verbatim evidence and the verb-precision note. The
cross-arc reconciliations (flagged ⚠️ AUDIT-CRITICAL below) are where the v1 audit went stale.

---

## The briefs, in narrative order

| # | Brief | Arc | The claims it governs |
|---|---|---|---|
| 00 | [00-meta](00-meta/BRIEF.md) | Playtest ledger + current deployment state | the "metrics can't see play-feel" thesis; deployed model = `gen_motif_full_fixed`; PR #41 **verified merged** |
| 01 | [01-classifier](01-classifier/BRIEF.md) | Phase 1 difficulty classifier | **82.9% acc / 0.835 macro F1** (label it the *classifier*, not the generator) |
| 02 | [02-generative-foundation](02-generative-foundation/BRIEF.md) | baseline→AR→factorized→calibrated decode | **most README numbers:** floor 0.053, AR 0.300/5.7×, factorized 0.763 / focal 0.748, ECE 0.17→0.01, KV-cache 33.4→3.6s (9.2×, **batch of 4**) |
| 03 | [03-typed-model](03-typed-model/BRIEF.md) | taps→holds, layered head, playable export | **rolls 0/675**; typed ~0.77; "holds at ~real rate" (near, not exact) |
| 04 | [04-conditioning](04-conditioning/BRIEF.md) | pattern prefs → radar → CFG → style; H11 | controllability; ⚠️ radar-works-but-disabled; ⚠️ **song-structure** honesty (H11 not robust) |
| 05 | [05-musical-features](05-musical-features/BRIEF.md) | chroma/HPSS/highres onset; the choreography axis | ⚠️ **"chaos smears"** (radar-CFG mechanism); features *used* ≠ musicality improved |
| 06 | [06-taste-critic](06-taste-critic/BRIEF.md) | **the headline thesis** | **AUC 0.964** (vs corrupted-real); **REAL 0.823>BASE 0.290>CHAOS 0.003**; v1 scored backwards; ⚠️ best-of-N **not playtested** |
| 07 | [07-chaos-placement](07-chaos-placement/BRIEF.md) | the keystone arc (2 superseded conclusions) | ⚠️ chaos **AMOUNT** controllable, **PLACEMENT** bounded; the "audio-ambiguity ceiling" is **refuted** |
| 08 | [08-phase3-joint](08-phase3-joint/BRIEF.md) | joint generative paradigm | ⚠️ **PARKED research, not shipped**; placement is a distribution (humans agree ~1/3) |
| 09 | [09-manifold-guidance](09-manifold-guidance/BRIEF.md) | the shipped `--style` path | ✅ `--style` manifold (shipped `radar_manifold.npz`), `--radar` disabled; **the corrected chaos framing's source** |
| 10 | [10-motif-arc](10-motif-arc/BRIEF.md) | H15→H19, the deployed model | ⚠️ motif surface **PARTIAL** (candle ships, sweep dead); knobs **not in the CLI**; deployed = `gen_motif_full_fixed` |
| 11 | [11-governor](11-governor/BRIEF.md) | foot fatigue / stamina / arc | ⚠️ **keep qualitative** ("tasteful edit"), not a %; governor is a *ceiling*, not a difficulty knob |
| 12 | [12-geometry-feasible](12-geometry-feasible/BRIEF.md) | feasible-region theory + easy-corner gate | ⚠️ region is **NOT "mapped"** (vouched ranges only); easy corner offline/2-songs |
| 13 | [13-roadmaps](13-roadmaps/BRIEF.md) | augmentation; constraint relaxation | scope honesty: fixed-BPM/max-2/16th-grid; tame-Hard has **two** causes; hands-filter **55% raw vs ~10% filtered** |

---

## The cross-arc reconciliations the v1 audit missed (check these first)

1. **Chaos.** "Chaos still smears" ([[05-musical-features]]) was the *radar-CFG* mechanism. The *deployed*
   chaos is manifold conditional-fill ([[09-manifold-guidance]]) + `onset_phase_calib` ([[07-chaos-placement]]) —
   **in-distribution-bounded, not a smear**, and **16th-share isn't the user dial**. The corrected README line
   must attribute "musical chaos" to the manifold path, never to raw radar/CFG.
2. **Radar vs style.** The trained radar knob *works* ([[04-conditioning]]) but is **disabled in deployment**
   because point-conditioning goes off-manifold; `--style` (manifold) is the shipped path ([[09-manifold-guidance]]).
3. **Song structure.** The model **tracks** local structure/accents given context, but global phrase-planning
   is the **open frontier**, and the free-running transition-drift effect is **not robust** ([[04-conditioning]] H11).
4. **"Mapped region."** v1 has **vouched** governor ranges ([[11-governor]]) + **one** offline boundary-walk
   ([[12-geometry-feasible]]). The joint feasible-region map is **v2, not done.** Never say "mapped/characterized."
5. **Best-of-N.** Built, **not playtested**, "little headroom at Hard" ([[06-taste-critic]]) — no quality claim.
6. **Deployed model lineage.** It's `gen_motif_full_fixed` ([[10-motif-arc]], [[00-meta]]); the model card's
   `gen_stage1/radar/style` lineage is **stale** (a known deferred item).

## Verb-precision legend (the discipline these briefs enforce)

- **measured** — a real offline number from a diag/eval (cite it, with its conditions).
- **vouched** — the user playtested it and stands behind it (qualitative; don't quantify).
- **mapped / characterized** — a region/space swept and bounded. **The project has NOT done this** for the
  joint settings region; reserve the word.
