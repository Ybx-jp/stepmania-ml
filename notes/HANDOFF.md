# HANDOFF — foot-physics baseline + jack-heaviness DONE; NEXT THREAD = the onset head

**Written 2026-06-27 for the next Claude.** This session built a learned-head-vs-physics comparison, then
decomposed *why the generator is jack-heavy* across four probes. The **next move is the ONSET HEAD** (the user's
call). Read this, then `notes/jack_heaviness_findings.md` + `notes/foot_physics_baseline.md`, then the memory
files. Env: conda `stepmania-chart-gen` (`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`). The two
load-bearing skills — **experiment-design** (attribution HARNESS→DATA→MODEL; now has **Rule 0 = check notes
first**) and **conditioning-mechanics** (exact decode math) — were used throughout and updated this session.

---

## 1. WHERE WE ARE
Deployed model unchanged: `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres) + the shipped governor
(`fatigue_penalty=2` default). This session was **diagnostic, not a model change**. We asked "is the learned
pattern head's footwork more human than a physics policy?" and "why is it jack-heavy?". Net answer: the
jack-heaviness is **both heads** — the **pattern head** is the proximate cause (it over-jacks dense sections;
`pattern_temperature=0.7` is too greedy) and the **onset head** is the contributing cause (audio-only, non-causal
→ blocky 8th-heavy/16th-absent rhythm + salience-chasing misallocation). The pattern-head lever is cheap
(`pattern_temperature` ↑, the governor permits it — Probe 4); the onset head is the deeper, bounded-hard thread =
**the next move**.

## 2. THE JACK-HEAVINESS RESULT (4 probes — `notes/jack_heaviness_findings.md`)
All native-mode (own onset head, radar-conditioned, density matched to REAL, governor toggled), 16 rich songs.
1. **Probe 1 (`probe_jack_temp.py`):** `pattern_temperature` IS a jack lever — Medium jackDist 0.31→0.17 as
   0.7→1.5, jumps rise toward real too. Hard resisted (→ Probe 2).
2. **Probe 2 (`probe_onset_rhythm.py`):** `onset_logit_scale` is a NO-OP under thresholding (0 frames differ —
   monotonic ranking). The onset head's contribution is RHYTHM: zero 16th-adjacent onsets (real 4–11%),
   8ths over-weighted, onset runs ~2× real → jack opportunity. Hard's blockiness is worst (explains Probe 1).
3. **Probe 3 (`probe_onset_sections.py`):** at MATCHED local density the model jacks far MORE than real, and
   real jackiness is density-INVARIANT (~1.07) while the model's RISES (corr +0.23) — the pattern head over-jacks
   dense sections (= low-temp greedy collapse). Misallocation: corr(model_dens, real_dens) ~0.48; model tracks
   audio salience (p_onset) MORE than real (0.62 vs 0.36) → over-notes loud/awkward, under-notes melodic-quiet
   (the user's "awkward over-noted / empty where active", quantified).
4. **Probe 4 (`probe_temp_governor.py`):** the fatigue governor lets `pattern_temperature` RISE without a jack
   blowup (maxRun bounded ~5 vs gov-OFF spiking to 22) while jumps recover. BUT transition-entropy (scramble)
   still climbs with temp regardless — the governor bounds FATIGUE, not musical structure, and that metric can't
   separate good variety from scramble. So raising temp >0.85 is a **by-ear** call; metrics favor ~1.0–1.2.

**Two experiment-design RETRACTIONS this session (all caught by a fair re-test):** (a) the original
`compare_foot_physics.py` fed the model REAL onsets via `onset_override` → pattern head OOD → maxRun-24 jacks; on
its OWN onsets the governed model matches real (maxRun ~4). (b) The "physics beats the head" verdict was an
`onset_override` + missing-radar artifact; the valid comparison is `compare_native.py`. Lesson burned in →
experiment-design **Rule 0** (grep notes/skills before designing — the onset-rhythm work was already scoped in
`sequence_aware_onset_plan.md`, and the long-jack fix in `foot_exertion_findings.md`).

## 3. THE NEXT MOVE — THE ONSET HEAD (user's call; start with Rule 0)
The onset head is audio-only + non-causal → it (a) places isolated 16ths not coherent runs ("awkward"), (b)
chases audio salience so it over-/under-notes sections vs the human. **READ FIRST:** `sequence_aware_onset_plan.md`
— this thread was ALREADY rigorously bounded (06-22): 16th placement is SEQUENCE-determined (note-context AUC
0.935 vs audio-only 0.649), but every cheap lever failed — fully-AR onset EXPLODES (density 0.73 vs 0.18),
scheduled sampling only dampens, frozen-context refinement can't bootstrap from the audio-only first pass
(anti-correlated C0). **Verdict then: reaching the ceiling needs a paradigm change (learn placement from
multiple human chartings, or a much stronger first pass), not a cheap decode lever.** Also `playtest_log.md` +
`foot_fatigue_design.md §8d`: the melodic under-placement ("ignored piano solo") is the onset head, an
audio-feature/retrain thread (the breathing-energy percussion-bias was REFUTED — p_onset reads melodic as +0.47).
- **What's genuinely NEW after this session** (not re-derive): the section-level misallocation is now MEASURED
  (Probe 3B — corr ~0.48, salience-chasing); and the pattern-head proximate cause is isolated (so an onset fix is
  about COHERENCE/allocation, not jacks per se). Decide: accept the 06-22 bound, or commit to the onset rebuild
  (sequence-aware / note-context head + scheduled sampling) or the audio-feature melodic angle. Cheapest fresh
  probe if pushing: does better section-level density allocation (e.g. a learned-from-real allocation prior)
  improve coherence without the AR explosion?

## 4. AWAITING USER: the pattern_temperature playtest (this session)
Installed to `~/sm-generated/T{0.7,1.0,1.2}_cond{OFF,ON}` (6 groups × 4 Hard songs: japa1=突撃ガラスのニーソ姫,
Deja Loin, OH WORLD, High School Love). Governor ON throughout; cond ON = `--match_radar --guidance 2.0`
(cranked), cond OFF = plain. **Binding question: does `pattern_temperature` >0.85 read as coherent or scrambly
by ear, now that the governor bounds jacks?** When the user reports back, log it in `playtest_log.md` (a SETUP
entry is already there) and — if higher temp is coherent — it's a shippable jack/jump fix with no retrain
(consider raising the `pattern_temperature` default / the H2 0.6–0.85 range). New exporter flag this session:
`--song_filter "a,b,c"` (+ `--difficulty_select` now works without `--groove_select`).

## 5. BRANCH / PR STATE
- **This session's work is on branch `claude/arrowvortex-linux-compat-75ri8n`** (started from `main` 9d3e7a8 +
  the cloud foot-physics commits 9f8871f/0abbdea). Committed, **NOT pushed, NO PR yet** — open one when ready.
  Commits: native comparison + jack probes + the skill Rule 0 + this handoff/memory/INDEX refresh.
- **Prior arcs (unchanged this session):** governor SHIPPED (PR #41 merged to `main`). **PR #42**
  (`release/v0.1.0-prep` → `main`, v0.1.0 + taste-critic interpretability) was **OPEN** at last handoff — status
  not touched this session; check before assuming. `main` protected by ruleset `protect-main` (id 18199761).

## 6. OTHER OPEN THREADS (none block the onset work)
- best-of-N reranking + V2 region-map (taste-critic prereq done) — `geometry_feasible_region.md`.
- Model UNDER-JUMPS (a separate air/density thread; do NOT tune the governor to it — conditioning-mechanics §8d).
- GDL/equivariance (pad symmetry) — v2/paper, parked.

## 7. DISCIPLINE (load-bearing)
- **experiment-design Rule 0 (new):** grep `notes/` + skills BEFORE designing a probe — this session re-derived/
  mis-attributed 3× until the notes were checked. **HARNESS→DATA→MODEL**: a surprising model-blaming result gets
  the fair re-test FIRST (every retraction above was caught that way).
- **conditioning-mechanics:** replicate `scripts/generate.py` for any probe that sets/measures a knob; native
  decode (tau from the SAME conditioned logits), governor needs `bpm`. §6 (onset_logit_scale no-op) and §7
  (`pattern_temperature` × governor) were updated this session.
- **Match metric to the property's resolution; one change at a time; coherence/play-feel is BY-EAR** (Rule 8).
  `playtest_log.md` = subjective only; quantitative results → `notes/*_findings.md`.
