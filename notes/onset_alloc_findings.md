# Onset 16th-unlock — allocation/selection learnability probe (2026-06-28)

Goal (user): unlock 16th-note placement via a LEARNED per-region density target (not the audio-energy
envelope the breathing arc uses, which smears event emphasis across the window), preserving event emphasis
AND fixing placement. Prerequisite question this probe answers: **is the 16th signal already in what the
deployed onset head sees, or is it a representation problem?** Harness:
`experiments/generation_typed/probe_onset_alloc.py` (deployed model `gen_motif_full_fixed`, 16 rich songs,
native conditioning replicated per conditioning-mechanics §1; seed 42).

## Decisive result — FRAME-LEVEL selection AUC ("given a phase-slot, did a human press it?")
Restricted to each phase band (§6: t%4 0=quarter, 2=8th, 1&3=16th-offbeat), AUC of the head's own `p_onset`
and of an out-of-fold (GroupKFold-by-song) logistic on the onset-encoder penultimate features:

| pool | band | n | base rate | p_onset AUC | encoder AUC |
|---|---|---|---|---|---|
| Med+Hard | quarter | 2880 | 0.732 | 0.666 | 0.633 |
| Med+Hard | 8th | 2880 | 0.597 | 0.580 | 0.582 |
| **Med+Hard** | **16th-off** | 5760 | **0.042** | **0.731** | **0.718** |

(ref: 06-22 `diag_seqcontext_probe` 16th-localization audio-only 0.649 vs note-context 0.935.)

**The head's `p_onset` ranks real 16th slots at AUC 0.73 — ABOVE the 06-22 audio-only reference — yet the
head places ~zero 16ths in deployment.** The encoder OOF logistic agrees (0.718), so it's a real,
transferable signal, not in-sample luck. → **READOUT/threshold problem, NOT representation.**

### Why the global tau buries it
Base rates: quarter slots pressed 73%, 8ths 60%, **16ths only 4%**. The single global density-quantile tau
is dominated by the quarter/8th backbone → it sits ABOVE almost every 16th-frame `p_onset`, so correctly-
ranked 16ths never clear the bar. Un-burying them is what `onset_phase_calib` (§6) already does GLOBALLY;
the new piece is making the un-burial LOCAL/learned so 16ths fire where real charts afford them (a global
offset smears — the H4/H16 chaos flood).

## What this does NOT show (discipline — experiment-design)
- **The 06-22 verdict is NOT contradicted.** That bounded reaching the 0.935 COHERENCE ceiling (needs
  note-context). This unlocks 16ths at ~0.73 placement — much better than zero, short of human run-coherence.
  Whether 0.73 reads well is a BY-EAR call (Rule 8).
- **The window-mean ALLOCATION cut was UNDERPOWERED — its null is a setup artifact, do not cite it.**
  s16_dens out-of-fold R² went MORE negative as features were added (envelope -0.65 → encoder -1.15 ALL pool)
  = overfitting on 16 songs; mean-pooling 128-dim feats over a 32-frame window erases the high-frequency 16th
  structure; the target (16th-density per window, real share ~0.04) is sparse/zero-inflated. The density
  sanity column DID work (encoder out-of-fold r≈0.42 ≈ Probe 3B's in-sample 0.48), proving the pipeline is
  fine for a dense target — the 16th-specific window setup was the problem.
- **The frame-level AUC section was added AFTER the window null** (post-hoc), but it is the better-powered,
  more direct measurement and its out-of-fold encoder confirmation guards against fishing. Flagged for honesty.

## Open question → next probe (generation)
The ALLOCATION half (which SECTIONS get the 16ths) is not cleanly measured. The 0.73 AUC is pooled across
sections, so `p_onset` is already higher in real-16th-dense sections — allocation may come along for free via
the ranking. Test: lower the LOCAL tau on 16th-phase frames (or scale `onset_phase_calib` per window) and
characterize WHERE the new 16ths land vs real (does a global un-burial smear into quarter-heavy sections, or
does the ranking keep it honest?). This is the lead-in to the build (a learned/local 16th-affordance gate).

## Connections
- conditioning-mechanics §6 (`onset_phase_calib`, `onset_logit_scale` no-op), §8c (stamina is CEILING-only —
  can thin but never UNLOCK; the unlock needs a two-sided/floor-lowering gate).
- [[jack-heaviness]] Probe 2/3B (blocky audio-only rhythm, zero 16th-adjacent onsets, salience-chasing).
- `sequence_aware_onset_plan.md` (06-22 coherence-ceiling bound — un-contradicted; this is a different, lower bar).
