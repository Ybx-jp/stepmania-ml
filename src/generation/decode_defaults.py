"""Single source of truth for the CANONICAL decode palette (the playtest-validated full stack).

Both the public CLI (`scripts/generate.py`) and the playtest exporter
(`experiments/generation_typed/export_typed_samples.py`) import `CANONICAL_DECODE` and use it for
their argparse defaults, so the two CANNOT silently drift apart. This module is the executable form
of the `generation-defaults` skill — change a value HERE and every entry point moves together.

Historical context: these scripts kept duplicate copies of the default list and repeatedly drifted
(e.g. `scripts/generate.py` shipped `pattern_temperature=0.7`, stamina off, and NO 16th-unlock long
after the exporter's canonical regime moved on). This module exists to make that class of bug
structurally impossible.

What is NOT in here (because it is per-song / per-invocation, not a static palette value):
  - `onset_threshold` (tau) — computed per song from the conditioned + phase-calibrated onset logits
  - `bpm` — per song; MANDATORY (no bpm -> the foot/stamina governors are silent)
  - `radar` / `style` / `motif` / `figure` / `guidance_scale` — set only when a groove knob is used
  - `hold_aware` / `no_jump_during_hold` / `no_cross_during_hold` — FORCED on by `enforce_playability`
"""

from __future__ import annotations

# The canonical runtime palette. Values mirror export_typed_samples.py's verified argparse defaults.
CANONICAL_DECODE: dict = {
    "type_temperature": 0.4,        # per-panel tap/hold/roll sampling temp (surfaces holds at rate)
    "pattern_temperature": 1.0,     # footwork sampling temp — real jack/jump balance (NOT the stale 0.7)
    "repetition_penalty": 1.0,
    "max_jack_run": 2,              # hard backstop: allow a justified 2-note jack, forbid 3+
    "fatigue_penalty": 2.0,         # per-NOTE foot governor (§8b); 0 disables
    "fatigue_free": 6.0,            # free zone before the fatigue ceiling bites
    "stamina_ceiling": 50.0,        # per-REGION density relief (§8c); needs fatigue_penalty; 0 disables
    "stamina_tau": 8.0,             # stamina slow-decay (beats)
    "stamina_scale": 15.0,          # excess-workload scale for the tau bump
    "stamina_breathe": 1.2,         # Stage-3 ARC: ceiling breathes with audio energy; 0 = flat
    "onset_phase_calib": (0.0, 1.0),  # ★ the 16th-UNLOCK (b8, b16); MUST also be applied to tau (see below)
    # HOLD-IN-STREAM fix (2026-07-02, playtest-validated): suppress hold-heads in dense STREAM sections (the type
    # head opens holds where a human streams; the pinned foot then forces jacks). Gated on local onset density with
    # a floor so SPARSE musical holds stay. floor 0.45 / penalty 8 = the by-ear sweet spot (japa1 "just right").
    "hold_stream_penalty": 8.0,     # 0 = off; the density-gated hold-head logit suppression
    "hold_stream_floor": 0.45,      # local onset density below which the penalty is exactly 0 (protects sparse holds)
    "hold_stream_win": 16,          # frames for the local-density gate
    # FOOTSWITCH policy (2026-07-02, playtest-validated): False = forbid footswitch footing -> same-panel runs must
    # be one-foot jacks. Playtest: OFF forced the model to ALTERNATE (more creative, less brutal voltage); japa1
    # "sooooo much better". Set the DEPLOYED default OFF for now; revisit a graded footswitch policy later.
    "footswitch": False,
}


def calib_arg_default() -> str:
    """The `onset_phase_calib` default formatted for a `type=str` argparse flag (e.g. '0.0,1.0')."""
    return ",".join(str(x) for x in CANONICAL_DECODE["onset_phase_calib"])


def parse_phase_calib(spec):
    """'b8,b16' string -> (float, float) tuple, or None for an empty/None spec."""
    if not spec:
        return None
    return tuple(float(x) for x in str(spec).split(","))


def phase_band_positions(subdiv=4):
    """The within-beat phase INDICES of the 8th and the two 16th-offbeat positions, for a `subdiv`-per-beat grid.

    `subdiv` = timesteps_per_beat: 4 = the legacy 16th grid (`t%4`), 12 = the data-layer-v2 48th grid (`t%12`).
    A beat spans indices 0..subdiv-1; the strong beat is index 0. Returns `(eighth, (sixteenth_a, sixteenth_b))`:
      - 8th  = the beat midpoint             = subdiv//2      (2 at subdiv=4; 6 at subdiv=12)
      - 16th = the quarter-beat off-positions = {subdiv//4, 3*subdiv//4}  ({1,3} at subdiv=4; {3,9} at subdiv=12)
    At subdiv=4 this is byte-identical to the old hard-coded {2} / {1,3}. NOTE (data-layer-v2, Phase 5): on the 48th
    grid the TRIPLET subdivisions (e.g. t%12 in {2,4,8,10}) are in NONE of these three bands — the 16th-unlock is a
    16th lever and must not silently boost triplets (that would be a new, unvalidated lever). Triplet placement comes
    from the model's learned weights, not a decode nudge; a triplet phase band is a deliberate follow-up gated on the
    Phase-6 by-ear result. See notes/data_layer_v2_scope.md / conditioning-mechanics §6.
    """
    return subdiv // 2, (subdiv // 4, 3 * subdiv // 4)


def apply_phase_calib(onset_logits, phase_calib, subdiv=4):
    """Add the per-phase 16th-unlock offset to a (T,) onset-logit tensor BEFORE the tau quantile.

    The phase grid (frame index t, `subdiv` timesteps/beat — 4=16th grid, 12=48th grid): the 8th and 16th-offbeat
    positions come from `phase_band_positions(subdiv)`; the strong beat (t%subdiv==0) and any triplet subdivisions
    are untouched. This offset MUST be applied identically (a) here, before the density quantile that sets tau, and
    (b) inside `generate()` (via the `onset_phase_calib` kwarg, passed the SAME `subdiv`); if tau is computed WITHOUT
    it, the boosted 16ths flood past the threshold (conditioning-mechanics §6 / generation-defaults §1a). Returns the
    logits unchanged when phase_calib is None.
    """
    if phase_calib is None:
        return onset_logits
    import torch
    b8, b16 = phase_calib
    e8, (s16a, s16b) = phase_band_positions(subdiv)
    ph = torch.arange(onset_logits.shape[0], device=onset_logits.device) % subdiv
    return onset_logits + torch.where(ph == e8, float(b8),
                                      torch.where((ph == s16a) | (ph == s16b), float(b16), 0.0))
