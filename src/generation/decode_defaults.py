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
    grid the TRIPLET subdivisions (t%12 in {2,4,8,10}) are in NONE of these three bands — the 16th-unlock is a 16th
    lever and must not silently boost triplets. Triplet placement comes from the model's learned weights; the OPT-IN
    triplet band (`triplet_band_positions`, the 3rd element of `onset_phase_calib`, default 0) is the deliberate
    follow-up lever, by-ear-gated. See notes/footspeed_floor_findings.md / conditioning-mechanics §6.
    """
    return subdiv // 2, (subdiv // 4, 3 * subdiv // 4)


def triplet_band_positions(subdiv=4):
    """The TRIPLET-only within-beat phase indices (the 6-per-beat positions NOT shared with the duple grid), for a
    `subdiv`-per-beat grid. Empty unless subdiv is divisible by 6 (triplets aren't representable otherwise — the
    16th grid subdiv=4 -> `()`). At subdiv=12 (48th grid): **{2,4,8,10}** = the 6 positions {0,2,4,6,8,10} minus the
    beat (0) and the duple 8th (subdiv//2=6). These get NO band from `phase_band_positions` (the Phase-5 no-triplet-
    band deferral). A triplet-calib offset on them (the optional 3rd element of `onset_phase_calib`) lets the model
    COMMIT to triplets where the audio affords them, resolving the duple/triplet HEDGE that under-places triplets
    (First of the Year gen occ 0.14 vs human 0.40). A NEW, by-ear-gated lever — default 0 (off). See
    notes/footspeed_floor_findings.md / conditioning-mechanics §6."""
    if subdiv % 6 != 0:
        return ()
    step = subdiv // 6
    e8 = subdiv // 2
    return tuple(k * step for k in range(1, 6) if k * step != e8)


def phase_calib_offset(T, phase_calib, subdiv=4, device=None):
    """The (T,) per-phase onset-logit offset for `onset_phase_calib = (b8, b16[, b_trip])` — the SINGLE source used
    by BOTH `apply_phase_calib` (tau side) and `generate()` (decode side), so they cannot drift. `b8` -> 8th frames,
    `b16` -> the two 16th-offbeat frames, optional `b_trip` -> the triplet-only frames (`triplet_band_positions`;
    empty on the 16th grid). Returns zeros when phase_calib is None."""
    import torch
    if phase_calib is None:
        return torch.zeros(T, device=device)
    b8, b16 = float(phase_calib[0]), float(phase_calib[1])
    b_trip = float(phase_calib[2]) if len(phase_calib) > 2 else 0.0
    e8, (s16a, s16b) = phase_band_positions(subdiv)
    ph = torch.arange(T, device=device) % subdiv
    off = torch.where(ph == e8, b8, torch.where((ph == s16a) | (ph == s16b), b16, 0.0))
    if b_trip:  # OPT-IN triplet band (subdiv%6==0); overrides the base offset on the triplet-only frames
        for tp in triplet_band_positions(subdiv):
            off = torch.where(ph == tp, b_trip, off)
    return off


def apply_phase_calib(onset_logits, phase_calib, subdiv=4):
    """Add the per-phase 16th-unlock (+ optional triplet-band) offset to a (T,) onset-logit tensor BEFORE the tau
    quantile.

    The phase grid (frame index t, `subdiv` timesteps/beat — 4=16th grid, 12=48th grid): the 8th and 16th-offbeat
    positions come from `phase_band_positions(subdiv)`; the strong beat (t%subdiv==0) is untouched; the TRIPLET
    positions (`triplet_band_positions`) are untouched UNLESS a 3rd `b_trip` element is given. This offset MUST be
    applied identically (a) here, before the density quantile that sets tau, and (b) inside `generate()` (via the
    `onset_phase_calib` kwarg, passed the SAME `subdiv`); if tau is computed WITHOUT it, the boosted onsets flood
    past the threshold (conditioning-mechanics §6 / generation-defaults §1a). Returns the logits unchanged when
    phase_calib is None.
    """
    if phase_calib is None:
        return onset_logits
    return onset_logits + phase_calib_offset(onset_logits.shape[0], phase_calib, subdiv, onset_logits.device)
