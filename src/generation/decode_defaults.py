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
}


def calib_arg_default() -> str:
    """The `onset_phase_calib` default formatted for a `type=str` argparse flag (e.g. '0.0,1.0')."""
    return ",".join(str(x) for x in CANONICAL_DECODE["onset_phase_calib"])


def parse_phase_calib(spec):
    """'b8,b16' string -> (float, float) tuple, or None for an empty/None spec."""
    if not spec:
        return None
    return tuple(float(x) for x in str(spec).split(","))


def apply_phase_calib(onset_logits, phase_calib):
    """Add the per-phase 16th-unlock offset to a (T,) onset-logit tensor BEFORE the tau quantile.

    The phase grid (16th resolution, frame index t): t%4==2 -> 8th, t%4 in {1,3} -> 16th-offbeat,
    t%4==0 -> quarter (untouched). This offset MUST be applied identically (a) here, before the
    density quantile that sets tau, and (b) inside `generate()` (via the `onset_phase_calib` kwarg);
    if tau is computed WITHOUT it, the boosted 16ths flood past the threshold (conditioning-mechanics
    §6 / generation-defaults §1a). Returns the logits unchanged when phase_calib is None.
    """
    if phase_calib is None:
        return onset_logits
    import torch
    b8, b16 = phase_calib
    ph = torch.arange(onset_logits.shape[0], device=onset_logits.device) % 4
    return onset_logits + torch.where(ph == 2, float(b8),
                                      torch.where((ph == 1) | (ph == 3), float(b16), 0.0))
