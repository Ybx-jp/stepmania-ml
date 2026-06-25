"""Mandatory pad-playability constraints for ANY chart export the user will PLAY.

These are NOT optional. Each traces to a specific "unplayable on pad" playtest finding
(notes/playtest_log.md). Every playtest/export path MUST run its generate() kwargs through
`enforce_playability()` so a forgotten flag fails LOUDLY instead of silently shipping an unplayable
chart (which confounds the user's subjective evaluation — the project's primary signal).

Deviation requires EXPLICIT user approval, passed as `override_reason` (prints a prominent warning;
the constraints are then respected-as-given rather than forced).
"""

# hard-required (the unplayable-on-pad constraints)
MANDATORY_PLAYABILITY = {
    "hold_aware": True,            # coherent holds / no orphans (notes/hold_aware_decode.md)
    "no_jump_during_hold": True,   # pad has ONE free foot while holding -> jump-during-hold unhittable ("Will Smith meme")
    "no_cross_during_hold": True,  # free foot can't fast-cross/jack while a hold pins the other (B4U; notes/hold_cross_decode.md)
}
# hard-required exertion cap (H13): a fast same-panel jack is one foot hammering one arrow at 16th speed =
# brutal/un-danceable. A finite cap MUST be present. Default injected = 2 (user-approved 2026-06-25): a 2-note
# 16th jack is sometimes musically justified, so allow the DOUBLE but hard-forbid 3+ at 16th speed. The graded
# escalation across spacings is the SOFT foot-exertion governor (generate(jack_penalty=..., bpm=...), default
# ~1.5 in the exporter) — see notes/foot_exertion_findings.md; H13 history notes/h13_exertion_findings.md.
MANDATORY_JACK_CAP = 2
# soft (warn, not fail): the arrow-coherence sweet spot (H2: greedy collapses, 1.0 over-randomizes)
PATTERN_TEMP_RANGE = (0.6, 0.85)

_WHY = {
    "hold_aware": "coherent holds / no orphans",
    "no_jump_during_hold": "one free foot while holding -> jump-during-hold is unhittable",
    "no_cross_during_hold": "free foot can't fast-cross/jack while a hold pins the other foot (B4U)",
}


def enforce_playability(gen_kwargs: dict, override_reason: str | None = None) -> dict:
    """Force the mandatory pad-playability constraints into gen_kwargs (in place, returned).

    - missing key -> set to its required value (the safe default).
    - present & WRONG -> raise SystemExit (refuse to ship unplayable), unless `override_reason` given,
      in which case print a prominent warning and KEEP the caller's value (deliberate, user-approved).
    - pattern_temperature outside the coherence range -> warn only.
    """
    violations = []
    for k, v in MANDATORY_PLAYABILITY.items():
        if k not in gen_kwargs:
            gen_kwargs[k] = v
        elif gen_kwargs[k] != v:
            violations.append(f"{k}={gen_kwargs[k]!r} (MUST be {v!r}: {_WHY[k]})")
    # exertion cap (H13): a finite positive cap MUST be present (None/0 = disabled = brutal fast jacks).
    mjr = gen_kwargs.get("max_jack_run")
    if "max_jack_run" not in gen_kwargs:
        gen_kwargs["max_jack_run"] = MANDATORY_JACK_CAP
    elif mjr is None or mjr < 1:
        violations.append(f"max_jack_run={mjr!r} (MUST be a positive cap, default {MANDATORY_JACK_CAP}: "
                          f"fast same-panel jacks are unplayably brutal — H13)")
    if violations:
        if override_reason:
            print("⚠️  PLAYABILITY OVERRIDE (user-approved): " + "; ".join(violations)
                  + f"   [reason: {override_reason}]", flush=True)
        else:
            raise SystemExit(
                "❌ PLAYABILITY VIOLATION — refusing to export a pad-unplayable chart:\n  "
                + "\n  ".join(violations)
                + "\nThese are MANDATORY for anything the user plays (see .claude/skills/playtest/SKILL.md)."
                  "\nTo deviate deliberately, pass override_reason (requires EXPLICIT user approval).")
    pt = gen_kwargs.get("pattern_temperature")
    if pt is not None and not (PATTERN_TEMP_RANGE[0] <= pt <= PATTERN_TEMP_RANGE[1]):
        print(f"⚠️  pattern_temperature {pt} outside the coherence range {PATTERN_TEMP_RANGE} (H2: "
              f"greedy collapses, >1.0 over-randomizes)", flush=True)
    return gen_kwargs
