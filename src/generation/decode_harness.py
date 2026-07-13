"""Shared DECODE-SIDE harness for the deployed scripts and probes — the correctness-critical pipeline that
every probe was re-deriving by hand (and getting wrong: see the conditioning-mechanics skill §3/§6).

This is the executable form of "a probe MUST build its conditioning EXACTLY as generate()/the exporter do, and
compute tau from the SAME conditioned + guided + phase-calibrated onset logits." Instead of copy-pasting that
pipeline (30+ probes did, several with their own buggy `calibrated_p_onset`), import it:

    from src.generation.decode_harness import conditioned_p_onset, compute_tau, phase_shares

It is DOGFOODED: scripts/generate.py and experiments/.../export_typed_samples.py compute their own tau through
`conditioned_p_onset`/`compute_tau`, so a probe that uses these helpers is matching the deployed path BY
CONSTRUCTION, not by remembering to.

Palette VALUES live in `decode_defaults.py` (CANONICAL_DECODE); this module is the decode MECHANICS.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from src.generation.decode_defaults import (  # re-exported for convenience (one import for the whole harness)
    CANONICAL_DECODE, apply_phase_calib, phase_band_positions, calib_arg_default, parse_phase_calib)

# the deployed 42-dim highres generator (generation-defaults skill §0). The single literal; probes import it.
DEPLOYED_CHECKPOINT = "checkpoints/gen_motif_full_fixed/best_val.pt"

# the ONE generator architecture (was copy-pasted in 79 probes as `d_model=128, num_layers=4, onset_layers=2`).
MODEL_ARCH = dict(d_model=128, num_layers=4, onset_layers=2)

# feature-space name -> (AudioFeatureConfig flags | None for base, audio_dim, cache_dir). The deployed model is
# 'highres' (42-dim). 'stage1'/'base' are the legacy 41/23-dim spaces (generation-defaults skill §0). One table
# instead of the per-probe `if features == ...` ladder that silently paired the wrong dim/cache with a checkpoint.
_FEATURE_SPECS = {
    "highres": (dict(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True, use_highres_onset=True),
                42, "cache/samples_v3"),
    # data-layer-v2 (notes/data_layer_v2_scope.md): SAME 42-dim channels, but gridded at the 48th subdivision
    # (timesteps_per_beat=12) so triplets resolve. Only the audio hop density + the metric_phase PERIOD change
    # (t%12 beat / t%48 measure — a triplet cell now differs from a 16th cell); audio_dim stays 42. MUST be paired
    # with StepManiaParser.for_v2() (same timesteps_per_beat) or audio frames and note cells mis-align — the
    # scope's "piecemeal drift" risk. Fresh cache dir (the grid changed → old cached features are stale).
    "highres_v2": (dict(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True, use_highres_onset=True,
                        timesteps_per_beat=12, beat_sync=True), 42, "cache/samples_v3_48th"),
    "stage1":  (dict(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True), 41, "cache/samples_v2"),
    "base":    (None, 23, "cache/samples"),
}

# what make_feature_extractor returns: the extractor (None for base), the audio_dim, and the on-disk cache dir.
FeatureSpec = namedtuple("FeatureSpec", ["extractor", "audio_dim", "cache_dir"])

__all__ = [
    "CANONICAL_DECODE", "apply_phase_calib", "phase_band_positions", "calib_arg_default", "parse_phase_calib",
    "DEPLOYED_CHECKPOINT", "MODEL_ARCH", "FeatureSpec",
    "conditioned_p_onset", "compute_tau", "phase_shares",
    "make_feature_extractor", "load_generator",
]


def conditioned_p_onset(model, memory, difficulty, *, radar=None, style=None,
                        guidance=1.0, phase_calib=None, extra_offset=None, subdiv=4, window=None, tail_hangover=0):
    """The deployed onset -> p_onset path (the tau source), built EXACTLY as generate()/the exporter do.

    Steps, in the order the decode uses them (conditioning-mechanics §3 + §6):
      1. conditioned onset logits  `model.onset_logits(memory, difficulty, radar, style)`
      2. CFG guidance blend  `ol = ol_u + g*(ol_cond - ol_u)`  — ONLY when guidance != 1 AND conditioning is set
         (guidance amplifies onset density only via radar/style; motif/figure are null on the onset path)
      3. the 16th-unlock phase-calib offset (shared `apply_phase_calib`; identical to what generate() applies)
      4. any extra per-frame logit offset (e.g. the exporter's sparse-harm-in-quiet calibrator)
      5. sigmoid -> p_onset (numpy)

    The caller MUST feed the returned p_onset to `compute_tau` with the SAME phase_calib already baked in here
    (which it is) — a tau computed on un-offset/unconditioned logits lets conditioning/calib flood past it.
    Pass `radar`/`style` exactly as the decode will (i.e. None when not conditioning), so tau matches decode.
    """
    import torch
    # `window` = the sliding-window onset for songs longer than the trained PE context (single-sourced with
    # generate() via model.onset_logits(window=) so tau matches the decode; conditioning-mechanics §3 + §6,
    # notes/byo_sliding_window_findings.md). None / short song => byte-identical single pass.
    ol = model.onset_logits(memory, difficulty, radar=radar, style=style, window=window,
                            tail_hangover=tail_hangover)[0]
    if guidance != 1.0 and (radar is not None or style is not None):
        ol_u = model.onset_logits(memory, difficulty, radar=None, style=None, window=window,
                                  tail_hangover=tail_hangover)[0]
        ol = ol_u + guidance * (ol - ol_u)
    ol = apply_phase_calib(ol, phase_calib, subdiv)
    if extra_offset is not None:
        ol = ol + extra_offset
    return torch.sigmoid(ol).detach().cpu().numpy()  # detach: safe to call outside a no_grad block (probe-friendly)


def compute_tau(p_onset, density, default=0.5):
    """tau = quantile(p_onset, 1 - density) — the onset threshold that targets `density` notes/frame.

    `density` priority is the CALLER's (explicit > manifold E[density|.] > source-chart); this just maps a chosen
    density to a threshold. Returns `default` (0.5) for a missing/non-positive density. NOTE (conditioning-
    mechanics §8c): with the stamina governor ON the per-frame onset decision is raised IN the AR loop, so the
    REALIZED density can be <= this target where workload is high — that is handled inside generate(), not here.
    """
    return float(np.quantile(p_onset, 1 - density)) if density and density > 0 else default


def phase_shares(onset_frames, subdiv=4):
    """Phase shares of onset FRAME INDICES: (quarter, eighth, sixteenth-offbeat), on a `subdiv`-per-beat grid.

    Phase grid (conditioning-mechanics §6): t%subdiv==0 -> quarter (backbone), the 8th and 16th-offbeat positions
    come from `phase_band_positions(subdiv)` (subdiv=4 = the legacy 16th grid t%4; subdiv=12 = the v2 48th grid t%12).
    Real Hard ~ (0.71, 0.25, 0.04). `onset_frames` = the frame indices that carry a note (e.g.
    `np.where(typed.any(-1))[0]`). Returns (q, e8, s16) fractions; (0,0,0) if empty. NOTE (v2, subdiv>4): TRIPLET
    frames are in none of the three bands, so on the 48th grid these three shares do NOT sum to 1 — the remainder is
    triplet/finer subdivision. That's intentional: this stays the quarter/8th/16th backbone metric; measure triplet
    placement with the v2 displacement probe, not by overloading this partition.
    """
    idx = np.asarray(onset_frames)
    if idx.size == 0:
        return 0.0, 0.0, 0.0
    e8, (s16a, s16b) = phase_band_positions(subdiv)
    ph = idx % subdiv
    return float((ph == 0).mean()), float((ph == e8).mean()), float(((ph == s16a) | (ph == s16b)).mean())


def chaos_onset_gate_offset(audio_np, gain, chaos=1.0, desmear=False, subdiv=4):
    """The chaos×onset GATE (scope: notes/chaos_onset_gate_scope.md) as a per-frame onset-logit offset (T,).

    Two variants (the flat 16th-unlock is a GLOBAL additive bias CFG smears uniformly — H4/referee):

    ADD (default): REPLACE the unlock with a LOCAL audio-keyed off-beat LIFT — raise an off-beat logit in
    proportion to local saliency, so high chaos ADDS off-beats where audio affords them and NONE in silence.
        Δlogit[t] = +gain · chaos · offbeat[t] · saliency[t]           (pair with onset_phase_calib=(0,0))
      Caveat (measured, gain=3): with the unlock OFF this OVER-corrects good songs to on-grid (strips the
      loved 16ths) because the audio off-beat cue is weak (AUC ~0.66) — can't PLACE expressive off-beats.

    DESMEAR (`desmear=True`): KEEP the unlock ON, and only SUBTRACT off-beat logits where saliency is LOW —
    the silent-intro / generic-pad smear — leaving high-saliency (musical) off-beats untouched.
        Δlogit[t] = −gain · offbeat[t] · (1 − saliency[t])            (pair with the deployed unlock (0,1.0))
      Rationale: removing off-beats in dead zones is a much easier ask of a weak cue than placing them, and
      the overload smear is UNIFORM (fires even in silence → a big low-saliency population), whereas good
      syncopation clusters on transients (high saliency → survives). No chaos scale (a fixed dead-zone guard).

    - saliency[t] = per-song norm01 of max(highres_onset dim41, perc_onset dim35) — the H4 local off-beat cue.
    - offbeat[t] = 0 quarter (t%subdiv==0), 0.5 8th (t%subdiv==subdiv//2), 1.0 every other off-beat (subdiv=4: {1,3}).

    Needs highres 42-dim audio. Caller MUST feed the result to BOTH tau (conditioned_p_onset extra_offset=)
    AND generate(onset_logit_offset=) — same tau-coupling as onset_phase_calib/harm_calib (cond-mechanics §6).
    Single-sourced here so the exporter flag and probe_chaos_onset_gate.py share IDENTICAL math.
    """
    import numpy as _np
    a = _np.asarray(audio_np)
    if a.shape[1] < 42:
        raise ValueError(f"chaos_onset_gate needs 42-dim highres audio (dims 41/35), got {a.shape[1]}")
    sal = _np.maximum(a[:, 41], a[:, 35]).astype(_np.float64)
    sal = (sal - sal.min()) / (_np.ptp(sal) + 1e-9)                 # per-song norm01 (silence -> ~0)
    ph = _np.arange(len(sal)) % subdiv
    offbeat = _np.where(ph == 0, 0.0, _np.where(ph == subdiv // 2, 0.5, 1.0))  # on-beat 0, 8th 0.5, other off-beat 1.0
    if desmear:
        return (-float(gain) * offbeat * (1.0 - sal)).astype(_np.float32)   # subtract in LOW-saliency off-beats
    return (float(gain) * float(chaos) * offbeat * sal).astype(_np.float32)  # add in HIGH-saliency off-beats


def make_feature_extractor(features="highres"):
    """features name -> FeatureSpec(extractor, audio_dim, cache_dir). Matches the exporter's --features ladder.

    'highres' = the deployed 42-dim space (cache/samples_v3, what gen_motif_full_fixed expects); 'stage1' = 41-dim
    legacy (samples_v2); 'base' = 23-dim legacy (samples, extractor None). Pairing the WRONG dim/cache with a
    checkpoint is a top cataloged failure (generation-defaults skill §0) — this table makes them move together.
    """
    if features not in _FEATURE_SPECS:
        raise ValueError(f"unknown features {features!r}; valid: {list(_FEATURE_SPECS)}")
    flags, audio_dim, cache_dir = _FEATURE_SPECS[features]
    if flags is None:
        return FeatureSpec(None, audio_dim, cache_dir)
    from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig  # lazy: keep module import light
    return FeatureSpec(AudioFeatureExtractor(AudioFeatureConfig(**flags)), audio_dim, cache_dir)


def load_generator(checkpoint, audio_dim, device, *, arch=None, strict=False, eval_mode=True):
    """Build LayeredTypedChartGenerator(audio_dim, **MODEL_ARCH), load the checkpoint, .eval(). Returns the model.

    `strict=False` by default: legacy checkpoints (gen_radar/gen_layered) predate the style_encoder, so those
    params stay at init (unused unless --reference) — matches how both deployed scripts load. Pass `arch=` only to
    deviate from the one canonical architecture (`MODEL_ARCH`).
    """
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator  # lazy: keep module import light
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, **(arch or MODEL_ARCH)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"], strict=strict)
    if eval_mode:
        model.eval()
    return model
