#!/usr/bin/env python3
"""The seq-appropriate DECODE SURFACE (2026-06-29). The deployed decode palette was co-evolved with the AUDIO onset
head; the seq head (causal, note-momentum-driven, no natural silences) needs adapted knobs. This module supplies the
two adaptations the audio head got for free, so a fair A/B can run (see notes/onset_placement_findings.md correction):

  build_rest_env  — the REST VALVE. The audio head's `p_onset` (non-causal, energy-tracking) box-smoothed over a
                    phrase + z-scored = an energy envelope (the SAME math as the deployed breathing arc, §8c). Passed
                    to rollout(rest_env=, rest_gain=) it biases the seq logit DOWN in quiet sections -> the rests the
                    note-momentum seq head never takes on its own ("never pauses").
  selfcal_tau     — ADAPTIVE tau. A global quantile assumes the audio head's calibrated non-causal p_onset; the seq
                    head's concentrated in-loop logits make a fixed tau cliff-y (62%->0%). Calibrate tau per song from
                    the head's OWN first-pass decision logits to hit a target density (iterate for the trajectory shift).
"""
import numpy as np, torch
from probe_seqonset_rollout import rollout


@torch.no_grad()
def build_rest_env(model, audio_np, diff, T, device, win=96):
    """z-scored phrase-smoothed AUDIO p_onset (the energy/silence signal) — the rest-valve envelope (T,). Mirrors the
    deployed breathing energy (cond-mech §8c): env = zscore(boxsmooth(p_onset, ~96)). NON-causal (audio is precomputed)."""
    audio_t = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    p = torch.sigmoid(model.onset_logits(model.encode_audio(audio_t), torch.tensor([diff], device=device)))[0, :T].cpu().numpy()
    w = max(int(win), 1)
    env = np.convolve(p, np.ones(w) / w, mode='same')                       # phrase box-smooth
    mu, sd = env.mean(), env.std()
    return (env - mu) / sd if sd > 1e-6 else np.zeros_like(env)             # z-score over the song


@torch.no_grad()
def selfcal_tau(model, head, audio_np, diff, T, device, target_d, rest_env=None, rest_gain=0.0,
                phase_pen=None, iters=8, lo=0.02, hi=0.98):
    """Per-song ADAPTIVE tau by BINARY SEARCH on realized free-run density (monotone: higher tau -> fewer onsets).
    Robust to the calibration cliff (a quantile-of-realized-logits iteration DIVERGES — collapsed a song to empty);
    the search can't collapse unless target_d is unreachable. Returns the calibrated tau (rollout threshold)."""
    if target_d <= 0:
        return float(hi)
    best = (1e9, float(hi))                                                  # (|d-target|, tau) — track closest achievable
    for _ in range(max(int(iters), 1)):
        tau = (lo + hi) / 2
        on = rollout(model, head, audio_np, diff, T, tau, device,
                     rest_env=rest_env, rest_gain=rest_gain, phase_pen=phase_pen)
        d = float(on.mean())
        if abs(d - target_d) < best[0]:                                     # the seq logits are concentrated -> density
            best = (abs(d - target_d), tau)                                 # vs tau can be a CLIFF (flood/collapse, no
        if d > target_d:                                                    # real-d point); track the closest tau seen
            lo = tau
        else:
            hi = tau
    return best[1]
