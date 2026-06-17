"""
Evaluation utilities for generated charts (Phase 2 Stage 0).

Two families of metrics, per docs/phase2_generative_design.md:

1. onset_density_metrics: rhythmic correctness without a model — onset F1
   (did we place a step where one belongs, ignoring exact panels) plus density
   and panel-accuracy-given-onset. Pure numpy; the workhorse for Stage 1 baselines.

2. DifficultyCritic: reuses the frozen Phase 1 classifier as a learned critic —
   given a generated chart + the song's audio, predict its difficulty and measure
   agreement with the requested target. This is the headline conditioning-fidelity
   metric for the generator.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch

from src.data.groove_radar import GrooveRadarCalculator
from src.data.stepmania_parser import TimingEvent

TIMESTEPS_PER_BEAT = 4  # 16th-note resolution


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _apply_mask(chart: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return chart
    mask = _to_numpy(mask).reshape(-1).astype(bool)
    return chart[mask[: chart.shape[0]]]


# ---- onset / density metrics ----------------------------------------------------

def onset_density_metrics(
    generated: Union[np.ndarray, torch.Tensor],
    reference: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Rhythmic-correctness metrics for a generated (T, 4) chart.

    Onset = any panel active at a timestep. If `reference` is given, computes
    onset precision/recall/F1 and panel-accuracy-on-shared-onsets against it;
    otherwise returns only density statistics for `generated`.

    Args:
        generated: (T, 4) binary chart.
        reference: optional ground-truth (T, 4) chart, same length.
        mask: optional (T,) validity mask (True = keep) applied to both.

    Returns:
        Dict of metrics. Density = fraction of timesteps with >=1 step.
    """
    gen = (_to_numpy(generated) > 0.5).astype(np.int64)
    gen = _apply_mask(gen, mask)
    gen_onset = gen.sum(axis=1) > 0

    out: Dict[str, float] = {
        "gen_density": float(gen_onset.mean()) if gen.shape[0] else 0.0,
        "gen_jump_rate": float((gen.sum(axis=1) == 2).mean()) if gen.shape[0] else 0.0,
        "n_timesteps": int(gen.shape[0]),
    }

    if reference is None:
        return out

    ref = (_to_numpy(reference) > 0.5).astype(np.int64)
    ref = _apply_mask(ref, mask)
    if ref.shape[0] != gen.shape[0]:
        raise ValueError(f"generated ({gen.shape[0]}) and reference ({ref.shape[0]}) length mismatch")
    ref_onset = ref.sum(axis=1) > 0

    tp = int(np.sum(gen_onset & ref_onset))
    fp = int(np.sum(gen_onset & ~ref_onset))
    fn = int(np.sum(~gen_onset & ref_onset))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Panel accuracy on timesteps where both have an onset (did we pick right arrows?).
    shared = gen_onset & ref_onset
    if shared.any():
        panel_acc = float(np.mean(np.all(gen[shared] == ref[shared], axis=1)))
    else:
        panel_acc = 0.0

    out.update({
        "onset_precision": precision,
        "onset_recall": recall,
        "onset_f1": f1,
        "panel_accuracy_on_onset": panel_acc,
        "ref_density": float(ref_onset.mean()) if ref.shape[0] else 0.0,
        "density_ratio": (out["gen_density"] / float(ref_onset.mean())) if ref_onset.mean() else 0.0,
    })
    return out


# ---- groove radar for a generated chart -----------------------------------------

def chart_groove_radar_vector(
    chart: Union[np.ndarray, torch.Tensor],
    bpm: float,
    calculator: Optional[GrooveRadarCalculator] = None,
) -> np.ndarray:
    """Compute the normalized 5-dim groove-radar vector for a generated (T, 4) chart.

    Approximates the dataset's pipeline for a Phase-1 (no-holds, fixed-BPM) chart so
    the result can be fed to the classifier. Returns [stream, voltage, air, freeze, chaos]
    each in [0, 1].
    """
    arr = (_to_numpy(chart) > 0.5).astype(np.float32)
    T = arr.shape[0]
    calc = calculator or GrooveRadarCalculator(timesteps_per_beat=TIMESTEPS_PER_BEAT)

    song_length_beats = T / TIMESTEPS_PER_BEAT
    song_length_seconds = song_length_beats * 60.0 / bpm if bpm > 0 else 0.0

    # Build note_beats; no holds in Phase 1 scope.
    note_beats = []
    active = np.argwhere(arr > 0)
    for t, panel in active:
        note_beats.append((t / TIMESTEPS_PER_BEAT, int(panel), "tap"))
    hold_info = {
        "holds": [],
        "total_hold_beats": 0.0,
        "note_beats": note_beats,
        "song_length_beats": song_length_beats,
    }
    timing_events = [TimingEvent(beat=0.0, value=float(bpm), event_type="bpm")]

    radar = calc.calculate(
        chart_tensor=arr,
        hold_info=hold_info,
        timing_events=timing_events,
        song_length_seconds=song_length_seconds,
        avg_bpm=float(bpm),
    )
    return radar.to_vector().astype(np.float32)


# ---- classifier-as-critic -------------------------------------------------------

DIFFICULTY_NAMES = ["Beginner", "Easy", "Medium", "Hard"]


class DifficultyCritic:
    """Wraps the frozen Phase 1 classifier to score generated charts.

    Reconstructs LateFusionClassifier from the model config (checkpoints store only
    weights), loads the chosen checkpoint, and predicts a difficulty class for a
    (chart, audio) pair — computing groove-radar features on the fly.

    Default checkpoint is the Phase 1 winner, standard_ordinal_multi.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/ordinal_exp/standard_ordinal_multi/best_val_loss.pt",
        model_config: Optional[Dict] = None,
        config_path: str = "config/model_config.yaml",
        head_type: str = "ordinal",
        ordinal_multi_output: bool = True,
        device: Optional[torch.device] = None,
    ):
        from src.models import LateFusionClassifier

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_config is None:
            import yaml
            with open(config_path) as f:
                model_config = yaml.safe_load(f)["classifier"]
        cfg = dict(model_config)
        cfg["head_type"] = head_type
        cfg["ordinal_multi_output"] = ordinal_multi_output

        self.model = LateFusionClassifier(cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self._radar_calc = GrooveRadarCalculator(timesteps_per_beat=TIMESTEPS_PER_BEAT)

    @torch.no_grad()
    def predict(
        self,
        chart: Union[np.ndarray, torch.Tensor],
        audio: Union[np.ndarray, torch.Tensor],
        bpm: float,
        groove_radar: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, Union[int, str, np.ndarray]]:
        """Predict difficulty class (0-3) for one generated chart + its song audio.

        Args:
            chart: (T, 4) generated chart.
            audio: (T, audio_dim) aligned audio features for the same song.
            bpm: chart BPM (for groove-radar computation if not supplied).
            groove_radar: optional precomputed (5,) radar vector.

        Returns:
            Dict with 'class' (int), 'name' (str), and 'probs' (np.ndarray).
        """
        chart_t = torch.as_tensor(_to_numpy(chart), dtype=torch.float32, device=self.device).unsqueeze(0)
        audio_t = torch.as_tensor(_to_numpy(audio), dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.ones(1, chart_t.shape[1], device=self.device)

        if groove_radar is None:
            groove_radar = chart_groove_radar_vector(chart, bpm, calculator=self._radar_calc)
        radar_t = torch.as_tensor(_to_numpy(groove_radar), dtype=torch.float32, device=self.device).unsqueeze(0)

        logits = self.model(audio_t, chart_t, mask_t, groove_radar=radar_t)
        if isinstance(logits, dict):
            logits = logits["logits"]

        if getattr(self.model, "head_type", None) == "ordinal":
            pred = self.model.predict_class_from_logits(logits)
            cum = torch.sigmoid(logits)
            ones = torch.ones(logits.shape[0], 1, device=logits.device)
            zeros = torch.zeros(logits.shape[0], 1, device=logits.device)
            ext = torch.cat([ones, cum, zeros], dim=1)
            probs = torch.clamp(ext[:, :-1] - ext[:, 1:], min=1e-8)
        else:
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

        cls = int(pred.item())
        return {"class": cls, "name": DIFFICULTY_NAMES[cls], "probs": probs.squeeze(0).cpu().numpy()}

    def agreement(
        self,
        charts: List[Union[np.ndarray, torch.Tensor]],
        audios: List[Union[np.ndarray, torch.Tensor]],
        targets: List[int],
        bpms: List[float],
    ) -> Dict[str, float]:
        """Conditioning-fidelity over a batch: exact and adjacent (|Δ|<=1) agreement,
        plus difficulty MAE between predicted and requested target classes.
        """
        preds = [self.predict(c, a, b)["class"] for c, a, b in zip(charts, audios, bpms)]
        preds = np.array(preds)
        tgts = np.array(targets)
        diff = np.abs(preds - tgts)
        return {
            "exact_agreement": float(np.mean(diff == 0)),
            "adjacent_agreement": float(np.mean(diff <= 1)),
            "difficulty_mae": float(np.mean(diff)),
            "n": int(len(preds)),
        }
