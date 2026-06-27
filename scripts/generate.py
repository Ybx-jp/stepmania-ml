#!/usr/bin/env python3
"""
Generate a playable StepMania chart from a SINGLE audio file (bring-your-own-song).

Unlike experiments/generation_typed/export_typed_samples.py (a dataset-bound A/B eval
harness that generates on held-out songs from the training set), this needs NO dataset:
point it at one audio file + a target difficulty and it writes a .sm you can drop into
StepMania.

It replicates the canonical decode path exactly:
  audio -> 42-dim highres features (BPM-aligned hop, like the dataset)
        -> manifold density target for the difficulty (source-chart-free E[density|difficulty,style])
        -> tau from the SAME conditioned onset logits generate() decodes from
        -> generate() with the shipped governor default (fatigue_penalty=2) + mandatory playability
        -> .sm

Usage:
  python scripts/generate.py --audio song.ogg --difficulty Hard
  python scripts/generate.py --audio song.ogg --difficulty Medium --bpm 174 --style "chaos=q0.7"

Weights: defaults to checkpoints/gen_motif_full_fixed/best_val.pt (the deployed 42-dim model).
Density/conditioning needs cache/radar_manifold.npz (shipped with the repo).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.data.stepmania_parser import StepManiaChart, TimingEvent
from src.data.dataset import DIFFICULTY_NAMES
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.sm_writer import charts_to_sm
from src.generation.playtest_export import enforce_playability
from src.generation.radar_manifold import RadarManifold

SR = 22050
TIMESTEPS_PER_BEAT = 4  # 16th-note resolution — must match the parser's hop formula
# representative StepMania meter per difficulty bucket (for the .sm header only)
DIFFICULTY_METER = {"Beginner": 2, "Easy": 4, "Medium": 6, "Hard": 9}


def estimate_bpm(audio_path: str) -> float:
    """Global tempo estimate (user can override with --bpm)."""
    import librosa
    y, sr = librosa.load(audio_path, sr=SR)
    tempo = librosa.beat.tempo(y=y, sr=sr)
    return float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)


def build_stub_chart(audio_path: str, bpm: float, duration: float, hop_length: int) -> StepManiaChart:
    """A minimal chart carrying only what extract_from_chart needs: offset, song_length, hop_length.
    No note data — we are GENERATING the notes, not reading them."""
    total_beats = duration * bpm / 60.0
    return StepManiaChart(
        title=Path(audio_path).stem, artist="", audio_file=audio_path,
        bpm=bpm, offset=0.0, sample_start=0.0, sample_length=0.0,
        timing_events=[TimingEvent(beat=0.0, value=bpm, event_type="bpm")],
        note_data=[], song_length_seconds=duration,
        timesteps_total=int(total_beats * TIMESTEPS_PER_BEAT), hop_length=hop_length,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate a StepMania chart from one audio file.")
    p.add_argument("--audio", required=True, help="path to an audio file (.ogg/.mp3/.wav)")
    p.add_argument("--difficulty", required=True, choices=list(DIFFICULTY_METER),
                   help="target difficulty bucket")
    p.add_argument("--out", default=None, help="output folder (default: <audio-stem>_sm/)")
    p.add_argument("--checkpoint", default="checkpoints/gen_motif_full_fixed/best_val.pt",
                   help="generator weights (default: the deployed 42-dim highres model)")
    p.add_argument("--bpm", type=float, default=None, help="song BPM (default: estimate it)")
    p.add_argument("--style", default=None,
                   help="optional groove feel, e.g. 'chaos=q0.7' (manifold conditional-fill path)")
    p.add_argument("--guidance", type=float, default=1.5, help="CFG scale for --style (default 1.5)")
    p.add_argument("--fatigue_penalty", type=float, default=2.0,
                   help="per-note foot governor (shipped default 2; 0 disables)")
    p.add_argument("--stamina_ceiling", type=float, default=None,
                   help="opt-in Stage-2 density relief (e.g. 25); needs --fatigue_penalty")
    p.add_argument("--stamina_breathe", type=float, default=None,
                   help="opt-in Stage-3 difficulty arc (e.g. 1.2); needs --stamina_ceiling")
    p.add_argument("--pattern_temperature", type=float, default=0.7,
                   help="footwork sampling temperature (coherence range ~0.6-0.85)")
    p.add_argument("--max_len", type=int, default=2048,
                   help="cap on generated frames (clamped to the model's trained context, 2048)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = PROJECT_ROOT / args.checkpoint
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}\n"
                         "Download the weights (see README) or pass --checkpoint.")
    manifold_path = PROJECT_ROOT / "cache/radar_manifold.npz"
    if not manifold_path.is_file():
        raise SystemExit(f"manifold not found: {manifold_path}\n"
                         "It ships with the repo and supplies the difficulty density target.")

    # 1. BPM -> the 16th-note-aligned hop the model was trained on
    bpm = args.bpm if args.bpm is not None else estimate_bpm(args.audio)
    if not (60.0 <= bpm <= 200.0):
        print(f"⚠️  BPM {bpm:.1f} is outside the trained range [60, 200] — output may be off-grid. "
              "Pass --bpm if the estimate looks wrong.")
    hop = int(SR * 60 / (bpm * TIMESTEPS_PER_BEAT))

    # 2. extract the 42-dim highres feature set via a stub chart (same pipeline as the dataset)
    import librosa
    duration = librosa.get_duration(path=args.audio)
    feat_ext = AudioFeatureExtractor(AudioFeatureConfig(
        use_chroma=True, use_hpss_onsets=True, use_metric_phase=True, use_highres_onset=True))
    stub = build_stub_chart(args.audio, bpm, duration, hop)
    feats = feat_ext.extract_from_chart(args.audio, stub)
    if feats is None:
        raise SystemExit(f"feature extraction failed for {args.audio}")
    audio_tensor = feats.get_aligned_features()  # (T, 42)
    if np.any(~np.isfinite(audio_tensor)):
        raise SystemExit("non-finite audio features — bad/corrupt audio?")
    # 3. model
    model = LayeredTypedChartGenerator(audio_dim=42, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"], strict=False)
    model.eval()

    # the model's positional encoding is a HARD context cap (trained length) — never feed more frames than that,
    # or the pos-encoding add throws a size mismatch. Longer songs are truncated to the context, with a warning.
    ctx = int(model.pos_encoding.pe.size(1))
    T = min(audio_tensor.shape[0], args.max_len, ctx)
    if audio_tensor.shape[0] > T:
        print(f"⚠️  song is {audio_tensor.shape[0]} frames; truncating to the model's {ctx}-frame context "
              f"(~{T * hop / SR:.0f}s). Charting past the trained context isn't supported yet.")
    audio = torch.from_numpy(audio_tensor[:T].astype(np.float32)).unsqueeze(0).to(device)

    diff_idx = list(DIFFICULTY_METER).index(args.difficulty)
    diff = torch.tensor([diff_idx], device=device)

    # 4. manifold: source-chart-free density target for this difficulty (+ optional --style feel)
    manifold = RadarManifold.load(manifold_path)
    tvec, tinfo = manifold.build_target(args.style or "", diff_idx)
    radar_for_gen = torch.from_numpy(tvec).unsqueeze(0).to(device)
    gen_density = tinfo["density"]
    print(f"BPM {bpm:.1f} | hop {hop} | {T} frames (~{T*hop/SR:.0f}s) | {args.difficulty} | "
          f"target density {gen_density:.3f}"
          + (f" | style '{args.style}'" if args.style else ""))

    # 5. tau from the SAME conditioned logits generate() decodes from (else conditioning floods past
    #    a tau calibrated on unconditioned p — conditioning-mechanics §8)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        ol = model.onset_logits(memory, diff, radar=radar_for_gen, style=None)[0]
        if args.guidance != 1.0 and args.style:
            ol_u = model.onset_logits(memory, diff, radar=None, style=None)[0]
            ol = ol_u + args.guidance * (ol - ol_u)
        p_onset = torch.sigmoid(ol).cpu().numpy()
    tau = float(np.quantile(p_onset, 1 - gen_density)) if gen_density and gen_density > 0 else 0.5

    # 6. generate with the shipped governor default + mandatory playability
    gen_kwargs = dict(
        onset_threshold=tau, type_sample=True, pattern_sample=True,
        pattern_temperature=args.pattern_temperature,
        max_jack_run=2,
        fatigue_penalty=(args.fatigue_penalty if args.fatigue_penalty and args.fatigue_penalty > 0 else None),
        stamina_ceiling=args.stamina_ceiling, stamina_breathe=args.stamina_breathe,
        bpm=bpm, radar=(radar_for_gen if args.style else None),
        style=None, guidance_scale=(args.guidance if args.style else 1.0),
    )
    enforce_playability(gen_kwargs, False)  # forces hold_aware / no_jump_during_hold / no_cross_during_hold
    with torch.no_grad():
        gen = pair_holds(model.generate(audio, diff, lengths=torch.tensor([T], device=device),
                                        **gen_kwargs)[0].cpu().numpy())

    # 7. write the .sm (single generated chart) + copy the audio next to it
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / f"{Path(args.audio).stem}_sm"
    out_dir.mkdir(parents=True, exist_ok=True)
    music = os.path.basename(args.audio)
    try:
        shutil.copy2(args.audio, out_dir / music)
    except Exception:
        pass
    sm = charts_to_sm(
        charts=[{"chart": gen, "difficulty_name": "Challenge",
                 "difficulty_value": DIFFICULTY_METER[args.difficulty], "author": "generated"}],
        bpm=bpm, title=Path(args.audio).stem, artist="", music=music, offset=0.0, typed=True,
    )
    (out_dir / "chart.sm").write_text(sm, encoding="utf-8")
    gen_d = float((gen != 0).any(1).mean())
    print(f"wrote {out_dir/'chart.sm'}  ({gen_d:.3f} realized density, "
          f"{int((gen[:, :] != 0).any(1).sum())} notes). Drop the folder into StepMania to play.")


if __name__ == "__main__":
    main()
