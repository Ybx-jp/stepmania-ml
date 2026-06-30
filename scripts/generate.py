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
import re
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
from src.generation.decode_defaults import CANONICAL_DECODE, calib_arg_default, parse_phase_calib
from src.generation.decode_harness import conditioned_p_onset, compute_tau

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
    p.add_argument("--out", default=None,
                   help="GROUP folder to write into; the song is nested as <out>/<title>/ so StepMania "
                        "sees Songs/<group>/<song>/ (default: ./Generated/)")
    p.add_argument("--checkpoint", default="checkpoints/gen_motif_full_fixed/best_val.pt",
                   help="generator weights (default: the deployed 42-dim highres model)")
    p.add_argument("--bpm", type=float, default=None, help="song BPM (default: estimate it)")
    p.add_argument("--style", action="append", default=None, metavar="DIM=VAL",
                   help="optional groove feel, e.g. 'chaos=q0.7'. Multidimensional: comma-separate "
                        "('chaos=high,freeze=low') OR repeat the flag ('--style chaos=high --style freeze=low'); "
                        "both merge into one manifold spec.")
    p.add_argument("--guidance", type=float, default=1.5, help="CFG scale for --style (default 1.5)")
    # ---- decode palette: defaults sourced from the CANONICAL single source of truth
    #      (src/generation/decode_defaults.py = the same values export_typed_samples.py uses = what the user
    #      plays). These are NOT opt-in tweaks; they ARE the deployed regime. Pass 0 to disable a governor.
    p.add_argument("--fatigue_penalty", type=float, default=CANONICAL_DECODE["fatigue_penalty"],
                   help="per-note foot governor (canonical 2; 0 disables)")
    p.add_argument("--fatigue_free", type=float, default=CANONICAL_DECODE["fatigue_free"],
                   help="foot-governor free zone before the ceiling bites (canonical 6)")
    p.add_argument("--stamina_ceiling", type=float, default=CANONICAL_DECODE["stamina_ceiling"],
                   help="Stage-2 per-region density relief (canonical 50; 0 disables; needs --fatigue_penalty)")
    p.add_argument("--stamina_breathe", type=float, default=CANONICAL_DECODE["stamina_breathe"],
                   help="Stage-3 difficulty arc — ceiling breathes with audio energy (canonical 1.2; 0 = flat)")
    p.add_argument("--pattern_temperature", type=float, default=CANONICAL_DECODE["pattern_temperature"],
                   help="footwork sampling temperature (canonical 1.0 — real jack/jump balance; NOT 0.7)")
    p.add_argument("--type_temperature", type=float, default=CANONICAL_DECODE["type_temperature"],
                   help="per-panel tap/hold/roll sampling temperature (canonical 0.4 — surfaces holds at rate)")
    p.add_argument("--onset_phase_calib", type=str, default=calib_arg_default(),
                   help="★ the 16th-UNLOCK 'b8,b16' (logit space): un-buries 16th-offbeats so they float with "
                        "the audio per song. Canonical '0.0,1.0' (a KNEE: ~0.5 calm song .. 2.0 dense). '' = off.")
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
    #    --style may be repeated and/or comma-separated; merge every occurrence into ONE manifold spec
    #    (parse_spec dedupes by dim, last value wins) so multidimensional groove works either way.
    style_spec = ",".join(args.style) if args.style else ""
    manifold = RadarManifold.load(manifold_path)
    tvec, tinfo = manifold.build_target(style_spec, diff_idx)
    radar_for_gen = torch.from_numpy(tvec).unsqueeze(0).to(device)
    gen_density = tinfo["density"]
    print(f"BPM {bpm:.1f} | hop {hop} | {T} frames (~{T*hop/SR:.0f}s) | {args.difficulty} | "
          f"target density {gen_density:.3f}"
          + (f" | style '{style_spec}'" if style_spec else ""))

    # the 16th-unlock offset (b8,b16) — applied to the onset logits BEFORE tau AND inside generate(); the two
    # MUST match or the calib floods past a tau computed without it (conditioning-mechanics §6 / generation-defaults §1a)
    phase_calib = parse_phase_calib(args.onset_phase_calib)
    # the radar fed to BOTH tau and the decode MUST be the same one, else tau is calibrated on a different
    # distribution than generate() decodes from (conditioning-mechanics §3). No --style -> radar=None (null token).
    radar_arg = radar_for_gen if style_spec else None

    # 5. tau via the shared decode harness (conditioned + guided + phase-calibrated, exactly as generate() decodes)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        p_onset = conditioned_p_onset(model, memory, diff, radar=radar_arg,
                                      guidance=args.guidance, phase_calib=phase_calib)
    tau = compute_tau(p_onset, gen_density)

    # 6. generate with the CANONICAL full-stack palette + mandatory playability (mirrors export_typed_samples.py)
    gen_kwargs = dict(
        onset_threshold=tau,
        type_sample=True, type_temperature=args.type_temperature,
        pattern_sample=True, pattern_temperature=args.pattern_temperature,
        repetition_penalty=CANONICAL_DECODE["repetition_penalty"],
        max_jack_run=CANONICAL_DECODE["max_jack_run"],
        onset_phase_calib=phase_calib,  # ★ the 16th-unlock (same offset baked into tau above)
        fatigue_penalty=(args.fatigue_penalty if args.fatigue_penalty and args.fatigue_penalty > 0 else None),
        fatigue_free=args.fatigue_free,
        stamina_ceiling=(args.stamina_ceiling if args.stamina_ceiling and args.stamina_ceiling > 0 else None),
        stamina_tau=CANONICAL_DECODE["stamina_tau"], stamina_scale=CANONICAL_DECODE["stamina_scale"],
        stamina_breathe=args.stamina_breathe,
        bpm=bpm, radar=radar_arg,  # SAME radar tau was computed from (conditioning-mechanics §3)
        style=None, guidance_scale=(args.guidance if style_spec else 1.0),
    )
    enforce_playability(gen_kwargs, False)  # forces hold_aware / no_jump_during_hold / no_cross_during_hold
    with torch.no_grad():
        gen = pair_holds(model.generate(audio, diff, lengths=torch.tensor([T], device=device),
                                        **gen_kwargs)[0].cpu().numpy())

    # 7. write a StepMania-shaped folder: <group>/<song>/{chart.sm, audio}.
    #    StepMania expects Songs/<group>/<song>/<files> — a song folder placed DIRECTLY in a songs folder
    #    becomes an empty group and won't appear. So --out is the GROUP folder (you drop it into Songs/);
    #    the song lives one level in, named after the track.
    title = Path(args.audio).stem
    song_name = re.sub(r'[<>:"/\\|?*]', "_", title).strip() or "song"
    group_dir = Path(args.out) if args.out else PROJECT_ROOT / "Generated"
    song_dir = group_dir / song_name
    song_dir.mkdir(parents=True, exist_ok=True)
    music = os.path.basename(args.audio)
    try:
        shutil.copy2(args.audio, song_dir / music)
    except Exception:
        pass
    sm = charts_to_sm(
        charts=[{"chart": gen, "difficulty_name": "Challenge",
                 "difficulty_value": DIFFICULTY_METER[args.difficulty], "author": "generated"}],
        bpm=bpm, title=title, artist="", music=music, offset=0.0, typed=True,
    )
    (song_dir / "chart.sm").write_text(sm, encoding="utf-8")
    gen_d = float((gen != 0).any(1).mean())
    print(f"wrote {song_dir/'chart.sm'}  ({gen_d:.3f} realized density, "
          f"{int((gen[:, :] != 0).any(1).sum())} notes).")
    print(f"   Drop the GROUP folder '{group_dir}' into your StepMania Songs directory "
          f"(it becomes the group; the song '{song_name}' sits inside it).")


if __name__ == "__main__":
    main()
