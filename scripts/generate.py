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
from src.data.stepmania_parser import StepManiaChart, TimingEvent
from src.data.dataset import DIFFICULTY_NAMES
from src.data.meter_detect import detect_triple_pref_audio  # audio duple/triplet classifier for --auto_b_trip (v2)
from src.generation.typed import pair_holds
from src.generation.sm_writer import charts_to_sm
from src.generation import sm_headers
from src.generation.playtest_export import enforce_playability
from src.generation.radar_manifold import RadarManifold
from src.generation.decode_defaults import (
    CANONICAL_DECODE, calib_arg_default, parse_phase_calib, grid_snap_offset)
from src.generation.decode_harness import (
    conditioned_p_onset, compute_tau, make_feature_extractor, load_generator, MODEL_ARCH)

SR = 22050
# 16th-note resolution — the v1/highres grid. v2/highres_v2 uses a 48th grid (12/beat); the ACTUAL subdivision
# is read per-run from the feature-extractor config (fspec.extractor.config.timesteps_per_beat), NOT this constant,
# so the stub-chart hop + timesteps_total, tau, decode, and the .sm writer all agree on whichever grid is active.
TIMESTEPS_PER_BEAT = 4
V2_MSL = 5400        # v2 training sequence length; the v1 max_len (2048) clips a 48th-grid song to ~1/3 (170 beats)
V2_CTX = 5504        # v2 positional-encoding capacity (a full ~3-min song on the finer grid; > V2_MSL by design)
V2_CHECKPOINT = "checkpoints/gen_motif_v2_48th_cont/best_val.pt"   # the by-ear-passed 48th-grid deploy candidate
# representative StepMania meter per difficulty bucket (for the .sm header only)
DIFFICULTY_METER = {"Beginner": 2, "Easy": 4, "Medium": 6, "Hard": 9}


def estimate_bpm(audio_path: str) -> float:
    """Global tempo estimate (user can override with --bpm)."""
    import librosa
    y, sr = librosa.load(audio_path, sr=SR)
    tempo = librosa.beat.tempo(y=y, sr=sr)
    return float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)


def build_stub_chart(audio_path: str, bpm: float, duration: float, hop_length: int,
                     subdiv: int = TIMESTEPS_PER_BEAT) -> StepManiaChart:
    """A minimal chart carrying only what extract_from_chart needs: offset, song_length, hop_length.
    No note data — we are GENERATING the notes, not reading them. `subdiv` = the grid resolution (4 = 16th /
    highres, 12 = 48th / highres_v2): it sets timesteps_total so extract_from_chart re-grids the audio to the
    SAME number of cells the model was trained on."""
    total_beats = duration * bpm / 60.0
    return StepManiaChart(
        title=Path(audio_path).stem, artist="", audio_file=audio_path,
        bpm=bpm, offset=0.0, sample_start=0.0, sample_length=0.0,
        timing_events=[TimingEvent(beat=0.0, value=bpm, event_type="bpm")],
        note_data=[], song_length_seconds=duration,
        timesteps_total=int(total_beats * subdiv), hop_length=hop_length,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate a StepMania chart from one audio file.")
    p.add_argument("--audio", required=True, help="path to an audio file (.ogg/.mp3/.wav)")
    p.add_argument("--difficulty", required=True, choices=list(DIFFICULTY_METER),
                   help="target difficulty bucket")
    p.add_argument("--out", default=None,
                   help="GROUP folder to write into; the song is nested as <out>/<title>/ so StepMania "
                        "sees Songs/<group>/<song>/ (default: ./Generated/)")
    # ---- .sm presentation / header (what StepMania shows for the song) ----------------------------------------
    p.add_argument("--title", default=None, help="#TITLE (default: the audio filename stem)")
    p.add_argument("--subtitle", default=None, help="#SUBTITLE")
    p.add_argument("--artist", default=None, help="#ARTIST (default: empty)")
    p.add_argument("--genre", default=None, help="#GENRE")
    p.add_argument("--credit", default=None, help="#CREDIT (charter credit)")
    p.add_argument("--banner", default=None, help="#BANNER image filename (must exist in the song folder)")
    p.add_argument("--background", default=None, help="#BACKGROUND image filename")
    p.add_argument("--cdtitle", default=None, help="#CDTITLE image filename")
    p.add_argument("--bgchanges", default=None,
                   help="#BGCHANGES value, e.g. a full 'beat=video.mp4=1.000=1=0=0=StretchNoLoop==' entry to play "
                        "a music video as the background. Usually easier via --inherit_from.")
    p.add_argument("--sample_start", type=float, default=None, help="#SAMPLESTART (music-wheel preview start, s)")
    p.add_argument("--sample_length", type=float, default=None, help="#SAMPLELENGTH (preview length, s)")
    p.add_argument("--display_bpm", default=None, help="#DISPLAYBPM shown on the wheel (a number, 'a:b', or '*')")
    p.add_argument("--inherit_from", default=None, metavar="SM|auto",
                   help="inherit presentation tags (subtitle/genre/credit/banner/background/cdtitle/bgchanges/"
                        "sample window/displaybpm) from a source .sm/.ssc chart AND copy the media it references "
                        "(banner, background, #BGCHANGES video, ...) into the new song folder. 'auto' = the "
                        ".sm/.ssc sitting next to --audio. Explicit flags above OVERRIDE inherited values. Timing "
                        "tags (#BPMS/#STOPS) are NOT inherited — the generator owns its own grid.")
    p.add_argument("--checkpoint", default=None,
                   help="generator weights (default: the deployed 42-dim highres model, or the v2 48th-grid "
                        "checkpoint when --features highres_v2). Pass a path to override.")
    p.add_argument("--features", choices=["highres", "highres_v2"], default="highres",
                   help="feature/grid space. highres = the deployed 42-dim 16th grid (gen_motif_full_fixed). "
                        "highres_v2 = the same 42 channels on the data-layer-v2 48th grid (timesteps_per_beat=12, "
                        "beat-sync) — resolves triplets, removes the 16th-grid triplet tax; auto-selects the v2 "
                        "checkpoint + 48th-grid .sm writer + v2 context length. Pair ONLY with a v2 checkpoint.")
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
                        "the audio per song. Canonical '0.0,1.0' (a KNEE: ~0.5 calm song .. 2.0 dense). '' = off. "
                        "A 3rd element (b_trip) is the v2 triplet band; with --auto_b_trip it applies per song.")
    # ---- v2 (48th-grid) decode flags — all are v1 no-ops (subdiv=4) BY CONSTRUCTION, so they're safe defaults
    p.add_argument("--no_fast_jump", action=argparse.BooleanOptionalAction, default=True,
                   help="[v2; DEFAULT ON] forbid a >=2-fresh-press JUMP at sub-16th spacing (a 24th/48th gap on the "
                        "48th grid) -> forces a playable SINGLE but keeps the onset. Closes the hole where a "
                        "sub-16th jump evades the fatigue governor. v1 (16th grid) = no-op. --no-no_fast_jump to A/B.")
    p.add_argument("--min_onset_gap", type=int, default=None,
                   help="[v2] FOOTSPEED FLOOR (frames): min spacing between onsets. None = auto (2 on the 48th grid "
                        "-> forbids 1-frame 48th flams, keeps 2-frame triplet-16ths; 1 on the 16th grid = no-op).")
    p.add_argument("--grid_snap", choices=["auto", "off", "all"], default="auto",
                   help="[v2; DEFAULT auto] veto onsets on the pure-48th cells {1,5,7,11}@subdiv=12 so busy low/mid "
                        "charts stay on the 16th grid (real 48th-usage ~0%% at Beginner/Easy/Medium). auto = ON for "
                        "difficulty <= Medium, OFF at Hard (fast 48th runs legit there). v1 grid = no-op.")
    p.add_argument("--grid_snap_keep_triplets", action=argparse.BooleanOptionalAction, default=True,
                   help="[v2] with --grid_snap, keep the triplet family {2,4,8,10} and veto ONLY {1,5,7,11}, so the "
                        "snap kills 48th jitter but preserves triplets (composes with --auto_b_trip).")
    p.add_argument("--auto_b_trip", action=argparse.BooleanOptionalAction, default=True,
                   help="[v2; DEFAULT ON] DUPLE/TRIPLET SWITCH: apply the triplet band (b_trip, the 3rd "
                        "--onset_phase_calib element; 0.7 if absent) ONLY to triplet-feel songs, per an audio meter "
                        "detector (triple_pref > --triple_pref_thresh). Duple songs get no band (no added busyness). "
                        "v1 grid = no-op. --no-auto_b_trip reverts to the fixed global b_trip.")
    p.add_argument("--triple_pref_thresh", type=float, default=0.0,
                   help="meter threshold for --auto_b_trip (triple_pref > this => triplet-feel => band on).")
    p.add_argument("--max_len", type=int, default=2048,
                   help="cap on generated frames (clamped to the model's trained context; auto-raised to the v2 "
                        "sequence length under --features highres_v2, where 2048 would clip the song to ~1/3)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_v2 = args.features == "highres_v2"
    # checkpoint default is grid-aware: v2 features REQUIRE a v2 (48th-grid) checkpoint or the weights mis-match
    # the audio grid. If --checkpoint was left unset, pick the matching deployed default.
    if args.checkpoint is None:
        args.checkpoint = V2_CHECKPOINT if is_v2 else "checkpoints/gen_motif_full_fixed/best_val.pt"
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
    # The safe single-hop grid holds well past the TRAINING gate [60,200]; warn only outside the widened
    # inference band [40,320] (StepManiaParser.for_inference). A gimmick-scale --bpm (e.g. 2467 copied from a
    # #BPMS scroll trick) would make hop≈0 and grid garbage — that's what the parser's gimmick guard catches
    # on the export path; here it's user-supplied, so just flag it.
    if not (40.0 <= bpm <= 320.0):
        print(f"⚠️  BPM {bpm:.1f} is outside the supported range [40, 320] — output may be off-grid "
              "(a gimmick/notation BPM will mis-grid the whole song). Pass --bpm if the estimate looks wrong.")

    # 2. extract the 42-dim feature set via a stub chart (same pipeline as the dataset)
    import librosa
    duration = librosa.get_duration(path=args.audio)
    fspec = make_feature_extractor(args.features)  # 42-dim highres (16th) or highres_v2 (48th); harness = single source
    # decode phase-band period, derived from the feature grid so it can't drift from the features: 4 = 16th grid
    # (the deployed highres), 12 = the data-layer-v2 48th grid. Threaded into the hop + stub AND both tau and
    # generate() below so the whole pipeline agrees on the grid (Phase 5; conditioning-mechanics §6).
    subdiv = fspec.extractor.config.timesteps_per_beat if fspec.extractor is not None else 4
    hop = int(SR * 60 / (bpm * subdiv))  # finer hop on the 48th grid so the audio is gridded at `subdiv`/beat
    stub = build_stub_chart(args.audio, bpm, duration, hop, subdiv=subdiv)
    feats = fspec.extractor.extract_from_chart(args.audio, stub)
    if feats is None:
        raise SystemExit(f"feature extraction failed for {args.audio}")
    audio_tensor = feats.get_aligned_features()  # (T, 42)
    if np.any(~np.isfinite(audio_tensor)):
        raise SystemExit("non-finite audio features — bad/corrupt audio?")
    # 3. model. v2 (48th grid): build the positional encoding at V2 capacity (5504) so a full ~3-min song fits —
    # the default 2048-frame build would index pe out of range past ~68s on the finer grid. The pe is a fixed
    # sinusoid, so a larger buffer is byte-identical to the checkpoint's over the shared range. Also raise --max_len
    # to the v2 sequence length (the 2048 default clips a 48th-grid song to ~1/3 — the 150-vs-450-tap truncation bug).
    v2_arch = None
    if is_v2:
        v2_arch = dict(MODEL_ARCH, max_len=V2_CTX)
        if args.max_len < V2_MSL:
            print(f"highres_v2: raising --max_len {args.max_len} -> {V2_MSL} (48th grid; 2048 clips to ~120 beats).")
            args.max_len = V2_MSL
    model = load_generator(ckpt, fspec.audio_dim, device, arch=v2_arch)  # builds + loads (strict=False) + .eval()

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
    # duple/triplet SWITCH (v2 only): peel the triplet-band magnitude (b_trip = the 3rd calib element, 0.7 if absent)
    # off the global calib so it can be applied PER SONG below; the 16th-unlock (b8,b16) stays global.
    _b8b16 = ((float(phase_calib[0]), float(phase_calib[1])) if phase_calib and len(phase_calib) >= 2 else (0.0, 1.0))
    auto_b_trip_val = None
    if args.auto_b_trip:
        auto_b_trip_val = float(phase_calib[2]) if (phase_calib and len(phase_calib) > 2) else 0.7
        phase_calib = _b8b16  # duple default (no band) unless the per-song detector says triplet
    song_calib = phase_calib
    if auto_b_trip_val is not None and subdiv != 4:  # v1 (16th grid) has an empty triplet band -> skip the detector
        r = detect_triple_pref_audio(args.audio, bpm)
        tp = r['triple_pref'] if r else None
        if tp is None:
            print("  [auto_b_trip] meter undetected -> global calib (no band)")
        else:
            is_trip = tp > args.triple_pref_thresh
            b = auto_b_trip_val if is_trip else 0.0
            song_calib = (_b8b16[0], _b8b16[1], b)
            print(f"  [auto_b_trip] triple_pref={tp:+.2f} -> "
                  + (f"TRIPLET (b_trip={b:.2f})" if is_trip else "duple (band off)"))
    # 16th-GRID SNAP (v2 only): veto the pure-48th cells so busy low/mid charts stay on-grid. Single-sourced into
    # BOTH tau (extra_offset=) and decode (onset_logit_offset=). auto = difficulty <= Medium; v1 (subdiv=4) = no-op.
    do_snap = subdiv != 4 and (args.grid_snap == 'all' or (args.grid_snap == 'auto' and diff_idx <= 2))
    snap_off = (grid_snap_offset(T, subdiv, keep_triplets=args.grid_snap_keep_triplets, device=device)
                if do_snap else None)
    # the radar fed to BOTH tau and the decode MUST be the same one, else tau is calibrated on a different
    # distribution than generate() decodes from (conditioning-mechanics §3). No --style -> radar=None (null token).
    radar_arg = radar_for_gen if style_spec else None

    # 5. tau via the shared decode harness (conditioned + guided + phase-calibrated + snap offset, as generate() decodes)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        p_onset = conditioned_p_onset(model, memory, diff, radar=radar_arg, guidance=args.guidance,
                                      phase_calib=song_calib, extra_offset=snap_off, subdiv=subdiv)
    tau = compute_tau(p_onset, gen_density)

    # 6. generate with the CANONICAL full-stack palette + mandatory playability (mirrors export_typed_samples.py)
    gen_kwargs = dict(
        onset_threshold=tau,
        type_sample=True, type_temperature=args.type_temperature,
        pattern_sample=True, pattern_temperature=args.pattern_temperature,
        repetition_penalty=CANONICAL_DECODE["repetition_penalty"],
        max_jack_run=CANONICAL_DECODE["max_jack_run"],
        min_onset_gap=args.min_onset_gap,  # v2 footspeed floor (frames); None -> auto per subdiv (2 on 48th; §8)
        no_fast_jump=args.no_fast_jump,    # v2: forbid >=2-fresh JUMP at sub-16th spacing (v1 no-op; §8d)
        onset_phase_calib=song_calib, subdiv=subdiv,  # ★ the 16th-unlock + per-song triplet band (baked into tau above)
        onset_logit_offset=snap_off,       # 16th-grid snap (same offset baked into tau above; None if not snapping)
        fatigue_penalty=(args.fatigue_penalty if args.fatigue_penalty and args.fatigue_penalty > 0 else None),
        fatigue_free=args.fatigue_free,
        stamina_ceiling=(args.stamina_ceiling if args.stamina_ceiling and args.stamina_ceiling > 0 else None),
        stamina_tau=CANONICAL_DECODE["stamina_tau"], stamina_scale=CANONICAL_DECODE["stamina_scale"],
        stamina_breathe=args.stamina_breathe,
        hold_stream_penalty=CANONICAL_DECODE["hold_stream_penalty"],  # suppress holds in dense streams (2026-07-02)
        hold_stream_floor=CANONICAL_DECODE["hold_stream_floor"], hold_stream_win=CANONICAL_DECODE["hold_stream_win"],
        footswitch=CANONICAL_DECODE["footswitch"],  # DEFAULT False: force one-foot jacks, model alternates (2026-07-02)
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
    title = args.title if args.title else Path(args.audio).stem
    song_name = re.sub(r'[<>:"/\\|?*]', "_", title).strip() or "song"
    group_dir = Path(args.out) if args.out else PROJECT_ROOT / "Generated"
    song_dir = group_dir / song_name
    song_dir.mkdir(parents=True, exist_ok=True)
    music = os.path.basename(args.audio)
    try:
        shutil.copy2(args.audio, song_dir / music)
    except Exception:
        pass

    # presentation header: start from an inherited source chart (its banner/background/#BGCHANGES video/etc.),
    # then let explicit CLI flags override. Timing tags (#BPMS/#STOPS) are never inherited — the generator owns
    # the grid (see src/generation/sm_headers.py).
    header = {}
    if args.inherit_from:
        src_sm = (sm_headers.find_sibling_chart(args.audio) if args.inherit_from == "auto"
                  else args.inherit_from)
        if src_sm and os.path.isfile(src_sm):
            header = sm_headers.read_header_tags(src_sm)
            copied = sm_headers.copy_media(header, os.path.dirname(src_sm), song_dir)
            print(f"   inherited header from {os.path.basename(src_sm)}"
                  + (f"; copied media {copied}" if copied else "")
                  + (" (BGCHANGES video will play)" if header.get("BGCHANGES") and copied else ""))
        else:
            print(f"   ⚠️  --inherit_from: no source chart found "
                  + ("next to the audio" if args.inherit_from == "auto" else f"at {args.inherit_from}"))
    # explicit flags win over inherited values (TAG -> arg)
    for tag, val in (("SUBTITLE", args.subtitle), ("GENRE", args.genre), ("CREDIT", args.credit),
                     ("BANNER", args.banner), ("BACKGROUND", args.background), ("CDTITLE", args.cdtitle),
                     ("BGCHANGES", args.bgchanges), ("DISPLAYBPM", args.display_bpm),
                     ("SAMPLESTART", None if args.sample_start is None else f"{args.sample_start:.3f}"),
                     ("SAMPLELENGTH", None if args.sample_length is None else f"{args.sample_length:.3f}")):
        if val is not None:
            header[tag] = val

    sm = charts_to_sm(
        charts=[{"chart": gen, "difficulty_name": "Challenge",
                 "difficulty_value": DIFFICULTY_METER[args.difficulty], "author": "generated"}],
        bpm=bpm, title=title, artist=(args.artist or ""), music=music, offset=0.0, typed=True,
        timesteps_per_beat=subdiv,  # 4 = 16th grid, 12 = the v2 48th grid -> 48 rows/measure so triplets land true
        header=header,
    )
    (song_dir / "chart.sm").write_text(sm, encoding="utf-8")
    gen_d = float((gen != 0).any(1).mean())
    print(f"wrote {song_dir/'chart.sm'}  ({gen_d:.3f} realized density, "
          f"{int((gen[:, :] != 0).any(1).sum())} notes).")
    print(f"   Drop the GROUP folder '{group_dir}' into your StepMania Songs directory "
          f"(it becomes the group; the song '{song_name}' sits inside it).")


if __name__ == "__main__":
    main()
