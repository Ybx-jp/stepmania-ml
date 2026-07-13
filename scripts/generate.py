#!/usr/bin/env python3
"""
Generate a playable StepMania chart from a SINGLE audio file (bring-your-own-song).

Unlike scripts/export_typed_samples.py (a dataset-bound A/B eval
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
  python scripts/generate.py --audio "https://youtu.be/XXXX" --difficulty Hard --bpm 174   # URL -> pulls .ogg

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
from src.data.offset_detect import detect_offset_audio  # audio start-offset detector (beat-anchoring; deaf-chore fix)
from src.data.youtube_audio import is_url, fetch_to_cache  # --audio may be a YouTube URL -> pull to a local .ogg
from src.data.audio_slice import parse_trim_spec, ensure_trimmed  # --trim-audio: keep a time range of the audio
from src.generation.typed import pair_holds
from src.generation.sm_writer import charts_to_sm
from src.generation import sm_headers
from src.generation.playtest_export import enforce_playability
from src.generation.radar_manifold import RadarManifold
from src.generation.decode_defaults import (
    CANONICAL_DECODE, calib_arg_default, parse_phase_calib, grid_snap_offset, UNIVERSAL_ONSET_WINDOW)
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


def _sparse_harm_offset(audio_np, gain, quiet_q, quiet_feat='total'):
    """The sparse-harm-in-quiet ONSET PHRASE CALIBRATOR (ported byte-for-byte from export_typed_samples.py):
    add a per-frame onset-logit boost `gain·quiet_gate·harm_onset` so a sparse harmonic/melodic event (a piano
    line in a lull) fires where the flat governor would bury it. `quiet_gate` -> 1 as the (box-smoothed) gate
    feature drops below the `quiet_q` percentile, 0 above it; dim-36 is the already-0-1 harmonic onset.
    `quiet_feat`: 'total' = dim-0 TOTAL energy (deployed default, byte-identical); 'perc' = dim-35 PERC_ONSET
    ABSENCE — the conditioning-mechanics §6 fix, which fires inside an energy-LOUD but percussion-absent melodic
    solo (a piano lead over a pad) that the total-energy gate MISSES (it once dumped boost onto a loud drum
    section). Rides the SAME onset_logit_offset slot as the grid snap, so tau MUST see it too (it does, below).
    See conditioning-mechanics §6 / notes/phrasing_coherence_findings.md."""
    gcol = 0 if quiet_feat == 'total' else 35                    # dim-0 total energy | dim-35 perc onset (its absence)
    e = audio_np[:, gcol].astype(np.float64)
    e = (e - e.min()) / (np.ptp(e) + 1e-9)                       # norm01
    w = 16; e = np.convolve(np.pad(e, w, mode='edge'), np.ones(2 * w + 1) / (2 * w + 1), mode='valid')  # boxsmooth
    q = np.percentile(e, quiet_q)
    quiet_gate = np.clip((q - e) / (q + 1e-6), 0.0, 1.0)         # 0 above the quiet quantile, ->1 as feature->0
    return (gain * quiet_gate * audio_np[:, 36]).astype(np.float32)   # harm onset (dim36) already 0-1


def build_stub_chart(audio_path: str, bpm: float, duration: float, hop_length: int,
                     subdiv: int = TIMESTEPS_PER_BEAT, offset: float = 0.0) -> StepManiaChart:
    """A minimal chart carrying only what extract_from_chart needs: offset, song_length, hop_length.
    No note data — we are GENERATING the notes, not reading them. `subdiv` = the grid resolution (4 = 16th /
    highres, 12 = 48th / highres_v2): it sets timesteps_total so extract_from_chart re-grids the audio to the
    SAME number of cells the model was trained on.
    `offset` (seconds, ≥0) = the BEAT-ANCHOR skip: extract_from_chart skips this many seconds of audio (its
    `start_sample = int(offset*sr) if offset>0`) so frame 0 lands on the first downbeat — the fix for the "deaf
    choreography" bug (the model chores on within-beat phase; a misaligned grid = phantom-grid placement). The
    effective (post-skip) length drives timesteps_total so the grid + note count stay right (no trailing pad)."""
    eff_duration = max(0.0, duration - max(0.0, offset))
    total_beats = eff_duration * bpm / 60.0
    return StepManiaChart(
        title=Path(audio_path).stem, artist="", audio_file=audio_path,
        bpm=bpm, offset=offset, sample_start=0.0, sample_length=0.0,
        timing_events=[TimingEvent(beat=0.0, value=bpm, event_type="bpm")],
        note_data=[], song_length_seconds=eff_duration,
        timesteps_total=int(total_beats * subdiv), hop_length=hop_length,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate a StepMania chart from one audio file.")
    p.add_argument("--audio", required=True,
                   help="path to an audio file (.ogg/.mp3/.wav) OR a YouTube/yt-dlp URL "
                        "(pulled to a local .ogg; requires yt-dlp+ffmpeg on PATH)")
    p.add_argument("--audio_quality", type=int, default=6, choices=range(0, 11), metavar="0-10",
                   help="Vorbis quality when re-encoding a URL pull or --trim-audio (higher = better; default 6)")
    p.add_argument("--trim-audio", "--trim_audio", dest="trim_audio", default=None, metavar="START[,END]",
                   help="chart only a time range of the audio. Timestamps are SS, M:SS, or H:MM:SS. "
                        "'0:04' = from 4s to the end; '0:04,2:14' = the 4s-2:14 interior. Applies to a "
                        "local file or a URL pull; the beat-anchor offset re-detects on the trimmed clip.")
    p.add_argument("--sm_difficulty", "--sm-difficulty", dest="sm_difficulty", default=None,
                   choices=["Beginner", "Easy", "Medium", "Hard", "Challenge", "Edit"],
                   help="StepMania difficulty SLOT written to the .sm (default: same as --difficulty). Set this to "
                        "place several generated difficulties of ONE song in distinct slots so they don't collide.")
    p.add_argument("--append_to", "--append-to", dest="append_to", default=None, metavar="CHART.sm",
                   help="append this difficulty into an EXISTING song's .sm as another difficulty slot (instead of "
                        "writing a new song folder). The target's #OFFSET/#BPMS/title/audio are kept; use the SAME "
                        "--bpm and --offset the target's charts were generated with so the beat grids match. Backs "
                        "the target up to <name>.sm.bak first.")
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
                   help="generator weights (default: auto — the v2 48th-grid checkpoint gen_motif_v2_48th_cont for "
                        "the default --features highres_v2, or the legacy gen_motif_full_fixed for --features highres). "
                        "Pass a path to override.")
    p.add_argument("--features", choices=["highres", "highres_v2"], default="highres_v2",
                   help="feature/grid space. highres_v2 (DEFAULT, the v1.0 canonical model) = the 42 channels on the "
                        "data-layer-v2 48th grid (timesteps_per_beat=12, beat-sync) — resolves triplets, removes the "
                        "16th-grid triplet tax; auto-selects the v2 checkpoint (gen_motif_v2_48th_cont) + 48th-grid "
                        ".sm writer + v2 context length. highres = the legacy 42-dim 16th grid (gen_motif_full_fixed) "
                        "— pair ONLY with a v1 checkpoint.")
    p.add_argument("--bpm", type=float, default=None,
                   help="song BPM. ★ STRONGLY RECOMMENDED for correct alignment — it grids the whole chart. Default "
                        "= auto-estimate (librosa), which is UNRELIABLE (octave / 2:3 metric errors, esp. fast songs) "
                        "and will mis-align the chart if wrong. Pass the real BPM whenever you know it.")
    p.add_argument("--offset", type=float, default=None,
                   help="song start #OFFSET in seconds (StepMania convention: negative = lead-in before beat 0, e.g. "
                        "-0.281). Default = AUTO-DETECT from audio (beat-anchors frame 0 to the first downbeat — the "
                        "fix for 'deaf choreography'). Pass your reference chart's #OFFSET to override the detector "
                        "(recommended when it flags low confidence — ~20%% of songs slip a half-beat).")
    p.add_argument("--no_auto_offset", action="store_true",
                   help="disable beat-anchoring entirely: grid from audio t=0 with #OFFSET 0 (the pre-fix behavior). "
                        "Use only to reproduce an old chart or if the detector misbehaves and you have no reference.")
    p.add_argument("--style", action="append", default=None, metavar="DIM=VAL",
                   help="optional groove feel, e.g. 'chaos=q0.7'. Multidimensional: comma-separate "
                        "('chaos=high,freeze=low') OR repeat the flag ('--style chaos=high --style freeze=low'); "
                        "both merge into one manifold spec.")
    p.add_argument("--guidance", type=float, default=1.5, help="CFG scale for --style (default 1.5)")
    p.add_argument("--target_density", type=float, default=None,
                   help="override the density target (notes/frame on the ACTIVE grid; the top-priority density lever, "
                        "conditioning-mechanics §6). Default = the manifold E[density|difficulty] — which is the "
                        "TRAINING-CORPUS AVERAGE for the bucket and can read SPARSE vs your own charts (corpus 'Hard' "
                        "≈ 2.7 notes/s @128bpm; a real meter-10 chart ≈ 3.5-5). Pass e.g. 0.14 (v2/48th grid) for a "
                        "denser Hard. NOTE: this is the final active-grid fraction — it BYPASSES the manifold + the "
                        "4/subdiv scaling, so specify it directly for the grid you're on (v2 = 48th).")
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
    p.add_argument("--stamina_breathe_local_win", type=int, default=None,
                   help="EXPERIMENTAL (long-song fix, notes/stamina_longsong_findings.md): z-normalize the breathe "
                        "envelope over a ROLLING window of this many FRAMES instead of the whole song. None (default) "
                        "= deployed whole-song z. ~3600 (48th grid) ≈ one training-song span; fixes the length-mis-"
                        "scoped arc on >130s songs.")
    p.add_argument("--onset_tail_hangover", type=str, default="auto",
                   help="tail-collapse fix (notes/playtest_log.md 2026-07-11/12): pad the audio past song-end by this "
                        "many FRAMES (SILENCE) so the onset head's FINAL window CENTERS on the true end instead of "
                        "leaving it at the under-trained trailing edge (fixes the tail quarter-backbone collapse on "
                        "songs longer than the onset window). 'auto' (DEFAULT) = W//2. 0/off = disabled. No-op if the "
                        "song fits the onset window (byte-identical).")
    p.add_argument("--onset_window", type=str, default="auto",
                   help="UNIVERSAL SUB-TRAIN-LENGTH WINDOW (notes/universal_window_findings.md): tile the ONSET head "
                        "over local-PE windows of this many FRAMES for EVERY song longer than it. 'auto' (default) = the "
                        "trained window (v2 V2_MSL=5400 / v1 the checkpoint PE size) = current behavior, only fires past "
                        "the trained context. A SMALLER value (e.g. 3600 = v2 p75 train length) also fixes SHORT-song "
                        "END-degeneration: the v2 abs-PE tail is under-trained past ~3500 (train len median 3120/max 5128) "
                        "so a song ending there fires ~30%% of its real tail notes; ~3600 restores tail recall + backbone "
                        "to human levels (probe_universal_window.py; by-ear gate pending). A song shorter than W = no-op "
                        "(byte-identical). Single-sourced into BOTH tau and generate() so they can't drift.")
    p.add_argument("--pattern_temperature", type=float, default=CANONICAL_DECODE["pattern_temperature"],
                   help="footwork sampling temperature (canonical 1.0 — real jack/jump balance; NOT 0.7)")
    p.add_argument("--type_temperature", type=float, default=CANONICAL_DECODE["type_temperature"],
                   help="per-panel tap/hold/roll sampling temperature (canonical 0.4 — surfaces holds at rate)")
    p.add_argument("--onset_phase_calib", type=str, default=calib_arg_default(),
                   help="★ the 16th-UNLOCK 'b8,b16' (logit space): un-buries 16th-offbeats so they float with "
                        "the audio per song. Canonical '0.0,1.0' (a KNEE: ~0.5 calm song .. 2.0 dense). '' = off. "
                        "A 3rd element (b_trip) is the v2 triplet band; with --auto_b_trip it applies per song.")
    p.add_argument("--harm_calib", type=float, default=0.0,
                   help="OPT-IN sparse-harm-in-quiet phrase calibrator (default 0 = off): boost the onset logit by "
                        "gain·quiet_gate·harm_onset so a sparse melodic event in a lull (a piano line) fires where "
                        "the flat governor buries it. ~10 to start. Rides the same slot as the grid snap; baked into "
                        "tau. Needs the 42-dim highres/highres_v2 features (perc/harm channels).")
    p.add_argument("--harm_quiet_q", type=float, default=40.0,
                   help="percentile of (smoothed) energy below which --harm_calib's quiet gate opens (default 40).")
    p.add_argument("--harm_quiet_feat", choices=["total", "perc"], default="total",
                   help="which feature --harm_calib's quiet gate keys on: 'total' (dim-0 total energy, deployed) or "
                        "'perc' (dim-35 perc-onset ABSENCE — cond-mech §6 fix: fires in an energy-LOUD but "
                        "percussion-absent melodic solo the total gate misses).")
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
    p.add_argument("--max_len", type=int, default=None,
                   help="optional cap on generated frames. DEFAULT = None = chart the WHOLE song (the positional "
                        "encoding is extended past the trained context; the model extrapolates coherently, validated "
                        "on v2). Set a value to deliberately truncate; a hard safety ceiling still applies.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _first_measure_rows(block: str) -> int:
    """Note rows in a #NOTES block's first measure = the grid resolution (e.g. 48 on the v2 grid, 16 on v1)."""
    n, started = 0, False
    for ln in block.splitlines():
        s = ln.strip()
        if re.fullmatch(r"[0-9MLF]{4}", s):
            n, started = n + 1, True
        elif s == "," and started:   # measures are comma-separated in the .sm
            break
    return n


def append_chart_to_sm(target: Path, new_block: str, gen_bpm: float, slot: str) -> None:
    """Splice a generated #NOTES block into an existing song's .sm as another difficulty, in place.

    The target's header (title, audio, #BPMS, and its possibly hand-tuned #OFFSET) is kept, so both
    difficulties share one grid + one offset. Two grid guards refuse an append that can't line up:
    a bpm mismatch (different beat spacing) or a rows-per-measure mismatch (different subdivision).
    """
    if not target.is_file():
        raise SystemExit(f"--append_to: target .sm not found: {target}")
    text = target.read_text(encoding="utf-8")
    if "#NOTES:" not in text:
        raise SystemExit(f"--append_to: {target} has no #NOTES block to append beside.")
    tgt_block = text[text.index("#NOTES:"):]
    m = re.search(r"#BPMS:\s*[0-9.]+\s*=\s*([0-9.]+)", text)          # guard 1: same tempo
    if m and abs(float(m.group(1)) - gen_bpm) > 0.05:
        raise SystemExit(f"--append_to: target bpm {float(m.group(1)):g} != --bpm {gen_bpm:g}; grids differ — "
                         "regenerate with a matching --bpm.")
    tgt_rpm, new_rpm = _first_measure_rows(tgt_block), _first_measure_rows(new_block)
    if tgt_rpm and new_rpm and tgt_rpm != new_rpm:                    # guard 2: same subdivision
        raise SystemExit(f"--append_to: target grid is {tgt_rpm} rows/measure but this chart is {new_rpm} — "
                         "different --features subdivision; regenerate to match.")
    if re.search(rf"(?m)^\s*{re.escape(slot)}:\s*$", tgt_block):
        print(f"⚠️  --append_to: target already has a '{slot}' difficulty — you'll end up with two. "
              "Pass a different --sm_difficulty if that's not intended.")
    off = re.search(r"#OFFSET:\s*(-?[0-9.]+)", text)
    print(f"ℹ️  --append_to: keeping the target's #OFFSET {off.group(1) if off else '?'} for BOTH difficulties — "
          "this chart must have been generated on the SAME --offset (beat-anchor) as the target's, or it will be "
          "off-grid against them.")
    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    target.write_text(text.rstrip() + "\n\n" + new_block.rstrip() + "\n", encoding="utf-8")
    print(f"appended '{slot}' difficulty → {target}  (backup: {backup})")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --audio may be a YouTube (or any yt-dlp) URL. Pull it to a local Vorbis .ogg (cached by
    # video id, so re-running on the same URL at another difficulty doesn't re-download) and
    # chart the local file — everything downstream sees a plain path. Title defaults to the
    # video title when --title wasn't given. BPM is still user's job (URLs carry no reliable BPM).
    if is_url(args.audio):
        try:
            audio_path, video_title = fetch_to_cache(args.audio, quality=args.audio_quality)
        except RuntimeError as e:
            raise SystemExit(f"could not fetch audio from URL: {e}")
        print(f"↓ pulled audio → {audio_path}")
        args.audio = str(audio_path)
        if args.title is None:
            args.title = video_title

    # --trim-audio: keep only a time range of the (local) audio, producing a new .ogg that BECOMES
    # args.audio here — so everything below (bpm, duration, offset detection, feature extraction,
    # generation) runs on the trimmed clip; the model never sees the untrimmed audio. Runs after the
    # URL pull so it slices whichever file we ended up with. Because offset detection therefore sees
    # the trimmed clip, cutting the intro just re-anchors #OFFSET to the clip's first beat.
    if args.trim_audio:
        try:
            trim_start, trim_end = parse_trim_spec(args.trim_audio)
            trimmed = ensure_trimmed(args.audio, trim_start, trim_end, quality=args.audio_quality)
        except (ValueError, RuntimeError) as e:
            raise SystemExit(f"--trim-audio: {e}")
        span = f"{trim_start:g}s..{'end' if trim_end is None else f'{trim_end:g}s'}"
        print(f"✂  trimmed audio to [{span}] → {trimmed}")
        args.audio = str(trimmed)

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

    # 1. BPM -> the 16th-note-aligned hop the model was trained on.
    # ⚠️ BPM is the load-bearing alignment lever: the audio is gridded at BPM·subdiv/60 and the model's metric-phase
    # channel assumes each frame IS that beat-fraction, so a WRONG bpm mis-grids the whole chart (notes drift off the
    # beat + OOD phase features). Auto-estimation (librosa.beat.tempo) is UNRELIABLE — it octave/2:3-errs, especially
    # on fast tracks (verified: ~40% of a personal set mis-estimated, fast hardcore under-reported by 2:3). So --bpm
    # is STRONGLY RECOMMENDED for correctness; the estimate is a convenience only, with a loud warning below.
    bpm = args.bpm if args.bpm is not None else estimate_bpm(args.audio)
    if args.bpm is None:
        print(f"⚠️  no --bpm given → ESTIMATED {bpm:.1f} BPM. Tempo estimation is UNRELIABLE (octave / 2:3 metric "
              f"errors, esp. on fast songs) and a wrong BPM mis-aligns the WHOLE chart. Verify this looks right; "
              f"pass --bpm <song BPM> for a correctly aligned chart.")
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

    # ★ BEAT-ANCHOR (the "deaf choreography" fix, notes/byo_offset_detection_findings.md): the model chores on
    # within-beat phase, so frame 0 MUST sit on a downbeat or it choreographs to a phantom grid. Resolve the
    # within-beat start phase and skip it during extraction (positive stub offset = extract_from_chart's
    # skip-to-beat), then write #OFFSET = -phase so the untrimmed copied audio still plays in time (§ below).
    beat_period = 60.0 / bpm
    if args.no_auto_offset:                              # escape hatch: pre-fix behavior (grid from t=0)
        anchor = 0.0
    elif args.offset is not None:                        # manual #OFFSET (negative lead-in) -> within-beat phase
        anchor = (-args.offset) % beat_period
        print(f"offset: using --offset {args.offset:+.3f}s → beat-anchor skip {anchor:.3f}s (#OFFSET {-anchor:+.3f})")
    else:                                                # audio detector (full-band onset pulse-train, ~7ms median)
        r = detect_offset_audio(args.audio, bpm)
        anchor = r["offset_sec"] if r else 0.0
        if r is None:
            print("⚠️  offset detector failed (silent/short audio) → grid from t=0. Pass --offset from a reference chart.")
        elif r["is_confident"]:
            print(f"offset: auto-detected {anchor:.3f}s (#OFFSET {-anchor:+.3f}), confidence {r['confidence']:.2f}")
        else:
            print(f"⚠️  offset auto-detected {anchor:.3f}s but LOW confidence ({r['confidence']:.2f}) — likely a "
                  f"half-beat slip (beat vs off-beat). Verify by ear; pass --offset <reference #OFFSET> if it's off.")
    stub = build_stub_chart(args.audio, bpm, duration, hop, subdiv=subdiv, offset=anchor)
    feats = fspec.extractor.extract_from_chart(args.audio, stub)
    if feats is None:
        raise SystemExit(f"feature extraction failed for {args.audio}")
    audio_tensor = feats.get_aligned_features()  # (T, 42)
    if np.any(~np.isfinite(audio_tensor)):
        raise SystemExit("non-finite audio features — bad/corrupt audio?")
    # 3. model — build at the CHECKPOINT's PE size (v2 = V2_CTX 5504; v1 = the 2048 default) so the stored pe buffer
    # loads without a size mismatch, THEN extend the PE below to cover the whole song.
    v2_arch = dict(MODEL_ARCH, max_len=V2_CTX) if is_v2 else None
    model = load_generator(ckpt, fspec.audio_dim, device, arch=v2_arch)  # builds + loads (strict=False) + .eval()

    # length: the PE was trained up to a fixed window (V2_MSL=5400 on v2), but the model EXTRAPOLATES past it
    # gracefully — validated on v2 at ~2x context: panel usage holds (~full 4-arrow entropy), density only thins
    # ~mildly late (partly the stamina governor). That's far better than TRUNCATING the rest of the song to silence
    # (the old behavior clipped ~23/34 real songs, some to <half). So we EXTEND the sinusoid PE to fit the whole song
    # (a fresh buffer, byte-identical over the trained range) instead of capping. --max_len (default None = whole
    # song) is an optional user truncation cap; a hard SAFETY_CAP guards against OOM / extreme extrapolation.
    needed = audio_tensor.shape[0]
    SAFETY_CAP = 24000  # ~13 min @128bpm on the 48th grid; beyond this, truncate rather than extrapolate wildly / OOM
    cap = SAFETY_CAP if args.max_len is None else min(args.max_len, SAFETY_CAP)
    T = min(needed, cap)
    trained_ctx = int(model.pos_encoding.pe.size(1))  # the checkpoint's PE length == the trained window
    # ONSET SLIDING WINDOW (notes/byo_sliding_window_findings.md): the non-causal ONSET encoder does NOT
    # extrapolate gracefully past the trained window — abs-PE COMPRESSES the tail onset-probability peaks, and
    # because density is a GLOBAL quantile (tau), those flattened peaks fall below tau so almost nothing fires
    # ("dead tail": Toulouse tail onsets 44 abs-PE vs 127 windowed). So we run the onset head over IN-DISTRIBUTION
    # local-PE windows of `onset_window` frames (single-sourced into BOTH tau and generate() so they can't drift).
    # The DECODER (choreography) DOES extrapolate gracefully (panel entropy ~1.0 at 2x context) and can't change
    # the onset count, so it keeps the extended absolute PE below — the two fixes compose. onset_window 'auto' = the
    # trained window (v2 = V2_MSL 5400; v1 = the checkpoint PE size); a song that fits is a no-op (byte-identical).
    # A SMALLER --onset_window (e.g. 3600) also fixes SHORT-song end-degeneration (the abs-PE tail is under-trained
    # past ~3500 even below the trained window; notes/universal_window_findings.md) — by-ear gate pending.
    _ow = str(args.onset_window).strip().lower()
    # 'auto' (default): v2 = the by-ear-validated UNIVERSAL window (3600, fires on short songs too); v1 = the trained
    # context (current behavior, the analysis is v2-specific). A positive number overrides on either grid; 0/off/none
    # DISABLES the universal window and reverts to the trained-window behavior (only tiles past the trained context).
    if _ow in ("auto", ""):
        onset_window = UNIVERSAL_ONSET_WINDOW if is_v2 else trained_ctx
    elif _ow in ("off", "none", "0"):
        onset_window = V2_MSL if is_v2 else trained_ctx          # disable universal window, keep long-song safety
    else:
        _v = int(float(_ow))
        onset_window = _v if _v > 0 else (V2_MSL if is_v2 else trained_ctx)
    # TAIL HANGOVER (the long-song backbone-collapse fix, notes/playtest_log.md 2026-07-11): the onset head's FINAL
    # window puts the song-end at its under-trained trailing edge -> the tail quarter-backbone collapses. Reflect-pad
    # the audio memory by `onset_tail_hangover` frames so a full window can CENTER on the true end. 'auto' = W//2 (the
    # minimum for a full window to place the end at its center); 0/off = disabled. Single-sourced into BOTH tau and
    # generate() (like onset_window) so they can't drift. No-op unless the song is longer than the onset window.
    _hv = str(args.onset_tail_hangover).strip().lower()
    onset_tail_hangover = (onset_window // 2) if _hv == "auto" else max(0, int(float(_hv)))
    if T > trained_ctx:
        from src.generation.transformer import PositionalEncoding
        model.pos_encoding = PositionalEncoding(model.pos_encoding.pe.shape[-1], max_len=T + 128).to(device)
        print(f"song is {needed} frames (~{needed * hop / SR:.0f}s) > trained context {trained_ctx}: onset head "
              f"uses a sliding window (in-distribution {onset_window}-frame windows so the tail fires normally); "
              f"the choreography decoder extrapolates its positional encoding {trained_ctx} → {T + 128} "
              f"(entropy-validated graceful extrapolation) to chart the WHOLE song.")
    if needed > T:
        print(f"⚠️  song ({needed} frames) exceeds the {'--max_len' if args.max_len else 'safety'} cap ({cap}); "
              f"charting only the first ~{T * hop / SR:.0f}s.")
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
    if subdiv != 4:  # the manifold is fit on the 16th grid (density = frac of 16th-frames); on the finer v2
        gen_density *= 4.0 / subdiv  # 48th grid the SAME notes/beat is a smaller frame-fraction, so scale by
        # 4/subdiv (=1/3 on the 48th grid) or the whole chart over-places ~subdiv/4x. Mirrors the exporter's
        # style_density fix (export_typed_samples.py); without a source chart this manifold value is the ONLY
        # density source here, so the bug hit every v2 BYO chart uniformly (notes/byo_audio_alignment_findings.md).
    density_src = "manifold E[density|diff]"
    if args.target_density is not None:  # explicit override: the top-priority density lever (conditioning-mechanics
        gen_density = args.target_density  # §6). Final ACTIVE-grid fraction — bypasses the manifold + 4/subdiv above.
        density_src = "--target_density"   # the manifold 'Hard' is corpus-average + can read sparse vs the user's charts
    print(f"BPM {bpm:.1f} | hop {hop} | {T} frames (~{T*hop/SR:.0f}s) | {args.difficulty} | "
          f"target density {gen_density:.3f} ({density_src}; ~{gen_density*bpm*subdiv/60:.1f} notes/s)"
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
    # per-frame onset-logit offsets that RIDE THE SAME SLOT and are single-sourced into BOTH tau (extra_offset=)
    # and decode (onset_logit_offset=) so they can't drift (conditioning-mechanics §6):
    #   (a) 16th-GRID SNAP (v2 only): veto the pure-48th cells so busy low/mid charts stay on-grid. auto =
    #       difficulty <= Medium; v1 (subdiv=4) = no-op.
    #   (b) OPT-IN sparse-harm-in-quiet phrase calibrator (--harm_calib): boost a sparse melodic event in a lull.
    onset_off = None
    do_snap = subdiv != 4 and (args.grid_snap == 'all' or (args.grid_snap == 'auto' and diff_idx <= 2))
    if do_snap:
        onset_off = grid_snap_offset(T, subdiv, keep_triplets=args.grid_snap_keep_triplets, device=device)
    if args.harm_calib > 0:
        if audio_tensor.shape[1] != 42:
            raise SystemExit("--harm_calib needs the 42-dim highres/highres_v2 features (perc/harm channels).")
        harm_t = torch.from_numpy(
            _sparse_harm_offset(audio_tensor[:T], args.harm_calib, args.harm_quiet_q, args.harm_quiet_feat)).to(device)
        onset_off = harm_t if onset_off is None else onset_off + harm_t
    # the radar fed to BOTH tau and the decode MUST be the same one, else tau is calibrated on a different
    # distribution than generate() decodes from (conditioning-mechanics §3). No --style -> radar=None (null token).
    radar_arg = radar_for_gen if style_spec else None

    # 5. tau via the shared decode harness (conditioned + guided + phase-calibrated + snap offset, as generate() decodes)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        p_onset = conditioned_p_onset(model, memory, diff, radar=radar_arg, guidance=args.guidance,
                                      phase_calib=song_calib, extra_offset=onset_off, subdiv=subdiv,
                                      window=onset_window,           # SAME window generate() decodes from (tau-coupling §3)
                                      tail_hangover=onset_tail_hangover)  # SAME hangover generate() uses (tau-coupling §3)
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
        onset_window=onset_window,  # ONSET sliding window for long songs (fixes the dead tail; no-op if song fits)
        onset_tail_hangover=onset_tail_hangover,  # reflect-pad past song-end so the final window centers on it (§ tail-collapse fix)
        onset_logit_offset=onset_off,      # 16th-grid snap + --harm_calib (same offset baked into tau above; None if neither)
        fatigue_penalty=(args.fatigue_penalty if args.fatigue_penalty and args.fatigue_penalty > 0 else None),
        fatigue_free=args.fatigue_free,
        stamina_ceiling=(args.stamina_ceiling if args.stamina_ceiling and args.stamina_ceiling > 0 else None),
        stamina_tau=CANONICAL_DECODE["stamina_tau"], stamina_scale=CANONICAL_DECODE["stamina_scale"],
        stamina_breathe=args.stamina_breathe, stamina_breathe_local_win=args.stamina_breathe_local_win,
        hold_stream_penalty=CANONICAL_DECODE["hold_stream_penalty"],  # suppress holds in dense streams (2026-07-02)
        hold_stream_floor=CANONICAL_DECODE["hold_stream_floor"], hold_stream_win=CANONICAL_DECODE["hold_stream_win"],
        footswitch=CANONICAL_DECODE["footswitch"],  # DEFAULT False: force one-foot jacks, model alternates (2026-07-02)
        hold_release_run=CANONICAL_DECODE["hold_release_run"],  # DEFECT-#3 free-foot-under-hold force-close (2026-07-12, §5c)
        hold_release_gap=CANONICAL_DECODE["hold_release_gap"], hold_max_beats=CANONICAL_DECODE["hold_max_beats"],
        bpm=bpm, radar=radar_arg,  # SAME radar tau was computed from (conditioning-mechanics §3)
        style=None, guidance_scale=(args.guidance if style_spec else 1.0),
    )
    enforce_playability(gen_kwargs, False)  # forces hold_aware / no_jump_during_hold / no_cross_during_hold
    with torch.no_grad():
        gen = pair_holds(model.generate(audio, diff, lengths=torch.tensor([T], device=device),
                                        **gen_kwargs)[0].cpu().numpy())

    # 7a. --append_to: splice this difficulty into an existing song's .sm instead of writing a new song folder.
    #     The target owns #OFFSET/#BPMS/title/audio; we contribute one #NOTES block (grid guards in the helper).
    if args.append_to:
        sm_difficulty = args.sm_difficulty or args.difficulty
        one = charts_to_sm(
            charts=[{"chart": gen, "difficulty_name": sm_difficulty,
                     "difficulty_value": DIFFICULTY_METER[args.difficulty], "author": "generated"}],
            bpm=bpm, title="", artist="", music="", offset=-anchor, typed=True,
            timesteps_per_beat=subdiv, header={})
        new_block = one[one.index("#NOTES:"):]
        append_chart_to_sm(Path(args.append_to), new_block, bpm, sm_difficulty)
        print(f"   ({float((gen != 0).any(1).mean()):.3f} realized density, "
              f"{int((gen != 0).any(1).sum())} notes)")
        return

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

    sm_difficulty = args.sm_difficulty or args.difficulty  # the .sm difficulty SLOT (defaults to --difficulty)
    sm = charts_to_sm(
        charts=[{"chart": gen, "difficulty_name": sm_difficulty,
                 "difficulty_value": DIFFICULTY_METER[args.difficulty], "author": "generated"}],
        bpm=bpm, title=title, artist=(args.artist or ""), music=music, offset=-anchor, typed=True,  # #OFFSET = -(beat-anchor); untrimmed audio plays in time
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
