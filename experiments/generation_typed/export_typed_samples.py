#!/usr/bin/env python3
"""
Export playable .sm song folders from the typed hold-aware generator.

Loads the style checkpoint and, for each of N val songs, generates a FULL-LENGTH
typed chart (taps + holds, hold-aware decoding, KV-cached) conditioned on the real
audio + difficulty, then writes a StepMania song folder: the original audio + a .sm
holding the generated chart (as "Challenge") and the original chart (for A/B), both
with hold/tail symbols. Drop a folder into StepMania and play it.

Step 3 (style): pass --reference <some_chart.sm> to generate every song IN THE STYLE OF
that chart, and --guidance 2-3 to amplify the effect (classifier-free guidance).

Usage:
    # plain (audio + difficulty only)
    python experiments/generation_typed/export_typed_samples.py \
        --data_dir data/ --audio_dir data/ --out_dir outputs/typed_samples --num_songs 8

    # in the style of a reference chart, amplified
    python experiments/generation_typed/export_typed_samples.py \
        --data_dir data/ --audio_dir data/ --out_dir outputs/style_samples --num_songs 8 \
        --reference "data/.../SomeSong/SomeSong.sm" --guidance 2.5
"""

import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, re, shutil, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.generation.typed_model import LayeredTypedChartGenerator, MOTIF_DIM
from src.generation.motif_codebook import FIGURE_CLASSES
from src.generation.typed import symbol_histogram, pair_holds
from src.generation.sm_writer import charts_to_sm
from src.generation.playtest_export import enforce_playability
from src.generation.evaluation import DifficultyCritic

DEFAULT_BPM = 150.0

# Sparse-harm-in-quiet phrase calibrator (Step 1 mechanism, notes/phrasing_coherence_findings.md). MUST match
# probe_phrasing_coherence.py:sparse_harm_offset exactly (boxsmooth win 16, norm01, highres dims energy=0 harm=36).
def _sparse_harm_offset(audio_np, gain, quiet_q):
    e = audio_np[:, 0].astype(np.float64)
    e = (e - e.min()) / (np.ptp(e) + 1e-9)                       # norm01
    w = 16; e = np.convolve(np.pad(e, w, mode='edge'), np.ones(2 * w + 1) / (2 * w + 1), mode='valid')  # boxsmooth
    q = np.percentile(e, quiet_q)
    quiet_gate = np.clip((q - e) / (q + 1e-6), 0.0, 1.0)         # 0 above the quiet quantile, ->1 as energy->0
    return (gain * quiet_gate * audio_np[:, 36]).astype(np.float32)   # harm onset (dim36) already 0-1


# ============================================================================================
# THE KNOB MAP — how all these args fit together (read before adding/trusting a flag)
# --------------------------------------------------------------------------------------------
# This script grew a LOT of knobs across experiments. They are NOT a flat bag — they are one
# decode PIPELINE with a lever at each stage. Read in decode order; STATUS tags tell you what to
# actually use. Authoritative mechanism: the `conditioning-mechanics` skill (§ refs below).
#
#   STATUS legend:  [LIVE] use it  ·  [DEFAULT] on unless you change it  ·  [OPT-IN] off until you pass it
#                   [DEPRECATED] superseded, kept only for legacy/ablation  ·  [TRAP] off-manifold/guarded
#                   [NICHE] narrow special-case
#
# 0. SELECT songs ....... --num_songs --groove_select --difficulty_select --song_filter --seed
#
# 1. CONDITION (one groove profile, fed to the onset+pattern heads). FOUR routes write the SAME
#    slot and OVERRIDE each other — pick EXACTLY ONE (precedence high→low; cond-mechanics §1-§3):
#       --reference / --reference_self  StyleEncoder latent from a full chart        [LIVE/NICHE]
#       --style "stream=high,..."        manifold partial-spec (the CORRECT radar)    [LIVE]
#       --match_radar                    the song's OWN 5-dim radar via the manifold  [LIVE]
#       --radar "chaos=0.9"              mean-pin — OFF-MANIFOLD SMEAR, errors w/o --radar_ood  [TRAP]
#    --guidance amplifies whichever is set (CFG; 1=off, 1.5-2.5 musical, >3 dissolves backbone). [LIVE]
#
# 2. ONSET / DENSITY (which frames fire). cond-mechanics §6.
#    density target priority:  --target_density  >  manifold E[density|style]  >  source chart.
#       --onset_phase_calib "b8,b16"  un-buries off-beats so 16ths float w/ audio  [LIVE, DEFAULT-ON "0,1.0"] the 16th lever ("knee not node", song-dep ~0.5-2.0)
#       --onset_phase_alloc "q,8,16"  flat per-phase QUOTA — SMEARS (exp-design Rule 13)   [DEPRECATED]
#       --onset_phase_penalty         downbeat gate; does NOT rescue chaos                 [NICHE]
#    (NOTE: onset is DECOUPLED from motif/figure by design — cond-mechanics §1. Don't re-couple.)
#
# 3. PATTERN (which panels): --pattern_temperature (1.0≈real) --repetition_penalty --jump_bias
#    --prefer --no_crossovers · H15 figure knobs: --motif (continuous) --figure (discrete, soft/capped)
#
# 4. TYPE (tap/hold/roll): --type_temperature
#
# 5. GOVERNORS (decode-time biomechanics; ALL need bpm [auto from chart] + fatigue on). cond-mech §8.
#    per-NOTE footwork:  --fatigue_penalty  the two-foot model, RELEASE default 2  [DEFAULT]
#                        --jack_penalty     OLD single-foot, superseded by fatigue  [DEPRECATED]
#                        --fatigue_free --max_jack_run (hard cap 2)
#    per-REGION density:  --stamina_ceiling  Stage-2 relief, DEFAULT 50              [DEFAULT, <=0=off]
#                         --stamina_breathe  Stage-3 difficulty ARC, DEFAULT 1.2     [DEFAULT, inert w/o ceiling]
#                         --stamina_tau --stamina_scale
#    NOT exposed here (use generate() validated defaults): stamina_breathe_floor=0.4 (the outro-collapse
#    fix), stamina_max_bump=0.45, stamina_breathe_win=96. Expose them only if you need to retune.
#    NOTE: this exporter's BARE DEFAULT IS THE ONE CANONICAL CONFIG (the `generation-defaults` skill) — model
#    gen_motif_full_fixed (42-dim highres) + FULL governor (fatigue 2 + stamina@50 + breathe 1.2) + pattern_temp
#    1.0 + onset_phase_calib "0,1.0" (the 16th-unlock). Run it with NO flags to reproduce what the user plays.
#    The shipped generate()/scripts/generate.py release center is SEPARATE and conservative (fatigue-only,
#    pattern_temp 0.7 = stale) — it is NOT the reference for matching playtests; mirror THIS exporter.
#
# 6. PLAYABILITY (MANDATORY — enforce_playability FORCES hold_aware + no_jump/cross_during_hold ON
#    regardless of the flags). --override_playability REASON to deviate (needs explicit approval).
#
# 7. OUTPUT: --out_dir --install --songs_dir --max_len --checkpoint --features
#
# NOTE on --fatigue_free: defaults 6.0 here vs 12.0 in generate(); BOTH are inside the vouched 6-12 range
#   (governor_release_region.md) — 12 = design-note default (barrier set high), 6 = the more-gating end this
#   exporter has always playtested. Not a bug, just two valid points in the band.
# ============================================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--out_dir', default='outputs/typed_samples')
    p.add_argument('--checkpoint', default='checkpoints/gen_motif_full_fixed/best_val.pt',
                   help='DEFAULT = the deployed model (42-dim H19 highres retrain; radar+motif+figure). '
                        'Legacy gen_style/gen_stage1 are 23/41-dim — pair them with --features base/stage1.')
    p.add_argument('--features', choices=['base', 'stage1', 'highres'], default='highres',
                   help='DEFAULT highres=42-dim (cache/samples_v3, what gen_motif_full_fixed expects). '
                        'stage1=41-dim (cache/samples_v2, legacy gen_stage1); base=23-dim (cache/samples, gen_style).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_songs', type=int, default=8)
    p.add_argument('--reference', type=str, default=None,
                   help='path to a reference .sm/.ssc chart: generate every song IN THE STYLE OF this chart (Step 3)')
    p.add_argument('--reference_difficulty', type=str, default=None,
                   help='which difficulty of the reference chart to use (name, e.g. Hard); default = hardest available')
    p.add_argument('--guidance', type=float, default=1.0,
                   help='classifier-free guidance scale; >1 amplifies the reference style (2-3 is a good range)')
    p.add_argument('--type_temperature', type=float, default=0.4)
    p.add_argument('--pattern_temperature', type=float, default=1.0,  # sample patterns for variety (greedy->Left/jacks)
                   help='which-panels sampling temperature; 1.0 matches real panel balance & jack rate')
    p.add_argument('--repetition_penalty', type=float, default=1.0,
                   help='>1 further discourages repeating the previous note; 1.0 already matches real')
    p.add_argument('--jump_bias', type=float, default=0.0, help='pattern preference: + = more jumps, - = fewer')
    p.add_argument('--no_crossovers', action='store_true', help='forbid crossover steps (foot automaton)')
    p.add_argument('--no_jump_during_hold', action='store_true',
                   help='forbid jumps while a hold is open (one free foot); pad-playable holds')
    p.add_argument('--override_playability', default=None, metavar='REASON',
                   help='DELIBERATELY deviate from the MANDATORY pad-playability constraints (hold_aware, '
                        'no_jump_during_hold, no_cross_during_hold). Requires EXPLICIT user approval; pass the '
                        'reason. Without this, those constraints are FORCED ON regardless of the flags below.')
    p.add_argument('--no_cross_during_hold', action='store_true',
                   help='forbid the free foot fast-crossing panels while a hold is open (the B4U one-foot '
                        'jacks-during-hold awkwardness; brings hold_burst ~6.9%%->4.7%% vs real 4.0%%)')
    p.add_argument('--onset_phase_alloc', type=str, default=None,
                   help='[DEPRECATED — SMEARS, prefer --onset_phase_calib] phase-aware onset threshold: target note shares "quarter,8th,16th" (real ~"0.707,0.252,0.041"). '
                        'Redistributes the density budget across phases so the model\'s own 16th confidence wins 16th '
                        'slots instead of losing to 8ths (which a single threshold buries). None = single threshold.')
    p.add_argument('--onset_phase_calib', type=str, default='0,1.0',
                   help='[LIVE — the validated 16th-unlock lever; NOW DEFAULT-ON] per-phase calibration offset '
                        '"b8,b16" (logit space) for VARIABLE per-song chaos: corrects the model\'s 16th '
                        'under-confidence so the 16th count floats with the audio (chaotic songs get many, calm '
                        'songs ~none). DEFAULT "0,1.0" = the unlock16_b10 by-ear sweet spot; it is "a KNEE not a '
                        'node" so the right b16 is song-dependent (~0.5 calm -> 1.0 -> 2.0 near-all-16ths on dense '
                        'songs). Preferred over the flat --onset_phase_alloc quota. "0,0" or "" = single threshold (off).')
    p.add_argument('--target_density', type=float, default=None,
                   help='override per-chart density (notes/frame) for the onset threshold; default = match the '
                        'source chart. Use to couple density to chaos (real high-chaos charts run ~0.34 vs ~0.22 '
                        'baseline) so raising chaos ADDS off-beats instead of replacing the quarter backbone.')
    p.add_argument('--onset_phase_penalty', type=float, default=0.0,
                   help='[NICHE] metric gate: off-beat onsets need higher confidence (on-beat 0, 8th -p, 16th -2p). '
                        '~0.5-1.5 restores the downbeat under chaos conditioning. 0 = off.')
    p.add_argument('--max_jack_run', type=int, default=2,
                   help='HARD 16th-jack cap: max consecutive same-panel 16th-adjacent presses. =2 (default, '
                        'user-approved) allows a justified 2-note 16th jack, hard-forbids 3+. 0/negative = off.')
    p.add_argument('--jack_penalty', type=float, default=0.0,
                   help='[DEPRECATED] OLD single-foot jack governor (lambda). SUPERSEDED by --fatigue_penalty (the two-foot model '
                        'generalizes it), so default 0 = off. Set >0 only to use the jack governor INSTEAD of fatigue. '
                        'notes/foot_exertion_findings.md')
    p.add_argument('--fatigue_penalty', type=float, default=2.0,
                   help='PER-FOOT FATIGUE governor (lambda) — the RELEASE per-note governor (default 2.0). Two-foot '
                        'biomechanical simulator; governs jacks AND jump streams via min-exertion footing (generalizes '
                        'the jack governor; required for --stamina_ceiling). Good range 1.5-3 (matches real jack dist, '
                        'density held); 0=off. notes/governor_release_region.md')
    p.add_argument('--fatigue_free', type=float, default=6.0,
                   help='free exertion zone for the fatigue governor (a rested jump/jack passes; only streams '
                        'gated). 6 and generate()\'s 12 are BOTH in the vouched 6-12 range (governor_release_'
                        'region.md): 12 = design-note default (barrier set high), 6 = the more-gating end this '
                        'exporter has always playtested. <6 jump-starves, >=18 silent.')
    p.add_argument('--stamina_ceiling', type=float, default=50.0,
                   help='STAGE-2 STAMINA governor (per-region DENSITY): a slow exertion accumulator thins UPCOMING '
                        'onset density only where sustained workload is high (a CEILING, never a global dent). Needs '
                        '--fatigue_penalty (the foot model supplies the cost). DEFAULT 50 (the full-governor playtest '
                        'value; 25 thins harder toward natural density). <=0 = off. notes/foot_fatigue_design.md "STAGE 2".')
    p.add_argument('--stamina_tau', type=float, default=8.0, help='stamina slow-decay (beats, ~several measures)')
    p.add_argument('--stamina_scale', type=float, default=15.0, help='excess-workload scale for the tau bump (tanh)')
    p.add_argument('--stamina_breathe', type=float, default=1.2,
                   help='[DEFAULT 1.2 = arc ON] STAGE-3 ARC: make the stamina ceiling BREATHE with audio energy (high '
                        'at climaxes -> keep the spicy notes, low in verses -> rest) = a difficulty arc. Needs '
                        '--stamina_ceiling (inert without it). 0 = flat (no arc). notes/foot_fatigue_design.md "STAGE 3".')
    p.add_argument('--harm_calib', type=float, default=0.0,
                   help='[STEP-1 phrase calibrator] sparse-harm-in-quiet onset logit boost (gain·quiet_gate·harm) '
                        'so the head allocates for a sparse melodic/harmonic event in a quiet phrase (the HSL '
                        'piano-solo under-placement). Needs --features highres (perc/harm channels). ~10 to start; '
                        '0 = off. notes/phrasing_coherence_findings.md.')
    p.add_argument('--harm_quiet_q', type=float, default=40.0,
                   help='energy percentile defining "quiet" for --harm_calib (frames below it get the boost).')
    p.add_argument('--motif', type=str, default=None,
                   help='H15 continuous motif-knob conditioning (gen_motif_full/local2), e.g. '
                        '"candle=3,trill=-2" or raw "3=3,10=-2". Aliases: candle=3, trill=10, jacksweep=0, '
                        'bracket=1. Sets a GLOBAL motif vector (CFG-amplifiable with --guidance).')
    p.add_argument('--figure', type=str, default=None,
                   help='H15 discrete figure-token conditioning (gen_motif_full), e.g. "sweep" -> a constant '
                        f'per-section figure schedule. One of {[c for c in FIGURE_CLASSES if c != "sparse"]}.')
    p.add_argument('--prefer', type=str, default=None, help='panel preference, e.g. "U,R" to favor Up+Right')
    # ⛔ --radar is DISABLED by default. It MEAN-PINS unset dims (others at the dataset mean), which is OFF-MANIFOLD:
    # the radar dims are correlated (stream/voltage/air/chaos r 0.71-0.92), so a single-dim pin at high values
    # SMEARS (chaos=0.9 g3 -> 16th-share 0.98, quarter backbone ~0) -- a knob-shaped artifact, NOT the deployed
    # knob. The CORRECT path is --style (RadarManifold conditional-fill + projection; coupled dims move together,
    # backbone preserved). See the conditioning-mechanics skill §2 + its misalignment catalog ("mean-pin vs
    # manifold conditional-fill"). Only --radar_ood (a deliberate, labeled "see the raw OOD reach" test) re-enables it.
    p.add_argument('--radar', type=str, default=None,
                   help='[TRAP] DISABLED (mean-pin = OFF-MANIFOLD smear, not the real knob). Use --style for the manifold '
                        'path. For a deliberate OOD reach test, pass --radar_ood too. See conditioning-mechanics skill §2.')
    p.add_argument('--radar_ood', action='store_true',
                   help='acknowledge that --radar is an off-manifold mean-pin (OOD smear) and run it anyway '
                        '(deliberate raw-reach test only). Without this, --radar errors out.')
    p.add_argument('--match_radar', action='store_true',
                   help="condition each song's generation on its OWN source-chart radar (with --guidance) so "
                        "the output matches the original's groove profile -- avoids profile drift when you "
                        "selected/expected a specific feel. Overrides --radar; pair with --guidance 1.5-2.5.")
    p.add_argument('--reference_self', action='store_true',
                   help="per-song style conditioning: encode each song's OWN source chart as the StyleEncoder "
                        "latent (the full-chart path vs match_radar's 5-dim summary). Pair with --guidance ~2.")
    p.add_argument('--style', type=str, default=None,
                   help="MANIFOLD-AWARE groove steering: a PARTIAL spec over named axes, e.g. "
                        "\"stream=high,chaos=low,air=low\". Unspecified dims are conditional-filled from the real "
                        "covariance and the whole point is projected onto the manifold, so coupled dims stay "
                        "coherent (vs --radar's pin-others-at-mean, which goes OOD). Levels low/mod/high (or "
                        "q0.9 / a raw 0-1 value); per-song difficulty. Pair with --guidance ~1.5. Overrides --radar.")
    p.add_argument('--max_len', type=int, default=1440)  # full 2-min songs (KV-cache makes it cheap)
    p.add_argument('--install', action='store_true',
                   help="After exporting, copy the set into the StepMania songs dir (no sudo).")
    p.add_argument('--songs_dir', default=None,
                   help="Destination for --install (default: $SM_SONGS_DIR or ~/sm-generated).")
    p.add_argument('--groove_select', default='none',
                   choices=['none', 'rich', 'stream', 'voltage', 'air', 'freeze', 'chaos'],
                   help="GROOVE-VALIDATE the set: pick songs that read strongly on this axis so the set "
                        "actually tests the hypothesis (freeze=holds, stream/voltage=density, air=jumps, "
                        "chaos=syncopation; 'rich'=strong across all). Reports the chosen songs' radar. "
                        "'none' = first-N by seed order (legacy).")
    p.add_argument('--difficulty_select', default=None, choices=['Beginner', 'Easy', 'Medium', 'Hard'],
                   help="Restrict groove-selected songs to this difficulty class (harder = more revealing).")
    p.add_argument('--song_filter', default=None,
                   help="Comma-separated case-insensitive substrings; keep only songs whose title/path matches "
                        "one (e.g. 'japa1,deja loin,oh world,high school love'). Applied after groove_select.")
    return p.parse_args()


def safe_name(s):
    s = re.sub(r'[^\w\- ]+', '', (s or 'untitled').strip(), flags=re.UNICODE).strip()
    return (s or 'untitled')[:60]


def typed_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def main():
    args = parse_args(); set_seed(args.seed)
    if args.radar and not args.radar_ood:  # fail FAST (before the slow data load) -- see conditioning-mechanics §2
        raise SystemExit(
            "⛔ --radar is DISABLED: it mean-pins unset dims (others at the dataset mean) = OFF-MANIFOLD, which\n"
            "   SMEARS at high single-dim values (a knob-shaped artifact, not the deployed knob). Use --style for\n"
            "   the manifold conditional-fill (e.g. --style \"chaos=q0.99\"). If you truly want the raw off-manifold\n"
            "   reach test, re-run with --radar_ood. See the conditioning-mechanics skill §2.")
    phase_alloc = ([float(x) for x in args.onset_phase_alloc.split(',')]
                   if args.onset_phase_alloc else None)  # phase-aware threshold shares (q,8th,16th)
    phase_calib = (tuple(float(x) for x in args.onset_phase_calib.split(','))
                   if args.onset_phase_calib else None)  # per-phase logit offset (b8, b16) for variable chaos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # H15 motif-knob conditioning (continuous): build a GLOBAL (1, MOTIF_DIM) vector from "name/idx=z" pairs.
    MOTIF_ALIAS = {'candle': 3, 'trill': 10, 'jacksweep': 0, 'bracket': 1}
    motif_vec = None
    if args.motif:
        motif_vec = torch.zeros(1, MOTIF_DIM)
        for tok in args.motif.split(','):
            key, val = tok.split('='); key = key.strip()
            idx = MOTIF_ALIAS.get(key, None)
            idx = int(key) if idx is None else idx
            motif_vec[0, idx] = float(val)
        motif_vec = motif_vec.to(device)
        print(f"motif conditioning: {args.motif} -> knob vector {motif_vec.cpu().numpy().round(1).tolist()}")
    # H15 figure-token conditioning (discrete): a constant per-section figure schedule built per song (needs T).
    figure_tok = None
    if args.figure:
        FIG_ALIAS = {'sweep': 'sweep/staircase', 'candle': 'candle/cross', 'jump': 'jump/bracket'}
        figure_tok = FIGURE_CLASSES.index(FIG_ALIAS.get(args.figure, args.figure))
        print(f"figure conditioning: '{args.figure}' -> token {figure_tok} ({FIGURE_CLASSES[figure_tok]})")
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=args.seed)
    if args.song_filter:  # restrict to named songs UP FRONT (before the pool cap) so they're loaded at all
        terms = [t.strip().lower() for t in args.song_filter.split(',') if t.strip()]
        val_files = [f for f in val_files if any(t in f.lower() for t in terms)]
        print(f"SONG-FILTER pre-restricted val_files to {len(val_files)} file(s) matching {terms}")
        if not val_files:
            raise SystemExit(f"--song_filter {terms!r} matched no val files under {args.data_dir}.")
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        msl = yaml.safe_load(f)['classifier']['max_sequence_length']
    # feature set: base (23-dim) vs stage1 (41-dim musical) vs highres (42-dim, + high-res onset)
    from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
    if args.features == 'highres':
        feat_ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                            use_metric_phase=True, use_highres_onset=True))
        audio_dim, cache = 42, 'cache/samples_v3'
    elif args.features == 'stage1':
        feat_ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
        audio_dim, cache = 41, 'cache/samples_v2'
    else:
        feat_ext, audio_dim, cache = None, 23, 'cache/samples'
    # widen the candidate pool when groove-selecting (parsing is cheap; audio is extracted only for the
    # chosen songs) so the selector has enough songs to find strong-on-axis ones.
    pool = args.num_songs * (40 if args.groove_select != 'none' else 8)
    ds = StepManiaDataset(chart_files=val_files[:pool], audio_dir=args.audio_dir,
                          max_sequence_length=msl, feature_extractor=feat_ext, cache_dir=cache)

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    # strict=False: gen_radar/gen_layered predate style_encoder; those params stay at init (unused unless --reference)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'], strict=False); model.eval()
    critic = DifficultyCritic(device=device)

    # Step 3: optional reference chart -> condition every generated song on its style
    style_vec, style_label = None, None
    if args.reference:
        ref_chart = ds.parser.parse_file(args.reference)
        if ref_chart is None:
            raise SystemExit(f"could not parse reference chart: {args.reference}")
        cands = [n for n in ref_chart.note_data if n.difficulty_name]
        if args.reference_difficulty:
            want = args.reference_difficulty.rstrip(':').strip().lower()
            cands = [n for n in cands if n.difficulty_name.rstrip(':').strip().lower() == want] or cands
        ref_nd = max(cands, key=lambda n: n.difficulty_value)  # hardest available (or filtered)
        ref_typed = ds.parser.convert_to_tensor_typed(ref_chart, ref_nd)[:args.max_len]
        ref_t = torch.from_numpy(ref_typed.astype(np.int64)).unsqueeze(0).to(device)
        ref_mask = torch.ones(1, ref_t.shape[1], dtype=torch.bool, device=device)
        with torch.no_grad():
            style_vec = model.encode_style(ref_t, ref_mask)  # (1,d) latent, reused for every song
        ref_dens = float((ref_typed != 0).any(1).mean())
        style_label = f"{ref_chart.title or Path(args.reference).stem} [{ref_nd.difficulty_name.rstrip(':').strip()}]"
        print(f"\nstyle reference: {style_label}  (density {ref_dens:.3f}, {len(ref_typed)} frames)  "
              f"guidance={args.guidance}")

    # groove-radar target: base at the dataset mean, override the requested dims
    RADAR_DIMS = ['stream', 'voltage', 'air', 'freeze', 'chaos']
    radar_vec = None
    if args.radar:  # guarded at main() entry -> only reached with --radar_ood (deliberate off-manifold reach test)
        radars = [m['groove_radar'].to_vector() for m in ds.valid_samples if 'groove_radar' in m]
        base = np.mean(radars, 0).astype(np.float32) if radars else np.full(5, 0.5, np.float32)
        for tok in args.radar.split(','):
            k, _, v = tok.strip().partition('=')
            k = k.strip().lower()
            if k in RADAR_DIMS:
                base[RADAR_DIMS.index(k)] = float(v)
        radar_vec = torch.from_numpy(base).unsqueeze(0).to(device)  # (1,5), reused for every song
        print(f"\ngroove radar target: {dict(zip(RADAR_DIMS, base.round(2).tolist()))}  guidance={args.guidance}")

    # manifold-aware steering: load the fitted manifold (cache, else fit from this dataset) for --style
    manifold = None
    if args.style:
        from src.generation.radar_manifold import RadarManifold
        mp = Path('cache/radar_manifold.npz')
        if mp.exists():
            manifold = RadarManifold.load(mp)
        else:
            manifold = RadarManifold.from_loaded_datasets(ds)
            print("⚠️  cache/radar_manifold.npz missing; fit from this split only "
                  "(run diag_radar_manifold.py to persist the full-data manifold).")
        manifold.parse_spec(args.style)  # validate axis names up front
        print(f"\nmanifold style: '{args.style}'  guidance={args.guidance} (per-song, difficulty-bucketed)")

    # build pattern-preference bias from CLI knobs
    from src.generation.typed import make_pattern_bias
    panel_prefs = None
    if args.prefer:
        names = {'L': 0, 'D': 1, 'U': 2, 'R': 3}
        panel_prefs = [0.0, 0.0, 0.0, 0.0]
        for tok in args.prefer.upper().split(','):
            if tok.strip() in names:
                panel_prefs[names[tok.strip()]] = 1.5
    pattern_bias = (make_pattern_bias(jump=args.jump_bias, panel_prefs=panel_prefs)
                    if (args.jump_bias or panel_prefs) else None)

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    # Groove-validate the set: pick songs that read strongly on the axis under test (else the playtest
    # can't reveal the hypothesis -- e.g. B4U has 3 holds, useless for a hold fix). Report the profile.
    if args.groove_select != 'none':
        from src.data.song_selection import select_by_groove, radar_table, RADAR_DIMS
        dcls = ['Beginner', 'Easy', 'Medium', 'Hard'].index(args.difficulty_select) if args.difficulty_select else None
        order = select_by_groove(ds, n=args.num_songs, by=args.groove_select, difficulty=dcls)
        print(f"\nGROOVE-SELECTED by '{args.groove_select}'"
              f"{f' (difficulty={args.difficulty_select})' if args.difficulty_select else ''} "
              f"-- {len(order)} songs, strongest first:")
        print(radar_table(ds, order))
    else:
        order = range(len(ds.valid_samples))

    if args.song_filter:  # keep only named songs (match title or chart_file path), preserving order
        terms = [t.strip().lower() for t in args.song_filter.split(',') if t.strip()]
        def _match(i):
            m = ds.valid_samples[i]
            hay = f"{m.get('chart', None) and getattr(m['chart'], 'title', '') or ''} {m['chart_file']}".lower()
            return any(t in hay for t in terms)
        order = [i for i in order if _match(i)]
        print(f"\nSONG-FILTERED to {terms}: {len(order)} matching sample(s).")
        if not order:
            raise SystemExit(f"--song_filter {terms!r} matched no songs. Check titles/paths in {args.data_dir}.")
    # difficulty restriction works WITH or WITHOUT groove_select (else the per-song loop takes the first
    # difficulty in dataset order, which can be Beginner -- wrong for a jack/coherence playtest).
    if args.difficulty_select and args.groove_select == 'none':
        dcls = ['Beginner', 'Easy', 'Medium', 'Hard'].index(args.difficulty_select)
        order = [i for i in order if ds.valid_samples[i]['difficulty_class'] == dcls]
        print(f"DIFFICULTY-FILTERED to {args.difficulty_select}: {len(order)} sample(s).")

    print(f"\n{'song':<34} {'diff':<8} {'gen_dens':>8} {'ref_dens':>8} {'holds':>6} {'critic':>9}")
    print("-" * 80)

    exported, seen = 0, set()
    for i in order:
        if exported >= args.num_songs:
            break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen:
            continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        audio_np = sample['audio'][:T].numpy().astype(np.float32)
        orig_typed = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        diff_idx = meta['difficulty_class']

        # radar conditioning: match this song's own source profile (so output ~ original feel), else the
        # fixed --radar target (or none).
        radar_for_gen = radar_vec
        style_density = None
        if manifold is not None:  # manifold-aware: build a coherent on-manifold target for THIS difficulty
            tvec, tinfo = manifold.build_target(args.style, diff_idx)
            radar_for_gen = torch.from_numpy(tvec).unsqueeze(0).to(device)
            style_density = tinfo['density']     # SOURCE-CHART-FREE density target (difficulty + style)
            if exported == 0:  # print the resolved target once (it varies slightly by difficulty)
                print(f"  -> target ({DIFFICULTY_NAMES[diff_idx]}): "
                      + " ".join(f"{d}={tvec[k]:.2f}" for k, d in enumerate(RADAR_DIMS))
                      + f"  Mahal d={tinfo['mahalanobis']:.2f}{' (projected)' if tinfo['projected'] else ''}"
                      + (f"  density~{style_density:.3f} (manifold, no source chart)" if style_density else ""))
        if args.match_radar:
            radar_for_gen = torch.from_numpy(
                meta['groove_radar'].to_vector().astype(np.float32)).unsqueeze(0).to(device)
        # style conditioning: per-song self-reference (encode this song's own source chart) vs global --reference
        style_for_gen = style_vec
        if args.reference_self:
            ref_t = torch.from_numpy(np.asarray(orig_typed)).long().unsqueeze(0).to(device)
            ref_mask = torch.ones(1, ref_t.shape[1], device=device)
            with torch.no_grad():
                style_for_gen = model.encode_style(ref_t, ref_mask)

        # onset threshold matched to THIS chart's real density
        audio = torch.from_numpy(audio_np).unsqueeze(0).to(device)
        diff = torch.tensor([diff_idx], device=device)
        with torch.no_grad():
            # tau MUST be computed from the SAME conditioned logits generate() decodes from, else radar/style
            # conditioning (which raises p broadly) floods past a tau calibrated on the unconditioned p.
            memory = model.encode_audio(audio)
            ol_onset = model.onset_logits(memory, diff, radar=radar_for_gen, style=style_for_gen)[0]
            if args.guidance != 1.0 and (radar_for_gen is not None or style_for_gen is not None):
                ol_u = model.onset_logits(memory, diff, radar=None, style=None)[0]
                ol_onset = ol_u + args.guidance * (ol_onset - ol_u)
            if phase_calib is not None:  # apply the SAME per-phase offset used inside generate() before tau
                b8, b16 = phase_calib; ph = torch.arange(ol_onset.shape[0], device=device) % 4
                ol_onset = ol_onset + torch.where(ph == 2, b8, torch.where((ph == 1) | (ph == 3), b16, 0.0))
            harm_off_t = None
            if args.harm_calib > 0:  # sparse-harm-in-quiet calibrator: tau MUST see the same offset generate() uses
                if audio_dim != 42:
                    raise SystemExit("--harm_calib needs --features highres (the 42-dim perc/harm channels).")
                harm_off_t = torch.from_numpy(_sparse_harm_offset(audio_np, args.harm_calib, args.harm_quiet_q)).to(device)
                ol_onset = ol_onset + harm_off_t
            p_onset = torch.sigmoid(ol_onset).cpu().numpy()
        real_density = float((orig_typed != 0).any(1).mean())
        # density target priority: explicit --target_density > manifold style density (SOURCE-CHART-FREE:
        # E[density | difficulty, style], so stream-as-a-knob works and no source chart is needed) > the
        # source chart's own density (eval convenience for A/B, NOT available for a brand-new song).
        # Raising chaos at FIXED density forces quarter->off-beat REPLACEMENT (backbone collapse); real charts
        # raise density WITH chaos (corr +0.63) -- the manifold density couples them so high-chaos lifts density.
        if args.target_density is not None:
            gen_density = args.target_density
        elif style_density is not None:
            gen_density = style_density
        else:
            gen_density = real_density
        tau = float(np.quantile(p_onset, 1 - gen_density)) if gen_density > 0 else 0.5

        gen_kwargs = dict(onset_threshold=tau, type_sample=True,
                          type_temperature=args.type_temperature,
                          pattern_sample=True, pattern_temperature=args.pattern_temperature,
                          repetition_penalty=args.repetition_penalty,
                          max_jack_run=(args.max_jack_run if args.max_jack_run and args.max_jack_run > 0 else None),
                          jack_penalty=(args.jack_penalty if args.jack_penalty and args.jack_penalty > 0 else None),
                          fatigue_penalty=(args.fatigue_penalty if args.fatigue_penalty and args.fatigue_penalty > 0 else None),
                          fatigue_free=args.fatigue_free,
                          stamina_ceiling=(args.stamina_ceiling if args.stamina_ceiling and args.stamina_ceiling > 0 else None),  # Stage-2 per-region density relief (needs fatigue_penalty); <=0 = off
                          stamina_tau=args.stamina_tau, stamina_scale=args.stamina_scale,
                          stamina_breathe=args.stamina_breathe,  # Stage-3 ARC: ceiling breathes with audio energy
                          bpm=float(meta['chart'].bpm),  # foot-exertion / fatigue governors need real BPM for press-rate
                          pattern_bias=pattern_bias, no_crossovers=args.no_crossovers,
                          onset_phase_penalty=args.onset_phase_penalty,
                          onset_phase_alloc=phase_alloc, onset_phase_calib=phase_calib,
                          onset_logit_offset=harm_off_t,  # STEP-1 sparse-harm-in-quiet phrase calibrator (None=off)
                          style=style_for_gen, guidance_scale=args.guidance, radar=radar_for_gen,
                          motif=motif_vec,  # H15 continuous motif knobs (global vector; None if --motif unset)
                          figure=(torch.full((1, T), figure_tok, dtype=torch.long, device=device)
                                  if figure_tok is not None else None))  # H15 discrete figure schedule
        if args.override_playability:  # user-approved deviation -> respect the explicit flags
            gen_kwargs.update(hold_aware=True, no_jump_during_hold=args.no_jump_during_hold,
                              no_cross_during_hold=args.no_cross_during_hold)
        enforce_playability(gen_kwargs, args.override_playability)  # MANDATORY pad-playability (forces them on)
        gen = pair_holds(model.generate(audio, diff, lengths=torch.tensor([T], device=device),
                                        **gen_kwargs)[0].cpu().numpy())

        chart_obj = meta['chart']
        bpm = float(chart_obj.bpm); music = os.path.basename(meta['audio_file'])
        title = (chart_obj.title or Path(meta['chart_file']).stem)
        dname = DIFFICULTY_NAMES[diff_idx]

        folder = out_root / f"{exported:02d}_{safe_name(title)}"
        folder.mkdir(parents=True, exist_ok=True)
        if os.path.exists(meta['audio_file']):
            try:
                shutil.copy2(meta['audio_file'], folder / music)
            except Exception:
                pass
        sm = charts_to_sm(
            charts=[
                {"chart": gen, "difficulty_name": "Challenge", "difficulty_value": nd.difficulty_value, "author": "generated"},
                {"chart": orig_typed, "difficulty_name": dname, "difficulty_value": nd.difficulty_value, "author": "original"},
            ],
            bpm=bpm, title=f"{title} (gen)", artist=chart_obj.artist or "",
            music=music, offset=float(chart_obj.offset), typed=True,
        )
        (folder / "chart.sm").write_text(sm, encoding="utf-8")

        h = symbol_histogram(gen)
        cpred = critic.predict(typed_binary(gen), audio_np[:, :23], bpm=DEFAULT_BPM)['name']  # critic is 23-dim Phase-1
        gen_d = float((gen != 0).any(1).mean())
        print(f"{safe_name(title)[:33]:<34} {dname:<8} {gen_d:>8.3f} {real_density:>8.3f} "
              f"{h['hold_head']:>6} {cpred:>9}")
        seen.add(meta['chart_file']); exported += 1

    print("-" * 80)
    print(f"Exported {exported} playable folders to {out_root}/ (chart.sm: Challenge=generated, "
          f"+ original; both with holds). Drop a folder into StepMania to play.")

    if args.install:
        from src.utils.sm_install import install_to_stepmania
        dests = install_to_stepmania(args.out_dir, args.songs_dir)
        print("\nInstalled to StepMania (no sudo):")
        for d in dests:
            print(f"  {d}")


if __name__ == '__main__':
    main()
