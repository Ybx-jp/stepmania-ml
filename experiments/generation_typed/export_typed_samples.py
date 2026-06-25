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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--out_dir', default='outputs/typed_samples')
    p.add_argument('--checkpoint', default='checkpoints/gen_style/best_val.pt')
    p.add_argument('--features', choices=['base', 'stage1', 'highres'], default='base',
                   help='base=23-dim (cache/samples); stage1=41-dim musical features (cache/samples_v2)')
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
                   help='phase-aware onset threshold: target note shares "quarter,8th,16th" (real ~"0.707,0.252,0.041"). '
                        'Redistributes the density budget across phases so the model\'s own 16th confidence wins 16th '
                        'slots instead of losing to 8ths (which a single threshold buries). None = single threshold.')
    p.add_argument('--onset_phase_calib', type=str, default=None,
                   help='per-phase calibration offset "b8,b16" (logit space; e.g. "0,0.19") for VARIABLE '
                        'per-song chaos: corrects the model\'s 16th under-confidence so the 16th count floats '
                        'with the audio (chaotic songs get many, calm songs ~none). Preferred over the flat '
                        '--onset_phase_alloc quota. None = single threshold.')
    p.add_argument('--target_density', type=float, default=None,
                   help='override per-chart density (notes/frame) for the onset threshold; default = match the '
                        'source chart. Use to couple density to chaos (real high-chaos charts run ~0.34 vs ~0.22 '
                        'baseline) so raising chaos ADDS off-beats instead of replacing the quarter backbone.')
    p.add_argument('--onset_phase_penalty', type=float, default=0.0,
                   help='metric gate: off-beat onsets need higher confidence (on-beat 0, 8th -p, 16th -2p). '
                        '~0.5-1.5 restores the downbeat under chaos conditioning. 0 = off.')
    p.add_argument('--max_jack_run', type=int, default=2,
                   help='HARD 16th-jack cap: max consecutive same-panel 16th-adjacent presses. =2 (default, '
                        'user-approved) allows a justified 2-note 16th jack, hard-forbids 3+. 0/negative = off.')
    p.add_argument('--jack_penalty', type=float, default=1.5,
                   help='SOFT foot-exertion governor (lambda): escalating BPM-aware penalty to extend a same-panel '
                        'run (penalty = lambda * accumulated exertion). Gates unnatural jack STREAMS (8th + long) '
                        'while keeping short justified ones; preserves density (re-routes to alternation). '
                        '0 = off; ~1.5 gentle (default), ~3 aggressive. Uses the song BPM. notes/foot_exertion_findings.md')
    p.add_argument('--motif', type=str, default=None,
                   help='H15 continuous motif-knob conditioning (gen_motif_full/local2), e.g. '
                        '"candle=3,trill=-2" or raw "3=3,10=-2". Aliases: candle=3, trill=10, jacksweep=0, '
                        'bracket=1. Sets a GLOBAL motif vector (CFG-amplifiable with --guidance).')
    p.add_argument('--figure', type=str, default=None,
                   help='H15 discrete figure-token conditioning (gen_motif_full), e.g. "sweep" -> a constant '
                        f'per-section figure schedule. One of {[c for c in FIGURE_CLASSES if c != "sparse"]}.')
    p.add_argument('--prefer', type=str, default=None, help='panel preference, e.g. "U,R" to favor Up+Right')
    p.add_argument('--radar', type=str, default=None,
                   help='groove-radar target as dim=val list over [stream,voltage,air,freeze,chaos], '
                        'e.g. "chaos=0.9,air=0.85"; unset dims default to the dataset mean. Use with --guidance to amplify.')
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
    return p.parse_args()


def safe_name(s):
    s = re.sub(r'[^\w\- ]+', '', (s or 'untitled').strip(), flags=re.UNICODE).strip()
    return (s or 'untitled')[:60]


def typed_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def main():
    args = parse_args(); set_seed(args.seed)
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
    if args.radar:
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
                          bpm=float(meta['chart'].bpm),  # foot-exertion governor needs real BPM for press-rate
                          pattern_bias=pattern_bias, no_crossovers=args.no_crossovers,
                          onset_phase_penalty=args.onset_phase_penalty,
                          onset_phase_alloc=phase_alloc, onset_phase_calib=phase_calib,
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
