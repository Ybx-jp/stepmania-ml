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
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import symbol_histogram, pair_holds
from src.generation.sm_writer import charts_to_sm
from src.generation.evaluation import DifficultyCritic

DEFAULT_BPM = 150.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--out_dir', default='outputs/typed_samples')
    p.add_argument('--checkpoint', default='checkpoints/gen_style/best_val.pt')
    p.add_argument('--features', choices=['base', 'stage1'], default='base',
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
    p.add_argument('--no_cross_during_hold', action='store_true',
                   help='forbid the free foot fast-crossing panels while a hold is open (the B4U one-foot '
                        'jacks-during-hold awkwardness; brings hold_burst ~6.9%%->4.7%% vs real 4.0%%)')
    p.add_argument('--onset_phase_penalty', type=float, default=0.0,
                   help='metric gate: off-beat onsets need higher confidence (on-beat 0, 8th -p, 16th -2p). '
                        '~0.5-1.5 restores the downbeat under chaos conditioning. 0 = off.')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        msl = yaml.safe_load(f)['classifier']['max_sequence_length']
    # feature set: base (23-dim) vs stage1 (41-dim musical features)
    if args.features == 'stage1':
        from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
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
            p_onset = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff))[0].cpu().numpy()
        real_density = float((orig_typed != 0).any(1).mean())
        tau = float(np.quantile(p_onset, 1 - real_density)) if real_density > 0 else 0.5

        gen = model.generate(audio, diff, lengths=torch.tensor([T], device=device),
                             onset_threshold=tau, type_sample=True,
                             type_temperature=args.type_temperature, hold_aware=True,
                             pattern_sample=True, pattern_temperature=args.pattern_temperature,
                             repetition_penalty=args.repetition_penalty,
                             pattern_bias=pattern_bias, no_crossovers=args.no_crossovers,
                             no_jump_during_hold=args.no_jump_during_hold,
                             no_cross_during_hold=args.no_cross_during_hold,
                             onset_phase_penalty=args.onset_phase_penalty,
                             style=style_for_gen, guidance_scale=args.guidance, radar=radar_for_gen)[0].cpu().numpy()
        gen = pair_holds(gen)

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
