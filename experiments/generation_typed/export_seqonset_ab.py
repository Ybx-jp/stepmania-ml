#!/usr/bin/env python3
"""BY-EAR A/B: deployed AUDIO onsets vs the SS head's SEQ onsets, everything else IDENTICAL. (2026-06-29)
Lineage seq-onset-arc.md. The binding gate (exp-design Rule 8) for fork (A): M1b-4 (AUC) and M1b-5 (taste critic)
both rank the seq head's free-run placement BELOW the deployed audio path — this export lets the user confirm by EAR.

ONE-CHANGE A/B (Rule 11) routed through the DEPLOYED generate() via the sanctioned `onset_override` (NO loop surgery,
Rule 14), canonical governor/playability config (generation-defaults skill). BOTH arms: same songs, same
gen_motif_full_fixed, density-matched to the seq head (d_seq), radar off, stamina OFF for both (override skips it),
per-note fatigue + playability ON. The ONLY difference is the ONSET TRAJECTORY. Each song -> ONE .sm with THREE
selectable charts so the user A/Bs on the same audio:
  Challenge = AUDIO onsets (deployed audio head + 16th-unlock onset_phase_calib, top-d_seq)
  Edit      = SEQ onsets   (the SS head's free-run rollout)
  <real>    = the ORIGINAL real chart (reference)
NOTE: stamina/breathe is OFF (override) -> this isolates PLACEMENT, it is not the full deployed density-arc feel.

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/export_seqonset_ab.py --load_head --n 4
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys, re, shutil
from pathlib import Path
import numpy as np, torch, yaml
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.playtest_export import enforce_playability
from src.generation.sm_writer import charts_to_sm
from src.data.dataset import DIFFICULTY_NAMES
from probe_seqcontext_frozenh import AD, DMODEL, CKPT, HReadConv
from probe_seqonset_rollout import rollout
from probe_seqonset_critic import DECODE, audio_onset


def safe_name(s):
    return re.sub(r'[^\w\-]+', '_', str(s)).strip('_')[:50] or 'song'


def gen_typed(model, A42, T, diff, onset_bool, bpm, device):
    """Deployed generate() with the supplied onsets via onset_override + canonical governor + MANDATORY playability.
    Returns the TYPED chart (pair_holds, symbols 0..4) for charts_to_sm(typed=True)."""
    kw = dict(onset_override=onset_bool.unsqueeze(0).to(device), bpm=float(bpm), **DECODE)
    enforce_playability(kw, "seq-onset by-ear A/B (onset supplied, panels deployed)")
    return pair_holds(model.generate(A42, diff, lengths=torch.tensor([T], device=device), **kw)[0].cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load_head', action='store_true'); ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--cap', type=int, default=640); ap.add_argument('--tau', type=float, default=0.55)
    ap.add_argument('--out', default="outputs/seqonset_ab",
                    help='BUILD dir (must NOT be under the songs root ~/sm-generated, or install rmtrees the source); '
                         'install_to_stepmania copies it to ~/sm-generated/<basename>')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    _, va_ds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                  max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')

    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=DMODEL, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head = HReadConv(DMODEL).to(device)
    head.load_state_dict(torch.load(PROJECT_ROOT / "cache/seqonset_ss_head.pt", map_location=device)); head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    cands = []
    for i in range(len(va_ds.valid_samples)):
        meta = va_ds.valid_samples[i]
        if meta['difficulty_class'] >= 3:
            cands.append((float(meta['groove_radar'].chaos), i))
    cands.sort(reverse=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"  exporting {args.n} A/B songs -> {out}\n", flush=True)
    n = 0
    for chaos, i in cands:
        if n >= args.n:
            break
        meta = va_ds.valid_samples[i]; s = va_ds[i]; T = min(int(s['mask'].sum().item()), args.cap)
        nd = next((x for x in meta['chart'].note_data if x.difficulty_name == meta['difficulty_name']
                   and x.difficulty_value == meta['difficulty_value']), None)
        if nd is None or T < 256:
            continue
        orig = np.asarray(va_ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        real_on = (orig != 0).any(1); i16 = (np.arange(T) % 4 == 1) | (np.arange(T) % 4 == 3)
        if real_on.sum() < 32 or (real_on & i16).sum() / max(real_on.sum(), 1) < 0.05:
            continue
        a = s['audio'][:T, :AD].numpy().astype(np.float32)
        A42 = torch.from_numpy(a).unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        bpm = float(meta['chart'].bpm)

        seq_on = torch.from_numpy(rollout(model, head, a, meta['difficulty_class'], T, args.tau, device)).bool()
        d_seq = float(seq_on.float().mean())
        aud_on = audio_onset(model, A42, diff, T, d_seq, device)            # density-matched deployed baseline
        seq_chart = gen_typed(model, A42, T, diff, seq_on, bpm, device)
        aud_chart = gen_typed(model, A42, T, diff, aud_on, bpm, device)

        title = meta['chart'].title or Path(meta['chart_file']).stem
        dname = DIFFICULTY_NAMES[meta['difficulty_class']]
        music = os.path.basename(meta['audio_file'])
        folder = out / f"{n:02d}_{safe_name(title)}"; folder.mkdir(parents=True, exist_ok=True)
        if os.path.exists(meta['audio_file']):
            try:
                shutil.copy2(meta['audio_file'], folder / music)
            except Exception:
                pass
        charts = [
            {"chart": aud_chart, "difficulty_name": "Challenge", "difficulty_value": nd.difficulty_value, "author": "audio-onset (deployed)"},
            {"chart": seq_chart, "difficulty_name": "Edit", "difficulty_value": nd.difficulty_value, "author": "seq-onset (SS head)"},
        ]
        if dname not in ("Challenge", "Edit"):
            charts.append({"chart": orig, "difficulty_name": dname, "difficulty_value": nd.difficulty_value, "author": "original"})
        sm = charts_to_sm(charts=charts, bpm=bpm, title=f"{title} (A/B)", artist=meta['chart'].artist or "",
                          music=music, offset=float(meta['chart'].offset), typed=True)
        (folder / "chart.sm").write_text(sm, encoding="utf-8")
        print(f"  {safe_name(title)[:30]:<32} chaos {chaos:4.1f}  d_seq {d_seq:.3f} (real {real_on.mean():.3f})  "
              f"Challenge=audio  Edit=seq", flush=True)
        n += 1

    from src.utils.sm_install import install_to_stepmania
    install_to_stepmania(str(out), None)
    print(f"\n  installed {n} A/B songs -> {out}", flush=True)
    print(f"  In StepMania: each song has Challenge=AUDIO onsets, Edit=SEQ onsets, plus the original. Same audio,", flush=True)
    print(f"  same density, same panels/governor — ONLY the onset placement differs. Play both, log which feels better.", flush=True)


if __name__ == '__main__':
    main()
