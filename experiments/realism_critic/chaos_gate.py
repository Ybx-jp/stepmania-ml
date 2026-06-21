#!/usr/bin/env python3
"""
A1 — decode-time SELECTIVE chaos gate (no retrain). See notes/chaos_mechanism_plan.md.

The trained chaos knob smears: a global conditioning scalar can only push off-beats up uniformly
(~5% on-beat vs real ~80-90%). Make syncopation EVENT-DRIVEN at decode by separating the two phase
classes (v2 of the gate — v1's multiplicative boost just swung the whole backbone bimodally):

  - on-beat BACKBONE: keep the top on-beat frames by p_base (the model is reliable on-beat).
  - off-beat ACCENTS: add the top-K off-beat frames by AUDIO saliency = the high-res onset (dim41),
    i.e. where there's an actual musical event (the model's own off-beat posterior is ~blind, since it's
    trained to suppress off-beats — so we key accents on audio, not p_base).
  - chaos_frac in [0,1] = fraction of the (fixed) density budget spent on off-beat accents. Total density
    held = real. So chaos_frac is a clean selective-syncopation dial; on-beat% ≈ 1-chaos_frac by design.

Offline metric is therefore the SELECTIVITY (do accents land on high-onset frames vs random?), not on-beat%
(which is the knob). Real test is the playtest: gated vs the smear, same off-beat count.
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
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.sm_writer import charts_to_sm

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
HIRES_DIM = 41
CHAOS_FRACS = [0.15, 0.30, 0.45, 0.60]  # fraction of density budget spent on off-beat accents


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/'); p.add_argument('--audio_dir', default='data/')
    p.add_argument('--num_songs', type=int, default=8)
    p.add_argument('--max_len', type=int, default=1024)
    p.add_argument('--export_frac', type=float, default=0.30, help='which chaos_frac to export for playtest')
    p.add_argument('--out_dir', default='outputs/chaos_gate')
    p.add_argument('--install', action='store_true')
    return p.parse_args()


def safe_name(s):
    s = re.sub(r'[^\w\- ]+', '', (s or 'untitled').strip(), flags=re.UNICODE).strip()
    return (s or 'untitled')[:60]


def gated_onset(p_base, sal_off, T, chaos_frac, target_density):
    """on-beat backbone (top p_base) + off-beat accents (top audio saliency sal_off). Total density fixed;
    chaos_frac = fraction of the budget spent on off-beat accents. Returns (mask, selectivity)."""
    t = np.arange(T)
    n_total = int(round(target_density * T))
    n_off = int(round(chaos_frac * n_total)); n_on = n_total - n_off
    on_idx = np.where(t % 4 == 0)[0]; off_idx = np.where(t % 4 != 0)[0]
    mask = np.zeros(T, dtype=bool)
    if n_on > 0 and len(on_idx):
        mask[on_idx[np.argsort(p_base[on_idx])[::-1][:n_on]]] = True
    sel_off = np.array([], dtype=int)
    if n_off > 0 and len(off_idx):
        sel_off = off_idx[np.argsort(sal_off[off_idx])[::-1][:n_off]]
        mask[sel_off] = True
    # selectivity: mean saliency at chosen off-beats vs all off-beats (1.0 = no selectivity)
    sel = (sal_off[sel_off].mean() / (sal_off[off_idx].mean() + 1e-8)) if len(sel_off) else float('nan')
    return mask, sel


def onbeat_frac(typed, T):
    note = (np.asarray(typed)[:T] != 0).any(1)
    t = np.arange(T)
    return float(note[t % 4 == 0].sum()) / max(int(note.sum()), 1)


def main():
    args = parse_args(); set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=val_files[:args.num_songs * 6], audio_dir=args.audio_dir,
                          max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()

    out = Path(args.out_dir); (out / "gated").mkdir(parents=True, exist_ok=True); (out / "smear").mkdir(parents=True, exist_ok=True)
    rows = []
    seen, used = set(), 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 128: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        audio42 = sample['audio'][:T].numpy()
        audio = torch.from_numpy(audio42[:, :HIRES_DIM]).unsqueeze(0).to(device)  # model gets 41-dim
        sal_off = audio42[:, HIRES_DIM]                                            # dim41 = audio saliency
        diff = torch.tensor([meta['difficulty_class']], device=device)
        orig = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        real_density = float((np.asarray(orig) != 0).any(1).mean())
        real_ob = onbeat_frac(orig, T)

        with torch.no_grad():
            p_base = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff, radar=None))[0].cpu().numpy()

        # gated decode at each chaos_frac
        gob, gsel = {}, {}
        gen_for_export = None
        for frac in CHAOS_FRACS:
            mask, selv = gated_onset(p_base, sal_off, T, frac, real_density)
            ov = torch.from_numpy(mask).unsqueeze(0).to(device)
            with torch.no_grad():
                g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_override=ov,
                                 type_sample=True, type_temperature=0.4, hold_aware=True,
                                 pattern_sample=True, pattern_temperature=0.7, no_jump_during_hold=True)[0].cpu().numpy()
            g = pair_holds(g)
            gob[frac] = onbeat_frac(g, T); gsel[frac] = selv
            if abs(frac - args.export_frac) < 1e-6:
                gen_for_export = g

        # trained-chaos smear reference (radar chaos=0.9, CFG g=2)
        radar = torch.zeros(1, 5, device=device); radar[0, 4] = 0.9
        with torch.no_grad():
            p_sm = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff, radar=radar))[0].cpu().numpy()
            tau = float(np.quantile(p_sm, 1 - real_density))
            gs = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                              type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                              pattern_temperature=0.7, no_jump_during_hold=True, radar=radar, guidance_scale=2.0)[0].cpu().numpy()
        smear_ob = onbeat_frac(pair_holds(gs), T)

        title = meta['chart'].title or Path(meta['chart_file']).stem
        rows.append((safe_name(title)[:24], real_ob, smear_ob, gob, gsel))

        # export gated@export_frac + smear, for playtest A/B
        if gen_for_export is not None:
            bpm = float(meta['chart'].bpm); music = os.path.basename(meta['audio_file'])
            dname = DIFFICULTY_NAMES[meta['difficulty_class']]
            for tag, root, g in [("gated", out / "gated", gen_for_export), ("smear", out / "smear", pair_holds(gs))]:
                folder = root / f"{used:02d}_{safe_name(title)}"; folder.mkdir(parents=True, exist_ok=True)
                if os.path.exists(meta['audio_file']):
                    try: shutil.copy2(meta['audio_file'], folder / music)
                    except Exception: pass
                sm = charts_to_sm(charts=[
                    {"chart": g, "difficulty_name": "Challenge", "difficulty_value": nd.difficulty_value, "author": f"chaos-{tag}"},
                    {"chart": np.asarray(orig), "difficulty_name": dname, "difficulty_value": nd.difficulty_value, "author": "original"},
                ], bpm=bpm, title=f"{title} ({tag})", artist=meta['chart'].artist or "", music=music,
                   offset=float(meta['chart'].offset), typed=True)
                (folder / "chart.sm").write_text(sm, encoding="utf-8")
        seen.add(meta['chart_file']); used += 1

    print(f"\n=== A1 selective chaos gate ({used} songs) ===")
    print("on-beat% (= 1-chaos_frac by design) + OFF-BEAT SELECTIVITY (mean dim41 at chosen accents / all")
    print("off-beats; >1 = accents land on louder audio events; ~1 = no better than random/smear)\n")
    hdr = f"{'song':<24} {'real_ob':>7} {'smear_ob':>8} | " + " ".join(f"f={f:g}".rjust(13) for f in CHAOS_FRACS)
    print(hdr); print("-" * len(hdr))
    for name, rob, sob, gob, gsel in rows:
        cells = " ".join(f"{100*gob[f]:>4.0f}%/{gsel[f]:>4.2f}x" for f in CHAOS_FRACS)
        print(f"{name:<24} {100*rob:>6.0f}% {100*sob:>7.1f}% | {cells}")
    print("-" * len(hdr))
    msel = {f: np.nanmean([gsel[f] for *_, gsel in rows]) for f in CHAOS_FRACS}
    mob = {f: 100*np.mean([gob[f] for *_, gob, _ in rows]) for f in CHAOS_FRACS}
    print(f"{'MEAN':<24} {100*np.mean([r[1] for r in rows]):>6.0f}% {100*np.mean([r[2] for r in rows]):>7.1f}% | "
          + " ".join(f"{mob[f]:>4.0f}%/{msel[f]:>4.2f}x" for f in CHAOS_FRACS))
    print(f"\nSelectivity >1 => off-beat accents land on real audio events (event-driven), not a smear.")
    print(f"Exported gated@frac={args.export_frac} vs smear to {args.out_dir}/ — playtest: does gated feel musical?")
    if args.install:
        from src.utils.sm_install import install_to_stepmania
        for d in install_to_stepmania(args.out_dir): print("installed:", d)


if __name__ == '__main__':
    main()
