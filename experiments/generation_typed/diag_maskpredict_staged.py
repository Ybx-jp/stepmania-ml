#!/usr/bin/env python3
"""
Phase-3 generation prototype (notes/phase3_generative_design.md): the core paradigm test. AR onset EXPLODES;
frozen-context refinement is stable but can't bootstrap from v4's bad C0. A JOINT generative model
(mask-and-predict: mask cells, predict from the rest, iterate) generates FROM SCRATCH on the learned chart
manifold -- no AR self-feedback loop, no dependence on a bad first pass. Does it produce STABLE, real-shaped
onset placement?

Onset-only prototype. Model predicts onset[t] from audio[t] (non-causal) + a PARTIAL onset context
(revealed frames = real onset, masked = 0) + a mask-indicator channel. Train with RANDOM masking (predict
masked onsets). Generate by ITERATIVE confidence-based unmasking from all-masked (MaskGIT-style). Measure the
generated chart vs real: density (stable? not exploded/collapsed), phase distribution, run-length; + the
teacher-forced AUC sanity (does it learn placement given partial real context).

Reads:
  generated density ~ real, phase dist ~ real, run-length ~ real, NO explosion/collapse -> joint generation
    is STABLE + real-shaped FROM SCRATCH (solves AR + refinement-bootstrap) -> paradigm viable -> next: panels
    + critic eval.
  exploded / collapsed / phase off -> joint gen doesn't fix it either.

  python experiments/generation_typed/diag_maskpredict_proto.py --epochs 8
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.models import LateFusionClassifier
from src.generation.sm_writer import charts_to_sm
from src.generation.playtest_export import enforce_playability
from src.data.dataset import DIFFICULTY_NAMES
import re, shutil

AD = 42


def safe_name(s):
    return re.sub(r'[^\w\-]+', '_', str(s)).strip('_')[:50] or 'song'
V4 = "checkpoints/gen_highres_v4/best_val.pt"; CRITIC = "checkpoints/realism_critic/best_val.pt"


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


@torch.no_grad()
def critic_score(critic, a23, chart, device):
    a = torch.from_numpy(a23).unsqueeze(0).to(device); c = torch.from_numpy(chart).unsqueeze(0).to(device)
    lo = critic(a, c, torch.ones(1, a.shape[1], device=device))
    if isinstance(lo, dict): lo = lo['logits']
    return float(torch.softmax(lo, 1)[0, 1])


def shuffle16(binary, rng):
    T = binary.shape[0]; t = np.arange(T); i16 = np.where((t % 4 == 1) | (t % 4 == 3))[0]
    out = binary.copy(); rows = [binary[k].copy() for k in i16 if binary[k].any()]; out[i16] = 0.0
    if rows:
        for r, k in zip(rows, rng.choice(i16, size=len(rows), replace=False)): out[k] = r
    return out


class MaskPredict(nn.Module):
    """audio (non-causal) + partial onset context [revealed_onset, is_revealed] (non-causal) -> onset logit."""
    def __init__(self, d=96):
        super().__init__()
        self.audio = nn.Sequential(nn.Conv1d(AD, d, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(d, d, 3, padding=2, dilation=2), nn.ReLU(),
                                   nn.Conv1d(d, d, 3, padding=4, dilation=4), nn.ReLU(),
                                   nn.Conv1d(d, d, 3, padding=8, dilation=8), nn.ReLU())
        self.ctx = nn.Sequential(nn.Conv1d(2, d, 7, padding=3), nn.ReLU(),
                                 nn.Conv1d(d, d, 7, padding=6, dilation=2), nn.ReLU(),
                                 nn.Conv1d(d, d, 7, padding=12, dilation=4), nn.ReLU())
        self.out = nn.Sequential(nn.Conv1d(2 * d, d, 1), nn.ReLU(), nn.Conv1d(d, 1, 1))

    def forward(self, audio, ctx2):  # audio (B,T,AD); ctx2 (B,T,2)=[revealed_onset, is_revealed]
        a = self.audio(audio.transpose(1, 2)); c = self.ctx(ctx2.transpose(1, 2))
        return self.out(torch.cat([a, c], 1)).squeeze(1)


def phase_frac(onset):
    t = np.arange(len(onset)); n = max(int(onset.sum()), 1)
    return (100 * onset[t % 4 == 0].sum() / n, 100 * onset[t % 4 == 2].sum() / n,
            100 * onset[(t % 4 == 1) | (t % 4 == 3)].sum() / n)


def run_mean(o):
    runs = []; c = 0
    for x in o:
        if x: c += 1
        elif c: runs.append(c); c = 0
    if c: runs.append(c)
    return float(np.mean(runs)) if runs else 0.0


def auc(s, l):
    l = l.astype(int); n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    return (r[l == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def collect(ds, cap, n, want_meta=False):
    out = []
    for i in range(min(len(ds.valid_samples), n)):
        s = ds[i]; meta = ds.valid_samples[i]; T = int(s['mask'].sum().item())
        nd = next((x for x in meta['chart'].note_data if x.difficulty_name == meta['difficulty_name']
                   and x.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd)); T = min(T, cap, typed.shape[0])
        if T < 128: continue
        a = s['audio'][:T, :AD].numpy().astype(np.float32); on = (typed[:T] != 0).any(1).astype(np.float32)
        if want_meta:
            out.append((a, on, to_binary(typed[:T]), int(meta['difficulty_class']),
                        meta['groove_radar'].to_vector().astype(np.float32)))
        else:
            out.append((a, on))
    return out


@torch.no_grad()
def gen_staged(m, audio, real_onset, device):
    """STAGED coarse-to-fine: place the BACKBONE (quarters then 8ths) first, then 16ths with the backbone as
    context. Oracle per-phase budget = the real chart's per-phase note counts (isolates PLACEMENT: density &
    phase match real by construction; the model picks WHICH frames). -> binary onset (T,)."""
    T = len(real_onset); A = torch.from_numpy(audio).unsqueeze(0).to(device)
    t = np.arange(T); bands = [t % 4 == 0, t % 4 == 2, (t % 4 == 1) | (t % 4 == 3)]   # quarter, 8th, 16th
    onset = np.zeros(T); revealed = np.zeros(T, bool)
    for band in bands:                                  # backbone (q, 8th) then 16ths, in order
        cnt = int((real_onset.astype(bool) & band).sum())
        ctx = np.stack([onset * revealed, revealed.astype(np.float32)], -1)[None].astype(np.float32)
        p = torch.sigmoid(m(A, torch.from_numpy(ctx).to(device)))[0].cpu().numpy()
        idx = np.where(band)[0]
        if cnt > 0:
            onset[idx[np.argsort(p[idx])[::-1][:cnt]]] = 1.0      # top-cnt by model confidence within band
        revealed[band] = True                                     # whole band now committed (note or no-note)
    return onset


def _gen_v4_panels(v4, A42, T, diff, R, onset_override, device):
    kw = dict(onset_override=onset_override, type_sample=True, type_temperature=0.4,
              pattern_sample=True, pattern_temperature=0.7, radar=R)
    enforce_playability(kw)   # MANDATORY pad-playability (hold_aware + no_jump/no_cross_during_hold)
    return pair_holds(v4.generate(A42, diff, lengths=torch.tensor([T], device=device), **kw)[0].cpu().numpy())


def export_playtest(m, v4, ds, args, device, rng):
    """STAGED mask-predict vs v4-baseline charts on chaotic Hard songs -> playable folders + install (A/B)."""
    # groove-validate: Hard songs with real 16ths, ranked by chaos
    cands = []
    for i in range(len(ds.valid_samples)):
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < 3: continue
        cands.append((float(meta['groove_radar'].chaos), i))
    cands.sort(reverse=True)
    out_stg = Path(args.export_dir); out_v4 = Path(args.export_dir + '_v4')
    out_stg.mkdir(parents=True, exist_ok=True); out_v4.mkdir(parents=True, exist_ok=True)
    print(f"\n=== exporting STAGED vs v4 (chaotic Hard) ===")
    n = 0
    for chaos, i in cands:
        if n >= args.export_songs: break
        meta = ds.valid_samples[i]; s = ds[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        nd = next((x for x in meta['chart'].note_data if x.difficulty_name == meta['difficulty_name']
                   and x.difficulty_value == meta['difficulty_value']), None)
        if nd is None or T < 256: continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        real_on = (orig != 0).any(1).astype(np.float32)
        rb = real_on.astype(bool); i16 = (np.arange(T) % 4 == 1) | (np.arange(T) % 4 == 3)
        if rb.sum() < 32 or (rb & i16).sum() / max(rb.sum(), 1) < 0.05:
            continue
        a = s['audio'][:T, :AD].numpy().astype(np.float32); A42 = torch.from_numpy(a).unsqueeze(0).to(device)
        diff = torch.tensor([meta['difficulty_class']], device=device)
        R = torch.from_numpy(meta['groove_radar'].to_vector().astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            stg_on = gen_staged(m, a, real_on, device)
            stg = _gen_v4_panels(v4, A42, T, diff, R, torch.from_numpy(stg_on).bool().unsqueeze(0).to(device), device)
            p = torch.sigmoid(v4.onset_logits(v4.encode_audio(A42), diff, radar=R))[0].cpu().numpy()
            v4_on = (p > np.quantile(p, 1 - float(real_on.mean()))).astype(np.float32)
            v4c = _gen_v4_panels(v4, A42, T, diff, R, torch.from_numpy(v4_on).bool().unsqueeze(0).to(device), device)
        chart_obj = meta['chart']; music = os.path.basename(meta['audio_file'])
        title = chart_obj.title or Path(meta['chart_file']).stem; dname = DIFFICULTY_NAMES[meta['difficulty_class']]
        for genc, root, tag in [(stg, out_stg, 'staged'), (v4c, out_v4, 'v4')]:
            folder = root / f"{n:02d}_{safe_name(title)}"; folder.mkdir(parents=True, exist_ok=True)
            if os.path.exists(meta['audio_file']):
                try: shutil.copy2(meta['audio_file'], folder / music)
                except Exception: pass
            sm = charts_to_sm(charts=[
                {"chart": genc, "difficulty_name": "Challenge", "difficulty_value": nd.difficulty_value, "author": tag},
                {"chart": orig, "difficulty_name": dname, "difficulty_value": nd.difficulty_value, "author": "original"}],
                bpm=float(chart_obj.bpm), title=f"{title} ({tag})", artist=chart_obj.artist or "",
                music=music, offset=float(chart_obj.offset), typed=True)
            (folder / "chart.sm").write_text(sm, encoding="utf-8")
        s16 = lambda o: 100 * (o.astype(bool) & ((np.arange(T) % 4 == 1) | (np.arange(T) % 4 == 3))).sum() / max(o.sum(), 1)
        print(f"  {safe_name(title)[:28]:<30} chaos {chaos:5.1f}  16th%: real {s16(real_on):.0f} staged {s16(stg_on):.0f} v4 {s16(v4_on):.0f}", flush=True)
        n += 1
    from src.utils.sm_install import install_to_stepmania
    install_to_stepmania(str(out_stg), None); install_to_stepmania(str(out_v4), None)
    print(f"  installed {n} songs: {out_stg} (staged) + {out_v4} (v4 baseline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export_dir', default=None); ap.add_argument('--export_songs', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--max_len', type=int, default=768)
    ap.add_argument('--max_train', type=int, default=1500); ap.add_argument('--gen_songs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--steps', type=int, default=10)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    tr_ds, va_ds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                      max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    print("collecting...", flush=True)
    train = collect(tr_ds, args.max_len, args.max_train)
    gen = collect(va_ds, args.max_len, args.gen_songs * 2, want_meta=True)[:args.gen_songs]
    print(f"train={len(train)} gen={len(gen)} songs", flush=True)
    rng = np.random.default_rng(42)
    pr = np.mean([y.mean() for _, y in train]); pw = torch.tensor((1 - pr) / pr, device=device)
    m = MaskPredict().to(device); opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    ckc = torch.load(CRITIC, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ckc['config']).to(device); critic.load_state_dict(ckc['model_state_dict']); critic.eval()
    v4 = LayeredTypedChartGenerator(audio_dim=42, d_model=128, num_layers=4, onset_layers=2).to(device)
    v4.load_state_dict(torch.load(V4, map_location=device)['model_state_dict']); v4.eval()

    def batch():
        idx = rng.choice(len(train), min(args.bs, len(train)), replace=False)
        ch = [train[j] for j in idx]; T = max(len(a) for a, _ in ch); B = len(ch)
        X = np.zeros((B, T, AD), np.float32); Y = np.zeros((B, T), np.float32)
        C = np.zeros((B, T, 2), np.float32); LM = np.zeros((B, T), bool)   # loss mask = masked & valid
        for b, (a, y) in enumerate(ch):
            t = len(y); X[b, :t] = a; Y[b, :t] = y
            r = rng.uniform(0.15, 1.0); rev = (rng.random(t) > r)          # reveal (1-r) fraction
            C[b, :t, 0] = y * rev; C[b, :t, 1] = rev.astype(np.float32)
            LM[b, :t] = ~rev                                               # predict the MASKED frames
        return (torch.from_numpy(X).to(device), torch.from_numpy(C).to(device),
                torch.from_numpy(Y).to(device), torch.from_numpy(LM).to(device))
    nb = (len(train) + args.bs - 1) // args.bs
    for ep in range(args.epochs):
        m.train()
        for _ in range(nb):
            X, C, Y, LM = batch()
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(m(X, C)[LM], Y[LM], pos_weight=pw)
            loss.backward(); opt.step()

    m.eval()
    if args.export_dir:
        export_playtest(m, v4, va_ds, args, device, rng); return

    # STAGED onset gen -> v4 panels (onset_override) -> taste critic. Compare REAL / shuf16 / v4-gen / staged.
    real_s, shuf_s, v4_s, st_s = [], [], [], []; st_ph, st_d = [], []
    real_ph = np.mean([phase_frac(y) for _, y, *_ in gen], 0)
    for a, y, rb, dcl, rad in gen:
        T = len(y); a23 = a[:, :23]
        real_s.append(critic_score(critic, a23, rb, device))
        shuf_s.append(critic_score(critic, a23, shuffle16(rb, rng), device))
        diff = torch.tensor([dcl], device=device); R = torch.from_numpy(rad).unsqueeze(0).to(device)
        A42 = torch.from_numpy(a).unsqueeze(0).to(device); d = float(y.mean())
        with torch.no_grad():
            p = torch.sigmoid(v4.onset_logits(v4.encode_audio(A42), diff, radar=R))[0].cpu().numpy()
            tau = float(np.quantile(p, 1 - d))
            v4g = v4.generate(A42, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                              type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                              pattern_temperature=0.7, no_jump_during_hold=True, radar=R)[0].cpu().numpy()
            v4_s.append(critic_score(critic, a23, to_binary(pair_holds(v4g)), device))
            # STAGED: mask-predict onset (oracle per-phase budget) -> v4 panels via onset_override
            stg = gen_staged(m, a, y, device)
            ov = torch.from_numpy(stg).bool().unsqueeze(0).to(device)
            sg = v4.generate(A42, diff, lengths=torch.tensor([T], device=device), onset_override=ov,
                             type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                             pattern_temperature=0.7, no_jump_during_hold=True, radar=R)[0].cpu().numpy()
        st_s.append(critic_score(critic, a23, to_binary(pair_holds(sg)), device))
        st_ph.append(phase_frac(stg)); st_d.append(stg.mean())
    sp = np.mean(st_ph, 0)
    print(f"\n=== STAGED mask-predict prototype ({len(gen)} songs) ===")
    print(f"  staged onset: density {np.mean(st_d):.3f} (real ~{np.mean([y.mean() for _,y,*_ in gen]):.3f}); "
          f"phase {sp[0]:.0f}/{sp[1]:.0f}/{sp[2]:.0f}% (real {real_ph[0]:.0f}/{real_ph[1]:.0f}/{real_ph[2]:.0f}%)")
    print(f"  taste-critic P(real)  [the placement verdict]:")
    print(f"    REAL              {np.mean(real_s):.3f}")
    print(f"    STAGED mask-pred  {np.mean(st_s):.3f}   (REAL - staged = {np.mean(real_s)-np.mean(st_s):+.3f})")
    print(f"    v4-gen (awkward)  {np.mean(v4_s):.3f}")
    print(f"    shuf16 (broken)   {np.mean(shuf_s):.3f}")
    print(f"  per-song staged>v4 {np.mean(np.array(st_s)>np.array(v4_s))*100:.0f}%  "
          f"staged>shuf16 {np.mean(np.array(st_s)>np.array(shuf_s))*100:.0f}%")
    print(f"\n  staged >> v4-gen and toward REAL -> joint+staged generation places GOOD 16ths -> paradigm WORKS.")
    print(f"  staged ~ v4-gen -> staged order didn't fix placement quality.")


if __name__ == '__main__':
    main()
