#!/usr/bin/env python3
"""PROBE: is the DEPLOYED generator's chart a good-enough C0 to recover the 16th-placement signal? (2026-06-28)
Re-opens the 06-22 seq-onset wall ("refinement can't bootstrap, audio-only is our only C0") — but that C0 was the
OLD audio-only gen_highres_v4 (scored 0.456, anti-correlated). Here C0 = the CURRENT deployed pipeline
(gen_motif_full_fixed + pattern_temp 1.0 + full governor). TARGET = REAL onset; only the note-CONTEXT source varies.

DESIGN (corrected — restore the positive control): the note-context branch needs LOTS of data (06-22 used ~1500
train songs). So TRAIN audio + both on the full REAL train split (free — real context needs no generation; this
restores audio~0.65 / both_real~0.94). Then EVAL the deployed-C0 val songs with the context source swapped at
TEST time: real (the in-domain ceiling) vs C0 (the deployed chart). This is also the realistic refiner setting:
train a refiner on real context, deploy it on generated context.
  audio    : audio-only (~0.65 floor)
  both/real: audio + REAL note-context (the ~0.94 ceiling)
  both/c0  : SAME trained net, eval-time context = the DEPLOYED chart  <-- the measurement
Read: both/c0 near both/real -> deployed C0 ~ real context -> refine-from-C0 can bootstrap (06-22 wall broken).
      both/c0 near audio (~ v4's 0.456) -> deployed C0 adds nothing over audio -> wall stands.
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml
from torch.utils.data import DataLoader, Subset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.playtest_export import enforce_playability
from diag_seqcontext_probe import Probe, auc          # reuse the EXACT 06-22 probe net + AUC
AD, NP = 42, 4
CKPT = "checkpoints/gen_motif_full_fixed/best_val.pt"

def extract_parallel(ds, idxs, cap_len, workers=4):
    """PARALLEL feature extraction via DataLoader workers (was the 1-core, 1-hour bottleneck).
    Returns per-song dict: idx, audio (T,AD), notes (T,4 bool->f32), diff, T. (chart from __getitem__ = typed.)"""
    loader = DataLoader(Subset(ds, list(idxs)), batch_size=8, num_workers=workers, shuffle=False,
                        collate_fn=lambda b: b)                 # identity collate -> list of sample dicts
    out = []
    for bi, batch in enumerate(loader):
        for it, gi in zip(batch, idxs[bi*8:bi*8+len(batch)]):
            T = int(it['mask'].sum().item())
            if T < 128: continue
            T = min(T, cap_len)
            audio = it['audio'][:T, :AD].numpy().astype(np.float32)
            notes = (it['chart'][:T].numpy() != 0).astype(np.float32)
            out.append(dict(idx=gi, audio=audio, notes=notes, diff=int(it['difficulty'].item()), T=T))
        print(f"    extracted {len(out)} songs ({(bi+1)*8} scanned)...", flush=True) if bi % 5 == 0 else None
    return out

def gen_c0(model, audio_t, diff_t, T, bpm, real_density, device):
    """Deployed canonical-config chart (the C0). Native mode: no radar/style/motif/figure, guidance 1.0."""
    with torch.no_grad():
        ol = model.onset_logits(model.encode_audio(audio_t), diff_t)[0]
        ph = torch.arange(ol.shape[0], device=device) % 4
        ol = ol + torch.where((ph == 1) | (ph == 3), torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))  # calib (0,1.0)
        p = torch.sigmoid(ol).cpu().numpy()
    tau = float(np.quantile(p, 1 - real_density)) if real_density > 0 else 0.5
    gk = dict(onset_threshold=tau, type_sample=True, type_temperature=0.4, pattern_sample=True,
              pattern_temperature=1.0, repetition_penalty=1.0, max_jack_run=2, fatigue_penalty=2.0,
              fatigue_free=6.0, stamina_ceiling=50.0, stamina_tau=8.0, stamina_scale=15.0, stamina_breathe=1.2,
              bpm=float(bpm), onset_phase_calib=(0.0, 1.0), guidance_scale=1.0)
    enforce_playability(gk, None)
    with torch.no_grad():
        gen = model.generate(audio_t, diff_t, lengths=torch.tensor([T], device=device), **gk)[0].cpu().numpy()
    return pair_holds(gen)[:T]

def collect_train(ds, max_songs, cap_len, cache, workers=4):
    """REAL audio + notes for the note-context branch, extracted in PARALLEL (DataLoader workers) and cached to
    our OWN npz (the dataset's index-cache was stale + not writing -> the 1-hour hang). No generation needed."""
    if cache.exists():
        d = np.load(cache, allow_pickle=True); print(f"loaded train cache ({len(d['data'])} songs): {cache}", flush=True); return list(d['data'])
    idxs = list(range(min(len(ds.valid_samples), max_songs)))
    songs = extract_parallel(ds, idxs, cap_len, workers)
    out = [(s['audio'], s['notes'], s['notes'].copy()) for s in songs]   # c0 slot = real (unused; train uses ctx='real')
    np.savez(cache, data=np.array(out, dtype=object)); print(f"cached train -> {cache}", flush=True)
    return out

def collect_c0(ds, model, cap_len, max_songs, device, cache):
    if cache.exists():
        d = np.load(cache, allow_pickle=True); print(f"loaded C0 cache: {cache} ({len(d['data'])} songs)", flush=True); return list(d['data'])
    out = []
    for i in range(min(len(ds.valid_samples), max_songs)):
        s = ds[i]; meta = ds.valid_samples[i]
        if meta['difficulty_class'] != 3: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))
        T = min(int(s['mask'].sum().item()), cap_len, typed.shape[0])
        if T < 128: continue
        audio_np = s['audio'][:T, :AD].numpy().astype(np.float32)
        rn = (typed[:T] != 0).astype(np.float32); real_d = float((rn.sum(-1) > 0).mean())
        c0 = gen_c0(model, torch.from_numpy(audio_np).unsqueeze(0).to(device), torch.tensor([3], device=device), T, meta['chart'].bpm, real_d, device)
        out.append((audio_np, rn, (c0[:T] != 0).astype(np.float32)))
        print(f"  gen {len(out):>3}: {str(getattr(meta['chart'],'title',''))[:30]:<30} T={T} real_d={real_d:.3f} c0_d={((c0[:T]!=0).any(1)).mean():.3f}", flush=True)
    np.savez(cache, data=np.array(out, dtype=object)); print(f"cached -> {cache}", flush=True)
    return out

def _batches(data, bs, ctx, rng, shuffle, device):
    idx = np.arange(len(data));  rng.shuffle(idx) if shuffle else None
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i+bs]]; T = max(len(a) for a,_,_ in chunk); B = len(chunk)
        X = np.zeros((B,T,AD),np.float32); Rn = np.zeros((B,T,NP),np.float32); Cn = np.zeros((B,T,NP),np.float32); M = np.zeros((B,T),bool)
        for b,(a,rn,cn) in enumerate(chunk):
            X[b,:len(a)]=a; Rn[b,:len(rn)]=rn; Cn[b,:len(cn)]=cn; M[b,:len(rn)]=True
        Y = (Rn.sum(-1) > 0).astype(np.float32)                      # TARGET = real onset (always)
        src = Rn if ctx == 'real' else Cn                            # CONTEXT source
        Nprev = np.zeros_like(src); Nprev[:,1:] = src[:,:-1]         # CAUSAL strictly-past
        yield (torch.from_numpy(X).to(device), torch.from_numpy(Nprev).to(device),
               torch.from_numpy(Y).to(device), torch.from_numpy(M).to(device))

def train_model(kind, train, device, epochs, bs, lr, pw):
    set_seed(42); m = Probe('audio' if kind == 'audio' else 'both').to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr); rng = np.random.default_rng(0)
    for _ in range(epochs):
        m.train()
        for X,Np,Y,M in _batches(train, bs, 'real', rng, True, device):
            opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(m(X,Np)[M], Y[M], pos_weight=pw)
            loss.backward(); opt.step()
    return m

def eval_model(m, val, ctx, device, bs):
    m.eval(); rng = np.random.default_rng(0); ps, ys, i16 = [], [], []
    with torch.no_grad():
        for X,Np,Y,M in _batches(val, bs, ctx, rng, False, device):
            p = torch.sigmoid(m(X,Np)).cpu().numpy(); B,T = Y.shape; t = np.arange(T)
            mask16 = ((t % 4 == 1)|(t % 4 == 3))[None].repeat(B,0); mm = M.cpu().numpy()
            ps.append(p[mm]); ys.append(Y.cpu().numpy()[mm]); i16.append(mask16[mm])
    P = np.concatenate(ps); Yv = np.concatenate(ys); I = np.concatenate(i16)
    return auc(P, Yv), auc(P[I], Yv[I])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_c0', type=int, default=120, help='cap on val songs to generate C0 for (eval set)')
    ap.add_argument('--max_train', type=int, default=800, help='real train songs for the note-context branch (parallel-extracted, cached)')
    ap.add_argument('--max_len', type=int, default=1024); ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=12); ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True, use_highres_onset=True))
    tr_ds, va_ds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                      max_sequence_length=msl, feature_extractor=ext, cache_dir=None)  # None = footgun-safe + the index-cache was stale/not-writing
    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()
    print("collecting REAL train (all difficulties) — PARALLEL extraction, own npz cache...", flush=True)
    train = collect_train(tr_ds, args.max_train, args.max_len, PROJECT_ROOT / "cache/seqctx_train_cache.npz", workers=4)
    print(f"  train songs = {len(train)}", flush=True)
    print("collecting deployed-C0 val songs (from cache if present)...", flush=True)
    val = collect_c0(va_ds, model, args.max_len, args.max_c0, device, PROJECT_ROOT / "cache/seqctx_c0_cache.npz")
    posrate = np.mean([(rn.sum(-1) > 0).mean() for _, rn, _ in train]); pw = torch.tensor((1 - posrate) / posrate, device=device)
    print(f"\ntrain={len(train)} (real) | eval={len(val)} (deployed-C0) | onset-rate {posrate:.3f}\n", flush=True)
    print(f"  {'predictor':<12} {'onset-AUC':>10} {'16th-AUC':>10}", flush=True)
    res = {}
    m_audio = train_model('audio', train, device, args.epochs, args.bs, args.lr, pw)
    res['audio'] = eval_model(m_audio, val, 'real', device, args.bs)        # ctx ignored for audio
    print(f"  {'audio':<12} {res['audio'][0]:>10.3f} {res['audio'][1]:>10.3f}", flush=True)
    m_both = train_model('both', train, device, args.epochs, args.bs, args.lr, pw)
    res['both_real'] = eval_model(m_both, val, 'real', device, args.bs)     # in-domain ceiling
    res['both_c0']   = eval_model(m_both, val, 'c0',   device, args.bs)     # the measurement
    print(f"  {'both/real':<12} {res['both_real'][0]:>10.3f} {res['both_real'][1]:>10.3f}", flush=True)
    print(f"  {'both/c0':<12} {res['both_c0'][0]:>10.3f} {res['both_c0'][1]:>10.3f}", flush=True)
    a, cr, c0 = res['audio'][1], res['both_real'][1], res['both_c0'][1]
    print(f"\n  16th-AUC: audio={a:.3f}  both/c0={c0:.3f}  both/real={cr:.3f}", flush=True)
    print(f"  POSITIVE CONTROL: both/real must be >> audio (~0.94 vs ~0.65); if not, run is underpowered.", flush=True)
    gap = (c0 - a) / max(cr - a, 1e-6)
    print(f"  both/c0 recovers {100*gap:.0f}% of the (real-context − audio) gap.", flush=True)
    print("  near both/real -> deployed C0 ~ real context -> refine-from-C0 can bootstrap (wall broken).", flush=True)
    print("  near audio (~v4's 0.456) -> deployed C0 adds nothing over audio -> 06-22 wall stands.", flush=True)

if __name__ == '__main__':
    main()
