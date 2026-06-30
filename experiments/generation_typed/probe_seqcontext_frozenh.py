#!/usr/bin/env python3
"""M1a PROBE: is the 16th-placement signal already decodable from the FROZEN deployed DECODER's hidden state h?
(2026-06-29; lineage seq-onset-arc.md, fork (A) build-sizing)

CONTEXT. The seq-onset wall is CLOSED NEGATIVE 4 ways: fine 16th PLACEMENT is a chart-sequence PRIOR, not in the
audio. The only remaining path to the 0.87 teacher-forced ceiling is a RETRAIN with a causal-AR onset head. Before
spending that retrain we size its CHEAPEST variant: a tiny onset head bolted onto the EXISTING decoder's per-frame
hidden state `h` (typed_model.py:635 `h = self._decoder_step_cached(...)`). `h[t]` already fuses audio (cross-attn
to memory) + the CAUSAL note history (self-attn over states[<t]) — exactly the probe's `both` information set, but
routed through the FROZEN deployed decoder (trained for the PATTERN head, "which panel given a note is placed").

THE QUESTION (representation sufficiency, NOT drift). Does the frozen decoder's h preserve the placement signal the
raw-note CNN extracts (the 0.871 `both_real` ceiling)? Teacher-forced over REAL typed states:
  audio     : Probe('audio') on the 42-dim audio (= the deployed onset head's info)            -> FLOOR  (~0.656)
  both_real : Probe('both')  on audio + raw causal real notes (the 06-28 ceiling)              -> CEILING (~0.871)  POSITIVE CONTROL
  frozen_h  : a matched 1x1 read-out head on the FROZEN decoder's h[t] (teacher-forced)        <- THE MEASUREMENT
All three predict onset[t] and are scored by 16th-localization AUC on held-out Hard val songs; h[t] and `both` use
the SAME causal info (notes[<t] + audio), so the comparison isolates "did the frozen learned compression keep it".

READ (build-sizing, not a wall re-test):
  frozen_h ~ both_real (0.87)  -> the frozen decoder ALREADY exposes placement -> M1b can be a tiny head on frozen h
                                  (cheapest build); the readout layer (typed_model.py:635) is the only new weight.
  frozen_h ~ audio (0.66)      -> the decoder compressed placement AWAY (kept only what the pattern head needs) ->
                                  M1b MUST add a dedicated causal note branch / unfreeze (the real Phase-2.6 build).
BOUNDARY (experiment-design Rule 9/10): this settles REPRESENTATION (is the signal in h), NOT DRIFT. A high number
does NOT prove the causal-AR head works at gen time — at gen time the head reads its OWN generated notes, and the
explosion (diag_ar_stability: density 0.73 vs real 0.18) is the separate, binding risk. h here is teacher-forced on
REAL notes (the upper bound on what a frozen-head readout could ever see). The drift gate is M1b.

  conda run -n stepmania-chart-gen python experiments/generation_typed/probe_seqcontext_frozenh.py --max_train 800
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
from src.generation.typed_model import LayeredTypedChartGenerator, _causal_mask
from diag_seqcontext_probe import Probe, auc, causal_conv   # reuse the EXACT 06-22 probe net + AUC (fair controls)
AD, NP = 42, 4
DMODEL = 128                                          # the deployed gen_motif_full_fixed width (probe_seqcontext_c0.py:148)
CKPT = "checkpoints/gen_motif_full_fixed/best_val.pt"


class HRead(nn.Module):
    """Read-out head on the frozen decoder's hidden state h (B,T,d_model). Capacity MATCHES Probe.out (two 1x1
    convs) so the frozen_h vs both_real comparison isolates the representation, not head capacity (Rule 11). 1x1
    only -> per-frame readout: h[t] already aggregated causal history via attention, so no extra temporal mixing."""
    def __init__(self, din, d=64):
        super().__init__()
        self.out = nn.Sequential(nn.Conv1d(din, d, 1), nn.ReLU(), nn.Conv1d(d, 1, 1))

    def forward(self, h):                              # h: (B,T,din)
        return self.out(h.transpose(1, 2)).squeeze(1)  # (B,T)


class HReadConv(nn.Module):
    """Capacity-matched control for HRead: the SAME 4-layer dilated CAUSAL conv stack as Probe's note branch
    (diag_seqcontext_probe.Probe.s1..s4), but over the frozen h instead of raw notes. Disambiguates 'the decoder
    compressed placement away' (stays ~0.76) from 'the signal IS in h but my 1x1 readout couldn't mix it across
    frames' (jumps toward 0.89). CAUSAL convs keep the same strictly-past discipline as h[t] (no future leak)."""
    def __init__(self, din, d=64):
        super().__init__()
        self.c1 = causal_conv(din, d, 3, 1); self.c2 = causal_conv(d, d, 3, 2)
        self.c3 = causal_conv(d, d, 3, 4); self.c4 = causal_conv(d, d, 3, 8)
        self.out = nn.Sequential(nn.Conv1d(d, d, 1), nn.ReLU(), nn.Conv1d(d, 1, 1))

    def forward(self, h):                              # h: (B,T,din)
        x = h.transpose(1, 2)
        for c in (self.c1, self.c2, self.c3, self.c4):
            x = torch.relu(c(x))
        return self.out(x).squeeze(1)                  # (B,T)


def extract_typed(ds, idxs, cap_len, workers=4, hard_only=False):
    """PARALLEL feature extraction keeping TYPED states (symbol ids, not binarized) + difficulty — the decoder needs
    typed states for _state_emb and difficulty for _cond. Returns dicts: audio (T,AD), typed (T,4 int), diff, T."""
    loader = DataLoader(Subset(ds, list(idxs)), batch_size=8, num_workers=workers, shuffle=False,
                        collate_fn=lambda b: b)
    out = []
    for bi, batch in enumerate(loader):
        for it in batch:
            diff = int(it['difficulty'].item())
            if hard_only and diff != 3:
                continue
            T = int(it['mask'].sum().item())
            if T < 128:
                continue
            T = min(T, cap_len)
            audio = it['audio'][:T, :AD].numpy().astype(np.float32)
            typed = it['chart'][:T].numpy().astype(np.int64)            # (T,4) symbol ids (0=empty)
            out.append(dict(audio=audio, typed=typed, diff=diff, T=T))
        if bi % 5 == 0:
            print(f"    extracted {len(out)} songs ({(bi + 1) * 8} scanned)...", flush=True)
    return out


def load_or_extract(ds, max_songs, cap_len, cache, hard_only, workers=4):
    if cache.exists():
        d = np.load(cache, allow_pickle=True); print(f"loaded cache ({len(d['data'])} songs): {cache}", flush=True)
        return list(d['data'])
    idxs = list(range(min(len(ds.valid_samples), max_songs)))
    songs = extract_typed(ds, idxs, cap_len, workers, hard_only)
    np.savez(cache, data=np.array(songs, dtype=object)); print(f"cached -> {cache} ({len(songs)} songs)", flush=True)
    return songs


@torch.no_grad()
def decode_hidden(model, audio_t, typed_t, diff_t, mask_t):
    """Mirror LayeredTypedChartGenerator._decode (typed_model.py:340) up to the decoder output h, NATIVE mode
    (radar/style/motif/figure=None, as gen_c0). h[t] sees states[<t] (causal) + full audio (cross-attn). eval-mode
    dropout is identity. Returns h (B,T,d_model)."""
    memory = model.encode_audio(audio_t)
    cond = model._cond(diff_t, None, None, None, None)                  # (B,1,d)
    tgt = model.dropout(model.pos_encoding(model._decoder_input(typed_t)) + cond)
    causal = _causal_mask(typed_t.shape[1], typed_t.device)
    pad = (~mask_t.bool()) if mask_t is not None else None
    return model.decoder(tgt=tgt, memory=memory, tgt_mask=causal,
                         tgt_key_padding_mask=pad, memory_key_padding_mask=pad)


@torch.no_grad()
def precompute_h(model, songs, device, cap_len, bs=8):
    """One frozen teacher-forced forward per song (NO AR loop -> cheap); store h (T,d_model) on each dict."""
    model.eval()
    for i in range(0, len(songs), bs):
        chunk = songs[i:i + bs]; T = max(s['T'] for s in chunk); B = len(chunk)
        X = np.zeros((B, T, AD), np.float32); St = np.zeros((B, T, NP), np.int64); M = np.zeros((B, T), bool)
        for b, s in enumerate(chunk):
            X[b, :s['T']] = s['audio']; St[b, :s['T']] = s['typed']; M[b, :s['T']] = True
        diff = torch.tensor([s['diff'] for s in chunk], device=device)
        h = decode_hidden(model, torch.from_numpy(X).to(device), torch.from_numpy(St).to(device),
                          diff, torch.from_numpy(M).to(device)).cpu().numpy().astype(np.float32)
        for b, s in enumerate(chunk):
            s['h'] = h[b, :s['T']]                                      # (T,d_model)
        print(f"    h for {min(i + bs, len(songs))}/{len(songs)}", flush=True) if (i // bs) % 10 == 0 else None
    return songs


def batches(data, bs, rng, shuffle, device):
    idx = np.arange(len(data))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i + bs]]; T = max(s['T'] for s in chunk); B = len(chunk)
        X = np.zeros((B, T, AD), np.float32); N = np.zeros((B, T, NP), np.float32)
        H = np.zeros((B, T, DMODEL), np.float32); M = np.zeros((B, T), bool)
        for b, s in enumerate(chunk):
            X[b, :s['T']] = s['audio']; N[b, :s['T']] = (s['typed'] != 0).astype(np.float32)
            H[b, :s['T']] = s['h']; M[b, :s['T']] = True
        Y = (N.sum(-1) > 0).astype(np.float32)                          # TARGET = real onset
        Nprev = np.zeros_like(N); Nprev[:, 1:] = N[:, :-1]              # CAUSAL strictly-past note context
        yield (torch.from_numpy(X).to(device), torch.from_numpy(Nprev).to(device),
               torch.from_numpy(H).to(device), torch.from_numpy(Y).to(device), torch.from_numpy(M).to(device))


def train_eval(kind, train, val, device, epochs, bs, lr, pw):
    set_seed(42)
    if kind == 'frozen_h':
        m = HRead(DMODEL).to(device)
        fwd = lambda X, Np, H: m(H)
    elif kind == 'frozen_h_conv':
        m = HReadConv(DMODEL).to(device)
        fwd = lambda X, Np, H: m(H)
    else:
        m = Probe(kind).to(device)
        fwd = lambda X, Np, H: m(X, Np)
    opt = torch.optim.Adam(m.parameters(), lr=lr); rng = np.random.default_rng(0)
    for _ in range(epochs):
        m.train()
        for X, Np, H, Y, M in batches(train, bs, rng, True, device):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(X, Np, H)[M], Y[M], pos_weight=pw)
            loss.backward(); opt.step()
    m.eval(); ps, ys, i16 = [], [], []
    with torch.no_grad():
        for X, Np, H, Y, M in batches(val, bs, rng, False, device):
            p = torch.sigmoid(fwd(X, Np, H)).cpu().numpy(); B, T = Y.shape; t = np.arange(T)
            mask16 = ((t % 4 == 1) | (t % 4 == 3))[None].repeat(B, 0); mm = M.cpu().numpy()
            ps.append(p[mm]); ys.append(Y.cpu().numpy()[mm]); i16.append(mask16[mm])
    P = np.concatenate(ps); Yv = np.concatenate(ys); I = np.concatenate(i16)
    return auc(P, Yv), auc(P[I], Yv[I])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_train', type=int, default=800, help='real train songs (powered control needs ~hundreds)')
    ap.add_argument('--max_val', type=int, default=400, help='val songs to scan (Hard-only kept; ~28 land)')
    ap.add_argument('--max_len', type=int, default=1024); ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=12); ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_cache = PROJECT_ROOT / "cache/seqctx_frozenh_train.npz"
    val_cache = PROJECT_ROOT / "cache/seqctx_frozenh_val.npz"

    def make_datasets():
        # Only built when a cache is MISSING (extraction needs the dataset). When both caches exist this whole
        # 4452-file re-parse is skipped (cache_dir=None re-parses every call) -> a re-run is h + heads only.
        cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
        tf, vf, _ = create_data_splits(cf, random_state=42)
        msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
        ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                       use_metric_phase=True, use_highres_onset=True))
        # cache_dir=None: the dataset index-cache is stale (786 vs 800) + identity-blind -> footgun ([[dataset-cache-footgun]])
        return create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                               max_sequence_length=msl, feature_extractor=ext, cache_dir=None)

    if train_cache.exists() and val_cache.exists():
        print("both feature caches present -> skipping dataset re-parse", flush=True)
        tr_ds = va_ds = None
    else:
        tr_ds, va_ds, _ = make_datasets()
    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=DMODEL, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()

    print("extracting REAL train (all difficulties, typed states)...", flush=True)
    train = load_or_extract(tr_ds, args.max_train, args.max_len, train_cache, hard_only=False)
    print("extracting REAL val (Hard only, typed states)...", flush=True)
    val = load_or_extract(va_ds, args.max_val, args.max_len, val_cache, hard_only=True)
    print(f"  computing frozen decoder h (teacher-forced) for train+val...", flush=True)
    precompute_h(model, train, device, args.max_len); precompute_h(model, val, device, args.max_len)

    posrate = np.mean([(s['typed'] != 0).any(-1).mean() for s in train]); pw = torch.tensor((1 - posrate) / posrate, device=device)
    print(f"\ntrain={len(train)} | eval={len(val)} (Hard) | onset-rate {posrate:.3f}\n", flush=True)
    print(f"  {'predictor':<12} {'onset-AUC':>10} {'16th-AUC':>10}", flush=True)
    res = {}
    for kind in ['audio', 'both', 'frozen_h', 'frozen_h_conv']:
        res[kind] = train_eval(kind, train, val, device, args.epochs, args.bs, args.lr, pw)
        label = 'both_real' if kind == 'both' else kind
        print(f"  {label:<14} {res[kind][0]:>10.3f} {res[kind][1]:>10.3f}", flush=True)

    a, cr, fh, fhc = res['audio'][1], res['both'][1], res['frozen_h'][1], res['frozen_h_conv'][1]
    gapden = max(cr - a, 1e-6)
    print(f"\n  16th-AUC: audio={a:.3f}  frozen_h(1x1)={fh:.3f}  frozen_h(conv)={fhc:.3f}  both_real={cr:.3f}", flush=True)
    print(f"  POSITIVE CONTROL (Rule 11): both_real must be >> audio (~0.87 vs ~0.66); else underpowered -> raise --max_train.", flush=True)
    if cr - a > 0.05:
        print(f"  gap recovered: frozen_h(1x1) {100 * (fh - a) / gapden:.0f}%  |  frozen_h(conv) {100 * (fhc - a) / gapden:.0f}%", flush=True)
        print(f"  conv >> 1x1 (toward both_real) -> signal IS in h, needs temporal mixing -> frozen-head M1b viable (cheap).", flush=True)
        print(f"  conv ~ 1x1 (both ~0.76)        -> decoder COMPRESSED ~half away -> M1b needs unfreeze / dedicated note branch.", flush=True)
    else:
        print(f"  !! both_real did not clear audio -> run UNDERPOWERED; do not interpret frozen_h. Raise --max_train.", flush=True)
    print(f"\n  BOUNDARY: settles REPRESENTATION (signal in h), NOT DRIFT. h is teacher-forced on REAL notes = the", flush=True)
    print(f"  upper bound a frozen-head readout could see; gen-time drift (diag_ar_stability) is the M1b gate.", flush=True)


if __name__ == '__main__':
    main()
