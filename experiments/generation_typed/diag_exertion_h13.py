#!/usr/bin/env python3
"""
H13 EXERTION probe — does the new-default generator over-produce physically-brutal FAST JACKS?
(playtest: "a 6-note jack on 1/16s is crazy"; notes/playtest_log.md 2026-06-22 quota-free entry.)

A "jack" = consecutive presses on the SAME single panel. At 16th spacing (adjacent onset frames) that is
ONE foot hammering at 16th speed = the exertion failure. Real charters break fast runs by ALTERNATING
panels (feet alternate). Two candidate mechanisms, and they need DIFFERENT fixes:
  (a) ONSET places long 16th RUNS real charts wouldn't  -> cap run length at the onset stage.
  (b) PATTERN head assigns one panel across a fast run instead of spreading feet -> exertion-aware pattern decode.

Experiment-design discipline (.claude/skills/experiment-design): measure what REAL Hard charts do FIRST,
and ISOLATE the variable. We compare, on the SAME Hard chaotic songs, three charts:
  REAL        — the human chart (reference; rule 5).
  DEPLOYMENT  — staged quota-free onset -> v4 panels (EXACTLY what the user played; deployment decode).
  CONTROL     — REAL onset -> v4 panels (same notes as real; ONLY the panel-assignment differs).
Attribution:
  DEPLOYMENT jacks >> REAL                      -> over-exertion is real, not imagined.
  CONTROL also jacks >> REAL                    -> the PATTERN HEAD is the culprit (mechanism b; decode-shared).
  CONTROL ~ REAL but DEPLOYMENT >> REAL         -> the staged ONSET places long fast runs (mechanism a).

Persists the staged onset model (MaskPredict) to checkpoints/gen_staged_onset/ so the new-default pipeline
is reproducible (train-or-load).

  python experiments/generation_typed/diag_exertion_h13.py --songs 12
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
import experiments.generation_typed.diag_maskpredict_staged as S  # reuse the staged pipeline

AD = S.AD
CKPT = PROJECT_ROOT / "checkpoints/gen_staged_onset/maskpredict.pt"


# ----- exertion metrics (computed on a typed chart's binary onset grid (T,4)) -----
def _onset_single_pid(binary):
    on = binary.any(1); nact = binary.sum(1); single = nact == 1
    pid = np.where(single, binary.argmax(1), -1)
    return on, single, pid


def jack_stats(binary):
    """FAST-jack metrics. A fast same-panel run = consecutive (16th-adjacent) onset frames, each a SINGLE
    panel, all the SAME panel. Returns (jack_pair_rate, runs>=4 per 1k onsets, max fast run, n_onsets)."""
    on, single, pid = _onset_single_pid(binary)
    T = len(on); n_on = int(on.sum())
    adj_single = jack_pair = 0
    for t in range(T - 1):
        if on[t] and single[t] and on[t + 1] and single[t + 1]:
            adj_single += 1
            if pid[t] == pid[t + 1]:
                jack_pair += 1
    runlen = []; cur = 0; curp = -2
    for t in range(T):
        if on[t] and single[t]:
            if cur > 0 and curp == pid[t]: cur += 1
            else:
                if cur >= 1: runlen.append(cur)
                cur = 1; curp = pid[t]
        else:
            if cur >= 1: runlen.append(cur)
            cur = 0; curp = -2
    if cur >= 1: runlen.append(cur)
    rate = jack_pair / adj_single if adj_single else float('nan')
    brutal = sum(1 for L in runlen if L >= 4)
    per1k = 1000 * brutal / max(n_on, 1)
    mx = max(runlen) if runlen else 0
    return rate, per1k, mx, n_on


def onset_run_stats(binary):
    """ANY-panel consecutive-onset runs (mechanism a: does the ONSET place long 16th runs?).
    Returns (frac of onsets in runs>=4, max onset run)."""
    on = binary.any(1); runs = []; c = 0
    for x in on:
        if x: c += 1
        elif c: runs.append(c); c = 0
    if c: runs.append(c)
    n_on = max(int(on.sum()), 1)
    in_long = sum(L for L in runs if L >= 4)
    return in_long / n_on, (max(runs) if runs else 0)


def train_or_load_maskpredict(train, device):
    m = S.MaskPredict().to(device)
    if CKPT.exists():
        m.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict']); m.eval()
        print(f"loaded MaskPredict from {CKPT}", flush=True); return m
    print("training MaskPredict (8 epochs) — replicates the staged-script training, then persists...", flush=True)
    rng = np.random.default_rng(42)
    pr = np.mean([y.mean() for _, y in train]); pw = torch.tensor((1 - pr) / pr, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); bs = 16

    def batch():
        idx = rng.choice(len(train), min(bs, len(train)), replace=False)
        ch = [train[j] for j in idx]; T = max(len(a) for a, _ in ch); B = len(ch)
        X = np.zeros((B, T, AD), np.float32); Y = np.zeros((B, T), np.float32)
        C = np.zeros((B, T, 2), np.float32); LM = np.zeros((B, T), bool)
        for b, (a, y) in enumerate(ch):
            t = len(y); X[b, :t] = a; Y[b, :t] = y
            r = rng.uniform(0.15, 1.0); rev = (rng.random(t) > r)
            C[b, :t, 0] = y * rev; C[b, :t, 1] = rev.astype(np.float32); LM[b, :t] = ~rev
        return (torch.from_numpy(X).to(device), torch.from_numpy(C).to(device),
                torch.from_numpy(Y).to(device), torch.from_numpy(LM).to(device))
    nb = (len(train) + bs - 1) // bs
    for ep in range(8):
        m.train()
        for _ in range(nb):
            X, C, Y, LM = batch(); opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(m(X, C)[LM], Y[LM], pos_weight=pw)
            loss.backward(); opt.step()
    m.eval(); CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': m.state_dict(), 'arch': 'MaskPredict', 'd': 96, 'audio_dim': AD}, CKPT)
    print(f"saved MaskPredict -> {CKPT}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--songs', type=int, default=12); ap.add_argument('--max_len', type=int, default=768)
    ap.add_argument('--max_train', type=int, default=1500); ap.add_argument('--gen_songs', type=int, default=40)
    ap.add_argument('--max_jack_run', type=int, default=None,
                    help='H13 exertion cap: max consecutive same-panel 16th-jack presses (apply to gen). None=off.')
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
    train = S.collect(tr_ds, args.max_len, args.max_train)
    gen_calib = S.collect(va_ds, args.max_len, args.gen_songs * 2, want_meta=True)[:args.gen_songs]
    m = train_or_load_maskpredict(train, device)

    # quota-free per-phase thresholds, calibrated on the Hard difficulty bucket (the new default)
    hard = [g for g in gen_calib if g[3] >= 3]
    if len(hard) < 6: hard = sorted(gen_calib, key=lambda g: -g[3])[:max(8, len(gen_calib) // 2)]
    threshs = S.calibrate_threshs(m, hard, device)
    print(f"QUOTA-FREE thresholds (Hard bucket, {len(hard)} songs): "
          f"q={threshs[0]:.3f} 8th={threshs[1]:.3f} 16th={threshs[2]:.3f}\n", flush=True)

    v4 = LayeredTypedChartGenerator(audio_dim=42, d_model=128, num_layers=4, onset_layers=2).to(device)
    v4.load_state_dict(torch.load(S.V4, map_location=device)['model_state_dict']); v4.eval()

    # groove-validate: Hard songs with real 16ths, ranked by chaos (same selection as the playtest export)
    cands = sorted(((float(va_ds.valid_samples[i]['groove_radar'].chaos), i)
                    for i in range(len(va_ds.valid_samples))
                    if va_ds.valid_samples[i]['difficulty_class'] >= 3), reverse=True)

    agg = {k: [] for k in ['real', 'deploy', 'control']}            # jack metrics rows: (rate, per1k, max)
    onruns = {k: [] for k in ['real', 'staged', 'v4free']}          # onset-run rows: (frac_long, max)
    print(f"{'song':<26} {'':>6} | jack-pair-rate  R/Dep/Ctl | runs>=4 /1k R/Dep/Ctl | maxrun R/Dep/Ctl")
    n = 0
    for chaos, i in cands:
        if n >= args.songs: break
        meta = va_ds.valid_samples[i]; s = va_ds[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        nd = next((x for x in meta['chart'].note_data if x.difficulty_name == meta['difficulty_name']
                   and x.difficulty_value == meta['difficulty_value']), None)
        if nd is None or T < 256: continue
        orig = np.asarray(va_ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        real_bin = S.to_binary(orig)
        real_on = (orig != 0).any(1).astype(np.float32)
        rb = real_on.astype(bool); i16 = (np.arange(T) % 4 == 1) | (np.arange(T) % 4 == 3)
        if rb.sum() < 32 or (rb & i16).sum() / max(rb.sum(), 1) < 0.05:
            continue
        a = s['audio'][:T, :AD].numpy().astype(np.float32); A42 = torch.from_numpy(a).unsqueeze(0).to(device)
        diff = torch.tensor([meta['difficulty_class']], device=device)
        R = torch.from_numpy(meta['groove_radar'].to_vector().astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            staged_on = S.gen_staged_thresh(m, a, T, threshs, device)
            mjr = args.max_jack_run
            ovr = None if (mjr and mjr >= 1) else "H13 offline diagnostic: measuring the uncapped baseline"
            dep = S._gen_v4_panels(v4, A42, T, diff, R, torch.from_numpy(staged_on).bool().unsqueeze(0).to(device), device, max_jack_run=mjr, override_reason=ovr)
            ctl = S._gen_v4_panels(v4, A42, T, diff, R, torch.from_numpy(real_on).bool().unsqueeze(0).to(device), device, max_jack_run=mjr, override_reason=ovr)
            p = torch.sigmoid(v4.onset_logits(v4.encode_audio(A42), diff, radar=R))[0].cpu().numpy()
            v4_on = (p > np.quantile(p, 1 - float(real_on.mean()))).astype(np.float32)
        charts = {'real': real_bin, 'deploy': S.to_binary(dep), 'control': S.to_binary(ctl)}
        js = {k: jack_stats(v) for k, v in charts.items()}
        for k in agg: agg[k].append(js[k][:3])
        onruns['real'].append(onset_run_stats(real_bin))
        onruns['staged'].append(onset_run_stats(staged_on[:, None].astype(bool)))   # (T,1): any(1)==onset
        onruns['v4free'].append(onset_run_stats(v4_on[:, None].astype(bool)))
        title = S.safe_name(meta['chart'].title or Path(meta['chart_file']).stem)[:24]
        print(f"{title:<26} c{chaos:4.1f} | "
              f"{js['real'][0]:.2f}/{js['deploy'][0]:.2f}/{js['control'][0]:.2f}        | "
              f"{js['real'][1]:4.1f}/{js['deploy'][1]:4.1f}/{js['control'][1]:4.1f}      | "
              f"{js['real'][2]:2d}/{js['deploy'][2]:2d}/{js['control'][2]:2d}", flush=True)
        n += 1

    def mean(rows, c): return np.nanmean([r[c] for r in rows])
    print(f"\n=== H13 EXERTION SUMMARY ({n} Hard chaotic songs) ===")
    print(f"  {'':<12} jack-pair-rate   runs>=4/1k-onset   max-fast-jack-run")
    for k, lab in [('real', 'REAL'), ('deploy', 'DEPLOYMENT'), ('control', 'CONTROL(real-onset)')]:
        print(f"  {lab:<20} {mean(agg[k],0):.3f}            {mean(agg[k],1):5.1f}             {mean(agg[k],2):4.1f}")
    print(f"\n  onset RUN-length (mechanism a — does the ONSET place long 16th runs?):")
    print(f"  {'':<12} frac onsets in runs>=4    max onset-run")
    for k, lab in [('real', 'REAL'), ('staged', 'STAGED onset'), ('v4free', 'v4 free onset')]:
        print(f"  {lab:<20} {mean(onruns[k],0):.3f}                  {mean(onruns[k],1):4.1f}")
    print("\n  READ: DEPLOYMENT>>REAL = over-exertion real. CONTROL>>REAL = pattern head (mechanism b).")
    print("        CONTROL~REAL but DEPLOYMENT>>REAL = staged onset long runs (mechanism a).")


if __name__ == '__main__':
    main()
