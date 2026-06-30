#!/usr/bin/env python3
"""M1b DRIFT GATE: does a causal onset head reading the DEPLOYED decoder's h survive FREE-RUN rollout? (2026-06-29)
Lineage seq-onset-arc.md. M1a (`probe_seqcontext_frozenh.py`) showed the frozen decoder's h carries the full
placement signal TEACHER-FORCED (conv readout 0.892 ≡ ceiling). But teacher-forcing feeds h built from REAL notes;
the BINDING question is DRIFT — at gen time the head reads h built from its OWN emitted notes, and the 06-22
de-risk found a naive AR onset head EXPLODES (free-run density 0.73 vs real 0.18; the onset→note→h→onset snowball).

THIS probe wires the M1a-trained conv head (HReadConv on h) into a step-by-step rollout that reuses the DEPLOYED
decoder's machinery (`_decoder_step_cached` + `pattern_head`, the same KV-cached decode as `generate()`), so the
h the head reads is the SAME representation the model authors — then free-runs and measures realized density +
onset run-length vs real. THE DRIFT MEASUREMENT = threshold-transfer:
  - CALIBRATE the onset threshold tau on TEACHER-FORCED h (real context) to hit the song's real density (sanity:
    teacher-forced density at tau ≈ real, confirms the head + tau).
  - APPLY that same tau during FREE-RUN (h built from own emitted notes). Density drift = free-run density ≫ real.
READ:
  free-run density ≈ real & run-length ≈ real -> the head is DRIFT-STABLE on h -> M1b integration is viable.
  free-run density ≫ real (toward 0.73) / runs explode -> snowball survives -> scheduled sampling MANDATORY first.
NOTE (experiment-design): this is a DIAGNOSTIC rollout (like diag_ar_stability), governors OFF, stateless taps —
it measures the PURE onset-from-h drift (density/run-length), NOT a playtest. Pattern = greedy (the model's own
choreography); the question is WHEN-density stability, not WHICH-panels.

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqonset_rollout.py
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import NUM_PANELS, NUM_PATTERNS
from src.generation.factorized import _LayerCache
from probe_seqcontext_frozenh import (AD, NP, DMODEL, CKPT, HReadConv, decode_hidden, precompute_h,
                                      load_or_extract, batches)

RF_WIN = 48   # rolling h-window for the head's causal conv (receptive field ~31 < 48 -> last-position output exact)


def train_head(model, train, device, epochs, bs, lr, pw):
    """Train HReadConv (the M1a conv readout) on TEACHER-FORCED h. Decoder frozen (h precomputed). BCE on real onset."""
    set_seed(42); head = HReadConv(DMODEL).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr); rng = np.random.default_rng(0)
    for _ in range(epochs):
        head.train()
        for X, Np, H, Y, M in batches(train, bs, rng, True, device):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(head(H)[M], Y[M], pos_weight=pw)
            loss.backward(); opt.step()
    head.eval()
    return head


def _runlen(onset_row):
    """Mean length of consecutive-onset runs (real Hard ~1.0; 06-22 explosion ~5.7)."""
    runs, c = [], 0
    for v in onset_row:
        if v:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)
    return float(np.mean(runs)) if runs else 0.0


@torch.no_grad()
def rollout(model, head, audio_np, diff, T, tau, device, pattern_sample=False, pat_temp=1.0, tf_states=None,
            seed_frames=0):
    """Onset[t] decided by head on the rolling h-window from the INCREMENTAL decode (`_decoder_step_cached`, the
    same KV-cached path as generate()). FEEDBACK: own emitted note (free-run) OR — when `tf_states` (T,4) is given
    AND (t < seed_frames OR seed_frames==0-with-tf-only) — the REAL note (control / warm-seed). With `seed_frames>0`
    + `tf_states`: feed REAL notes for t<seed_frames then switch to OWN (cold-start vs sustain test). Returns onset(T,)."""
    model.eval()
    audio_t = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    diff_t = torch.tensor([diff], device=device)
    memory = model.encode_audio(audio_t)
    cond = model._cond(diff_t, None, None, None, None)                       # (1,1,d)
    caches = [_LayerCache(layer, memory) for layer in model.decoder.layers]
    panel_bits = ((torch.arange(1, NUM_PATTERNS + 1, device=device).unsqueeze(-1)
                   >> torch.arange(NUM_PANELS, device=device)) & 1)         # (15,4)
    prev_emb = model.bos.expand(1, 1, -1)
    hbuf = []                                                               # rolling list of h_t (1,d)
    onset = np.zeros(T, dtype=bool)
    for t in range(T):
        pe_t = model.pos_encoding.pe[:, t:t + 1]
        h = model._decoder_step_cached(prev_emb + pe_t + cond, caches)[:, -1]   # (1,d) — SAME h as generate()
        hbuf.append(h)
        win = torch.stack(hbuf[-RF_WIN:], dim=1)                            # (1,w,d)
        ol = head(win)[:, -1]                                              # (1,) onset logit at the last position
        on = bool((torch.sigmoid(ol) > tau).item())
        onset[t] = on
        use_real = tf_states is not None and (seed_frames == 0 or t < seed_frames)
        if use_real:                                                       # CONTROL / warm-seed: feed back the REAL note
            state = torch.from_numpy(tf_states[t]).long().unsqueeze(0).to(device)
        elif on:                                                           # free-run: feed back the OWN emitted note
            pl = model.pattern_head(h)                                     # (1,15)
            if pattern_sample:
                pat = torch.multinomial(torch.softmax(pl / pat_temp, -1), 1).squeeze(-1)
            else:
                pat = pl.argmax(-1)
            state = (panel_bits[pat].bool().long())                       # (1,4) tap symbol=1 on active panels
        else:
            state = torch.zeros(1, NUM_PANELS, dtype=torch.long, device=device)
        prev_emb = model._state_emb(state.unsqueeze(1))                   # (1,1,d) AR feedback
    return onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_train', type=int, default=800); ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=12); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--cap', type=int, default=512, help='rollout length cap (per song; speed)')
    ap.add_argument('--n_eval', type=int, default=12, help='Hard val songs to roll out')
    ap.add_argument('--pattern_sample', action='store_true')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_cache = PROJECT_ROOT / "cache/seqctx_frozenh_train.npz"
    val_cache = PROJECT_ROOT / "cache/seqctx_frozenh_val.npz"
    assert train_cache.exists() and val_cache.exists(), "run probe_seqcontext_frozenh.py first (builds the caches)"
    train = load_or_extract(None, args.max_train, args.cap, train_cache, hard_only=False)
    val = load_or_extract(None, 400, args.cap, val_cache, hard_only=True)[:args.n_eval]

    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=DMODEL, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()

    print("training the causal onset head (HReadConv on teacher-forced h)...", flush=True)
    precompute_h(model, train, device, args.cap)
    posrate = np.mean([(s['typed'] != 0).any(-1).mean() for s in train]); pw = torch.tensor((1 - posrate) / posrate, device=device)
    head = train_head(model, train, device, args.epochs, args.bs, args.lr, pw)

    print("\nper-song DRIFT (tau calibrated on teacher-forced h -> applied to incremental rollout):", flush=True)
    SEED = 32
    print(f"  {'song':>4} {'real_d':>7} {'tf_par_d':>9} {'TFroll_d':>9} {'FREE_d':>8} {'seed_d':>7} {'real_run':>9} {'FREE_run':>9}", flush=True)
    precompute_h(model, val, device, args.cap)                            # teacher-forced (parallel) h for tau calibration
    rows = []
    torch.set_grad_enabled(False)
    for i, s in enumerate(val):
        T = min(s['T'], args.cap); real_on = (s['typed'][:T] != 0).any(-1)
        real_d = float(real_on.mean()); real_run = _runlen(real_on)
        p_tf = torch.sigmoid(head(torch.from_numpy(s['h'][:T]).unsqueeze(0).to(device))[0]).cpu().numpy()
        tau = float(np.quantile(p_tf, 1 - real_d)) if real_d > 0 else 0.5  # match real density on teacher-forced (parallel) h
        tf_par_d = float((p_tf > tau).mean())                             # sanity: should ≈ real_d
        on_tf = rollout(model, head, s['audio'][:T], s['diff'], T, tau, device, tf_states=s['typed'][:T].astype(np.int64))
        tf_roll_d = float(on_tf.mean())                                   # CONTROL: incremental h + REAL context
        on = rollout(model, head, s['audio'][:T], s['diff'], T, tau, device, args.pattern_sample)
        free_d = float(on.mean()); free_run = _runlen(on)
        on_seed = rollout(model, head, s['audio'][:T], s['diff'], T, tau, device, args.pattern_sample,
                          tf_states=s['typed'][:T].astype(np.int64), seed_frames=SEED)
        seed_d = float(on_seed[SEED:].mean()) if T > SEED else 0.0        # density AFTER the warm-seed (cold-start test)
        rows.append((real_d, tf_par_d, tf_roll_d, free_d, seed_d, real_run, free_run))
        print(f"  {i:>4} {real_d:>7.3f} {tf_par_d:>9.3f} {tf_roll_d:>9.3f} {free_d:>8.3f} {seed_d:>7.3f} {real_run:>9.2f} {free_run:>9.2f}", flush=True)

    R = np.array(rows)
    rd, tfp, tfr, fd, sd, rr, fr = R.mean(0)
    print(f"\n  MEAN  real_d={rd:.3f}  tf_parallel={tfp:.3f}  TF_rollout={tfr:.3f}  FREE_run={fd:.3f}"
          f"  seed{SEED}_after={sd:.3f}  | real_run={rr:.2f}  FREE_run={fr:.2f}", flush=True)
    print(f"  COLD-START vs SUSTAIN: seed{SEED}_after = density AFTER a {SEED}-frame REAL warm-seed. seed≈real ->", flush=True)
    print(f"    cold-start only (fixable w/ audio-anchor/seed); seed≈0 too -> can't sustain from own context (deeper).", flush=True)
    print(f"  CONTROL (Rule 11): TF_rollout must ≈ real_d ({rd:.2f}); if it collapses too, the incremental h ≠ training", flush=True)
    print(f"    h (a HARNESS bug) and FREE is uninterpretable. Only if TF_rollout fires ~real is FREE a real drift number.", flush=True)
    if abs(tfr - rd) > 0.4 * rd:
        print(f"  !! TF_rollout ({tfr:.3f}) FAR from real ({rd:.3f}) -> suspect incremental-h mismatch; debug harness before MODEL.", flush=True)
    else:
        ratio = fd / max(rd, 1e-6)
        verdict = ("COLLAPSE to empty" if fd < 0.5 * rd else
                   "EXPLODE" if ratio > 1.3 or fr > 1.6 * rr else "STABLE")
        print(f"  DRIFT (control passed): FREE-run density = {ratio:.2f}× real, run {fr:.2f} vs {rr:.2f} -> **{verdict}**", flush=True)
        if verdict == "STABLE":
            print(f"  => the head is drift-stable on h -> M1b integration viable (next: wire into generate()).", flush=True)
        else:
            print(f"  => exposure-bias drift survives ({verdict}) -> scheduled sampling / audio-anchor MANDATORY before integration.", flush=True)
    print(f"\n  BOUNDARY: governors OFF, stateless taps, greedy pattern — pure onset-from-h drift (density/run-length).", flush=True)


if __name__ == '__main__':
    main()
