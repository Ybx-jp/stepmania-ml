#!/usr/bin/env python3
"""
H4 high-res-onset retrain (see notes/h4_offbeat_signal_findings.md). Same layered typed generator +
groove-radar conditioning as train_stage1.py, but trained on the 42-dim feature set: the 41 Stage-1
musical features (23 base + chroma 12 + HPSS 2 + metric phase 4) PLUS a 1-dim high-res onset feature
(onset detected at hop~128 / n_fft~512, max-pooled into each 16th cell). From cache/samples_v3.

Goal: H4. The shipped grid-hop onset is nearly phase-flat and barely predicts off-beat note placement
(off-beat AUC ~0.53 ≈ chance), so the model renders chaos as a uniform global smear — it has no local
off-beat cue. The high-res-pooled onset recovers that cue (off-beat AUC 0.53->0.66; gate confirmed in
diag_confirm_highres_feature.py), giving the onset head + chaos conditioning something local to key on.

H4-v2 (see h4_offbeat_signal_findings.md Result 5). The v1 retrain (train_highres.py) ran but the model
IGNORED the new feature: warm-started from a model already converged WITHOUT it + zero-init new column +
frame-wise CE that barely weights the ~5% off-beat frames => no gradient pressure. New-col weight norm
0.127 (rank 1/42), teacher-forced ablation KL = 0.0000 (feature unused). v2 supplies INCENTIVE:

  1. RANDOM-init the new conv column at full magnitude (~ scale of existing cols), not zero — so the
     feature affects the output from step 0 and training must keep-or-suppress it (not grow-from-dead).
  2. OFF-BEAT-UPWEIGHTED onset loss (--offbeat_weight): off-beat frames (t%4 != 0) weigh more in the
     onset BCE, so getting syncopation right actually moves the loss. This is the key lever — the v1
     objective gave the feature nothing to do.

H4-v3 (chaos = no 16ths; see notes/chaos_mechanism_plan.md no-16ths localization). v2's 3x off-beat weight
lumped 8ths and 16ths together; but the model produces ZERO 16ths (p_on @16th 0.169 vs 0.43 @8th, never
> threshold) -> rhythm caps at 8ths -> chaos radar 0. The high-res feature raises 16th p_on (0.169->0.204)
but 16ths are 4% of notes so the loss ignores them. v3: a GRADUATED phase weight that HEAVILY weights 16th
frames (--w16, default 15) over 8ths (--w8, default 3) over quarters (1), forcing the model to use the
high-res feature to predict 16ths. Same random-init high-res column + gen_stage1 warm-start as v2.

Usage:
    python experiments/generation_typed/train_highres_v3.py --data_dir data/ --audio_dir data/ \
        --epochs 15 --batch_size 8 --w8 3 --w16 15
"""

import warnings, os
warnings.filterwarnings('ignore')
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import (NUM_PATTERNS, NUM_TYPES, NUM_PANELS, SYMBOL_NAMES,
                                   symbol_histogram, pair_holds, panels_to_pattern)
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0
WARM_CKPT = "checkpoints/gen_stage1/best_val.pt"  # warm-start from the trained Stage-1 model (41-dim)
FIRST_CONV = "audio_encoder.net.0.conv.weight"   # input-channel dim changes 41->42; expand, don't reset
TYPE_NAMES = ['tap', 'hold_head', 'tail', 'roll_head']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_train_len', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_freeze', type=int, default=0,
                   help='Stage 1: keep 0 — the fresh 41-dim first conv must train from epoch 0')
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--type_weight', choices=['none', 'inv_sqrt'], default='none',
                   help='none = calibrated focal (sample types at true rate); inv_sqrt over-inflates holds')
    p.add_argument('--eval_songs', type=int, default=64)
    p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--checkpoint_dir', default='checkpoints/gen_highres_v3')
    p.add_argument('--cache_dir', default='cache/samples_v3', help='42-dim cache (41 musical + high-res onset)')
    p.add_argument('--w8', type=float, default=3.0, help='onset-loss weight on 8th frames (t%%4 == 2)')
    p.add_argument('--w16', type=float, default=15.0,
                   help='onset-loss weight on 16th frames (t%%4 in {1,3}); heavy -> forces the model to use '
                        'the high-res feature to predict 16ths (they are 4%% of notes, else ignored).')
    p.add_argument('--cfg_drop', type=float, default=0.15, help='classifier-free guidance: prob of dropping radar conditioning per batch')
    p.add_argument('--patience', type=int, default=3, help='early stopping: stop after this many epochs with no val_total improvement')
    return p.parse_args()


def typed_binary(typed):
    t = np.asarray(typed)
    return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def collect_typed(ds, cap):
    out = []
    for i in range(len(ds)):
        sample = ds[i]; meta = ds.valid_samples[i]
        T = int(sample['mask'].sum().item())
        nd = next((n for n in meta['chart'].note_data
                   if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        typed_full = ds.parser.convert_to_tensor_typed(meta['chart'], nd)
        T = min(T, cap, typed_full.shape[0])
        out.append({'typed': typed_full[:T].astype(np.int64),
                    'audio': sample['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(meta['difficulty_class']),
                    'radar': meta['groove_radar'].to_vector().astype(np.float32)})  # (5,) normalized target
    return out


def focal_bce(logits, targets, gamma, weight=None):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits); p_t = p * targets + (1 - p) * (1 - targets)
    loss = (1 - p_t) ** gamma * bce
    if weight is not None:
        return (weight * loss).sum() / weight.sum().clamp(min=1e-8)  # weighted mean over frames
    return loss.mean()


def focal_ce(logits, targets, gamma, weight=None):
    logp = nn.functional.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    loss = -((1 - p_t) ** gamma) * logp_t
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)

    chart_files = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(chart_files, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        max_seq_len = yaml.safe_load(f)['classifier']['max_sequence_length']
    ext42 = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                     use_metric_phase=True, use_highres_onset=True))
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files,
                                          test_files=[], audio_dir=args.audio_dir,
                                          max_sequence_length=max_seq_len,
                                          feature_extractor=ext42, cache_dir=args.cache_dir)
    print("Warming caches (cache/samples_v2; pre-warm with warm_cache_v2.py)...")
    train_ds.warm_cache(show_progress=True); val_ds.warm_cache(show_progress=True)
    print("Collecting typed samples...")
    train = collect_typed(train_ds, args.max_train_len)
    val = collect_typed(val_ds, args.max_train_len)
    print(f"train={len(train)} val={len(val)}")
    audio_dim = train[0]['audio'].shape[1]
    target_density = float(np.mean([(s['typed'] != 0).any(1).mean() for s in val]))

    try:
        import mlflow; mlflow.set_experiment("stepmania-chart-generator"); mlflow_on = True
    except ImportError:
        mlflow_on = False

    assert audio_dim == 42, f"expected 42-dim features (use_highres_onset), got {audio_dim}"
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    ck = torch.load(WARM_CKPT, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    print(f"warm-started {model.load_from_factorized(sd)} tensors from gen_stage1")
    # Expand the first conv (41->42 input channels): copy the trained channels, and RANDOM-init the new
    # column at the scale of the existing columns (v2: NOT zero — v1's zero-init never grew, KL=0). This
    # makes the high-res feature affect the output from step 0, so training keeps-or-suppresses it.
    with torch.no_grad():
        old_w = sd[FIRST_CONV]                                   # (d_model, 41, k)
        new_w = dict(model.named_parameters())[FIRST_CONV]       # (d_model, 42, k)
        assert new_w.shape[1] == old_w.shape[1] + 1, (old_w.shape, new_w.shape)
        new_w[:, :old_w.shape[1], :].copy_(old_w.to(new_w.device))
        col_std = old_w.std()                                    # per-existing-column scale
        new_w[:, old_w.shape[1], :].normal_(0.0, float(col_std))
        new_col_norm = new_w[:, old_w.shape[1], :].norm()
        print(f"expanded {FIRST_CONV}: {tuple(old_w.shape)} -> {tuple(new_w.shape)} "
              f"(new column RANDOM-init, std={float(col_std):.4f}, norm={float(new_col_norm):.4f} "
              f"vs existing-mean {float(old_w.norm(dim=(0,2)).mean()):.4f})")
    model.freeze_audio_encoder(True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # mild inv_sqrt class weights for the 4 TYPES (tap dominant ~20:1) on active panels
    type_counts = np.ones(NUM_TYPES)
    for s in train:
        act = s['typed'][s['typed'] != 0]  # active-panel symbols 1..4
        type_counts += np.bincount(act - 1, minlength=NUM_TYPES)
    if args.type_weight == 'inv_sqrt':
        type_w = np.clip(np.sqrt(type_counts.sum() / (NUM_TYPES * type_counts)), 0, 5.0)
    else:  # none -> uniform (calibrated focal; rely on sampling for hold rate)
        type_w = np.ones(NUM_TYPES)
    type_weight = torch.tensor(type_w, dtype=torch.float32, device=device)
    print("type class weights:", {TYPE_NAMES[i]: round(float(type_w[i]), 2) for i in range(NUM_TYPES)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batches(samples, bs, shuffle):
        idx = np.arange(len(samples))
        if shuffle: rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [samples[j] for j in idx[i:i + bs]]

    def to_tensors(batch):
        T = max(len(s['typed']) for s in batch); B = len(batch)
        audio = torch.zeros(B, T, audio_dim); states = torch.zeros(B, T, NUM_PANELS, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool); diff = torch.zeros(B, dtype=torch.long)
        radar = torch.zeros(B, 5)
        for b, s in enumerate(batch):
            t = len(s['typed'])
            audio[b, :t] = torch.from_numpy(s['audio']); states[b, :t] = torch.from_numpy(s['typed'])
            mask[b, :t] = True; diff[b] = s['difficulty']; radar[b] = torch.from_numpy(s['radar'])
        # pattern target (B,T) and type target (B,T,4)
        active = (states != 0)
        pat = torch.from_numpy(panels_to_pattern(active.numpy())).clamp(min=0)  # (B,T)
        typ = (states - 1).clamp(min=0)  # (B,T,4) symbol->type idx (valid on active)
        return (audio.to(device), states.to(device), mask.to(device), diff.to(device),
                pat.to(device), typ.to(device), active.to(device), radar.to(device))

    def losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active):
        onset_t = (states != 0).any(-1).float()
        # graduated phase-weighted onset loss: quarter 1, 8th w8, 16th w16 (heavy -> forces 16th learning).
        B, T = ol.shape
        t = torch.arange(T, device=ol.device)
        pw = torch.ones(T, device=ol.device)
        pw[t % 4 == 2] = float(args.w8)
        pw[(t % 4 == 1) | (t % 4 == 3)] = float(args.w16)
        phase_w = pw.unsqueeze(0).expand(B, T)
        o = focal_bce(ol[mask], onset_t[mask], args.focal_gamma, weight=phase_w[mask])
        sel = mask & (onset_t > 0.5)
        p = focal_ce(pat_l[sel], pat_t[sel], args.focal_gamma) if sel.any() else torch.zeros((), device=device)
        act = active & mask.unsqueeze(-1)
        t = focal_ce(typ_l[act], typ_t[act], args.focal_gamma, weight=type_weight) if act.any() else torch.zeros((), device=device)
        return o, p, t

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best = float('inf'); bad = 0; best_path = Path(args.checkpoint_dir) / "best_val.pt"
    if mlflow_on:
        mlflow.start_run(run_name="gen-highres-v3"); mlflow.log_params({'epochs': args.epochs, 'lr': args.lr, 'features': '42-dim: stage1 + high-res onset', 'warm_start': 'gen_stage1', 'w8': args.w8, 'w16': args.w16, 'new_col_init': 'random'})

    for epoch in range(args.epochs):
        if epoch == args.warmup_freeze:
            model.freeze_audio_encoder(False); print(f"  [epoch {epoch+1}] unfroze audio encoder")
        model.train(); tr = [0.0, 0.0, 0.0]; nb = 0
        for batch in batches(train, args.batch_size, True):
            audio, states, mask, diff, pat_t, typ_t, active, radar = to_tensors(batch)
            optimizer.zero_grad()
            cond_radar = None if rng.random() < args.cfg_drop else radar  # CFG: drop conditioning sometimes
            ol, pat_l, typ_l = model(audio, states, diff, mask, radar=cond_radar)
            o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
            (o + p + t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tr[0] += o.item(); tr[1] += p.item(); tr[2] += t.item(); nb += 1
        model.eval(); v = [0.0, 0.0, 0.0]; vnb = 0
        with torch.no_grad():
            for batch in batches(val, args.batch_size, False):
                audio, states, mask, diff, pat_t, typ_t, active, radar = to_tensors(batch)
                ol, pat_l, typ_l = model(audio, states, diff, mask, radar=radar)
                o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
                v[0] += o.item(); v[1] += p.item(); v[2] += t.item(); vnb += 1
        v = [x / max(1, vnb) for x in v]; vt = sum(v)
        print(f"  epoch {epoch+1}/{args.epochs}  train(o={tr[0]/max(1,nb):.3f} pat={tr[1]/max(1,nb):.3f} typ={tr[2]/max(1,nb):.3f})  "
              f"val(o={v[0]:.3f} pat={v[1]:.3f} typ={v[2]:.3f} tot={vt:.3f})" + ("  *" if vt < best else ""))
        if mlflow_on: mlflow.log_metrics({'val_onset': v[0], 'val_pattern': v[1], 'val_type': v[2], 'val_total': vt}, step=epoch)
        if vt < best:
            best = vt; bad = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_total': vt}, best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop @ epoch {epoch+1}: no val_total improvement for {args.patience} epochs"); break

    # ---- eval ----
    print(f"\nLoading best (val_total={best:.4f}) for eval...")
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict']); model.eval()
    eval_set = val[:args.eval_songs]
    critic = DifficultyCritic(device=device)

    # teacher-forced type recall on active panels (does it learn holds without none-bias?)
    tc = np.zeros(NUM_TYPES); tt = np.zeros(NUM_TYPES)
    with torch.no_grad():
        for batch in batches(eval_set, args.batch_size, False):
            audio, states, mask, diff, pat_t, typ_t, active, radar = to_tensors(batch)
            _, _, typ_l = model(audio, states, diff, mask, radar=radar)
            pred = typ_l.argmax(-1)
            for ty in range(NUM_TYPES):
                m = active & mask.unsqueeze(-1) & (typ_t == ty)
                tt[ty] += int(m.sum()); tc[ty] += int((m & (pred == ty)).sum())
    type_recall = {TYPE_NAMES[i]: (tc[i] / tt[i] if tt[i] else float('nan')) for i in range(NUM_TYPES)}

    onset_logits_all = []
    with torch.no_grad():
        for s in eval_set:
            a = torch.from_numpy(s['audio']).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
            rad = torch.from_numpy(s['radar']).unsqueeze(0).to(device)
            onset_logits_all.append(torch.sigmoid(model.onset_logits(model.encode_audio(a), d, radar=rad))[0].cpu().numpy())
    tau = float(np.quantile(np.concatenate(onset_logits_all), 1 - target_density))

    f1s, dens, gen_syms, preds, tgts, holds, orphans = [], [], {k: 0 for k in SYMBOL_NAMES}, [], [], 0, 0
    for i in range(0, len(eval_set), 8):
        batch = eval_set[i:i + 8]
        L = min(args.max_gen_len, max(len(s['typed']) for s in batch)); B = len(batch)
        audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
        radar = torch.zeros(B, 5)
        for b, s in enumerate(batch):
            t = min(len(s['typed']), L); audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t
            diff[b] = s['difficulty']; radar[b] = torch.from_numpy(s['radar'])
        gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                             onset_threshold=tau, greedy=True, type_sample=True, radar=radar.to(device)).cpu().numpy()
        for b, s in enumerate(batch):
            t = int(lengths[b]); g_raw = gen[b, :t]
            for pnl in range(NUM_PANELS):
                col = g_raw[:, pnl]; h = int(((col == 2) | (col == 4)).sum()); tl = int((col == 3).sum())
                holds += h; orphans += abs(h - tl)
            g = pair_holds(g_raw)
            m = onset_density_metrics((g != 0).astype(np.float32), reference=(s['typed'][:t] != 0).astype(np.float32))
            f1s.append(m['onset_f1']); dens.append((g != 0).any(1).mean())
            for k, val_ in symbol_histogram(g).items(): gen_syms[k] += val_
            # critic is the Phase-1 classifier (23-dim audio encoder) -> feed only the original 23 dims
            preds.append(critic.predict(typed_binary(g), s['audio'][:t, :23], bpm=DEFAULT_BPM)['class']); tgts.append(s['difficulty'])
    dd = np.abs(np.array(preds) - np.array(tgts))
    real_taps = sum(symbol_histogram(s['typed'])['tap'] for s in eval_set)
    real_holds = sum(symbol_histogram(s['typed'])['hold_head'] for s in eval_set)

    print("\n" + "=" * 76)
    print("  LAYERED TYPED GENERATOR")
    print("=" * 76)
    print(f"  onset_F1 {np.mean(f1s):.3f}  density {np.mean(dens):.3f} (real {target_density:.3f})  "
          f"crit_adj {np.mean(dd<=1):.3f}  crit_mae {np.mean(dd):.3f}")
    print(f"  teacher-forced type recall (active panels): " + " ".join(f"{k}={type_recall[k]:.2f}" for k in TYPE_NAMES))
    print(f"  generated symbols: {gen_syms}")
    gh = gen_syms['hold_head']; gt = gen_syms['tap']
    print(f"  generated tap:hold = {gt}:{gh} ({gt/max(gh,1):.1f}:1)   real tap:hold = {real_taps/max(real_holds,1):.1f}:1")
    print(f"  raw hold orphans (pre-pairing): {orphans}/{holds}")
    print("=" * 76)
    if mlflow_on:
        mlflow.log_metrics({'gen_onset_f1': float(np.mean(f1s)), 'gen_density': float(np.mean(dens)),
                            'crit_adj': float(np.mean(dd <= 1)), 'hold_recall': float(type_recall['hold_head'])})
        mlflow.end_run()


if __name__ == '__main__':
    main()
