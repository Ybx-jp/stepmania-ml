#!/usr/bin/env python3
"""
Interpretability for the taste critic — PHASE A: the VALIDATION GATE (run first, believe nothing until it passes).

Per notes/taste_critic_interpretability_plan.md + the experiment-design skill: saliency on a SATURATED classifier
(the critic is near-binary) is a textbook way to get a confidently-wrong map. So before interpreting any real
chart, prove the attribution method localizes a KNOWN, INJECTED defect — using the critic's OWN training negative
types (panels-scramble, audio-shift).

Two attribution methods, cross-checked:
  - INTEGRATED GRADIENTS on the LOGIT MARGIN (z_real - z_fake), NOT P(real). The logit keeps dynamic range when
    the probability saturates; IG (path integral from a baseline) is the standard saturation fix.
  - BLOCK-REPAIR (occlusion-style, saturation-proof ground truth): restore a block of the corrupted chart to
    clean, measure how much the margin recovers. The block that recovers the most margin IS where the defect is.

GATE PASSES iff, on high-margin REAL charts with a corruption injected into a known window [a,b]:
  (1) the corruption actually drops the margin (the cue exists), AND
  (2) BOTH block-repair and IG concentrate their mass in [a,b] (localization ratio > THRESH), AND
  (3) the two methods agree (their per-frame saliencies correlate).
Only then do we trust IG on real charts (Phase B: REAL/MEANPIN/MANIFOLD/BASE, the off-grid hypothesis).

Usage:
    python experiments/realism_critic/critic_saliency.py --data_dir data/ --audio_dir data/ --gate_songs 6
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.models import LateFusionClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--critic', default='checkpoints/realism_critic/best_val.pt')
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--gate_songs', type=int, default=6, help='# high-margin REAL charts to inject defects into')
    p.add_argument('--scan_songs', type=int, default=40, help='# val songs to scan for high-margin REAL charts')
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--ig_steps', type=int, default=32)
    p.add_argument('--block', type=int, default=32, help='block size for block-repair granularity')
    p.add_argument('--corrupt_frac', type=float, default=0.5,
                   help='fraction of the chart to scramble (contiguous) for the localization test')
    p.add_argument('--loc_thresh', type=float, default=2.0, help='localization ratio to PASS')
    return p.parse_args()


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def margin(critic, audio, chart, mask):
    """Logit margin z_real - z_fake (B,). Pre-softmax: keeps range when P(real) saturates."""
    logits = critic(audio, chart, mask)
    if isinstance(logits, dict): logits = logits['logits']
    return logits[:, 1] - logits[:, 0]


def integrated_gradients(critic, audio, chart, mask, target='chart', steps=32):
    """IG of the logit margin w.r.t. `target` input, baseline=zeros. Returns per-frame attribution (T,)
    = sum over feature dim of (input-baseline)*avg_grad. The non-target input is held at its actual value."""
    a0, c0 = audio.detach(), chart.detach()
    base_a = torch.zeros_like(a0); base_c = torch.zeros_like(c0)
    grads = []
    for k in range(1, steps + 1):
        alpha = k / steps
        a = (base_a + alpha * (a0 - base_a)).clone().requires_grad_(target == 'audio')
        c = (base_c + alpha * (c0 - base_c)).clone().requires_grad_(target == 'chart')
        m = margin(critic, a, c, mask).sum()
        g = torch.autograd.grad(m, a if target == 'audio' else c)[0]
        grads.append(g.detach())
    avg_grad = torch.stack(grads).mean(0)
    inp, base = (a0, base_a) if target == 'audio' else (c0, base_c)
    attr = ((inp - base) * avg_grad)[0]          # (T, F)
    return attr.sum(-1).cpu().numpy()            # (T,)


def block_repair(critic, audio, chart_corrupt, chart_clean, mask, block):
    """Saturation-proof ground truth: for each block of frames, restore corrupt->clean and measure margin
    recovery. Returns per-block Δmargin and the per-frame series (block value broadcast)."""
    T = chart_corrupt.shape[1]
    with torch.no_grad():
        base_m = margin(critic, audio, chart_corrupt, mask).item()
        per_frame = np.zeros(T)
        for s in range(0, T, block):
            e = min(s + block, T)
            repaired = chart_corrupt.clone(); repaired[:, s:e] = chart_clean[:, s:e]
            dm = margin(critic, audio, repaired, mask).item() - base_m
            per_frame[s:e] = dm
    return per_frame


def localization_ratio(per_frame_saliency, a, b, valid_T):
    """(mean |saliency| inside [a,b]) / (mean |saliency| outside). >1 = concentrates in the injected window.
    Returns NaN when the saliency is ~flat-zero everywhere (no signal to localize) — avoids the divide-by-~0
    blowup that makes a no-signal case look like astronomical localization."""
    s = np.abs(per_frame_saliency[:valid_T])
    if s.max() < 1e-6:
        return float('nan')          # truly no signal anywhere -> nothing to localize
    win = np.zeros(valid_T, bool); win[a:b] = True
    inside = s[win].mean() if win.any() else 0.0
    outside = s[~win].mean() if (~win).any() else 0.0
    if inside < 1e-6:
        return 0.0                   # signal exists but NOT in the window -> localization failed
    ratio = inside / max(outside, s.max() * 1e-3)   # relative floor: outside~0 = perfect localization
    return float(min(ratio, 999.0))                 # cap so "all signal in window" reads as 999, not inf


def scramble_panels(frame_rows, rng):
    """Per-frame: keep the onset COUNT, permute WHICH panels are active (kills arrow coherence). Mirrors the
    critic's `panels` training negative. frame_rows: (W,4) binary -> (W,4) binary."""
    out = np.zeros_like(frame_rows)
    for i, row in enumerate(frame_rows):
        k = int(row.sum())
        if k == 0: continue
        panels = rng.choice(4, size=k, replace=False)
        out[i, panels] = 1.0
    return out


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=args.seed)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    _, val_ds, _ = create_datasets(train_files=[], val_files=val_files, test_files=[], audio_dir=args.audio_dir,
                                   max_sequence_length=msl, feature_extractor=ext, cache_dir=args.cache_dir)
    val_ds.warm_cache(show_progress=False)

    ck = torch.load(PROJECT_ROOT / args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device)
    critic.load_state_dict(ck['model_state_dict']); critic.eval()

    # gather REAL charts, keep the highest-margin ones (the critic is confident they're real -> a clean
    # case to inject a defect into and watch the margin fall)
    cand = []
    for i in range(len(val_ds)):
        if len(cand) >= args.scan_songs: break
        s = val_ds[i]; meta = val_ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        if T < 128: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        tf = to_binary(val_ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T])
        a23 = s['audio'][:T, :23].numpy().astype(np.float32)
        cand.append({'a23': a23, 'chart': tf, 'T': T, 'diff': int(meta['difficulty_class'])})
    rng = np.random.default_rng(args.seed)
    for c in cand:
        a = torch.from_numpy(c['a23']).unsqueeze(0).to(device); ch = torch.from_numpy(c['chart']).unsqueeze(0).to(device)
        m = torch.ones(1, c['T'], device=device)
        with torch.no_grad(): c['margin'] = margin(critic, a, ch, m).item()
    cand.sort(key=lambda c: -c['margin'])
    gate = cand[:args.gate_songs]
    margins_str = ', '.join('%.1f' % c['margin'] for c in gate)
    print(f"scanned {len(cand)} REAL charts; using top-{len(gate)} by margin (margins {margins_str})")

    # ---- Stage 0: CUE-EXISTS sanity. Does a WHOLE-chart panels-scramble collapse the margin (as in training,
    #      where `panels` negatives scored P(real)~0.03)? If not, the critic doesn't key on arrow coherence and
    #      there is nothing to localize. This is the prerequisite for the localization test below.
    print("\n" + "=" * 72)
    print("  STAGE 0 — cue exists? whole-chart panels-scramble margin collapse")
    print("=" * 72)
    full_drops = []
    for c in gate:
        T = c['T']; a = torch.from_numpy(c['a23']).unsqueeze(0).to(device); m = torch.ones(1, T, device=device)
        clean = torch.from_numpy(c['chart']).unsqueeze(0).to(device)
        full_np = scramble_panels(c['chart'], rng)
        with torch.no_grad():
            dm = margin(critic, a, clean, m).item() - margin(critic, a, torch.from_numpy(full_np).unsqueeze(0).to(device), m).item()
        full_drops.append(dm)
    fd = np.array(full_drops)
    print(f"  whole-chart scramble Δmargin: mean={fd.mean():+.2f}  (min {fd.min():+.2f}, max {fd.max():+.2f})")
    cue_exists = fd.mean() > 1.0
    print(f"  cue exists (mean drop > 1.0): {'YES' if cue_exists else 'NO'}")

    # ---- Localization: scramble a CONTIGUOUS fraction (sized to register on a globally-pooled critic), check
    #      both methods concentrate their saliency in that window.
    print("\n" + "=" * 72)
    print(f"  LOCALIZATION — scramble a contiguous {args.corrupt_frac:.0%} window, do repair+IG localize it?")
    print("=" * 72)
    repair_ratios, ig_ratios, agreements, drops = [], [], [], []
    for n, c in enumerate(gate, 1):
        T = c['T']; W = args.block
        a = torch.from_numpy(c['a23']).unsqueeze(0).to(device)
        m = torch.ones(1, T, device=device)
        clean = torch.from_numpy(c['chart']).unsqueeze(0).to(device)
        wlen = int(round(args.corrupt_frac * T))
        astart = (T - wlen) // 2; aend = astart + wlen
        corrupt_np = c['chart'].copy(); corrupt_np[astart:aend] = scramble_panels(c['chart'][astart:aend], rng)
        corrupt = torch.from_numpy(corrupt_np).unsqueeze(0).to(device)
        with torch.no_grad():
            drop = margin(critic, a, clean, m).item() - margin(critic, a, corrupt, m).item()
        repair = block_repair(critic, a, corrupt, clean, m, W)              # ground truth
        ig = integrated_gradients(critic, a, corrupt, m, 'chart', args.ig_steps)  # the gradient method
        rr = localization_ratio(repair, astart, aend, T)
        ir = localization_ratio(ig, astart, aend, T)
        agr = float(np.corrcoef(np.abs(repair[:T]), np.abs(ig[:T]))[0, 1]) if np.std(ig[:T]) > 1e-9 else float('nan')
        repair_ratios.append(rr); ig_ratios.append(ir); agreements.append(agr); drops.append(drop)
        print(f"  [{n}/{len(gate)}] diff={c['diff']} win[{astart}:{aend}]  Δmargin(drop)={drop:+.2f}  "
              f"repair-loc={rr:.1f}x  IG-loc={ir:.1f}x  agree(|repair|,|IG|)={agr:+.2f}")

    rr, ir, ag, dr = (np.array(repair_ratios), np.array(ig_ratios), np.array(agreements), np.array(drops))
    print("=" * 72)
    print(f"  mean: Δmargin-drop={dr.mean():+.2f} | repair-loc={np.nanmean(rr):.1f}x | "
          f"IG-loc={np.nanmean(ir):.1f}x")
    # The gate VALIDATES the attribution method we'll trust in Phase B. Perturbation/repair is the candidate
    # (it asks the critic's own question: change the chart, watch the margin). Gradient-IG-from-EMPTY is the
    # other candidate but is a priori suspect here: an empty baseline makes IG measure note PRESENCE, while the
    # critic's cue is arrow CONFIGURATION at fixed count -> we expect it to under-localize, and report it as such.
    p_cue = cue_exists and dr.mean() > 0.5
    p_repair = np.nanmean(rr) > args.loc_thresh
    verdict = 'PASS (perturbation/repair)' if (p_cue and p_repair) else 'FAIL'
    print(f"  GATE: cue-exists & corruption-drops-margin {'Y' if p_cue else 'N'} | "
          f"repair localizes (>{args.loc_thresh}x) {'Y' if p_repair else 'N'}  ->  {verdict}")
    print(f"  method note: gradient-IG-from-empty localizes only {np.nanmean(ir):.1f}x -> NOT suitable for this "
          f"critic (measures note presence, not arrow configuration). Phase B uses perturbation/repair saliency.")
    if not (p_cue and p_repair):
        print("  *** Do NOT interpret saliency on real charts until repair localizes a known defect. ***")


if __name__ == '__main__':
    main()
