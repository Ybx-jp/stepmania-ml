#!/usr/bin/env python3
"""M1b-5 TASTE-CRITIC A/B: is the seq-onset head's free-run placement MUSICALLY good, even though it doesn't match
THE real chart's exact 16ths (the AUC-vs-real metric is too strict for valid ALTERNATIVE phrasing)? (2026-06-29)
Lineage seq-onset-arc.md. M1b-4 (`probe_seqonset_placement.py`) found free-run 16th-AUC ≤ the audio floor — but AUC
is measured vs ONE reference chart and penalizes musically-valid-but-different placement. The fair gate for
MUSICALITY is the realism/taste critic (a learned P(real), not exact-match) and by-ear (exp-design Rule 8). This is
the critic gate; if it passes, export for by-ear.

ONE-CHANGE A/B (exp-design Rule 11 + Rule 15) routed through the DEPLOYED generate() via the SANCTIONED
`onset_override` input (NO surgery on the loop — Rule 14). Both arms: same songs, same deployed
`gen_motif_full_fixed`, same canonical governor/playability config (generation-defaults skill), same density, radar
off — the ONLY difference is the ONSET TRAJECTORY. `onset_override` skips stamina for BOTH arms (controlled);
per-note fatigue + playability run. Density is matched to the seq head's realized count (d_seq) so only PLACEMENT
differs. Arms scored by the realism critic P(real):
  REAL        : the real chart                                            (CONTROL: must be HIGH ~0.8)
  shuf16      : real with 16ths shuffled                                  (CONTROL: must be LOW — critic sees 16th placement)
  AUDIO@real  : deployed audio onset head + 16th-unlock, at REAL density  (the deployed baseline, reference)
  AUDIO@dseq  : same, density-matched to the seq head                     (the FAIR one-change baseline vs SEQ)
  SEQ@dseq    : the SS head's FREE-RUN onsets (rollout)                    <- MEASUREMENT
Baseline is the STRONGEST deployed onset path (audio head WITH onset_phase_calib=(0,1.0), the 16th-unlock) so SEQ
must beat the capability we ALREADY ship, not a strawman (Rule 15).

READ:
  SEQ ≳ AUDIO@dseq and toward REAL -> the free-run placement is MUSICALLY good; AUC-vs-real was too strict ->
    proceed to by-ear export (the binding gate).
  SEQ ≲ AUDIO@dseq / ~ shuf16 -> placement is bad even by the LENIENT musicality gate -> M1b-4 stands, BANK.
CONTROL GATE: REAL >> shuf16 (the critic must separate good/broken placement) before interpreting SEQ vs AUDIO.

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqonset_critic.py --load_head
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
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
from src.models import LateFusionClassifier
from probe_seqcontext_frozenh import AD, DMODEL, CKPT, HReadConv
from probe_seqonset_rollout import rollout
from diag_maskpredict_staged import critic_score, to_binary, shuffle16

CRITIC = "checkpoints/realism_critic/best_val.pt"

# Canonical deployed decode palette (generation-defaults skill §1), onset-related knobs dropped (onset is supplied
# via onset_override; stamina is skipped under override for BOTH arms). Per-note fatigue + playability stay on.
DECODE = dict(type_sample=True, type_temperature=0.4, pattern_sample=True, pattern_temperature=1.0,
              repetition_penalty=1.0, max_jack_run=2, fatigue_penalty=2.0, fatigue_free=6.0)


def gen_panels(model, A42, T, diff, onset_bool, bpm, device):
    """Deployed generate() with the supplied onset trajectory injected via onset_override; canonical governor +
    MANDATORY playability (enforce_playability). radar=None (canonical default). Returns binary chart (T,4)."""
    kw = dict(onset_override=onset_bool.unsqueeze(0).to(device), bpm=float(bpm), **DECODE)
    enforce_playability(kw, "seq-onset critic A/B (onset supplied, panels deployed)")
    g = model.generate(A42, diff, lengths=torch.tensor([T], device=device), **kw)[0].cpu().numpy()
    return to_binary(pair_holds(g))


def audio_onset(model, A42, diff, T, density, device):
    """The DEPLOYED audio onset path: native audio onset logits + the 16th-unlock onset_phase_calib=(0,1.0), then
    the per-song density quantile on the SAME offset logits (generation-defaults §2/§1a). Top-`density` frames."""
    with torch.no_grad():
        ol = model.onset_logits(model.encode_audio(A42), diff)[0]                      # (T,) native
        ph = torch.arange(T, device=device) % 4
        ol = ol + torch.where(ph == 2, 0.0, torch.where((ph == 1) | (ph == 3), 1.0, 0.0))  # 16th-unlock (b8=0,b16=1)
        p = torch.sigmoid(ol).cpu().numpy()
    tau = float(np.quantile(p, 1 - density))
    return torch.from_numpy(p > tau).bool()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load_head', action='store_true'); ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--cap', type=int, default=512); ap.add_argument('--tau', type=float, default=0.55,
                    help='SS head free-run threshold (M1b-3 plateau; density slightly over real)')
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
    ckc = torch.load(CRITIC, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ckc['config']).to(device)
    critic.load_state_dict(ckc['model_state_dict']); critic.eval()
    rng = np.random.default_rng(42)

    # groove-validate: Hard songs with real 16ths, ranked by chaos (the population that exercises 16th placement)
    cands = []
    for i in range(len(va_ds.valid_samples)):
        meta = va_ds.valid_samples[i]
        if meta['difficulty_class'] >= 3:
            cands.append((float(meta['groove_radar'].chaos), i))
    cands.sort(reverse=True)

    print(f"  {'song':<26} {'real_d':>7} {'d_seq':>6} | {'REAL':>5} {'shuf':>5} {'A@rl':>5} {'A@ds':>5} {'SEQ':>5}", flush=True)
    rows = []
    for chaos, i in cands:
        if len(rows) >= args.n:
            break
        meta = va_ds.valid_samples[i]; s = va_ds[i]; T = min(int(s['mask'].sum().item()), args.cap)
        nd = next((x for x in meta['chart'].note_data if x.difficulty_name == meta['difficulty_name']
                   and x.difficulty_value == meta['difficulty_value']), None)
        if nd is None or T < 256:
            continue
        orig = np.asarray(va_ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        real_on = (orig != 0).any(1); i16 = (np.arange(T) % 4 == 1) | (np.arange(T) % 4 == 3)
        if real_on.sum() < 32 or (real_on & i16).sum() / max(real_on.sum(), 1) < 0.05:    # needs real 16ths to test
            continue
        a = s['audio'][:T, :AD].numpy().astype(np.float32); a23 = a[:, :23]
        A42 = torch.from_numpy(a).unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        bpm = float(meta['chart'].bpm); real_d = float(real_on.mean())

        seq_on = torch.from_numpy(rollout(model, head, a, meta['difficulty_class'], T, args.tau, device)).bool()
        d_seq = float(seq_on.float().mean())
        aud_rl = audio_onset(model, A42, diff, T, real_d, device)          # deployed baseline @ real density
        aud_ds = audio_onset(model, A42, diff, T, d_seq, device)           # density-matched baseline (fair vs SEQ)

        sc_real = critic_score(critic, a23, to_binary(orig), device)
        sc_shuf = critic_score(critic, a23, shuffle16(to_binary(orig), rng), device)
        sc_arl = critic_score(critic, a23, gen_panels(model, A42, T, diff, aud_rl, bpm, device), device)
        sc_ads = critic_score(critic, a23, gen_panels(model, A42, T, diff, aud_ds, bpm, device), device)
        sc_seq = critic_score(critic, a23, gen_panels(model, A42, T, diff, seq_on, bpm, device), device)
        rows.append((sc_real, sc_shuf, sc_arl, sc_ads, sc_seq, real_d, d_seq))
        print(f"  {meta['chart'].title[:24]:<26} {real_d:>7.3f} {d_seq:>6.3f} | {sc_real:>5.2f} {sc_shuf:>5.2f} "
              f"{sc_arl:>5.2f} {sc_ads:>5.2f} {sc_seq:>5.2f}", flush=True)

    R = np.array(rows); m = R.mean(0)
    print(f"\n  === taste-critic P(real), mean over {len(rows)} chaotic Hard songs (density d_seq {m[6]:.3f} vs real {m[5]:.3f}) ===", flush=True)
    print(f"  REAL          {m[0]:.3f}   <- CONTROL (high)", flush=True)
    print(f"  AUDIO@real_d  {m[2]:.3f}   (deployed baseline, real density)", flush=True)
    print(f"  AUDIO@d_seq   {m[3]:.3f}   <- FAIR one-change baseline (density-matched to SEQ)", flush=True)
    print(f"  SEQ@d_seq     {m[4]:.3f}   <- MEASUREMENT (SS head free-run onsets)", flush=True)
    print(f"  shuf16        {m[1]:.3f}   <- CONTROL (broken/low)", flush=True)

    if m[0] - m[1] <= 0.05:
        print(f"\n  !! CONTROL FAILED: REAL {m[0]:.3f} not >> shuf16 {m[1]:.3f} -> critic can't separate placement here; do not interpret.", flush=True)
        return
    seq_vs_aud = m[4] - m[3]
    pct = np.mean(R[:, 4] > R[:, 3]) * 100
    print(f"\n  CONTROL ok (REAL {m[0]:.3f} >> shuf16 {m[1]:.3f}). SEQ − AUDIO@d_seq = {seq_vs_aud:+.3f}  (SEQ wins {pct:.0f}% of songs)", flush=True)
    if seq_vs_aud >= 0.03:
        print(f"  => SEQ placement scores ABOVE the deployed baseline by the LENIENT musicality gate -> AUC-vs-real was", flush=True)
        print(f"     too strict (valid alternative phrasing) -> PROCEED to by-ear export (the binding gate).", flush=True)
    elif seq_vs_aud <= -0.03:
        print(f"  => SEQ placement scores BELOW the deployed baseline even by the lenient gate -> placement-hollow CONFIRMED", flush=True)
        print(f"     (M1b-4 stands; not just an over-strict metric) -> BANK fork (A), fall to the nearest-shippable.", flush=True)
    else:
        print(f"  => SEQ ≈ AUDIO (|Δ|<0.03): the seq head is NO BETTER than the deployed audio path by the critic. By-ear", flush=True)
        print(f"     (Rule 8) is the tiebreaker, but there is no critic-measured GAIN to justify the generate() wiring cost.", flush=True)
    print(f"\n  BOUNDARY: critic = a learned P(real) proxy, near-binary (taste-critic-transfer memo); by-ear stays the", flush=True)
    print(f"  binding gate. Stamina off (override) for both arms; per-note fatigue + playability on; radar off.", flush=True)


if __name__ == '__main__':
    main()
