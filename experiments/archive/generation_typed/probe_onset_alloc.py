#!/usr/bin/env python3
"""PROBE: is per-window note ALLOCATION (and 16th-density specifically) LEARNABLE from what the onset
head ALREADY sees?  The cheap, decisive gate before building a learned-from-real density target.

CONTEXT (notes/sequence_aware_onset_plan.md, jack_heaviness_findings.md, conditioning-mechanics §8c):
the deployed onset head is audio-only + non-causal and thresholded by a SINGLE global tau, so it
(a) places ~zero 16th-adjacent onsets and (b) allocates density by audio SALIENCE more than humans do
(Probe 3B: corr(model_dens, real_dens) ~0.48; corr(model_dens, p_onset) 0.62 vs real 0.36). The breathing
arc (Stage 3) modulates a CEILING from a boxsmoothed p_onset envelope -- it can THIN but never UNLOCK a
16th, and the boxsmooth SMEARS event emphasis across the window.

The proposed fix is a LEARNED per-window density target (two-sided: can lower tau to add 16ths where real
charts afford them), with selection still by the sharp per-frame p_onset ranking (preserves emphasis).
This probe asks ONLY the prerequisite: can we PREDICT what real charts do per window from the head's
features?  If yes -> it's a READOUT problem (the signal is there, the global tau discards it) -> build the
target.  If no -> it's a REPRESENTATION problem -> the 06-22 verdict (note-context / richer audio) stands.

THREE nested predictors (cheapest -> richest), each ridge-regressed with GroupKFold-by-song (out-of-fold
scoring; no song leaks train->test) + StandardScaler fit on train folds only:
  envelope  : window-mean p_onset                         (1)   = the scalar the breathe arc uses today
  psummary  : window mean/std/max p_onset                 (3)   = a cheap richer readout of the same head
  encoder   : window-mean onset-encoder penultimate feats (128) = "is the signal in the representation?"
All three also get difficulty + radar appended (constant within a song -> cross-song signal only).

TARGETS (per window, win frames):  density (press rate) ; sixteenth_density (16th-offbeat presses / win,
phase t%4 in {1,3}, §6) -- the HEADLINE, always defined ; sixteenth_share (16th / all presses, windows
with >=1 press) -- reported as secondary.

Decision rule (FIXED before reading the number, experiment-design Rule 9):
  encoder beats envelope on sixteenth_density AND reaches a usefully high out-of-fold r (>~0.5)
    -> READOUT problem -> the learned-target build is justified.
  encoder ~= envelope ~= low
    -> REPRESENTATION problem -> 06-22 stands (note-context / richer features), no cheap target.

  python experiments/generation_typed/probe_onset_alloc.py [--songs 16] [--win 32] [--folds 5]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.playability_metrics import ACTIVE_SYMBOLS
from compare_foot_physics import load_songs, DIFF_NAMES

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def onset_feats(model, song, device):
    """Replicate onset_logits (typed_model.py:321-330) but TAP the encoder output BEFORE onset_head.
    Returns p_onset (T,) and penultimate features (T, d_model), matching the DEPLOYED conditioning
    (diff + radar; motif/figure decoupled per conditioning-mechanics §1)."""
    T = song['len']
    audio = torch.from_numpy(song['audio']).unsqueeze(0).to(device)
    dt = torch.tensor([song['diff']], device=device)
    radar_t = torch.from_numpy(song['radar']).unsqueeze(0).to(device)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        cond = model._cond(dt, radar_t, None, motif=None)              # onset conditioning (no motif)
        x = model.dropout(model.pos_encoding(memory) + cond)           # dropout is identity in eval()
        feat = model.onset_encoder(x)                                  # (1,T,d) penultimate
        p = torch.sigmoid(model.onset_head(feat).squeeze(-1))[0].cpu().numpy()[:T]
    return p, feat[0].cpu().numpy()[:T]


def window_rows(song, p, feat, win):
    """Per-window feature rows + targets for ONE song. Returns dict of arrays (nW each)."""
    real = np.asarray(song['real']); T = song['len']
    press = np.isin(real, ACTIVE_SYMBOLS).any(1)                       # (T,) press mask
    t = np.arange(T)
    is16 = np.isin(t % 4, (1, 3))                                      # 16th-offbeat phase (§6)
    nW = (T + win - 1) // win
    out = defaultdict(list)
    for w in range(nW):
        sl = slice(w * win, (w + 1) * win)
        pm = press[sl]; npress = int(pm.sum())
        n16 = int((press[sl] & is16[sl]).sum())
        pw = p[sl]
        # --- targets ---
        out['density'].append(pm.mean())
        out['s16_dens'].append(n16 / win)
        out['s16_share'].append(n16 / npress if npress > 0 else np.nan)
        # --- features (p-summary) ---
        out['f_env'].append([pw.mean()])
        out['f_psum'].append([pw.mean(), pw.std(), pw.max()])
        out['f_enc'].append(feat[sl].mean(0))                         # (d,) window-mean penultimate
        out['diff'].append(song['diff'])
    out['radar'] = [song['radar']] * nW
    return out


def cv_score(X, y, groups, folds):
    """Out-of-fold Pearson r and R^2 for ridge (scaler fit on train folds only)."""
    m = np.isfinite(y)
    X, y, groups = X[m], y[m], groups[m]
    if len(np.unique(groups)) < folds or len(y) < folds * 2:
        return float('nan'), float('nan')
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        pipe.fit(X[tr], y[tr]); pred[te] = pipe.predict(X[te])
    ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
    r = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-9 else float('nan')
    return r, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=16)
    ap.add_argument("--by", default="rich")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_songs(args, device)
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    # collect all windows, tagged by song-id (group) and difficulty
    acc = defaultdict(list); gid = []; diff = []
    fr = defaultdict(list)                                              # FRAME-level (the well-powered cut)
    for sidx, s in enumerate(songs):
        p, feat = onset_feats(model, s, device)
        rows = window_rows(s, p, feat, args.win)
        n = len(rows['density'])
        for k in ('density', 's16_dens', 's16_share', 'f_env', 'f_psum', 'f_enc', 'radar', 'diff'):
            acc[k].extend(rows[k])
        gid.extend([sidx] * n); diff.extend(rows['diff'])
        # frame-level: phase, real press, p_onset, encoder feats
        T = s['len']; t = np.arange(T)
        press = np.isin(np.asarray(s['real']), ACTIVE_SYMBOLS).any(1)
        fr['phase'].append(t % 4); fr['press'].append(press); fr['p'].append(p)
        fr['feat'].append(feat); fr['gid'].append(np.full(T, sidx)); fr['diff'].append(np.full(T, s['diff']))
    gid = np.array(gid); diff = np.array(diff)
    radar = np.array(acc['radar'], dtype=float); dcol = np.array(acc['diff'], dtype=float)[:, None]
    feats = {
        'envelope': np.hstack([np.array(acc['f_env']),  dcol, radar]),
        'psummary': np.hstack([np.array(acc['f_psum']), dcol, radar]),
        'encoder':  np.hstack([np.array(acc['f_enc']),  dcol, radar]),
    }
    targets = {'density': np.array(acc['density'], float),
               's16_dens': np.array(acc['s16_dens'], float),
               's16_share': np.array(acc['s16_share'], float)}

    print(f"\nPROBE — section ALLOCATION learnability  {len(songs)} songs, win={args.win} "
          f"frames, {args.folds}-fold GroupKFold-by-song (out-of-fold)")
    print(f"total windows: {len(gid)}   (ref: Probe 3B corr(model_dens,real_dens) ~0.48; head realizes ~0 16ths)\n")

    # pooled (all difficulties) + Med+Hard pool (Rule 12)
    pools = {'ALL': np.ones(len(gid), bool)}
    medhard = np.isin(diff, [2, 3])
    if medhard.sum() > args.folds * 3:
        pools['Med+Hard'] = medhard

    for pool_name, pmask in pools.items():
        print(f"================ pool: {pool_name}  ({int(pmask.sum())} windows) ================")
        hdr = f"   {'target':>12} | " + " | ".join(f"{fn:>16}" for fn in feats)
        print(hdr); print("   " + "-" * (len(hdr) - 3))
        for tname, y in targets.items():
            cells = []
            for fn, X in feats.items():
                r, r2 = cv_score(X[pmask], y[pmask], gid[pmask], args.folds)
                cells.append(f"r={r:+.2f} R2={r2:+.2f}")
            print(f"   {tname:>12} | " + " | ".join(f"{c:>16}" for c in cells))
        print()

    # ---- FRAME-LEVEL SELECTION cut (well-powered: thousands of frames, not 384 windows) ----
    # "given a phase-slot, does the head know if a human pressed it?"  AUC of p_onset, and of an
    # out-of-fold logistic on the encoder feats, restricted to each phase band. The 16th band is the
    # SELECTION half of the two-sided gate: if p_onset ranks real 16ths high, a lowered LOCAL tau would
    # surface them (global-tau-burial); if AUC ~chance, the head can't place 16ths even with a perfect budget.
    P = {k: np.concatenate(v) for k, v in fr.items() if k != 'feat'}
    FT = np.concatenate(fr['feat'])
    print("================ FRAME-LEVEL selection: AUC for 'is this slot a real press?' ================")
    print(f"   {'phase band':>14} {'n_frames':>9} {'base_rate':>9} {'p_onset AUC':>12} {'encoder AUC':>12}")
    bands = [('quarter', (0,)), ('8th', (2,)), ('16th-off', (1, 3))]
    for pool_name, dsel in (('ALL', [0, 1, 2, 3]), ('Med+Hard', [2, 3])):
        dm = np.isin(P['diff'], dsel)
        if dm.sum() == 0:
            continue
        print(f"   -- pool {pool_name} --")
        for bname, phs in bands:
            bm = dm & np.isin(P['phase'], phs)
            y = P['press'][bm].astype(int)
            if y.sum() < 10 or (1 - y).sum() < 10:
                continue
            auc_p = roc_auc_score(y, P['p'][bm])
            # out-of-fold encoder logistic
            X = FT[bm]; g = P['gid'][bm]; pred = np.full(len(y), np.nan)
            if len(np.unique(g)) >= args.folds:
                for tr, te in GroupKFold(n_splits=args.folds).split(X, y, g):
                    if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
                        continue
                    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5))
                    pipe.fit(X[tr], y[tr]); pred[te] = pipe.predict_proba(X[te])[:, 1]
            mok = np.isfinite(pred)
            auc_e = roc_auc_score(y[mok], pred[mok]) if mok.sum() > 10 and len(np.unique(y[mok])) == 2 else float('nan')
            print(f"   {bname:>14} {int(bm.sum()):>9} {y.mean():>9.3f} {auc_p:>12.3f} {auc_e:>12.3f}")
    print("   (ref 06-22 diag_seqcontext_probe: 16th-localization audio-only 0.649 vs note-context 0.935)\n")

    print("READ: HEADLINE = s16_dens (16th-offbeat density per window). If `encoder` r clears `envelope` by a "
          "real margin AND tops ~0.5 -> the 16th signal is IN the head's features, the global tau discards it -> "
          "a learned-target READOUT fix is justified (build it). If encoder~=envelope~=low -> REPRESENTATION "
          "problem -> note-context / richer audio (sequence_aware_onset_plan.md 06-22 verdict stands). "
          "density is the sanity column (should be well-predicted; it's most of what the head already does).")


if __name__ == "__main__":
    main()
