#!/usr/bin/env python3
"""
Stage 2a — train a REALISM CRITIC (discriminator) that scores P(real human chart | audio, chart).
See notes/stage2_realism_critic_plan.md.

The critic reuses the Phase-1 LateFusionClassifier backbone (dual audio/chart encoders + fusion +
backbone + masked pooling) with a binary real/fake head (num_classes=2), warm-started from the Phase-1
difficulty classifier. Three example types:
  positive      (audio_i, real_chart_i)            label 1
  neg-generated (audio_i, gen_chart_i)             label 0   (gen from gen_stage1, recommended decode)
  neg-mismatch  (audio_i, real_chart_j, j!=i)      label 0   (forces audio-grounding, not a density detector)

Validates: ROC-AUC real-vs-generated; that mismatched pairs score low; and the TASTE METRIC sanity —
P(real) should rank the playtested sets base > chaos.

Usage:
    python experiments/realism_critic/train_critic.py --data_dir data/ --audio_dir data/ \
        --max_train_songs 1500 --epochs 12
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
from src.models import LateFusionClassifier
from src.generation.typed_model import LayeredTypedChartGenerator

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"            # generator we want to critique/improve (41-dim)
CLS_CKPT = "checkpoints/ordinal_exp/standard_ordinal_multi/best_val_loss.pt"  # Phase-1 warm-start


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--max_train_songs', type=int, default=1500)
    p.add_argument('--max_val_songs', type=int, default=300)
    p.add_argument('--cache_dir', default='cache/samples_v2')
    p.add_argument('--checkpoint_dir', default='checkpoints/realism_critic')
    p.add_argument('--no_warmstart', action='store_true')
    return p.parse_args()


def to_binary(typed):
    t = np.asarray(typed); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def collect(ds, cap, max_len):
    out = []
    for i in range(len(ds)):
        if len(out) >= cap: break
        s = ds[i]; meta = ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), max_len)
        if T < 64: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        tf = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        out.append({'audio41': s['audio'][:T].numpy().astype(np.float32),
                    'real': to_binary(tf), 'difficulty': int(meta['difficulty_class']), 'T': T})
    return out


@torch.no_grad()
def gen_fakes(model, songs, device, max_len, target_density):
    """Generate one chart per song from the generator (recommended decode), density-matched."""
    model.eval()
    for k in range(0, len(songs), 8):
        chunk = songs[k:k + 8]; L = max(s['T'] for s in chunk); B = len(chunk)
        audio = torch.zeros(B, L, 41); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
        for b, s in enumerate(chunk):
            audio[b, :s['T']] = torch.from_numpy(s['audio41']); lengths[b] = s['T']; diff[b] = s['difficulty']
        audio = audio.to(device); diff = diff.to(device)
        # per-chunk density-matched threshold from the model's own onset logits
        p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff)).cpu().numpy()
        for b, s in enumerate(chunk):
            tau = float(np.quantile(p[b, :s['T']], 1 - max((s['real'] != 0).any(1).mean(), 1e-3)))
            g = model.generate(audio[b:b+1], diff[b:b+1], lengths=lengths[b:b+1].to(device),
                               onset_threshold=tau, type_sample=True, type_temperature=0.4, hold_aware=True,
                               pattern_sample=True, pattern_temperature=0.7, no_jump_during_hold=True)[0].cpu().numpy()
            s['gen'] = to_binary(g[:s['T']])


def build_examples(songs):
    """(audio_idx, chart_key, chart_idx, label): positive, neg-gen, neg-mismatch (1:1:1)."""
    n = len(songs); rng = np.random.default_rng(0); ex = []
    for i in range(n):
        ex.append((i, 'real', i, 1))
        ex.append((i, 'gen', i, 0))
        j = int(rng.integers(0, n))
        while j == i and n > 1: j = int(rng.integers(0, n))
        ex.append((i, 'real', j, 0))  # real chart from a DIFFERENT song over this audio
    return ex


def make_batch(songs, ex_batch, device):
    L = max(songs[a]['T'] for a, _, _, _ in ex_batch); B = len(ex_batch)
    audio = torch.zeros(B, L, 23); chart = torch.zeros(B, L, 4); mask = torch.zeros(B, L); y = torch.zeros(B, dtype=torch.long)
    for b, (ai, key, ci, lab) in enumerate(ex_batch):
        T = songs[ai]['T']
        audio[b, :T] = torch.from_numpy(songs[ai]['audio41'][:, :23])  # critic uses the 23-dim slice
        c = songs[ci][key]                                             # (Tc,4); crop/pad to this audio's T
        Tc = min(len(c), T); chart[b, :Tc] = torch.from_numpy(c[:Tc])
        mask[b, :T] = 1.0; y[b] = lab
    return audio.to(device), chart.to(device), mask.to(device), y.to(device)


@torch.no_grad()
def p_real(model, audio23, chart, mask):
    logits = model(audio23, chart, mask)
    if isinstance(logits, dict): logits = logits['logits']
    return torch.softmax(logits, 1)[:, 1]


def roc_auc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels)
    order = np.argsort(s); y = y[order]
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0: return float('nan')
    ranks = np.argsort(np.argsort(s)) + 1
    auc = (ranks[np.asarray(labels) == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(cf, random_state=args.seed)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=msl,
                                          feature_extractor=ext, cache_dir=args.cache_dir)
    train_ds.warm_cache(show_progress=False); val_ds.warm_cache(show_progress=False)
    print("collecting real samples...")
    train = collect(train_ds, args.max_train_songs, args.max_len)
    val = collect(val_ds, args.max_val_songs, args.max_len)
    print(f"train songs={len(train)} val songs={len(val)}")

    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()
    print("generating fakes (train)..."); gen_fakes(gen, train, device, args.max_len, None)
    print("generating fakes (val)...");   gen_fakes(gen, val, device, args.max_len, None)
    del gen; torch.cuda.empty_cache()

    # critic = LateFusionClassifier with a binary head, no radar/projection (judge audio+chart only)
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']
    cfg = dict(cfg); cfg['num_classes'] = 2; cfg['head_type'] = 'classification'
    cfg['use_groove_radar'] = False; cfg['use_projection_head'] = False
    if args.no_warmstart:
        critic = LateFusionClassifier(cfg).to(device)
    else:
        critic = LateFusionClassifier.from_pretrained(CLS_CKPT, cfg, device=str(device))  # encoders transfer, head fresh
    print(f"critic params: {sum(p.numel() for p in critic.parameters()):,}")
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_auc = -1.0; best_path = Path(args.checkpoint_dir) / "best_val.pt"
    train_ex = build_examples(train); val_ex = build_examples(val)

    for epoch in range(args.epochs):
        critic.train(); rng.shuffle(train_ex); tot = 0.0; nb = 0
        for k in range(0, len(train_ex), args.batch_size):
            audio, chart, mask, y = make_batch(train, train_ex[k:k + args.batch_size], device)
            opt.zero_grad()
            logits = critic(audio, chart, mask)
            loss = crit(logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        # eval
        critic.eval(); sc, lb = [], []; cat = {'real': [], 'gen': [], 'mismatch': []}
        with torch.no_grad():
            for k in range(0, len(val_ex), args.batch_size):
                eb = val_ex[k:k + args.batch_size]
                audio, chart, mask, y = make_batch(val, eb, device)
                pr = p_real(critic, audio, chart, mask).cpu().numpy()
                for (ai, key, ci, lab), p in zip(eb, pr):
                    sc.append(p); lb.append(lab)
                    cat['real' if (key == 'real' and lab == 1) else ('gen' if key == 'gen' else 'mismatch')].append(p)
        auc = roc_auc(sc, lb)
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.3f}  val real-vs-all AUC={auc:.3f}  "
              f"P(real): real={np.mean(cat['real']):.3f} gen={np.mean(cat['gen']):.3f} mismatch={np.mean(cat['mismatch']):.3f}"
              + ("  *" if auc > best_auc else ""))
        if auc > best_auc:
            best_auc = auc
            torch.save({'model_state_dict': critic.state_dict(), 'config': cfg, 'epoch': epoch, 'val_auc': auc}, best_path)

    print(f"\nbest val AUC={best_auc:.3f} -> {best_path}")
    print("Next: score the playtested sets (base vs chaos) with eval_taste.py to validate the metric.")


if __name__ == '__main__':
    main()
