#!/usr/bin/env bash
# One-time physical cleanup of GITIGNORED artifacts (they exist only in the primary checkout).
#
#   ⚠ Run in the PRIMARY checkout, from the repo root, when NO training/probe/Claude session is
#     active — it relocates files that running code may hold open. Nothing is deleted, only moved;
#     every move is echoed so it can be reversed.
#
#   Dry-run by default:      bash tools/migrate_artifacts.sh
#   Actually move things:    bash tools/migrate_artifacts.sh --apply
#
# What it does:
#   1. cache/  → probe RESULT files (*.csv, *.log, *.txt) move to outputs/probe_results/.
#      Functional artifacts stay: samples*/ feature caches, *.npz fitted artifacts, *.pt probe
#      heads, manifest.json files.
#   2. repo root → stray build/train logs move to outputs/logs/.
#   3. checkpoints/ → everything NOT on the KEEP list moves to checkpoints/archive/.
#      (Old checkpoints stay loadable at their archived path; move back to rerun an old probe.)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1
run() { echo "  $*"; [[ $APPLY -eq 1 ]] && "$@" || true; }
[[ $APPLY -eq 1 ]] || echo "== DRY RUN (pass --apply to execute) =="

# Live checkpoints that stay at their canonical paths (code/skills reference them):
KEEP_CKPT=(
  gen_motif_v2_48th_cont     # DEPLOYED v2 model (both CLIs' default)
  gen_motif_full_fixed       # legacy v1 deployed model (--features highres)
  gen_style                  # legacy 23-dim model (exporter compat)
  realism_critic realism_critic_graded realism_critic_graded_v2   # critic family
  ordinal_exp                # classifier evaluate.py example checkpoint
  archive
)

echo "== 1. cache/ result files -> outputs/probe_results/"
mkdir -p outputs/probe_results
for f in cache/*.csv cache/*.log cache/*.txt; do
  [[ -e "$f" ]] || continue
  run mv "$f" outputs/probe_results/
done

echo "== 2. root logs -> outputs/logs/"
mkdir -p outputs/logs
for f in cache_v2_build.log train_v2_48th.log train_v2_48th_cont.log probe_v2grid.log out.txt; do
  [[ -e "$f" ]] || continue
  run mv "$f" outputs/logs/
done

echo "== 3. dead checkpoints -> checkpoints/archive/"
mkdir -p checkpoints/archive
for d in checkpoints/*/; do
  name=$(basename "$d")
  keep=0
  for k in "${KEEP_CKPT[@]}"; do [[ "$name" == "$k" ]] && keep=1 && break; done
  [[ $keep -eq 1 ]] && continue
  run mv "$d" checkpoints/archive/
done
# loose root checkpoint files from the earliest classifier era
for f in checkpoints/best_val_loss.pt checkpoints/last.pt; do
  [[ -e "$f" ]] || continue
  run mv "$f" checkpoints/archive/
done

echo "== done. verify with: python tools/cache.py status ; ls checkpoints ; pytest tests/ -k layout"
