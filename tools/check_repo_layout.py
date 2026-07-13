#!/usr/bin/env python3
"""VALIDATOR: the repo layout conventions in experiments/README.md must hold for all TRACKED files.

Sibling of tools/check_export_defaults.py (doc<->code alignment); this one is layout<->convention
alignment. Runs from anywhere inside the repo; judges `git ls-files` (what the repo ships), not the
working tree, so local scratch/ignored files never trip it.

Rules:
  R1  No *.py at the repo root — probes go in experiments/probes/, tools in tools/, CLIs in scripts/.
  R2  train_*.py under experiments/ (outside archive/) must be on the LIVE_TRAINERS allowlist.
      Adding a trainer is a deliberate act: update the allowlist in the same commit.
  R3  No *.csv / *.log tracked under cache/ — cache/ holds regenerable feature caches and fitted
      artifacts only; probe results belong in outputs/probe_results/ (gitignored) or notes/.
Exit 0 = aligned; exit 1 = violations listed on stdout.
"""
import subprocess
import sys
from pathlib import Path

# Live trainers (experiments/, non-archive). Deliberately short — see experiments/README.md rule 6.
LIVE_TRAINERS = {
    "experiments/generation_typed/train_motif_figure.py",       # legacy v1 deployed (gen_motif_full_fixed)
    "experiments/generation_typed/train_motif_figure_v2.py",    # deployed v2 48th (gen_motif_v2_48th_cont)
    "experiments/generation_factorized/train_factorized.py",    # factorized lineage head
    "experiments/generation_transformer/train_transformer.py",  # transformer lineage head
    "experiments/realism_critic/train_graded_critic_v2.py",     # graded taste/realism critic (explore branch)
    "experiments/realism_critic/train_critic.py",
    "experiments/realism_critic/train_graded_critic.py",
}


def repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def violations(files: list[str]) -> list[str]:
    bad = []
    for f in files:
        p = Path(f)
        if p.suffix == ".py" and len(p.parts) == 1:
            bad.append(f"R1 root .py: {f}  (move to experiments/probes/ or tools/)")
        if (
            p.name.startswith("train_")
            and p.suffix == ".py"
            and p.parts[0] == "experiments"
            and "archive" not in p.parts
            and f not in LIVE_TRAINERS
        ):
            bad.append(f"R2 unlisted trainer: {f}  (archive it, or add to LIVE_TRAINERS deliberately)")
        if p.parts[0] == "cache" and p.suffix in {".csv", ".log"}:
            bad.append(f"R3 result file in cache/: {f}  (belongs in outputs/probe_results/ or notes/)")
    return sorted(bad)


def main() -> int:
    root = repo_root()
    bad = violations(tracked_files(root))
    if bad:
        print(f"LAYOUT VIOLATIONS ({len(bad)}):")
        for b in bad:
            print(" ", b)
        return 1
    print("ALIGNED ✓  (repo layout matches experiments/README.md conventions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
