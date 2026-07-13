"""Repo layout conventions (experiments/README.md) hold for all tracked files.

Thin pytest wrapper around tools/check_repo_layout.py so the layout is enforced by the suite,
not just by remembering to run the tool.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_repo_layout  # noqa: E402


def test_repo_layout_aligned():
    root = check_repo_layout.repo_root()
    bad = check_repo_layout.violations(check_repo_layout.tracked_files(root))
    assert not bad, "layout violations:\n" + "\n".join(bad)
