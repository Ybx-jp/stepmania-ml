#!/usr/bin/env python
"""COMPAT SHIM — the canonical exporter moved to scripts/export_typed_samples.py (Phase-2 reorg, 2026-07-12).

Every historical command, note, and skill referencing this path keeps working: the shim executes the
real module's code in this namespace, so both `python experiments/generation_typed/export_typed_samples.py ...`
and `import export_typed_samples` behave identically to the real file. New work should reference
scripts/export_typed_samples.py directly.
"""
from pathlib import Path as _Path

_real = _Path(__file__).resolve().parents[2] / "scripts" / "export_typed_samples.py"
__file__ = str(_real)  # so the real module's PROJECT_ROOT (parents[1] of __file__) resolves correctly
exec(compile(_real.read_text(), str(_real), "exec"))
