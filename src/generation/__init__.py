"""
Phase 2 generative utilities.

Stage 0 infrastructure for audio-conditioned chart generation:
- tokenizer: panel-state <-> token-id mapping for sequence modeling
- sm_writer: render a (T, 4) chart tensor back to a playable .sm file

See docs/phase2_generative_design.md.
"""

from .tokenizer import (
    ChartTokenizer,
    NUM_PANEL_STATES,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    VOCAB_SIZE,
)
from .sm_writer import tensor_to_sm, write_sm, charts_to_sm
from .evaluation import (
    onset_density_metrics,
    chart_groove_radar_vector,
    DifficultyCritic,
)

__all__ = [
    "ChartTokenizer",
    "NUM_PANEL_STATES",
    "PAD_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "VOCAB_SIZE",
    "tensor_to_sm",
    "write_sm",
    "charts_to_sm",
    "onset_density_metrics",
    "chart_groove_radar_vector",
    "DifficultyCritic",
]
