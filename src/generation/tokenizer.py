"""
Chart tokenizer for autoregressive generation.

A Phase 1 chart timestep is a (4,) binary vector over panels [Left, Down, Up, Right].
There are 2^4 = 16 possible panel-states. We map each timestep to a single token id
so a chart becomes a length-T sequence over a small vocabulary:

    token ids  0 .. 15   -> the 16 panel-states
    token id   16        -> PAD
    token id   17        -> BOS
    token id   18        -> EOS
    VOCAB_SIZE = 19

Panel-state <-> token encoding is a lossless bit-packing for Phase 1 scope
(steps and jumps only; no holds/rolls). Bit i corresponds to panel i:

    token = sum(panel_state[i] << i  for i in range(4))
    e.g. [0,1,0,1] (Down+Right) -> 0b1010 = 10

See docs/phase2_generative_design.md.
"""

from typing import Optional

import numpy as np
import torch

NUM_PANELS = 4
NUM_PANEL_STATES = 1 << NUM_PANELS  # 16

PAD_TOKEN = NUM_PANEL_STATES        # 16
BOS_TOKEN = NUM_PANEL_STATES + 1    # 17
EOS_TOKEN = NUM_PANEL_STATES + 2    # 18
VOCAB_SIZE = NUM_PANEL_STATES + 3   # 19

SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)

# Precomputed (16, 4) lookup: token id -> panel-state row, for fast vectorized decode.
# Row s has bit i set iff panel i is active in state s.
_STATE_TO_PANELS = np.array(
    [[(s >> i) & 1 for i in range(NUM_PANELS)] for s in range(NUM_PANEL_STATES)],
    dtype=np.float32,
)
# Bit weights [1, 2, 4, 8] for packing a (T, 4) row into a state index.
_BIT_WEIGHTS = (1 << np.arange(NUM_PANELS)).astype(np.int64)


class ChartTokenizer:
    """Lossless (T, 4) chart-tensor <-> token-id-sequence converter.

    Stateless; instances exist only to namespace the API and expose the vocab.
    """

    num_panel_states = NUM_PANEL_STATES
    pad_token = PAD_TOKEN
    bos_token = BOS_TOKEN
    eos_token = EOS_TOKEN
    vocab_size = VOCAB_SIZE

    # ---- single-timestep helpers ------------------------------------------------

    @staticmethod
    def panel_state_to_token(panel_state) -> int:
        """Encode one (4,) binary panel-state into a token id 0..15."""
        arr = np.asarray(panel_state).reshape(-1)
        if arr.shape[0] != NUM_PANELS:
            raise ValueError(f"panel_state must have {NUM_PANELS} entries, got {arr.shape[0]}")
        bits = (arr > 0.5).astype(np.int64)
        return int(bits @ _BIT_WEIGHTS)

    @staticmethod
    def token_to_panel_state(token: int) -> np.ndarray:
        """Decode a panel-state token id 0..15 into a (4,) float binary row."""
        if not (0 <= token < NUM_PANEL_STATES):
            raise ValueError(
                f"token {token} is not a panel-state (expected 0..{NUM_PANEL_STATES - 1})"
            )
        return _STATE_TO_PANELS[token].copy()

    # ---- full-chart helpers -----------------------------------------------------

    @staticmethod
    def encode(chart: "np.ndarray | torch.Tensor", add_special: bool = False) -> torch.Tensor:
        """Encode a (T, 4) binary chart tensor into a (T,) LongTensor of token ids.

        Args:
            chart: (T, 4) array/tensor, values in {0, 1} (thresholded at 0.5).
            add_special: if True, prepend BOS and append EOS -> length T + 2.
        """
        if isinstance(chart, torch.Tensor):
            arr = chart.detach().cpu().numpy()
        else:
            arr = np.asarray(chart)
        if arr.ndim != 2 or arr.shape[1] != NUM_PANELS:
            raise ValueError(f"chart must be (T, {NUM_PANELS}), got {arr.shape}")

        bits = (arr > 0.5).astype(np.int64)
        tokens = bits @ _BIT_WEIGHTS  # (T,)

        if add_special:
            tokens = np.concatenate([[BOS_TOKEN], tokens, [EOS_TOKEN]])
        return torch.from_numpy(tokens).long()

    @staticmethod
    def decode(tokens: "np.ndarray | torch.Tensor") -> np.ndarray:
        """Decode a (T,) sequence of token ids back to a (T', 4) binary chart tensor.

        Special tokens (PAD/BOS/EOS) are dropped, so T' <= T.
        """
        if isinstance(tokens, torch.Tensor):
            arr = tokens.detach().cpu().numpy()
        else:
            arr = np.asarray(tokens)
        arr = arr.reshape(-1).astype(np.int64)

        keep = arr < NUM_PANEL_STATES  # drop PAD/BOS/EOS
        states = arr[keep]
        if states.size == 0:
            return np.zeros((0, NUM_PANELS), dtype=np.float32)
        return _STATE_TO_PANELS[states].copy()
