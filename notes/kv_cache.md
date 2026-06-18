# KV-Cache for Full-Length Generation (Stage 3 infra)

*2026-06-18. Branch `gen/kv-cache`.*

`generate()` re-ran the whole panel decoder over the growing token sequence at every
step — O(T²) — so generation eval was capped at 768 frames. Added `generate_cached()`:
a [[KV-cache]]d inference path that processes only the new token each step against
cached keys/values — O(T) — enabling full 1440-frame (2-minute) songs.

## Implementation

`nn.TransformerDecoder` has no incremental-decode API, so the cached path reuses the
trained layer weights via manual attention (no retraining, weight-compatible):
- per layer, a growing self-attention K/V cache (append the new token's K/V each step);
- cross-attention K/V to the fixed audio memory computed **once** per layer;
- post-norm + relu to match `nn.TransformerDecoderLayer` exactly.

Helpers `_project` / `_attend` / `_LayerCache` in `src/generation/factorized.py`;
`F.scaled_dot_product_attention` for the core attention (same op MHA uses internally).

## Verification

- **Bit-identical to non-cached** `generate()` (greedy, same onset): 0/600 timesteps
  differ on a random model; identical on the real focal checkpoint at T=1440. Regression
  test `test_kv_cache_matches_noncached` (14 generation tests pass).
- **Speedup scales with length** (O(T²) → O(T)):
  - T=1024: 9.1s → 2.0s (4.5×)
  - T=1440: 33.4s → 3.6s (**9.2×**), batch of 4

## Result

Full 2-minute (1440-frame) chart generation is now practical (~3.6s for a batch of 4 on
the RTX 3060), removing the 768-frame eval cap. Same args as `generate()`
(threshold / Bernoulli / calibration / `onset_override` / panel sampling all supported).

Use `generate_cached` for inference and full-length eval; `generate` remains as the
simple reference implementation the cache is tested against.
