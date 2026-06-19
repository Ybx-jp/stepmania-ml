# Conditioning Roadmap — richer control knobs

*2026-06-18. Where to take the typed generator next: more ways to steer what it produces.*

Today the generator conditions on **audio** (cross-attention) + **target difficulty** (an embedding
added at every decoder position). We want more knobs: a specific **groove-radar profile**, **pattern
preferences** (more jumps, no crossovers, …), and **match a reference chart's style**.

These split across two surfaces:

- **Decode-time steering** — bias/constrain the existing model at generation. No retraining. Immediate,
  but limited to things expressible as logit nudges/constraints (we've already used this for density,
  hold rate, and pattern variety).
- **Trained conditioning** — feed a new control signal as input and train the model to obey it.
  More powerful and general; costs a retrain. Generalizes the difficulty embedding.

## Unified design

**Trained conditioning vector** `c`, added at every decoder position (today it's just the difficulty
embedding). Extend to a sum of:
- difficulty embedding (existing),
- **groove-radar projection**: `Linear(5 -> d)` of a target radar [stream, voltage, air, freeze, chaos],
- **style embedding**: a latent from a small chart encoder (for reference matching).

Training signal is free: each real chart's own radar (via `GrooveRadarCalculator`) and its own
encoder-latent are the teacher targets; at inference you set whatever target you want.

**Classifier-free guidance (CFG)** to make trained knobs actually bite: randomly drop the conditioning
during training (replace `c` with a learned null vector ~10-20% of the time); at inference, extrapolate
`logit = uncond + g*(cond - uncond)` with guidance scale `g>1` to push generation toward the condition.
Without it, weak conditioning is often ignored.

**Decode-time controls** (no training), layered on top:
- **pattern logit bias** (a 15-vector added to the pattern head): favor/suppress specific panel
  patterns -> "more jumps" (boost 2-panel patterns), panel preferences, avoid specific patterns.
- **foot-state transition mask**: track likely stepping foot; mask crossovers / impossible same-foot
  transitions at decode -> playability + "no crossovers" preference.
- temperature / threshold / repetition_penalty (already have).

## The three requested knobs, mapped

1. **Specific groove-radar profile** -> trained radar conditioning (+ CFG). The headline knob:
   "high-voltage, low-freeze." Evaluate by generating with target radar, recomputing the output's
   radar, and checking it matches. We already compute radar per chart, so this is the cleanest big win.
2. **Pattern preferences** -> mostly decode-time (pattern logit bias + foot/transition mask). "More
   jumps", panel prefs, "no crossovers" all expressible as biases/constraints. Cheap, immediate.
3. **Match a reference chart's style** -> two tiers:
   - *Cheap*: extract the reference's stats (radar, jump/jack rate, panel balance) and feed them through
     the radar-conditioning + pattern-bias knobs. Reuses 1 & 2.
   - *Full*: a learned **style embedding** — encode the reference chart with a small chart encoder
     (reuse the Phase 1 chart encoder / contrastive groove embeddings) into a latent, condition on it,
     train with same-song/augmented pairs. The real "match this chart."

## Evaluation (per knob)

Every knob needs a **conditioning-fidelity** metric — set target, measure achieved:
- radar: cosine/MAE between target radar and the generated chart's recomputed radar.
- pattern preference: did jump rate / panel balance move toward the request?
- style: similarity (groove-radar / embedding distance) of output to the reference.
- always check onset_F1 + difficulty critic don't regress.

## Suggested sequencing

- **Step 1 (cheap, immediate): decode-time pattern preferences + foot/transition constraints.**
  Gives "more jumps / no crossovers / panel prefs" now, no retrain, and improves playability. Builds
  the decode-time control surface.
- **Step 2 (headline): trained groove-radar conditioning + CFG.** The "specific radar profile" knob;
  generalizes the conditioning vector and sets up everything downstream. One retrain.
- **Step 3 (ambitious): reference-chart style embedding.** Reuse Phase 1 chart encoder; "match this
  chart." Builds on Step 2's conditioning infrastructure.

Recommended start: **Step 1** (immediate user-facing control, zero training risk), then **Step 2** as
the first trained conditioning knob.
