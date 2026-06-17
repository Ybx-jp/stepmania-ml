# Contrastive Learning Experiments: Fixing Representation Rigidity

## The Problem

When loading a pretrained classifier (80% validation accuracy) and attempting to fine-tune it with contrastive learning, the model **refused to learn**. Validation accuracy stayed stuck at exactly 80% throughout training, with no improvement in the contrastive objective.

This is a classic case of **representation rigidity** in transfer learning.

---

## What is Representation Rigidity?

When you pretrain a model on a task (difficulty classification) and then try to add a new objective (groove radar similarity), the model can become "geometrically frozen." Here's what happens:

### The Lifecycle of Representation Learning

```
Fresh model → Too floppy, learns nothing meaningful
    ↓
Pretrained model → Well-organized, good at original task
    ↓
Pretrained + New Objective → TOO RIGID, refuses to adapt
```

### Why This Happens

1. **Classification task is already solved** (80% accuracy)
   - Gradients from classification loss are **tiny and stable**
   - The model is sitting in a comfortable local minimum
   - No pressure to change the representation

2. **Contrastive gradients are too weak**
   - The new objective conflicts with the stable geometry
   - But the conflict isn't strong enough to overcome the inertia
   - Contrastive loss gradients get "averaged away"

3. **No strong conflict between objectives**
   - The model can satisfy both losses without moving much
   - Difficulty and groove similarity might partially correlate
   - The path of least resistance is to stay put

### The Mental Model

Think of it like this:

> **Pretraining gives you axes** (e.g., "this is a Hard chart" axis).
>
> **Multitask fine-tuning must be allowed to bend those axes** to accommodate the new task (e.g., "this has high Stream" axis).
>
> But if the old axes are working perfectly, why would the model bother bending them?

---

## The Diagnosis: What We Checked

### 1. Gradient Flow
✅ **Parameters are trainable** (`requires_grad=True`)
✅ **Optimizer includes all parameters**
❌ **But gradient magnitudes are probably tiny in the backbone**

### 2. Loss Landscape
✅ **Classification loss is stable** (task already solved)
❌ **Contrastive loss may not be decreasing meaningfully**
❌ **Classification gradients dominate; contrastive gradients are weak**

### 3. Representation Change
❌ **Embeddings don't drift from initial state**
❌ **Cosine similarity between epoch 0 and epoch N is ~0.99**
❌ **Backbone parameters barely update**

### The Smoking Gun

If you compute the cosine similarity between embeddings at epoch 0 vs epoch 10, and you see something like **0.995**, that means the representation moved by only **0.5%**. The backbone didn't budge.

---

## The Solution: Controlled Plasticity

The fix is to create **controlled plasticity** — forcing the model to adapt to the new objective without destroying what it already learned.

### Core Insight

> If the backbone never moves, accuracy never dips.
> If accuracy never dips, the backbone never moved.

**We WANT validation accuracy to drop temporarily** as proof that the representation is being reshaped. It should recover later.

---

## Two Experimental Approaches

We implemented two parallel experiments to test different strategies:

| | **Experiment A: Aggressive** | **Experiment B: Conservative** |
|---|---|---|
| **Strategy** | Freeze classifier head during warmup | Freeze only low-level encoders |
| **Mechanics** | 5 epochs with `cls_weight=0`, then unfreeze | Freeze audio/chart encoders permanently |
| **Loss Weights** | `cls_weight=0.3`, `contrastive_weight=2.0` | `cls_weight=0.8`, `contrastive_weight=2.0` |
| **Expected Drop** | Val acc: 80% → 70-75% (warmup) | Val acc: stable 78-80% |
| **Recovery** | Val acc: 78-82% (fine-tune) | Slower adaptation |
| **Risk** | Higher disruption | Safer, but slower |

### Common Changes (Both Experiments)

Both experiments share these improvements:

1. **Boosted contrastive pressure:**
   - `triplet_margin: 2.0` (up from 1.0)
   - `margin_scale: 1.0` (up from 0.5)
   - `contrastive_weight: 2.0` (up from 1.0)

2. **Harder triplet mining:**
   - `positive_percentile: 15.0` (down from 20.0) — stricter positives
   - `negative_percentile: 85.0` (up from 80.0) — harder negatives

3. **Full diagnostics:**
   - Per-module gradient norms
   - Embedding drift tracking
   - Triplet margin statistics

---

## Experiment A: Aggressive (Recommended)

### The Strategy

**Phase 1: Warmup (Epochs 1-5)**
- Freeze the classifier head completely
- Set classification loss weight to **zero**
- Let contrastive loss **own** the backbone
- Force the representation to reshape for groove similarity

**Phase 2: Fine-tune (Epochs 6+)**
- Unfreeze classifier head
- Increase classification weight to 0.3
- Both losses co-train
- Classification performance recovers

### Why This Works

1. **Eliminates the stable gradient source**
   No classification gradients = no anchor holding the geometry in place

2. **Creates strong conflict**
   Contrastive loss is the only objective → must adapt or fail

3. **Allows recovery**
   Classification comes back later and "fits" into the new geometry

### Expected Timeline

```
Epoch 1-5:  Classifier frozen, cls_weight=0
            Val acc drops: 80% → 75% → 72% → 70%
            Contrastive loss: 1.5 → 0.8 → 0.5
            ✅ THIS IS GOOD! Backbone is moving!

Epoch 6:    PHASE TRANSITION
            Unfreeze classifier, cls_weight=0.3

Epoch 6-20: Both objectives co-train
            Val acc recovers: 70% → 75% → 79% → 81%
            Contrastive loss: 0.5 → 0.3
            ✅ Both objectives satisfied!
```

### The Psychology of This Approach

This is like telling the model:

> "Forget about classification for a moment. Just focus on making similar charts close together. I don't care if you temporarily mess up difficulty prediction — we'll fix that later."

---

## Experiment B: Conservative

### The Strategy

**Selective Unfreezing:**
- **Freeze:** `audio_encoder`, `chart_encoder` (low-level features)
- **Train:** `fusion_module`, `backbone`, `pooling`, `classifier_head`, `projection_head`

**Rationale:**
- Low-level encoders learn audio/chart primitives (beats, note patterns)
- These are universally useful and shouldn't be changed
- High-level reasoning (fusion, backbone) needs to adapt for similarity
- Less disruption, but also less freedom to reshape

### Why This Works

1. **Preserves learned features**
   Audio/chart patterns stay intact

2. **Allows high-level adaptation**
   The backbone can still learn groove-based reasoning

3. **Safer but slower**
   Less risk of catastrophic forgetting, but may adapt more slowly

### Expected Behavior

```
Epoch 1-20: Continuous training
            Val acc: 80% → 79% → 78% → 79% → 80%
            Contrastive loss: 1.5 → 1.0 → 0.7

            ✅ Stable, gradual improvement
            ❌ Less dramatic representation shift
```

---

## Implementation Details

### Architecture Overview

```
Input (audio + chart)
    ↓
[audio_encoder]  [chart_encoder]  ← Experiment B: FROZEN
    ↓                ↓
    └────[fusion_module]────        ← Experiment B: trainable
              ↓
        [backbone]                   ← Both: trainable (but needs pressure!)
              ↓
        [pooling]                    ← Both: trainable
              ↓
    ┌─────────┴──────────┐
    ↓                    ↓
[classifier_head]  [projection_head]
    ↓                    ↓
Difficulty          Embeddings
(classification)    (contrastive)
```

### Warmup Schedule (Experiment A)

```python
# Epoch 0: Initialize
if warmup_epochs > 0:
    freeze_modules(['classifier_head'])
    classification_weight = 0.0

# Epoch 1-5: Warmup phase
for epoch in range(1, warmup_epochs + 1):
    train_with_contrastive_only()

# Epoch 6: Phase transition
if current_epoch == warmup_epochs:
    unfreeze_modules(['classifier_head'])
    classification_weight = 0.3
    print("PHASE TRANSITION: Warmup → Fine-tune")

# Epoch 6+: Fine-tune phase
for epoch in range(warmup_epochs + 1, num_epochs):
    train_with_both_losses()
```

### Selective Freezing (Experiment B)

```python
selective_unfreeze = [
    'fusion_module',
    'backbone',
    'pooling',
    'classifier_head',
    'projection_head',
    'radar_mlp'
]

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only specified modules
for name, param in model.named_parameters():
    for module in selective_unfreeze:
        if name.startswith(module):
            param.requires_grad = True
```

---

## Diagnostic System

We built comprehensive diagnostics to understand exactly what's happening:

### 1. Gradient Norms

**What:** Measure `||∇θ||` for each module after backward pass

**Why:** Tells us if gradients are flowing to the backbone

**Red flags:**
- Backbone gradient norm < 1e-5 (too small!)
- Classifier gradient norm 100x larger than backbone (imbalanced!)

**Good signs:**
- Backbone gradient norm ~ 1e-3 to 1e-1
- Backbone/classifier ratio > 0.1

### 2. Embedding Drift

**What:** Cosine similarity and L2 distance between epoch 0 and epoch N embeddings

**Why:** Direct measurement of representation change

**Metrics:**
```python
# Normalize embeddings
emb_0 = embeddings_epoch_0 / ||embeddings_epoch_0||
emb_N = embeddings_epoch_N / ||embeddings_epoch_N||

# Cosine similarity (higher = less change)
cosine_sim = mean(emb_0 · emb_N)  # Want this < 0.95

# L2 distance (higher = more change)
l2_dist = mean(||emb_0 - emb_N||)  # Want this > 0.1
```

**Red flags:**
- Cosine similarity > 0.99 → No movement
- L2 distance < 0.05 → Backbone didn't adapt

**Good signs:**
- Cosine similarity: 0.85-0.95 → Significant but not catastrophic change
- L2 distance > 0.1 → Meaningful representation shift

### 3. Triplet Margin Statistics

**What:** Track `d(anchor, negative) - d(anchor, positive)`

**Why:** Measures separation quality in embedding space

**Target:**
- Mean margin should increase toward `triplet_margin` (2.0)
- Margin improvement > 0.5 over training
- Positive margin ratio > 80% (most triplets satisfied)

---

## How to Use

### Running Experiment A (Aggressive)

```python
# In notebooks/contrastive_training.ipynb
EXPERIMENT = 'A'

# Load pretrained model
model = LateFusionClassifier.from_pretrained(
    checkpoint_path='models/classifier/best_val_loss.pt',
    config=model_config,
    device='cuda'
)

# Create trainer (warmup parameters loaded from config)
trainer = ContrastiveTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    config=config,
    warmup_epochs=5,           # From config
    warmup_cls_weight=0.0,     # From config
    finetune_cls_weight=0.3    # From config
)

# Train!
history = trainer.fit()
```

### Running Experiment B (Conservative)

```python
# In notebooks/contrastive_training.ipynb
EXPERIMENT = 'B'

# Load pretrained model with selective freezing
selective_freeze = ['audio_encoder', 'chart_encoder']
model = LateFusionClassifier.from_pretrained(
    checkpoint_path='models/classifier/best_val_loss.pt',
    config=model_config,
    selective_freeze=selective_freeze,
    device='cuda'
)

# OR: Use selective_unfreeze in trainer
trainer = ContrastiveTrainer(
    model=model,
    ...,
    selective_unfreeze=['fusion_module', 'backbone', 'pooling',
                        'classifier_head', 'projection_head', 'radar_mlp']
)

history = trainer.fit()
```

---

## Interpreting Results

### Success Indicators

#### Experiment A Success:
1. **Warmup (Epochs 1-5):**
   - ✅ Val acc drops below 80% (ideally to 70-75%)
   - ✅ Contrastive loss decreases rapidly (>30% reduction)
   - ✅ Embedding L2 drift > 0.1 by epoch 5
   - ✅ Backbone gradient norms > 1e-3

2. **Fine-tune (Epochs 6+):**
   - ✅ Val acc recovers to 78-82%
   - ✅ Contrastive loss continues to decrease
   - ✅ Triplet margin increases toward 2.0
   - ✅ Both gradient norms remain balanced

#### Experiment B Success:
1. **Throughout Training:**
   - ✅ Val acc stable (78-80%)
   - ✅ Contrastive loss decreases steadily
   - ✅ Embedding drift > 0.05
   - ✅ Gradient flow concentrated in unfrozen modules

### Failure Indicators

❌ **Validation accuracy never dips below 79.5%** (Experiment A)
→ Classifier head wasn't actually frozen, or classification weight still too high

❌ **Contrastive loss doesn't decrease much (<10%)**
→ Margins too small, need harder negatives, or backbone still frozen

❌ **Embedding drift < 0.03**
→ Representation barely changed, need more aggressive approach

❌ **Backbone gradient norm < 1e-5**
→ Gradients not reaching backbone, check freezing logic

### Diagnostic Plots

After training, check `checkpoints/contrastive/diagnostics/training_curves.png`:

**Panel 1: Gradient Norms**
- Look for: Backbone norm should be visible (not flat at zero)
- Experiment A: Should spike after epoch 6 (classifier unfreezes)
- Experiment B: Should be stable but non-zero

**Panel 2: Triplet Distances**
- Look for: Green (A-P) and red (A-N) lines separating over time
- Target: A-N distance > A-P distance by at least margin (2.0)

**Panel 3: Triplet Margin**
- Look for: Upward trend toward 2.0
- Should be positive for >80% of triplets

**Panel 4: Cosine Similarity**
- Look for: Downward trend from 1.0
- Target: End between 0.85-0.95
- If stuck at 0.99: backbone didn't move

**Panel 5: L2 Distance**
- Look for: Upward trend
- Target: > 0.1 by end of training
- If < 0.05: representation barely changed

**Panel 6: Gradient Ratio**
- Look for: Ratio > 0.1 (backbone getting 10% of classifier gradients)
- Experiment A: Should jump at epoch 6

---

## Theoretical Background

### Why Contrastive Learning?

Difficulty classification alone teaches:
- "This chart is Hard"
- "This chart is Easy"

But it doesn't capture **groove similarity**:
- Two Hard charts might feel completely different (technical vs stamina)
- Stream patterns create similarity across difficulty levels

Contrastive learning adds:
- "These two charts feel similar (high Stream, low Air)"
- "These charts feel different despite same difficulty"

### Triplet Loss with Adaptive Margins

Standard triplet loss:
```
L = max(0, d(a,p) - d(a,n) + margin)
```

Our adaptive version:
```
margin = base_margin + margin_scale * ||radar_a - radar_n||

L = max(0, d(a,p) - d(a,n) + margin)
```

**Why adaptive?**
- If anchor and negative have very different groove radars (e.g., Stream=10 vs Stream=90), demand more separation in embedding space
- If they're similar in radar space, allow smaller embedding separation
- This creates **geometry that respects groove radar structure**

### The Warmup Trick

This is inspired by curriculum learning and staged training:

1. **Stage 1: Learn the new task in isolation**
   Contrastive objective reshapes the representation freely

2. **Stage 2: Reconcile both tasks**
   Classification objective "fits into" the new geometry

This is similar to:
- **Discriminative fine-tuning** (different LR per layer)
- **Gradual unfreezing** (ULMFit approach)
- **Warm-up in transformers** (low LR → high LR)

But instead of learning rate, we control **which objectives are active**.

---

## Troubleshooting

### "Experiment A: Val acc never drops below 79%"

**Diagnosis:** Classifier head is still receiving gradients

**Fix:**
1. Check that `warmup_epochs > 0` in config
2. Verify `classifier_head` appears in frozen modules list during warmup
3. Check training log for "WARMUP PHASE: Freezing classifier_head" message
4. Increase `warmup_epochs` to 10

### "Experiment B: Contrastive loss decreases but embeddings don't drift"

**Diagnosis:** Model is moving projection head, not backbone

**Fix:**
1. Check which modules are actually trainable: print trainable param count
2. Verify `selective_unfreeze` includes `backbone` and `fusion_module`
3. Try unfreezing top 50% of encoders too
4. Increase `contrastive_weight` to 3.0

### "Both experiments: Triplet margin stays near zero"

**Diagnosis:** Negatives are too easy, or margin is too high

**Fix:**
1. Reduce `positive_percentile` to 10.0 (even stricter positives)
2. Increase `negative_percentile` to 90.0 (even harder negatives)
3. Check that `same_difficulty_only=true` (prevents trivial solutions)
4. Lower `triplet_margin` to 1.5 initially

### "Gradients are NaN"

**Diagnosis:** Exploding gradients or numerical instability

**Fix:**
1. Check `gradient_clip_norm` is set (should be 1.0)
2. Reduce learning rate by 10x
3. Reduce `contrastive_weight` to 1.0
4. Check for extreme outliers in groove radar values

---

## Next Steps After Success

Once you've successfully trained with contrastive learning:

### 1. Evaluate Similarity Quality
- Extract embeddings for validation set
- Compute k-NN retrieval: given a query chart, find most similar charts
- Check if similar groove radar patterns cluster together

### 2. Ablation Studies
- Try intermediate warmup durations (2, 10, 15 epochs)
- Experiment with different classification weights (0.1, 0.5, 0.8)
- Test other contrastive losses (InfoNCE, NTXent)

### 3. Use the Embeddings
- Chart recommendation: "If you like this chart, try these"
- Difficulty clustering: Discover sub-categories within difficulty levels
- Chart generation: Use embeddings to condition generative model

### 4. Scale Up
- Try on larger dataset
- Experiment with harder negatives (semi-hard mining)
- Add more augmentation (chart patterns, audio perturbations)

---

## Key Takeaways

1. **Pretrained models can be geometrically rigid**
   Just because parameters are trainable doesn't mean they'll move

2. **Expect (and want!) temporary accuracy drops**
   If val acc never dips, the representation probably didn't change

3. **Diagnostics are essential**
   Gradient norms, embedding drift, and triplet margins tell the real story

4. **Controlled plasticity is the solution**
   Either freeze the old objective (Experiment A) or freeze low-level features (Experiment B)

5. **Boost the new objective aggressively**
   Higher margins, harder negatives, and stronger loss weights force adaptation

6. **Different approaches for different risk tolerances**
   Aggressive (A) for maximum representation shift, Conservative (B) for safety

---

## References & Further Reading

### Academic Papers
- **ULMFit** (Howard & Ruder, 2018): Gradual unfreezing for transfer learning
- **SimCLR** (Chen et al., 2020): Contrastive learning framework
- **Triplet Networks** (Schroff et al., 2015): FaceNet and triplet loss
- **Curriculum Learning** (Bengio et al., 2009): Progressive difficulty in training

### Related Concepts
- **Catastrophic forgetting**: Why fine-tuning can destroy pretrained knowledge
- **Multi-task learning**: Training on multiple objectives simultaneously
- **Metric learning**: Learning embedding spaces with distance-based losses
- **Hard negative mining**: Selecting difficult examples to improve learning

### Implementation Resources
- PyTorch metric learning library: https://github.com/KevinMusgrave/pytorch-metric-learning
- Papers with Code - Contrastive Learning: https://paperswithcode.com/methods/category/contrastive-learning

---

## Credits

This approach was developed through collaborative debugging and draws inspiration from:
- Transfer learning literature (ULMFit, progressive fine-tuning)
- Contrastive learning best practices (hard negative mining, temperature scaling)
- Multi-task learning theory (gradient balancing, loss weighting)
- Empirical debugging (tracking gradients, visualizing representations)

The key insight came from recognizing that **representation rigidity is a feature, not a bug** — the model is correctly maintaining its well-learned features. The solution is to temporarily break that rigidity in a controlled way, then let it stabilize again with both objectives satisfied.
