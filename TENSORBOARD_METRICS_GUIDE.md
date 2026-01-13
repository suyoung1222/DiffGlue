# TensorBoard Metrics Guide for DiffGlue Training

## Overview
TensorBoard logs are automatically saved during training to help you monitor the training progress. This guide explains what each metric means, where they're saved, and what to look for to assess training stability.

---

## Where TensorBoard Logs Are Saved

### Location
```
/project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/diffglue_data/outputs/training/{experiment_name}/
```

### Key Files
- **`events.out.tfevents.*`** - Binary TensorBoard event files containing all metrics
- **`config.yaml`** - Configuration file for your training run
- **`checkpoints/`** - Saved model weights at various stages

### How to View
```bash
# Navigate to your training directory
cd /project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/diffglue_data/outputs/training/

# Launch TensorBoard
tensorboard --logdir={experiment_name} --port=6006

# Open in browser: http://localhost:6006
```

---

## Understanding the Training Metrics

### **training/assignment_nll** (NLL = Negative Log Likelihood)

**What it is:**
- The main loss for the feature matching assignment task
- Measures how well the model predicts correct and incorrect matches
- Weighted combination of positive and negative match losses

**Formula:**
```
assignment_nll = 0.5 * nll_pos + 0.5 * nll_neg
(default weighting of 0.5 for each)
```

**Components:**
- **nll_pos**: Negative log likelihood for CORRECT matches (should be high confidence)
- **nll_neg**: Negative log likelihood for INCORRECT/non-matchable features (should be low confidence)

**Why it matters:**
- **Primary indicator of model learning** - Should decrease steadily during training
- Lower values = better matching predictions
- Sudden spikes = potential training instability (bad batch, NaN gradients, etc.)

**Ideal behavior:**
```
Initial:     ~2.0-5.0  (model is learning to distinguish matches)
Mid-training: ~0.5-1.5  (model improves)
End-training: ~0.1-0.5  (well-trained model)
```

**Code location:** [DiffGlue/scripts/models/utils/losses.py](DiffGlue/scripts/models/utils/losses.py#L29-L60)

---

### **training/confidence**

**What it is:**
- Loss for the token confidence predictor (auxiliary task)
- Predicts whether the model is confident about its matches at each refinement iteration
- Uses Binary Cross Entropy (BCE) loss to predict match correctness

**Why it matters:**
- **Calibration indicator** - Model learns to be confident when it's right, uncertain when it's wrong
- Helps the model produce more reliable confidence scores for filtering matches
- Useful for uncertainty quantification in downstream tasks

**Ideal behavior:**
```
Initial:      ~0.7-1.0  (random guessing)
Mid-training: ~0.3-0.5  (learning confidence)
End-training: ~0.05-0.2 (well-calibrated)
```

**How it works:**
- For each refinement iteration i (0 to N-1), the model predicts:
  - Whether matches found at iteration i match the final predictions
  - Loss = sum of all iteration losses / N iterations
- Averaged across both image pairs (desc0 and desc1)

**Code location:** [DiffGlue/scripts/models/matchers/diffglue.py](DiffGlue/scripts/models/matchers/diffglue.py#L80-L100)

---

### **training/nll_pos**

**What it is:**
- NLL component for CORRECT matches only
- Measures how well the model scores positive correspondences

**Ideal behavior:**
```
Should be: LOW (model assigns high probability to correct matches)
Typical range: 0.1-0.5
If HIGH (>1.0): Model struggles to find correct matches
```

**What to watch:**
- If nll_pos decreases while nll_neg increases → overfitting to negatives
- Should decrease steadily with training

---

### **training/nll_neg**

**What it is:**
- NLL component for INCORRECT/non-matchable matches
- Measures how well the model rejects false positives
- Includes both:
  - **Unmatched features in image 0** (features with no correspondence in image 1)
  - **Unmatched features in image 1** (features with no correspondence in image 0)

**Ideal behavior:**
```
Should be: MODERATE (model rejects bad matches but not too aggressively)
Typical range: 0.3-1.0
Balance: Should be comparable to nll_pos
```

**What to watch:**
- If much higher than nll_pos → model over-rejects matches (too many false negatives)
- If much lower than nll_pos → model under-rejects matches (too many false positives)
- Should gradually decrease and stabilize

---

### **training/num_matchable**

**What it is:**
- Average number of features in image 0 that have ground truth matches in image 1
- Purely informational metric (not a loss)

**Why it matters:**
- Helps understand the data distribution
- Indicates dataset difficulty and balance
- Large number = many possible matches (challenging)
- Small number = sparse correspondences

**Typical range:**
```
Depends on dataset, but usually 10-100 per image
Low values (<5): Sparse correspondences, harder problem
High values (>200): Dense correspondences, easier problem
```

---

### **training/num_unmatchable**

**What it is:**
- Average number of features that DON'T have matches (unmatched keypoints)
- Sum of unmatchable features from both images / 2

**Why it matters:**
- Indicates class imbalance in training data
- More unmatchable features = harder negative samples to learn
- If very high: Dataset has many spurious features or occlusions

**Watch for:**
- Extreme imbalance (e.g., 95% unmatched) → may need loss weighting adjustment
- Should be relatively stable across batches

---

### **training/lr** (Learning Rate)

**What it is:**
- Current learning rate being used by the optimizer
- Changes according to lr_schedule configuration

**Ideal behavior:**
```
Should follow your scheduling:
- Start high (1e-3 to 1e-4)
- Gradually decrease or remain constant
- Sudden drops indicate schedule trigger
```

**What to watch:**
- Verify lr schedule is working as intended
- Large jumps in loss after lr changes indicate sensitivity
- If lr too high → loss oscillates
- If lr too low → training progress is too slow

**Code location:** [DiffGlue/scripts/train.py](DiffGlue/scripts/train.py#L572-576)

---

### **training/epoch**

**What it is:**
- Current training epoch number
- Simply tracks progress through dataset

**What to watch:**
- Useful for correlating metrics with training stage
- Helps identify if problems occur at specific epochs

---

### **training/row_norm** (Validation Metric)

**What it is:**
- Average sum of log-probabilities per row in assignment matrix (excluding null row)
- Measures how much probability mass is distributed across matches

**Interpretation:**
```
row_norm = sum(exp(log_assignment[:-1])) / num_features
```

**Ideal behavior:**
```
Range: 0 to (number of features in image 1)
Typical: 1.0 - 10.0 (features distribute matches across possibilities)
Low (<0.5): Model makes very confident decisions (possibly too sharp)
High (>50): Model distributes probability too uniformly (uncertain)
```

**What to watch:**
- Too low → model collapses to one-hot assignments (risky)
- Too high → model is too uncertain
- Should stabilize during training

---

## Validation Metrics (val/*)

These appear at intervals specified by `eval_every_iter`:

### **val/assignment_nll**
- Validation NLL on held-out data
- Should follow training loss but be slightly higher
- If much higher → overfitting
- If tracking training loss closely → good generalization

### **val/nll_pos, val/nll_neg**
- Same components as training, computed on validation set
- Similar interpretation as training versions

### **val/row_norm**
- Validation version of row_norm
- Should be similar to training row_norm

---

## What to Look For: Training Stability Indicators

### ✅ **GOOD SIGNS** (Stable, Healthy Training)

1. **assignment_nll Curves**
   ```
   ✓ Smooth, monotonic decrease
   ✓ No sudden spikes or NaNs
   ✓ Validation NLL close to training NLL
   ✓ Typical trajectory: 2.0 → 0.5 → 0.1
   ```

2. **Component Balance**
   ```
   ✓ nll_pos ≈ nll_neg (roughly similar magnitudes)
   ✓ Both decreasing together during training
   ✓ No sudden divergence between pos and neg
   ```

3. **Confidence Loss**
   ```
   ✓ Steady decrease from ~0.8 to ~0.1
   ✓ Follows assignment_nll curve (slightly offset)
   ✓ No oscillations
   ```

4. **Learning Rate**
   ```
   ✓ Decreases smoothly according to schedule
   ✓ No erratic jumps during stable training
   ```

5. **Validation Metrics**
   ```
   ✓ Track training metrics (slightly higher)
   ✓ Validation NLL within 10-30% of training NLL
   ✓ Smooth curves without overfitting spike
   ```

### ❌ **WARNING SIGNS** (Potential Problems)

1. **Loss Spikes/Divergence**
   ```
   ✗ Sudden jump in assignment_nll
   ✗ NaN or Inf values (loss goes to infinity)
   ✗ Loss oscillates wildly instead of decreasing
   Action: Check batch quality, reduce learning rate, enable gradient clipping
   ```

2. **Validation Divergence**
   ```
   ✗ Val loss >> training loss (e.g., 2-3x higher)
   ✗ Val loss increases while training decreases
   ✗ Sharp divergence after certain epoch
   Action: Likely overfitting → reduce model capacity or increase regularization
   ```

3. **Component Imbalance**
   ```
   ✗ nll_pos >> nll_neg (or vice versa)
   ✗ One component stops decreasing
   ✗ Sudden divergence between pos and neg
   Action: May need to adjust nll_balancing parameter (default 0.5)
   ```

4. **Stalled Progress**
   ```
   ✗ Loss plateaus very early (epoch 1-2)
   ✗ No improvement for many iterations
   ✗ Metrics stuck at random initialization values
   Action: Learning rate too low, bad weight initialization, or architecture issue
   ```

5. **Confidence Issues**
   ```
   ✗ Confidence loss doesn't decrease
   ✗ Confidence loss higher than assignment_nll
   ✗ Confidence loss suddenly increases
   Action: Model may be outputting unreliable confidence scores
   ```

6. **Data Issues**
   ```
   ✗ num_matchable or num_unmatchable vary wildly between batches
   ✗ Extreme class imbalance (num_unmatchable >> num_matchable)
   Action: Check data loading, potential data contamination or mislabeling
   ```

---

## How Metrics Are Logged in Code

### Logging Location
[DiffGlue/scripts/train.py](DiffGlue/scripts/train.py#L560-580)

```python
# Every log_every_iter iterations (default: 200)
if it % conf.train.log_every_iter == 0:
    # Collect loss components
    losses, metrics = loss_fn(pred, data)
    
    # Log to TensorBoard
    for k, v in losses.items():
        writer.add_scalar("training/" + k, v, tot_n_samples)
    
    writer.add_scalar("training/lr", optimizer.param_groups[0]["lr"], tot_n_samples)
    writer.add_scalar("training/epoch", epoch, tot_n_samples)
```

### Loss Computation
[DiffGlue/scripts/models/matchers/diffglue.py](DiffGlue/scripts/models/matchers/diffglue.py#L750-800)

```python
# Main loss from final predictions
nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)

# Loss metrics include:
# - assignment_nll: main loss
# - nll_pos: positive matching loss
# - nll_neg: negative/unmatched loss
# - num_matchable: count of matchable features
# - num_unmatchable: count of unmatchable features

# Confidence loss (auxiliary)
losses["confidence"] = sum of token_confidence losses across refinement steps
```

---

## Configuration Parameters Affecting Metrics

In your config file or training args:

```yaml
train:
  log_every_iter: 200          # How often to log metrics
  eval_every_iter: 1000        # How often to validate
  
  lr: 0.001                    # Learning rate
  lr_schedule:
    type: exp                  # Learning rate schedule type
    exp_div_10: 10            # Exponential decay parameter
  
  clip_grad: 1.0              # Gradient clipping threshold (prevents spikes)

model:
  loss:
    nll_balancing: 0.5        # Balance between pos and neg loss
    gamma_f: 0.0              # Focal loss parameter (0 = no focal)
  
  width_confidence: [0, 2]    # Which refinement iterations compute confidence
```

---

## Debugging Checklist

If training looks unstable, check in this order:

### 1. **Check for NaNs**
```bash
# Look for NaN messages in logs
tail -f /project/.../outputs/training/{experiment}/log.txt | grep -i nan
```

### 2. **Verify Hyperparameters**
```bash
# Check saved config
cat /project/.../outputs/training/{experiment}/config.yaml | grep -A 10 "train:"
```

### 3. **Inspect Data**
- num_matchable/unmatchable should be consistent
- If varies wildly → potential data loading issue

### 4. **Check Gradients** (if enabled)
- Look at grad/* metrics in TensorBoard
- Ensure gradient norms are reasonable (not too small or large)
- Very small gradients → vanishing gradient problem
- Very large gradients → exploding gradient problem (use clip_grad)

### 5. **Monitor GPU Memory**
- Loss spikes sometimes indicate OOM (out of memory)
- Check NVIDIA GPU usage: `nvidia-smi`

### 6. **Review Learning Rate**
- If loss spikes at certain iterations, check if lr_schedule fired
- Try reducing learning rate if loss is volatile

---

## Example Metric Trajectories

### **Healthy Training (Good)**
```
Iteration 0-1000:
  assignment_nll: 2.5 → 1.5
  nll_pos: 1.8 → 0.8
  nll_neg: 3.2 → 2.2
  confidence: 0.8 → 0.5
  Validation NLL: 2.6 → 1.6 (slightly higher than training)
  
Iteration 1000-5000:
  assignment_nll: 1.5 → 0.3
  nll_pos: 0.8 → 0.15
  nll_neg: 2.2 → 0.45
  confidence: 0.5 → 0.1
  Validation NLL: 1.6 → 0.35 (still tracks training)
  
Result: Model converges smoothly, validation follows training
```

### **Overfitting** (Diverging Val Loss)
```
Iteration 0-2000:
  training assignment_nll: 2.5 → 0.5 ✓
  validation assignment_nll: 2.5 → 2.0 ✗ (doesn't decrease)
  
Problem: Model memorizes training data but doesn't generalize
Solution: Add regularization, reduce model capacity, or get more training data
```

### **Exploding Loss** (NaN)
```
Iteration 0-500: Stable ~1.5
Iteration 500: assignment_nll → inf/NaN
Gradient norms: Suddenly very large

Problem: Gradient explosion
Solution: Enable gradient clipping (clip_grad: 1.0), reduce learning rate
```

---

## Summary Table: Quick Reference

| Metric | Ideal Range | Decreases? | Tracks Val? | If High | If Low |
|--------|------------|-----------|-----------|---------|---------|
| assignment_nll | 0.1-0.5 | ✓ Yes | ✓ Slightly higher | Bad matches | - |
| nll_pos | 0.1-0.5 | ✓ Yes | ✓ Slightly higher | Miss true matches | - |
| nll_neg | 0.3-1.0 | ✓ Yes | ✓ Slightly higher | Too many false pos | Too harsh |
| confidence | 0.05-0.2 | ✓ Yes | ✓ Slightly higher | Bad calibration | - |
| num_matchable | Dataset dependent | - | - | More hard negatives | Sparse data |
| num_unmatchable | Dataset dependent | - | - | More false positives | Few outliers |
| lr | Decreasing | ✓ Per schedule | - | Too slow learning | Too fast/unstable |
| row_norm | 1.0-10.0 | ~ Stable | ✓ Similar | Too uncertain | Too confident |

---

## Additional Resources

- **Training Script:** [DiffGlue/scripts/train.py](DiffGlue/scripts/train.py)
- **Loss Functions:** [DiffGlue/scripts/models/utils/losses.py](DiffGlue/scripts/models/utils/losses.py)
- **Model Definition:** [DiffGlue/scripts/models/matchers/diffglue.py](DiffGlue/scripts/models/matchers/diffglue.py)
- **PyTorch TensorBoard Docs:** https://pytorch.org/docs/stable/tensorboard.html

---

*Last updated: January 2026*
