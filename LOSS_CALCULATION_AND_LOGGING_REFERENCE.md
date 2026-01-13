# DiffGlue Loss Calculation & Logging Flow

## Data Flow Diagram

```
Training Loop (train.py)
    ↓
1. Forward Pass: pred = model(data)
    ↓
2. Loss Calculation: losses, _ = loss_fn(pred, data)
    ↓
3. Backward Pass: loss.backward()
    ↓
4. Logging (every log_every_iter): 
    writer.add_scalar("training/" + k, v, tot_n_samples)
    ↓
5. TensorBoard File: /outputs/training/{experiment}/events.out.tfevents.*
```

---

## 1️⃣ WHERE LOSSES ARE LOGGED

### File: [DiffGlue/scripts/train.py](DiffGlue/scripts/train.py#L560-580)

**Lines 560-580: Main Logging Code**

```python
# ===== LOSS COMPUTATION =====
# Line 493: Call loss function
with autocast(enabled=args.mixed_precision is not None, dtype=mp_dtype):
    data = batch_to_device(data, device, non_blocking=True)
    pred = model(data)
    losses, _ = loss_fn(pred, data)          # 👈 losses dict created here
    loss = torch.mean(losses["total"])       # Get total loss for backprop

# ===== LOGGING SECTION =====
# Lines 559-580: Log metrics to TensorBoard
if it % conf.train.log_every_iter == 0:
    # Aggregate losses across distributed processes
    for k in sorted(losses.keys()):
        if args.distributed:
            losses[k] = losses[k].sum(-1)
            torch.distributed.reduce(losses[k], dst=0)
            losses[k] /= train_loader.batch_size * args.n_gpus
        losses[k] = torch.mean(losses[k], -1)
        losses[k] = losses[k].item()  # Convert tensor to Python float
    
    # Print to console
    if rank == 0:
        str_losses = [f"{k} {v:.3E}" for k, v in losses.items()]
        logger.info(
            "[E {} | it {}] loss {{{}}}".format(
                epoch, it, ", ".join(str_losses)
            )
        )
        
        # ⭐ THIS IS WHERE DATA GOES TO TENSORBOARD ⭐
        for k, v in losses.items():
            writer.add_scalar("training/" + k, v, tot_n_samples)  # Line 573
        
        writer.add_scalar("training/lr", optimizer.param_groups[0]["lr"], tot_n_samples)    # Line 574-575
        writer.add_scalar("training/epoch", epoch, tot_n_samples)                           # Line 577
```

**Key Points:**
- Line 28: `from torch.utils.tensorboard import SummaryWriter`
- Line 249: `writer = SummaryWriter(log_dir=str(output_dir))`
- Line 573: `writer.add_scalar("training/" + k, v, tot_n_samples)` ← **This line writes to TensorBoard**
- `tot_n_samples` is the global iteration counter (x-axis in TensorBoard)

**What gets logged:**
- All keys in the `losses` dictionary (see section 3 below for what's in it)
- Learning rate
- Current epoch

---

## 2️⃣ WHERE LOSSES ARE CALCULATED (High Level)

### File: [DiffGlue/scripts/models/matchers/diffglue.py](DiffGlue/scripts/models/matchers/diffglue.py#L745-795)

**Lines 745-795: Loss Calculation in Model's loss() method**

```python
def loss(self, pred, data):
    """
    pred: model predictions (descriptors, log_assignments, matches)
    data: ground truth data (gt_matches0, gt_matches1, gt_assignment)
    returns: dict of losses for logging
    """
    
    # ===== STEP 1: Get log_assignment matrices for final predictions =====
    def loss_params(pred, i):
        la, _ = self.log_assignment[i](
            pred["ref_descriptors0"][:, i], 
            pred["ref_descriptors1"][:, i]
        )
        return {"log_assignment": la}
    
    # ===== STEP 2: Compute main NLL loss from FINAL predictions =====
    # Line 754: Call NLLLoss (defined in losses.py)
    nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
    #         ↓
    # Returns:
    # - nll: scalar loss value
    # - gt_weights: ground truth assignment weights (used for all iterations)
    # - loss_metrics: dict with {assignment_nll, nll_pos, nll_neg, num_matchable, num_unmatchable}
    
    N = pred["ref_descriptors0"].shape[1]  # Number of refinement iterations
    losses = {
        "matcher_total": nll,
        "last": nll.clone().detach(),
        **loss_metrics  # 👈 Unpack {assignment_nll, nll_pos, nll_neg, num_matchable, num_unmatchable}
    }
    
    if self.training:
        losses["confidence"] = torch.zeros_like(nll)  # Initialize confidence loss
    
    losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1).detach()
    
    # ===== STEP 3: Accumulate losses across refinement iterations =====
    if self.training:
        loss_terms = [nll]  # Start with final iteration loss
        confidence_terms = []
        
        for i in range(N):  # Loop through each refinement iteration
            # Get assignments for iteration i
            params_i = loss_params(pred, i)
            
            # Compute NLL for this iteration
            nll_i, _, _ = self.loss_fn(params_i, data, weights=gt_weights)
            
            # Weight by iteration (later iterations have higher weight)
            weight = self.conf.loss.gamma ** (N - i) if self.conf.loss.gamma > 0.0 else (i + 1)
            loss_terms.append(nll_i * weight)
            
            # Compute confidence loss for this iteration
            confidence_terms.append(
                self.token_confidence[i].loss(
                    pred["ref_descriptors0"][:, i],
                    pred["ref_descriptors1"][:, i],
                    params_i["log_assignment"],
                    pred["log_assignment"],
                ) / N
            )
        
        # Aggregate all loss terms
        losses["matcher_total"] = torch.stack(loss_terms).sum(0)
        losses["confidence"] = torch.stack(confidence_terms).sum(0) if confidence_terms else torch.zeros_like(nll)
    
    # ===== STEP 4: Add other losses (epipolar, etc.) =====
    if "T_0to1" in data:
        L_epi = sampson_epipolar_loss(...)
        losses["epipolar"] = L_epi
    
    # ... more loss terms ...
    
    # ===== FINAL STEP: Combine all losses =====
    losses["total"] = losses["matcher_total"] + ...  # All loss components combined
    
    return losses, {}
```

**Summary:**
- Main loss computation happens in `self.loss_fn()` which calls the `NLLLoss` class
- Losses are accumulated across N refinement iterations
- Confidence loss is computed as auxiliary task
- All losses combined into `losses["total"]` for backpropagation

---

## 3️⃣ WHERE INDIVIDUAL LOSS COMPONENTS ARE CALCULATED

### File: [DiffGlue/scripts/models/utils/losses.py](DiffGlue/scripts/models/utils/losses.py)

**Lines 5-27: weight_loss() function - Core math**

```python
def weight_loss(log_assignment, weights, gamma=0.0):
    """
    Compute NLL components from log_assignment matrix
    
    log_assignment: shape [batch, m+1, n+1]
        m = # features in image 0
        n = # features in image 1
        Last row/col = null/unmatched class
    
    weights: binary mask indicating true matches
    """
    b, m, n = log_assignment.shape
    m -= 1  # Remove null row
    n -= 1  # Remove null col

    # ===== MATCHING LOSS =====
    loss_sc = log_assignment * weights  # Element-wise multiply: log-prob × match indicator
    
    # Count matchable/unmatchable features
    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)  # Unmatched in image 0
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)  # Unmatched in image 1
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)  # Matched pairs
    
    # ===== COMPUTE NLL COMPONENTS =====
    # NLL for POSITIVE matches (should maximize log_assignment)
    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))  # Sum log-probs of true matches
    nll_pos /= num_pos.clamp(min=1.0)            # Average over # of matches
    
    # NLL for NEGATIVE/unmatched features
    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)       # Features unmatched in image 0
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)       # Features unmatched in image 1
    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)  # Average negatives
    
    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0
```

**Lines 29-60: NLLLoss class - Combines components**

```python
class NLLLoss(nn.Module):
    """
    Main loss class that returns all loss components
    """
    default_conf = {
        "nll_balancing": 0.5,     # Balance between positive and negative losses
        "gamma_f": 0.0,           # Focal loss parameter
    }

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]  # Model's log probabilities
        
        # Get ground truth weights if not provided
        if weights is None:
            weights = self.loss_fn(log_assignment, data)
        
        # ===== COMPUTE COMPONENTS =====
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.conf.gamma_f
        )
        
        # ===== COMBINE INTO FINAL NLL =====
        nll = (
            self.conf.nll_balancing * nll_pos +      # 0.5 × positive loss
            (1 - self.conf.nll_balancing) * nll_neg   # 0.5 × negative loss
        )

        # ===== RETURN ALL METRICS FOR LOGGING =====
        return (
            nll,  # Total loss for backprop
            weights,  # For reuse in other iterations
            {
                "assignment_nll": nll,          # 👈 Main loss logged in TensorBoard
                "nll_pos": nll_pos,             # 👈 Logged
                "nll_neg": nll_neg,             # 👈 Logged
                "num_matchable": num_pos,       # 👈 Logged
                "num_unmatchable": num_neg,     # 👈 Logged
            },
        )

    def nll_loss(self, log_assignment, data):
        """Create weight matrix from ground truth"""
        m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        positive = data["gt_assignment"].float()  # Ground truth positive matches
        neg0 = (data["gt_matches0"] == -1).float()  # Unmatched in image 0
        neg1 = (data["gt_matches1"] == -1).float()  # Unmatched in image 1

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive  # Mark true matches
        weights[:, :m, -1] = neg0      # Mark unmatched in 0
        weights[:, -1, :n] = neg1      # Mark unmatched in 1
        return weights
```

---

## 4️⃣ CONFIDENCE LOSS CALCULATION

### File: [DiffGlue/scripts/models/matchers/diffglue.py](DiffGlue/scripts/models/matchers/diffglue.py#L80-105)

**Lines 80-105: TokenConfidence class**

```python
class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(dim, 1),  # Dense layer: descriptor → confidence score
            nn.Sigmoid()        # Sigmoid: output in [0, 1]
        )
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def loss(self, desc0, desc1, la_now, la_final):
        """
        Compute confidence loss: does model know if its current assignment is correct?
        
        desc0, desc1: descriptors at current refinement iteration
        la_now: log_assignment at current iteration
        la_final: log_assignment at final iteration (ground truth)
        """
        # Get confidence predictions (logits before sigmoid)
        logit0 = self.token[0](desc0.detach()).squeeze(-1)  # [batch, m]
        logit1 = self.token[0](desc1.detach()).squeeze(-1)  # [batch, n]
        
        # Detach predictions to avoid backprop issues
        la_now, la_final = la_now.detach(), la_final.detach()
        
        # Compute ground truth: did current iteration's matches match final predictions?
        correct0 = (
            la_final[:, :-1, :].max(-1).indices ==   # Best match in final
            la_now[:, :-1, :].max(-1).indices        # Best match in current iteration
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices ==   # Best match in final
            la_now[:, :, :-1].max(-2).indices        # Best match in current iteration
        )
        
        # Compute BCE loss: predict if assignments are correct
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1) +
            self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0
```

---

## 5️⃣ COMPLETE FLOW SUMMARY

### Training Loop Structure (train.py, lines 485-577)

```
while epoch < num_epochs:
    for it, batch in enumerate(train_loader):
        
        # Step 1: Forward pass
        data = batch_to_device(batch, device)
        pred = model(data)
        
        # Step 2: Compute losses
        losses, _ = model.loss(pred, data)  # Returns dict:
        #   {
        #       "assignment_nll": tensor,
        #       "nll_pos": tensor,
        #       "nll_neg": tensor,
        #       "num_matchable": tensor,
        #       "num_unmatchable": tensor,
        #       "confidence": tensor,
        #       "row_norm": tensor,
        #       "total": tensor,
        #       ... (other loss terms)
        #   }
        
        loss = torch.mean(losses["total"])
        
        # Step 3: Backward pass
        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Step 4: Log metrics (every log_every_iter iterations)
        if it % log_every_iter == 0:
            for k, v in losses.items():
                writer.add_scalar("training/" + k, v, tot_n_samples)  # ← To TensorBoard!
            writer.add_scalar("training/lr", lr, tot_n_samples)
            writer.add_scalar("training/epoch", epoch, tot_n_samples)
        
        # Step 5: Validation (every eval_every_iter iterations)
        if it % eval_every_iter == 0:
            val_results = evaluate(model, val_loader)
            for k, v in val_results.items():
                writer.add_scalar("val/" + k, v, tot_n_samples)  # ← Validation metrics!
        
        tot_n_samples += batch_size
```

---

## 6️⃣ KEY FILES AND LINES REFERENCE

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **Loss Computation** | [diffglue.py](DiffGlue/scripts/models/matchers/diffglue.py) | 745-820 | Model's `loss()` method - computes all loss terms |
| **Loss Components** | [losses.py](DiffGlue/scripts/models/utils/losses.py) | 5-60 | `NLLLoss` class - breaks down into pos/neg components |
| **Weight Calculation** | [losses.py](DiffGlue/scripts/models/utils/losses.py) | 5-27 | `weight_loss()` function - core NLL math |
| **Confidence Loss** | [diffglue.py](DiffGlue/scripts/models/matchers/diffglue.py) | 80-105 | `TokenConfidence` class - predicts match correctness |
| **TensorBoard Logging** | [train.py](DiffGlue/scripts/train.py) | 560-580 | Logs losses to TensorBoard |
| **TensorBoard Init** | [train.py](DiffGlue/scripts/train.py) | 249 | Creates SummaryWriter |
| **Loss Function Call** | [train.py](DiffGlue/scripts/train.py) | 493 | Calls `loss_fn(pred, data)` in forward pass |

---

## 7️⃣ HOW TO TRACE A SPECIFIC METRIC

Example: Following `training/assignment_nll`

```
1. TensorBoard shows: training/assignment_nll = 0.234

2. Trace back to source:
   train.py:573  →  writer.add_scalar("training/" + "assignment_nll", v, ...)
   
3. Find where "assignment_nll" key is created:
   diffglue.py:756  →  losses = {..., **loss_metrics}
   
4. loss_metrics comes from:
   losses.py:54  →  "assignment_nll": nll
   
5. nll is computed by:
   losses.py:49-52  →  nll = 0.5 * nll_pos + 0.5 * nll_neg
   
6. Which comes from:
   losses.py:45  →  nll_pos, nll_neg, ... = weight_loss(log_assignment, weights)
   
7. weight_loss computes:
   losses.py:18  →  nll_pos = -loss_sc[:, :m, :n].sum((-1, -2)) / num_pos
   losses.py:24  →  nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)
```

---

## 8️⃣ IMPORTANT: Order of Execution

```python
# In train.py, line 493
pred = model(data)  # Model forward pass

# In model (diffglue.py)
losses, _ = model.loss(pred, data)  # Calls:
    # → diffglue.py:754
    #   nll, gt_weights, loss_metrics = self.loss_fn(...)
    #       # → losses.py:forward()
    #       #   nll_pos, nll_neg, ... = weight_loss(...)
    #       #   returns {"assignment_nll": nll, "nll_pos": ..., ...}

# Back in train.py, line 573
writer.add_scalar("training/assignment_nll", losses["assignment_nll"], tot_n_samples)
```

**The flow is:**
```
model.forward(data)
  ↓
model.loss(pred, data)  [diffglue.py line 745+]
  ↓
NLLLoss.forward(pred, data)  [losses.py line 40+]
  ↓
weight_loss(log_assignment, weights)  [losses.py line 5+]
  ↓
Returns loss values
  ↓
Logged to TensorBoard  [train.py line 573]
  ↓
Saved to: /outputs/training/{exp}/events.out.tfevents.*
```

