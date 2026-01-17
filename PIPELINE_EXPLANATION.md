# DiffGlue Training Pipeline: Detailed Module Explanation

This document explains each module in your training pipeline diagram and maps them to the corresponding code implementation.

## Pipeline Overview

Your pipeline implements a Geometry-Guided Diffusion Matching (GGDM) approach that alternates between appearance-based matching (LoFTR transformer) and geometry-based refinement (epipolar constraints via diffusion).

---

## 1. Input Image Pair (I₁, I₂)

**Diagram**: Input images fed into the pipeline

**Code Location**: 
- **Data Loading**: `DiffGlue/scripts/datasets/` (various dataset implementations)
- **Entry Point**: `TwoViewPipeline._forward()` at `two_view_pipeline.py:129`
  ```python
  pred0 = self.extract_view(data, "0")  # Extract view 0
  pred1 = self.extract_view(data, "1")  # Extract view 1
  ```

**Explanation**: 
- Images come as `data["view0"]["image"]` and `data["view1"]["image"]`
- The pipeline expects `[B, C, H, W]` tensors where B=batch, C=channels (1 or 3), H=height, W=width

**Improvements**:
- ✅ Already supports both RGB and grayscale images
- 💡 Consider adding image normalization/standardization checks in preprocessing

---

## 2. Feature Encoder (CNN/ViT Backbone)

**Diagram**: 
```
Feature Encoder (CNN / ViT Backbone)
F₁ ∈ R^{N₁×C}, F₂ ∈ R^{N₂×C}
```

**Code Location**:
- **Configuration**: `TwoViewPipeline.default_conf.extractor` at `two_view_pipeline.py:24-27`
- **Extraction**: `TwoViewPipeline.extract_view()` at `two_view_pipeline.py:119-127`
- **Common Extractors**: SuperPoint, DINOv2, etc. (in `models/` directory)

**Key Implementation**:
```python
# two_view_pipeline.py:123-124
if self.conf.extractor.name and not skip_extract:
    pred_i = {**pred_i, **self.extractor(data_i)}
```

**Outputs**:
- `keypoints0`, `keypoints1`: [B, N, 2] pixel coordinates
- `descriptors0`, `descriptors1`: [B, N, C] feature descriptors (C=256 typically)

**Explanation**:
- The encoder extracts sparse keypoints and dense descriptors from each image
- For detector-free methods (LoFTR-style), this can be a CNN backbone that outputs dense features
- The features F₁, F₂ are the descriptors that get fed into the transformer

**Improvements**:
- ✅ Flexible extractor system (can swap SuperPoint, DINOv2, etc.)
- 💡 Consider caching features if extractor is not trainable to speed up training
- 💡 Add feature dimension validation to catch mismatches early

---

## 3. LoFTR Coarse Transformer with Cross-Attention

**Diagram**:
```
LoFTR Coarse Transformer
Cross-Attention
S₀ = F₁F₂ᵀ
```

**Code Location**:
- **Main Module**: `DiffGlue` class at `diffglue/diffglue.py:540`
- **Transformer Layers**: `TransformerLayer` at `diffglue/diffglue.py:411`
- **Cross-Attention**: `CrossBlock` at `diffglue/diffglue.py:305`
- **Forward Pass**: `DiffGlue.forward()` at `diffglue/diffglue.py:609-706`

**Key Implementation**:

**Cross-Attention Block** (`diffglue.py:331-363`):
```python
# CrossBlock.forward() - computes cross-attention between F₁ and F₂
def forward(self, x0, x1, encoding0=None, encoding1=None, mask=None):
    qk0, qk1 = self.map_(self.to_qk, x0, x1)  # Query-Key projections
    v0, v1 = self.map_(self.to_v, x0, x1)     # Value projections
    
    # Multi-head attention
    sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)  # S = F₁F₂ᵀ
    attn01 = F.softmax(sim, dim=-1)  # Attention from 0→1
    attn10 = F.softmax(sim.transpose(-2, -1), dim=-1)  # Attention from 1→0
    
    m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)  # Aggregated features
    m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
    
    return x0 + self.ffn(torch.cat([x0, m0], -1)), \
           x1 + self.ffn(torch.cat([x1, m1], -1))
```

**Transformer Layer** (`diffglue.py:442-481`):
```python
# TransformerLayer.forward() - full transformer with self-attn + cross-attn
for i in range(self.atten_layers):
    desc0 = self.self_attn[i](desc0, encoding0)  # Self-attention
    desc1 = self.self_attn[i](desc1, encoding1)
    
    # Cross-attention with adjacency matrix (from previous iteration)
    desc0, desc1 = self.cross_attn[i](desc0, desc1, mask)
```

**DiffGlue Forward** (`diffglue.py:668-678`):
```python
# Multiple transformer layers process features
for i in range(self.conf.n_layers):
    desc0, desc1 = self.transformers[i](
        desc0, desc1, encoding0, encoding1, time_embd, adj_mat_fore[...,:-1,:-1]
    )
```

**Explanation**:
- The transformer processes descriptors through multiple layers
- Each layer applies self-attention (within each image) then cross-attention (between images)
- The similarity matrix S₀ = F₁F₂ᵀ is computed implicitly via attention
- `adj_mat_fore` can contain attention bias from previous refinement iterations (feedback mechanism)

**Improvements**:
- ✅ Supports FlashAttention for efficiency
- ✅ Supports gradient checkpointing for memory savings
- 💡 The attention bias mechanism (`_attention_bias`) could be better documented
- 💡 Consider adding attention visualization utilities for debugging

---

## 4. Soft Match Matrix (DualSoftmax)

**Diagram**:
```
Soft Match Matrix
M₀ = DualSoftmax(S₀)
(Optional Sinkhorn)
```

**Code Location**:
- **DualSoftmax**: `sigmoid_log_double_softmax()` at `diffglue/diffglue.py:484-496`
- **Match Assignment**: `MatchAssignment` class at `diffglue/diffglue.py:499-518`
- **Usage**: `DiffGlue.forward()` at `diffglue/diffglue.py:681`

**Key Implementation**:

**DualSoftmax Function** (`diffglue.py:484-496`):
```python
def sigmoid_log_double_softmax(sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor):
    """
    Creates log assignment matrix using dual softmax + matchability scores.
    
    Formula:
        M_ij = log_softmax(S_i·) + log_softmax(S_·j) + log_sigmoid(z0_i) + log_sigmoid(z1_j)
    
    Where:
        S = similarity matrix (S₀ from transformer)
        z0, z1 = matchability scores (learned per-keypoint)
    """
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)  # Row-wise softmax
    scores1 = F.log_softmax(sim.transpose(-1, -2), 2).transpose(-1, -2)  # Col-wise softmax
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties  # Dual softmax
    
    # Dustbin rows/columns for unmatched points
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))  # Row dustbin
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))  # Col dustbin
    
    return scores  # Log-space match matrix [B, M+1, N+1]
```

**MatchAssignment Class** (`diffglue.py:499-518`):
```python
class MatchAssignment(nn.Module):
    def forward(self, desc0, desc1):
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        sim = torch.einsum("bmd,bnd->bmn", mdesc0 / d**0.25, mdesc1 / d**0.25)  # S₀
        z0 = self.matchability(desc0)  # Matchability for image 0
        z1 = self.matchability(desc1)  # Matchability for image 1
        scores = sigmoid_log_double_softmax(sim, z0, z1)  # M₀
        return scores, sim
```

**Explanation**:
- DualSoftmax enforces both row-wise and column-wise normalization (doubly-stochastic constraint)
- Matchability scores (z₀, z₁) allow the model to express uncertainty about matches
- The "+1" dimensions are "dustbin" entries for unmatched keypoints
- This is in log-space for numerical stability (M₀ = exp(scores))

**Improvements**:
- ✅ Dual softmax ensures proper normalization
- 💡 **Sinkhorn normalization is mentioned in the diagram but not implemented** - consider adding as an option for stricter doubly-stochastic constraints
- 💡 Consider adding temperature scaling for sharper/softer distributions

---

## 5. Initial Geometry Estimate (Weighted 8-Point)

**Diagram**:
```
Initial Geometry Estimate
E₀ = Weighted 8-point
(Optional / Fallback)
```

**Code Location**:
- **Function**: `weighted_eight_point()` at `models/utils/geometry_guidance.py:99-199`
- **Usage**: `AlternatingRefinement.estimate_essential_matrix()` at `alternating_refinement.py:400-414`
- **Initial Call**: `AlternatingRefinement.forward()` at `alternating_refinement.py:556`

**Key Implementation**:

**Weighted 8-Point Algorithm** (`geometry_guidance.py:99-199`):
```python
def weighted_eight_point(kpts0, kpts1, weights, K0, K1, normalize=True):
    """
    Estimate essential matrix E from soft correspondences using weighted least squares.
    
    Algorithm:
    1. Normalize keypoints to camera coordinates: x' = K^{-1} * x
    2. Build constraint matrix A where each row is: A_ij = vec(x1_j * x0_i^T)
    3. Apply weights: A_weighted = sqrt(weights) * A
    4. SVD: A_weighted = U * S * V^T, take last column of V^T as E (reshaped to 3x3)
    5. Project E onto essential matrix manifold (enforce rank-2 constraint)
    
    Returns:
        E: [B, 3, 3] essential matrix (x1^T * E * x0 = 0)
        F: [B, 3, 3] fundamental matrix (F = K1^{-T} * E * K0^{-1})
    """
    # Normalize to camera coordinates
    kpts0_norm = normalize_keypoints_for_E(kpts0, K0)  # [B, N, 2]
    kpts1_norm = normalize_keypoints_for_E(kpts1, K1)  # [B, M, 2]
    
    # Build constraint matrix A [B, N*M, 9]
    A = build_constraint_matrix(kpts0_norm, kpts1_norm)
    
    # Apply weights: weighted least squares
    w_sqrt = torch.sqrt(weights.clamp(min=1e-8)).unsqueeze(-1)
    A_weighted = A * w_sqrt
    
    # SVD to find null space (smallest singular value)
    U, S, Vh = torch.linalg.svd(A_weighted, full_matrices=False)
    e = Vh[:, -1, :]  # [B, 9] - last row (smallest singular value)
    E = e.view(B, 3, 3)
    
    # Project to essential matrix manifold (rank-2, equal singular values)
    E = project_to_essential_manifold(E)
    
    return E, F
```

**Initial Estimation** (`alternating_refinement.py:540-556`):
```python
# After initial transformer forward pass
M_k = self.soft_assignment_to_matches(pred["sample"], scale=scale)  # M₀
E_k = self.estimate_essential_matrix(M_k, kpts0, kpts1, K0, K1)  # E₀
```

**Explanation**:
- The 8-point algorithm finds the essential matrix E that satisfies: x₁^T E x₀ = 0 for all correspondences
- "Weighted" means each correspondence (i,j) is weighted by M₀[i,j] (confidence from soft matches)
- Essential matrix has 5 DOF (3 for rotation, 2 for translation direction, scale is ambiguous)
- The algorithm uses SVD to find the null space of the constraint system

**Improvements**:
- ✅ Handles numerical stability with normalization
- ✅ Projects to essential matrix manifold (enforces rank-2 constraint)
- 💡 Consider adding RANSAC wrapper for robust estimation in presence of outliers
- 💡 Consider adding confidence thresholding (only use high-confidence matches)

---

## 6. Alternating Refinement Loop (k = 1..K)

**Diagram**: Main loop that alternates between diffusion refinement and geometry updates

**Code Location**:
- **Main Loop**: `AlternatingRefinement.forward()` at `alternating_refinement.py:505-666`
- **Configuration**: `TwoViewPipeline.default_conf.refinement` at `two_view_pipeline.py:36-53`

**Key Implementation** (`alternating_refinement.py:573-651`):
```python
# MAIN LOOP: for k = 1 to K
for k in range(1, self.num_refinement_iters):
    # 1. Geometry conditioning
    # 2. Guided denoising
    # 3. Projection
    # 4. Feedback to TransformerHead
```

**Explanation**:
- The loop alternates between:
  1. **Geometry-guided diffusion** (refines matches using epipolar constraints)
  2. **Geometry estimation** (updates E from refined matches)
  3. **Feedback to transformer** (bias attention based on refined matches)

**Improvements**:
- ✅ Well-structured loop with clear phases
- 💡 Consider adding early stopping if geometry converges (E_k ≈ E_{k-1})
- 💡 Consider adaptive iteration count (fewer iterations for easy cases)

---

## 7. Geometry-Guided Diffusion Refinement

**Diagram**:
```
Geometry-Guided Diffusion Refinement
x_T ← noisy(M_{k-1})
for t = T … 1:
    sθ ← ScoreNet(x_t | F₁,F₂,E_{k-1})
    ∇geo ← Epipolar Gradient(E_{k-1})
    x_{t-1} ← ReverseStep(x_t, sθ−λ∇geo)
```

**Code Location**:
- **Diffusion**: `GaussianDiffusion.ddim_sample_loop_progressive()` at `diffusers/gaussian_diffusion.py:660-723`
- **Geometry Guidance**: `epipolar_gradient()` at `models/utils/geometry_guidance.py:298-351`
- **Guidance Function**: `AlternatingRefinement.create_geometry_guidance_fn()` at `alternating_refinement.py:355-398`
- **Score Network**: `DiffGlue.forward()` at `diffglue/diffglue.py:609` (acts as score network)

**Key Implementation**:

**Epipolar Gradient** (`geometry_guidance.py:298-351`):
```python
def epipolar_gradient(soft_matches, kpts0, kpts1, E, K0, K1, error_type="sampson"):
    """
    Compute gradient of epipolar constraint w.r.t. soft match matrix.
    
    The epipolar loss for soft matches is:
        L_epi = Σ_{i,j} M_{ij} * epipolar_error(i, j)
    
    The gradient w.r.t. M is simply the pairwise epipolar errors:
        ∂L_epi/∂M_{ij} = epipolar_error(i, j)
    
    This gradient guides diffusion: score_guided = score - λ * ∇L_epi
    """
    # Compute pairwise epipolar errors
    errors = compute_epipolar_error_pairwise(kpts0, kpts1, E, K0, K1, error_type)  # [B, N, M]
    
    # Add dustbin rows/cols (zero gradient for dustbin entries)
    grad = torch.zeros(B, 1, N + 1, M + 1, device=device, dtype=dtype)
    grad[:, 0, :N, :M] = errors
    return grad
```

**Diffusion with Guidance** (`gaussian_diffusion.py:703-721`):
```python
for i in indices:  # T down to 1
    t = th.tensor([i] * shape[0], device=device)
    
    # Compute geometry guidance
    geometry_guidance = None
    if geometry_guidance_fn is not None and geometry_weight > 0:
        geometry_guidance = geometry_guidance_fn(adj_mat, model_kwargs)  # ∇geo
    
    # DDIM step with guided score
    out = self.ddim_sample(
        model,  # ScoreNet
        adj_mat,  # x_t
        t,
        geometry_guidance=geometry_guidance,  # ∇geo
        geometry_weight=geometry_weight,  # λ
    )
    # Inside ddim_sample: score_guided = score - λ * geometry_guidance
    adj_mat = out["sample"]  # x_{t-1}
```

**Score Network** (`diffglue.py:609-706`):
```python
def forward(self, adj_mat_fore, timesteps, data: dict):
    """
    DiffGlue acts as the score network (ScoreNet) in diffusion.
    
    Input:
        adj_mat_fore: [B, 1, N+1, M+1] noisy match matrix (x_t)
        timesteps: [B] diffusion timesteps (t)
        data: contains F₁, F₂, E_{k-1} (via keypoints, descriptors, etc.)
    
    Output:
        adj_mat: [B, 1, N+1, M+1] predicted score/noise (sθ)
    """
    # Process features through transformer
    # ... (uses F₁, F₂ from data)
    
    # Predict match matrix
    scores, _ = self.log_assignment[i](desc0, desc1)
    adj_mat = scores.unsqueeze(1).clone()
    
    return {"adj_mat": adj_mat}  # This is the score sθ
```

**Explanation**:
- **Score Network (DiffGlue)**: Predicts the "score" (gradient of log probability) at timestep t
- **Geometry Guidance**: Epipolar gradient penalizes matches that violate epipolar constraint
- **Guided Denoising**: Combined score = score_network - λ * epipolar_gradient
- **DDIM Sampling**: Deterministic reverse process (faster than full diffusion)

**Improvements**:
- ✅ Clean separation between score network and guidance
- 💡 **Current implementation uses 1-step DDIM (not full T steps)** - see `gaussian_diffusion.py:695`. This is efficient but may limit refinement capability. Consider making timesteps configurable.
- 💡 Consider time-dependent guidance weight λ(t) (stronger guidance at high noise, weaker at low noise)
- 💡 Consider using Sampson error instead of symmetric error for better numerical stability

---

## 8. Refined Soft Matches

**Diagram**:
```
Refined Soft Matches
M̃_k = DualSoftmax(x₀)
(Optional Sinkhorn)
```

**Code Location**:
- **Conversion**: `AlternatingRefinement.soft_assignment_to_matches()` at `alternating_refinement.py` (implied, similar to initial)
- **Usage**: `alternating_refinement.py:623`

**Key Implementation** (`alternating_refinement.py:618-623`):
```python
# After diffusion refinement
M_tilde_k = self.soft_assignment_to_matches(pred["sample"], scale=scale)
# This converts the diffusion output x₀ back to a soft match matrix M̃_k
```

**Explanation**:
- After diffusion, the output x₀ is in the same space as the initial match matrix
- Converted back to probability space via normalization (dual softmax)
- M̃_k is the geometrically-refined version of M_{k-1}

**Improvements**:
- ✅ Same dual softmax as initial matching
- 💡 Same comment about Sinkhorn normalization

---

## 9. Geometry Projection

**Diagram**:
```
Geometry Projection
E_k = Weighted 8-point
(Stability Check)
```

**Code Location**:
- **Estimation**: `alternating_refinement.py:627`
- **Reliability Check**: `alternating_refinement.is_reliable()` (implied at line 630)

**Key Implementation** (`alternating_refinement.py:625-635`):
```python
# E_k ← Weighted8Point(M̃_k)
E_k = self.estimate_essential_matrix(M_tilde_k, kpts0, kpts1, K0, K1)

# IsReliable check: if not reliable, keep E_{k-1}
reliable_k = self.is_reliable(E_k, M_tilde_k, kpts0, kpts1, K0, K1)
E_k = torch.where(
    reliable_k.view(-1, 1, 1).expand_as(E_k),
    E_k,
    E_prev,  # Keep previous if unreliable
)
```

**Explanation**:
- Re-estimate essential matrix from refined matches
- Reliability check prevents using degenerate/unstable estimates
- Falls back to previous estimate if current one is unreliable

**Improvements**:
- ✅ Stability check prevents degradation
- 💡 Consider logging reliability metrics to understand failure cases
- 💡 Consider adaptive guidance weight based on reliability (weaker guidance if unreliable)

---

## 10. Feedback to LoFTR

**Diagram**:
```
Feedback to LoFTR
Attention Bias:
bias = γ · log(M̃_k)
```

**Code Location**:
- **Bias Computation**: `AlternatingRefinement.compute_attention_bias()` at `alternating_refinement.py:343-353`
- **Feedback**: `alternating_refinement.py:641-647`
- **Usage in Transformer**: `DiffGlue.forward()` receives `_attention_bias` via `data` dict

**Key Implementation**:

**Attention Bias** (`alternating_refinement.py:343-353`):
```python
def compute_attention_bias(self, soft_matches, eps=1e-6):
    """
    Compute attention bias from refined matches.
    
    B^(k) = γ · log(M̃_k + ε)
    
    This bias is added to the attention logits in the transformer,
    guiding it to focus on high-confidence matches from previous iteration.
    """
    return self.feedback_scale * torch.log(soft_matches + eps)
```

**Feedback Loop** (`alternating_refinement.py:641-647`):
```python
if self.feedback_to_loftr and k < self.num_refinement_iters - 1:
    # B^(k) ← γ log(M̃_k + ε)
    attention_bias = self.compute_attention_bias(M_tilde_k)
    
    # Store bias for next iteration's TransformerHead
    data["_attention_bias"] = attention_bias
```

**Usage in Transformer** (`diffglue.py:1096`):
```python
# Get attention bias from alternating refinement loop (if provided)
attention_bias = data.get("_attention_bias", None)
# This would be used in the transformer (though current implementation may not use it directly)
```

**Explanation**:
- The refined match matrix M̃_k is converted to attention bias
- Log-space bias is added to attention logits (before softmax)
- This encourages the transformer to focus on geometrically-consistent matches
- The feedback creates a closed loop: transformer → matches → geometry → bias → transformer

**Note on Implementation**:
- The `_attention_bias` is computed and stored in `data`, but the actual feedback mechanism works through the diffusion process itself
- The feedback happens via `adj_mat_fore` (noisy match matrix from previous iteration) being passed to the transformer
- The attention bias computation exists but may be intended for future use or alternative feedback mechanisms

**Improvements**:
- ✅ Feedback mechanism works through diffusion adjacency matrix
- 💡 **Consider implementing explicit attention bias**: Add `_attention_bias` to attention logits in `CrossBlock` for more direct feedback (currently computed but not used)
- 💡 Consider different bias formulations (e.g., temperature-scaled, clipped)

---

## 11. LoFTR Context Update

**Diagram**:
```
LoFTR Context Update
S_k = CoarseHead(F₁,F₂,b)
M_k = DualSoftmax(S_k)
```

**Code Location**:
- **Transformer Forward**: `DiffGlue.forward()` at `diffglue/diffglue.py:609`
- **Next Iteration**: Implicitly happens when `diffuser(matcher, data)` is called again in the loop

**Key Implementation** (`alternating_refinement.py:611-616`):
```python
# Run diffusion with geometry guidance
# This internally calls matcher (DiffGlue) which runs transformer
pred = diffuser(
    matcher,  # DiffGlue transformer
    data,  # Contains F₁, F₂ and _attention_bias
    geometry_guidance_fn=geometry_guidance_fn,
    geometry_weight=self.geometry_guidance_weight,
)
# Inside diffuser → matcher.forward() → transformer → S_k
```

**Explanation**:
- The transformer is re-run with updated attention bias
- This produces a new similarity matrix S_k that incorporates geometric feedback
- The cycle continues: S_k → M_k → E_k → bias → S_{k+1} → ...

**Improvements**:
- ✅ Closed-loop refinement
- 💡 Consider caching transformer features if geometry doesn't change much
- 💡 Consider warm-starting transformer from previous iteration's features

---

## 12. Final Outputs

**Diagram**:
```
Final Outputs
- Soft Matches M_K
- Relative Pose (R,t)
```

**Code Location**:
- **Pose Decomposition**: `alternating_refinement.py:657-664`
- **Function**: `decompose_essential_matrix()` at `models/utils/geometry_guidance.py:354-390`

**Key Implementation** (`alternating_refinement.py:653-664`):
```python
# ============================================
# FINAL POSE
# ============================================

# E ← Weighted8Point(M_K)
pred["estimated_E"] = E_k

# (R, t) ← DecomposeEssential(E)
R, t = decompose_essential_matrix(E_k)
pred["estimated_R"] = R
pred["estimated_t"] = t

return pred
```

**Essential Matrix Decomposition** (`geometry_guidance.py:354-390`):
```python
def decompose_essential_matrix(E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose essential matrix E into rotation R and translation t.
    
    E = [t]_× R where [t]_× is the skew-symmetric matrix of t.
    
    Algorithm (based on Hartley & Zisserman):
    1. SVD: E = U * diag(1,1,0) * V^T
    2. Two possible rotations: R = U * W * V^T or U * W^T * V^T
    3. Translation: t = U[:, 2] (third column)
    4. Choose solution with points in front of both cameras
    """
    U, S, Vh = torch.linalg.svd(E)
    
    # Ensure proper essential matrix (two equal singular values)
    S_corrected = torch.tensor([1.0, 1.0, 0.0], device=E.device)
    E_corrected = U @ torch.diag(S_corrected) @ Vh
    
    # Decompose to R and t
    W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=E.device)
    R1 = U @ W @ Vh
    R2 = U @ W.T @ Vh
    t = U[:, :, 2]  # Translation direction
    
    # Choose solution with positive depth (chirality check)
    # ... (implementation details)
    
    return R, t
```

**Explanation**:
- Essential matrix E encodes relative pose up to scale
- Decomposition gives rotation R and translation direction t (scale is ambiguous)
- Multiple solutions exist; choose one that satisfies chirality (points in front of cameras)

**Improvements**:
- ✅ Standard decomposition algorithm
- 💡 Consider returning all 4 solutions and using additional constraints (e.g., known scale) to disambiguate
- 💡 Consider adding pose uncertainty estimates (from match confidence)

---

## Summary of Suggested Improvements

### High Priority
1. **Verify attention bias integration**: Ensure `_attention_bias` is actually used in transformer attention computation
2. **Sinkhorn normalization**: Add as optional alternative to dual softmax for stricter constraints
3. **Multi-step diffusion**: Currently uses 1-step DDIM; consider making timesteps configurable

### Medium Priority
4. **Early stopping**: Stop refinement loop if geometry converges
5. **Time-dependent guidance weight**: λ(t) for adaptive geometry guidance
6. **Reliability logging**: Log when geometry estimates are unreliable to understand failure cases
7. **RANSAC wrapper**: Add robust estimation option for essential matrix

### Low Priority
8. **Attention visualization**: Add utilities to visualize attention patterns
9. **Feature caching**: Cache extractor features if not trainable
10. **Warm-starting**: Initialize transformer from previous iteration's features

---

## Code Flow Summary

```
TwoViewPipeline._forward()
  ↓
extract_view() → F₁, F₂ (features)
  ↓
alternating_refinement.forward() [if enabled]
  ↓
  Initial: diffuser(matcher, data) → M₀, E₀
  ↓
  Loop k=1..K:
    ↓
    Geometry Guidance: epipolar_gradient(E_{k-1})
    ↓
    Diffusion: diffuser(matcher, data, geometry_guidance_fn) → x₀
    ↓
    Projection: soft_assignment_to_matches(x₀) → M̃_k
    ↓
    Geometry: weighted_eight_point(M̃_k) → E_k
    ↓
    Feedback: compute_attention_bias(M̃_k) → data["_attention_bias"]
    ↓
    (next iteration uses bias in transformer)
  ↓
  Final: decompose_essential_matrix(E_K) → R, t
```
