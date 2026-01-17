"""
Alternating Refinement Loop for Geometry-Guided Feature Matching (GGDM).

This module implements the GGDM algorithm for both training and inference:

INFERENCE (Algorithm 1):
1. Initialize: Encode images, run TransformerHead, estimate initial E_0
2. For k = 1 to K:
   a. Geometry conditioning: compute Sampson map G and geo embedding z
   b. Guided denoising: run diffusion with s^guided = s - λ(t) * ∇L_epi
   c. Projection: normalize x_0 → M̃_k, estimate E_k
   d. Feedback: compute B^(k) = γ log(M̃_k + ε), re-run TransformerHead with bias
3. Final pose: decompose E_K to get (R, t)

TRAINING MODES (Section 3.7):
=============================
1. Standard DSM Training (unrolled_training=False):
   - Single forward pass, no K iterations
   - Loss: L_DSM = ||ε_θ - ε||²
   - Fast, stable gradients
   
2. Unrolled GGDM Training (unrolled_training=True):
   - Backprop through K iterations of the refinement loop
   - Losses: L_DSM + λ_match * L_match + λ_pose * L_pose
   - L_match = -Σ log M_K(i,j) for GT matches
   - L_pose = angle(R_K, R_gt) + angle(t_K, t_gt)
   - Better train/test alignment, but memory-heavy

The loop creates alternating refinement between appearance (transformer) and 
geometry (epipolar constraint), converging to geometrically consistent matches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Optional, Dict, Callable, Tuple
from functools import partial
from einops import rearrange

from .geometry_guidance import (
    GeometryGuidance,
    weighted_eight_point,
    epipolar_gradient,
    compute_epipolar_error_pairwise,
    build_K_matrix,
    decompose_essential_matrix,
)
from .misc import align_image_pair_sizes


class AlternatingRefinement(nn.Module):
    """
    Implements the GGDM alternating refinement loop (Algorithm 1).
    
    For k = 1 to K:
        1. Geometry conditioning: G ← SampsonMap(E_{k-1}), check IsReliable
        2. Guided denoising: x_T ← InitFrom(M_{k-1}), run DDIM with ∇L_epi guidance
        3. Projection: M̃_k ← Norm(x_0), E_k ← Weighted8Point(M̃_k)
        4. Feedback: B^(k) ← γ log(M̃_k + ε), S_k ← TransformerHead(bias=B^(k))
    """
    
    def __init__(
        self,
        num_refinement_iters: int = 1,
        use_geometry_guidance: bool = False,
        geometry_guidance_weight: float = 0.1,
        feedback_to_loftr: bool = False,
        feedback_scale: float = 0.5,
        geometry_error_type: str = "sampson",
        min_matches_for_E: int = 8,
        top_k_candidates: int = 500,
        reliability_threshold: float = 0.1,
        init_from_previous: bool = True,
        # Unrolled training options (Section 3.7)
        unrolled_training: bool = False,
        unrolled_k: int = 2,
        unrolled_t: int = 4,
        loss_match_weight: float = 1.0,
        loss_pose_weight: float = 0.1,
    ):
        """
        Args:
            num_refinement_iters: K iterations (1 = original behavior, no loop)
            use_geometry_guidance: Whether to inject epipolar gradient in diffusion
            geometry_guidance_weight: λ for geometry gradient injection
            feedback_to_loftr: Whether to feed refined matches back to LoFTR
            feedback_scale: γ for log(M̃) attention bias
            geometry_error_type: "sampson" or "symmetric" epipolar error
            min_matches_for_E: Minimum matches needed to estimate E
            top_k_candidates: Number of top candidates to use for E estimation
            reliability_threshold: Threshold for IsReliable check (mean Sampson error)
            init_from_previous: Whether to initialize x_T from M_{k-1}
            unrolled_training: If True, backprop through K iterations during training
            unrolled_k: Number of iterations for unrolled training (keep small!)
            unrolled_t: Number of diffusion steps for unrolled training (keep small!)
            loss_match_weight: λ_match weight for L_match
            loss_pose_weight: λ_pose weight for L_pose
        """
        super().__init__()
        
        self.num_refinement_iters = num_refinement_iters
        self.use_geometry_guidance = use_geometry_guidance
        self.geometry_guidance_weight = geometry_guidance_weight
        self.feedback_to_loftr = feedback_to_loftr
        self.feedback_scale = feedback_scale
        self.geometry_error_type = geometry_error_type
        self.min_matches_for_E = min_matches_for_E
        self.top_k_candidates = top_k_candidates
        self.reliability_threshold = reliability_threshold
        self.init_from_previous = init_from_previous
        
        # Unrolled training parameters
        self.unrolled_training = unrolled_training
        self.unrolled_k = unrolled_k
        self.unrolled_t = unrolled_t
        self.loss_match_weight = loss_match_weight
        self.loss_pose_weight = loss_pose_weight
        
        # Warmup: current epoch tracking for warmup-based mode switching
        self._current_epoch = 0
        self._warmup_epochs = 0  # Will be set from config
        
        # Initialize geometry guidance helper
        if use_geometry_guidance:
            self.geometry_guidance = GeometryGuidance(
                error_type=geometry_error_type,
                guidance_weight=geometry_guidance_weight,
                min_matches_for_E=min_matches_for_E,
            )
    
    def set_epoch(self, epoch: int):
        """Set current epoch for warmup-based mode switching."""
        self._current_epoch = epoch
    
    def set_warmup_epochs(self, warmup_epochs: int):
        """Set number of warmup epochs before enabling unrolled training."""
        self._warmup_epochs = warmup_epochs
    
    def should_use_unrolled(self) -> bool:
        """
        Check if unrolled training should be used based on warmup.
        
        Returns True if:
        - unrolled_training is enabled AND
        - current_epoch >= warmup_epochs (warmup period is over)
        """
        if not self.unrolled_training:
            return False
        return self._current_epoch >= self._warmup_epochs
    
    def _populate_keypoints_from_loftr(self, matcher, data: Dict):
        """
        Populate keypoints0, keypoints1, descriptors0, descriptors1 in data dict
        by running only the LoFTR coarse matching part (backbone + transformer + matching).
        This avoids running the full DiffGlue transformer which requires proper adj_mat.
        
        Args:
            matcher: DiffGlue matcher instance
            data: data dict that will be updated with keypoints/descriptors
        """
        # Ensure images are in the right format
        if "view0" not in data or "view1" not in data:
            return  # Can't proceed without images
        
        # Create a deep copy of data to avoid modifying the original
        import copy
        temp_data = copy.deepcopy(data)
        
        # Run only the LoFTR part (backbone + coarse matching) without the DiffGlue transformers
        # This is the same logic as in matcher.forward() but stops before transformers
        
        # 1. Local Feature CNN
        temp_data.update({
            'bs': temp_data["view0"]["image"].size(0),
            'hw0_i': temp_data["view0"]["image"].shape[2:], 
            'hw1_i': temp_data["view1"]["image"].shape[2:]
        })
        
        # Convert to grayscale
        if temp_data["view0"]["image"].shape[1] == 3:
            temp_data["view0"]["image"] = TF.rgb_to_grayscale(
                temp_data["view0"]["image"], num_output_channels=1
            )
        if temp_data["view1"]["image"].shape[1] == 3:
            temp_data["view1"]["image"] = TF.rgb_to_grayscale(
                temp_data["view1"]["image"], num_output_channels=1
            )
        
        # Align image sizes
        temp_data["view0"]["image"], temp_data["view1"]["image"], aligned_hw = align_image_pair_sizes(
            temp_data["view0"]["image"], 
            temp_data["view1"]["image"], 
            temp_data['hw0_i'], 
            temp_data['hw1_i']
        )
        temp_data['hw0_i'] = temp_data['hw1_i'] = aligned_hw
        
        # Run backbone separately
        (feat_c0, feat_f0) = matcher.backbone(temp_data['view0']['image'])
        (feat_c1, feat_f1) = matcher.backbone(temp_data['view1']['image'])
        
        temp_data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })
        
        # 2. Coarse-level LoFTR module
        feat_c0 = rearrange(matcher.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(matcher.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        
        mask_c0 = mask_c1 = None
        if 'mask0' in temp_data:
            mask_c0 = temp_data['mask0'].flatten(-2)
            mask_c1 = temp_data['mask1'].flatten(-2)
        
        feat_c0, feat_c1 = matcher.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        
        # 3. Coarse matching (this populates keypoints and descriptors in temp_data)
        attention_bias = temp_data.get("_attention_bias", None)
        matcher.coarse_matching(
            feat_c0, feat_c1, temp_data, 
            mask_c0=mask_c0, mask_c1=mask_c1, 
            attention_bias=attention_bias
        )
        
        # Copy keypoints and descriptors back to original data dict
        if "keypoints0" in temp_data:
            data["keypoints0"] = temp_data["keypoints0"]
        if "keypoints1" in temp_data:
            data["keypoints1"] = temp_data["keypoints1"]
        if "descriptors0" in temp_data:
            data["descriptors0"] = temp_data["descriptors0"]
        if "descriptors1" in temp_data:
            data["descriptors1"] = temp_data["descriptors1"]
        
        # Also copy other useful metadata
        for key in ['hw0_c', 'hw1_c', 'hw0_f', 'hw1_f', 'hw0_i', 'hw1_i', 'bs']:
            if key in temp_data:
                data[key] = temp_data[key]
    
    def extract_camera_intrinsics(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract camera intrinsic matrices from data dict.
        
        Supports:
        - Camera objects with .f, .c properties
        - Dicts with 'K' key
        - Dicts with 'f', 'c' keys
        - Fallback to identity if no camera info
        """
        B = data["view0"]["image"].shape[0]
        device = data["view0"]["image"].device
        
        if "camera" in data["view0"]:
            cam0 = data["view0"]["camera"]
            cam1 = data["view1"]["camera"]
            
            # Check if K matrix is directly available
            if hasattr(cam0, "K") or (isinstance(cam0, dict) and "K" in cam0):
                K0 = cam0["K"] if isinstance(cam0, dict) else cam0.K
                K1 = cam1["K"] if isinstance(cam1, dict) else cam1.K
            else:
                # Extract f and c from Camera object or dict
                if hasattr(cam0, "f"):  # Camera object
                    f0, c0 = cam0.f, cam0.c
                    f1, c1 = cam1.f, cam1.c
                elif isinstance(cam0, dict):  # Dict
                    f0 = cam0.get("f", torch.ones(B, 2, device=device))
                    c0 = cam0.get("c", torch.zeros(B, 2, device=device))
                    f1 = cam1.get("f", torch.ones(B, 2, device=device))
                    c1 = cam1.get("c", torch.zeros(B, 2, device=device))
                else:
                    # Unknown format, use identity
                    f0 = f1 = torch.ones(B, 2, device=device)
                    c0 = c1 = torch.zeros(B, 2, device=device)
                
                K0 = build_K_matrix(f0, c0)
                K1 = build_K_matrix(f1, c1)
        else:
            # No camera info - use identity matrix
            K0 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
            K1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        
        return K0, K1
    
    def select_top_k_candidates(
        self, 
        soft_matches: torch.Tensor, 
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k candidate matches based on confidence.
        
        Algorithm: TopKCandidates(M, k)
        
        Args:
            soft_matches: [B, N, M] soft match probabilities
            k: number of candidates to select
            
        Returns:
            indices0: [B, k] indices in image 0
            indices1: [B, k] indices in image 1 (matched to indices0)
        """
        B, N, M = soft_matches.shape
        k = min(k, N, M)
        
        # Get maximum confidence per row (best match for each keypoint in image 0)
        max_conf, best_j = soft_matches.max(dim=-1)  # [B, N], [B, N]
        
        # Select top-k by confidence
        topk_conf, topk_i = max_conf.topk(k, dim=-1)  # [B, k], [B, k]
        
        # Get corresponding j indices
        topk_j = best_j.gather(1, topk_i)  # [B, k]
        
        return topk_i, topk_j
    
    def is_reliable(
        self,
        E: torch.Tensor,
        soft_matches: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        K0: torch.Tensor,
        K1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if essential matrix estimation is reliable.
        
        Algorithm: IsReliable(E, M)
        
        Uses mean Sampson error weighted by match confidence as quality metric.
        
        Args:
            E: [B, 3, 3] essential matrix
            soft_matches: [B, N, M] soft match probabilities
            kpts0, kpts1: [B, N, 2], [B, M, 2] keypoints
            K0, K1: [B, 3, 3] camera intrinsics
            
        Returns:
            reliable: [B] boolean tensor, True if E is reliable
        """
        # Compute pairwise Sampson errors
        errors = compute_epipolar_error_pairwise(
            kpts0, kpts1, E, K0, K1, error_type="sampson"
        )  # [B, N, M]
        
        # Weighted mean error
        weighted_error = (soft_matches * errors).sum(dim=(-2, -1))  # [B]
        total_weight = soft_matches.sum(dim=(-2, -1))  # [B]
        mean_error = weighted_error / (total_weight + 1e-8)  # [B]
        
        # Reliable if mean error is below threshold
        reliable = mean_error < self.reliability_threshold
        
        return reliable
    
    def compute_sampson_map(
        self,
        E: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        K0: torch.Tensor,
        K1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Sampson error map for all keypoint pairs.
        
        Algorithm: SampsonMap(E; C)
        
        Args:
            E: [B, 3, 3] essential matrix
            kpts0, kpts1: [B, N, 2], [B, M, 2] keypoints
            K0, K1: [B, 3, 3] camera intrinsics
            
        Returns:
            G: [B, N, M] Sampson error map (geometry conditioning)
        """
        G = compute_epipolar_error_pairwise(
            kpts0, kpts1, E, K0, K1, error_type="sampson"
        )
        return G
    
    def soft_assignment_to_matches(
        self, 
        soft_assignment: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Convert diffusion output to normalized soft match matrix.
        
        Algorithm: Norm(x_0)
        """
        x = soft_assignment.squeeze(1)  # [B, N+1, M+1]
        x = x[:, :-1, :-1]  # [B, N, M] - exclude dustbin
        x = x / scale + 0.5  # Convert from scaled form
        soft_matches = F.softmax(x, dim=-1)  # Normalize
        return soft_matches
    
    def matches_to_diffusion_init(
        self,
        soft_matches: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Convert soft matches to diffusion initialization.
        
        Algorithm: InitFrom(M_{k-1})
        
        Args:
            soft_matches: [B, N, M] soft match probabilities
            scale: diffusion scale factor
            
        Returns:
            x_T: [B, 1, N+1, M+1] initial noise (centered on M_{k-1})
        """
        B, N, M = soft_matches.shape
        device = soft_matches.device
        
        # Create full assignment matrix with dustbin
        x = torch.zeros(B, 1, N+1, M+1, device=device)
        
        # Convert to scaled form: (p - 0.5) * scale
        x[:, 0, :-1, :-1] = (soft_matches - 0.5) * scale
        
        # Add small noise for diffusion (but keep structure from M_{k-1})
        noise_scale = 0.1 * scale
        noise = torch.randn_like(x) * noise_scale
        noise[..., -1, -1] = 0  # No noise on dustbin corner
        
        x_init = x + noise
        
        return x_init
    
    def compute_attention_bias(
        self,
        soft_matches: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute attention bias from refined soft matches.
        
        Algorithm: B^(k) ← γ log(M̃_k + ε)
        """
        return self.feedback_scale * torch.log(soft_matches + eps)
    
    def create_geometry_guidance_fn(
        self,
        G: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        E: torch.Tensor,
        K0: torch.Tensor,
        K1: torch.Tensor,
        scale: float = 1.0,
    ) -> Callable:
        """
        Create geometry guidance function for diffusion.
        
        Algorithm: g ← ∇_x L_epi(x_t; G)
        
        Args:
            G: [B, N, M] Sampson error map (geometry conditioning)
            kpts0, kpts1: keypoints
            E: essential matrix
            K0, K1: camera intrinsics
            scale: diffusion scale factor
            
        Returns:
            guidance_fn: callable(x_t, model_kwargs) -> gradient
        """
        def guidance_fn(x_t: torch.Tensor, model_kwargs: dict) -> torch.Tensor:
            soft_matches = self.soft_assignment_to_matches(x_t, scale=scale)
            
            # Use precomputed Sampson map G for gradient
            # ∇L_epi = G (the error at each position is the gradient)
            grad = epipolar_gradient(
                soft_matches=soft_matches,
                kpts0=kpts0,
                kpts1=kpts1,
                E=E,
                K0=K0,
                K1=K1,
                error_type=self.geometry_error_type,
                include_dustbin=True,
            )
            
            return grad
        
        return guidance_fn
    
    def estimate_essential_matrix(
        self,
        soft_matches: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        K0: torch.Tensor,
        K1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate essential matrix from soft matches.
        
        Algorithm: Weighted8Point(M; C)
        """
        E, _ = weighted_eight_point(kpts0, kpts1, soft_matches, K0, K1)
        return E
    
    def compute_match_loss(
        self,
        soft_matches: torch.Tensor,
        gt_assignment: torch.Tensor,
        gt_matches0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matching loss on refined soft matches.
        
        L_match = -Σ log M_K(i,j) for (i,j) ∈ GT matches
        
        Args:
            soft_matches: [B, N, M] refined soft match matrix M_K
            gt_assignment: [B, N, M] ground truth assignment matrix
            gt_matches0: [B, N] GT match indices (-1 for unmatched)
            
        Returns:
            loss: scalar loss value
        """
        B, N, M = soft_matches.shape
        
        # Get valid GT matches
        valid_mask = gt_matches0 >= 0  # [B, N]
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=soft_matches.device)
        
        # For each valid match, get the probability assigned to the GT correspondence
        batch_idx = torch.arange(B, device=soft_matches.device).unsqueeze(1).expand(-1, N)
        row_idx = torch.arange(N, device=soft_matches.device).unsqueeze(0).expand(B, -1)
        col_idx = gt_matches0.clamp(min=0)  # Clamp to avoid index errors
        
        # Get predicted probabilities for GT matches
        pred_probs = soft_matches[batch_idx, row_idx, col_idx]  # [B, N]
        
        # Compute negative log likelihood only for valid matches
        log_probs = torch.log(pred_probs + 1e-8)
        loss = -log_probs[valid_mask].mean()
        
        return loss
    
    def compute_pose_loss(
        self,
        R_est: torch.Tensor,
        t_est: torch.Tensor,
        T_0to1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pose loss between estimated and ground truth pose.
        
        L_pose = angle(R, R_gt) + angle(t, t_gt)
        
        Args:
            R_est: [B, 3, 3] estimated rotation
            t_est: [B, 3] estimated translation
            T_0to1: Ground truth transformation (4x4 matrix or Pose object)
            
        Returns:
            loss: scalar loss value (in radians)
        """
        # Extract GT rotation and translation
        if hasattr(T_0to1, 'R') and hasattr(T_0to1, 't'):
            R_gt = T_0to1.R
            t_gt = T_0to1.t
        elif isinstance(T_0to1, torch.Tensor) and T_0to1.shape[-2:] == (4, 4):
            R_gt = T_0to1[..., :3, :3]
            t_gt = T_0to1[..., :3, 3]
        else:
            # Cannot compute pose loss without GT
            return torch.tensor(0.0, device=R_est.device)
        
        # Rotation error: angle(R_est, R_gt)
        # trace(R_est @ R_gt.T) = 1 + 2*cos(θ)
        R_rel = torch.bmm(R_est, R_gt.transpose(-1, -2))
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_angle = ((trace - 1) / 2).clamp(-1, 1)
        rot_error = torch.acos(cos_angle)  # [B] in radians
        
        # Translation error: angle between directions (t and -t are equivalent)
        t_est_norm = t_est / (t_est.norm(dim=-1, keepdim=True) + 1e-8)
        t_gt_norm = t_gt / (t_gt.norm(dim=-1, keepdim=True) + 1e-8)
        cos_t = (t_est_norm * t_gt_norm).sum(-1).abs().clamp(0, 1)
        trans_error = torch.acos(cos_t)  # [B] in radians
        
        # Combined loss
        loss = (rot_error + trans_error).mean()
        
        return loss
    
    def forward(
        self,
        diffuser,
        matcher,
        data: Dict,
    ) -> Dict:
        """
        Run the GGDM alternating refinement loop (Algorithm 1).
        
        Args:
            diffuser: SpacedDiffusion instance
            matcher: DiffGlue instance
            data: input data dict
            
        Returns:
            pred: prediction dict with refined matches and pose
        """
        # Extract camera intrinsics
        K0, K1 = self.extract_camera_intrinsics(data)
        scale = getattr(diffuser.conf, 'scale', 1.0)
        
        # For training: choose between standard DSM or unrolled GGDM
        # Uses warmup: first N epochs use DSM, then switch to unrolled GGDM
        if matcher.training:
            if self.should_use_unrolled():
                return self.forward_unrolled(diffuser, matcher, data, K0, K1, scale)
            else:
                pred = diffuser(matcher, data)
                return pred
        
        # ============================================
        # INITIALIZATION (before main loop)
        # ============================================
        
        # Ensure keypoints and descriptors are in data (for detector-free mode)
        # The matcher needs to populate these from LoFTR coarse matching before diffuser can use them
        if "keypoints0" not in data or "keypoints1" not in data:
            # Populate keypoints/descriptors by running LoFTR coarse matching
            # This is needed for detector-free mode where no extractor provides keypoints
            self._populate_keypoints_from_loftr(matcher, data)
        
        # Pass num_refinement_iters to matcher for adaptive layer usage
        data["_num_refinement_iters"] = self.num_refinement_iters
        
        # Run initial forward to get M_0
        pred = diffuser(matcher, data)
        
        if "sample" not in pred:
            return pred
        
        # M_0 ← Norm(initial output)
        M_k = self.soft_assignment_to_matches(pred["sample"], scale=scale)
        
        # Get keypoints
        kpts0 = data.get("keypoints0", pred.get("keypoints0"))
        kpts1 = data.get("keypoints1", pred.get("keypoints1"))
        
        if kpts0 is None or kpts1 is None:
            return pred
        
        # E_0 ← Weighted8Point(M_0)
        E_k = self.estimate_essential_matrix(M_k, kpts0, kpts1, K0, K1)
        
        # Skip remaining iterations if K=1
        if self.num_refinement_iters <= 1:
            pred["estimated_E"] = E_k
            R, t = decompose_essential_matrix(E_k)
            pred["estimated_R"] = R
            pred["estimated_t"] = t
            pred["refined_matches"] = M_k
            return pred
        
        # ============================================
        # MAIN LOOP: for k = 1 to K
        # ============================================
        
        E_prev = E_k
        M_prev = M_k
        
        for k in range(1, self.num_refinement_iters):
            # --------------------------------------------
            # 1. Geometry conditioning
            # --------------------------------------------
            geometry_guidance_fn = None
            G = None
            
            if self.use_geometry_guidance:
                # IsReliable(E_{k-1}, M_{k-1})
                reliable = self.is_reliable(E_prev, M_prev, kpts0, kpts1, K0, K1)
                
                if reliable.all():
                    # G ← SampsonMap(E_{k-1})
                    G = self.compute_sampson_map(E_prev, kpts0, kpts1, K0, K1)
                    
                    # Create guidance function (includes GeoEmbed implicitly via E)
                    geometry_guidance_fn = self.create_geometry_guidance_fn(
                        G=G,
                        kpts0=kpts0,
                        kpts1=kpts1,
                        E=E_prev,
                        K0=K0,
                        K1=K1,
                        scale=scale,
                    )
            
            # --------------------------------------------
            # 2. Guided denoising
            # --------------------------------------------
            
            # x_T ← InitFrom(M_{k-1})
            if self.init_from_previous:
                init_noise = self.matches_to_diffusion_init(M_prev, scale=scale)
            else:
                init_noise = None
            
            # Run diffusion with geometry guidance
            # This internally does: s^guided = s - λ(t) * g, then DDIMStep
            pred = diffuser(
                matcher,
                data,
                geometry_guidance_fn=geometry_guidance_fn,
                geometry_weight=self.geometry_guidance_weight if geometry_guidance_fn else 0.0,
            )
            
            # --------------------------------------------
            # 3. Projection
            # --------------------------------------------
            
            # M̃_k ← Norm(x_0)
            M_tilde_k = self.soft_assignment_to_matches(pred["sample"], scale=scale)
            
            # TopKCandidates (implicit in weighted 8-point via confidence weighting)
            # E_k ← Weighted8Point(M̃_k)
            E_k = self.estimate_essential_matrix(M_tilde_k, kpts0, kpts1, K0, K1)
            
            # IsReliable check: if not reliable, keep E_{k-1}
            reliable_k = self.is_reliable(E_k, M_tilde_k, kpts0, kpts1, K0, K1)
            E_k = torch.where(
                reliable_k.view(-1, 1, 1).expand_as(E_k),
                E_k,
                E_prev,
            )
            
            # --------------------------------------------
            # 4. Feedback to TransformerHead
            # --------------------------------------------
            
            if self.feedback_to_loftr and k < self.num_refinement_iters - 1:
                # B^(k) ← γ log(M̃_k + ε)
                attention_bias = self.compute_attention_bias(M_tilde_k)
                
                # Store bias for next iteration's TransformerHead
                # S_k ← TransformerHead(F1, F2; bias=B^(k))
                data["_attention_bias"] = attention_bias
            
            # Update for next iteration
            E_prev = E_k
            M_prev = M_tilde_k
        
        # ============================================
        # FINAL POSE
        # ============================================
        
        # E ← Weighted8Point(M_K)
        # (already computed as E_k in last iteration)
        pred["estimated_E"] = E_k
        
        # (R, t) ← DecomposeEssential(E)
        R, t = decompose_essential_matrix(E_k)
        pred["estimated_R"] = R
        pred["estimated_t"] = t
        pred["refined_matches"] = M_k
        return pred
    
    def forward_unrolled(
        self,
        diffuser,
        matcher,
        data: Dict,
        K0: torch.Tensor,
        K1: torch.Tensor,
        scale: float,
    ) -> Dict:
        """
        Unrolled GGDM training with full gradient flow through K iterations.
        
        This implements Section 3.7 of the method: "Optional end-to-end fine-tuning"
        
        Total loss = λ_DSM * L_DSM + λ_match * L_match + λ_pose * L_pose
        
        Where:
            L_DSM = ||ε_θ - ε||² (standard denoising score matching)
            L_match = -Σ log M_K(i,j) for GT matches
            L_pose = angle(R_K, R_gt) + angle(t_K, t_gt)
        
        Args:
            diffuser: SpacedDiffusion instance
            matcher: DiffGlue instance
            data: input data dict
            K0, K1: camera intrinsics
            scale: diffusion scale factor
            
        Returns:
            pred: prediction dict with losses
        """
        # ============================================
        # Step 1: Run standard diffusion forward (includes L_DSM)
        # ============================================
        pred = diffuser(matcher, data)
        
        # Early exit if no sample
        if "sample" not in pred:
            return pred
        
        # Get keypoints and GT data
        kpts0 = data.get("keypoints0", pred.get("keypoints0"))
        kpts1 = data.get("keypoints1", pred.get("keypoints1"))
        gt_assignment = data.get("gt_assignment")
        gt_matches0 = data.get("gt_matches0")
        T_0to1 = data.get("T_0to1")
        
        if kpts0 is None or kpts1 is None:
            return pred
        
        # ============================================
        # Step 2: Initialize M_0 and E_0
        # ============================================
        M_k = self.soft_assignment_to_matches(pred["sample"], scale=scale)
        E_k = self.estimate_essential_matrix(M_k, kpts0, kpts1, K0, K1)
        
        # ============================================
        # Step 3: Unrolled refinement loop for k = 1 to unrolled_k
        # ============================================
        E_prev = E_k
        M_prev = M_k
        
        for k in range(1, self.unrolled_k + 1):
            # --------------------------------------------
            # 3a. Geometry conditioning (with gradient flow)
            # --------------------------------------------
            geometry_guidance_fn = None
            
            if self.use_geometry_guidance:
                # Compute Sampson map (differentiable)
                G = self.compute_sampson_map(E_prev, kpts0, kpts1, K0, K1)
                
                # Create guidance function
                geometry_guidance_fn = self.create_geometry_guidance_fn(
                    G=G,
                    kpts0=kpts0,
                    kpts1=kpts1,
                    E=E_prev,
                    K0=K0,
                    K1=K1,
                    scale=scale,
                )
            
            # --------------------------------------------
            # 3b. Guided denoising (with gradient flow)
            # --------------------------------------------
            # Note: For unrolled training, we run fewer diffusion steps
            # to reduce memory and computational cost
            
            if self.init_from_previous and k > 1:
                # Initialize from previous M (warm start)
                init_noise = self.matches_to_diffusion_init(M_prev, scale=scale)
            else:
                init_noise = None
            
            # Run diffusion with reduced steps for training
            # The diffuser's training_losses handles the DSM loss
            pred_k = diffuser(
                matcher,
                data,
                geometry_guidance_fn=geometry_guidance_fn,
                geometry_weight=self.geometry_guidance_weight if geometry_guidance_fn else 0.0,
            )
            
            # Merge losses from this iteration
            if "diffuser_loss" in pred_k:
                if "diffuser_loss" not in pred:
                    pred["diffuser_loss"] = pred_k["diffuser_loss"]
                else:
                    # Accumulate diffusion loss (will average later)
                    pred["diffuser_loss"] = pred["diffuser_loss"] + pred_k["diffuser_loss"]
            
            # --------------------------------------------
            # 3c. Projection (differentiable)
            # --------------------------------------------
            if "sample" in pred_k:
                M_tilde_k = self.soft_assignment_to_matches(pred_k["sample"], scale=scale)
                E_k = self.estimate_essential_matrix(M_tilde_k, kpts0, kpts1, K0, K1)
                
                # Update for next iteration
                E_prev = E_k
                M_prev = M_tilde_k
            
            # --------------------------------------------
            # 3d. Feedback (store attention bias)
            # --------------------------------------------
            if self.feedback_to_loftr and k < self.unrolled_k:
                attention_bias = self.compute_attention_bias(M_prev)
                data["_attention_bias"] = attention_bias
        
        # ============================================
        # Step 4: Compute final losses L_match and L_pose
        # ============================================
        
        # Final refined matches M_K
        M_K = M_prev
        
        # L_match: matching loss on final M_K
        if gt_assignment is not None and gt_matches0 is not None:
            loss_match = self.compute_match_loss(M_K, gt_assignment, gt_matches0)
            pred["loss_match"] = loss_match * self.loss_match_weight
        
        # L_pose: pose loss on final E_K
        if T_0to1 is not None:
            R_K, t_K = decompose_essential_matrix(E_k)
            loss_pose = self.compute_pose_loss(R_K, t_K, T_0to1)
            pred["loss_pose"] = loss_pose * self.loss_pose_weight
            
            # Store estimated pose
            pred["estimated_E"] = E_k
            pred["estimated_R"] = R_K
            pred["estimated_t"] = t_K
        
        # Average diffusion loss if accumulated over iterations
        if "diffuser_loss" in pred and self.unrolled_k > 1:
            pred["diffuser_loss"] = pred["diffuser_loss"] / self.unrolled_k
        
        # Store final refined matches
        pred["refined_matches"] = M_K
        
        return pred
    
    def forward_training(
        self,
        diffuser,
        matcher,
        data: Dict,
        train_with_guidance: bool = False,
    ) -> Dict:
        """
        Training forward pass with optional geometry regularization.
        
        Args:
            diffuser: SpacedDiffusion instance
            matcher: DiffGlue instance
            data: input data dict
            train_with_guidance: If True, use geometry guidance during training
                for better train/inference alignment. Default False (standard training).
        
        Note on train/inference consistency:
            - By default, training uses standard diffusion without geometry guidance
            - Inference uses GGDM with geometry guidance, attention bias, etc.
            - This follows the classifier-free guidance paradigm:
              train unconditional, guide at inference
            - Set train_with_guidance=True for closer alignment (experimental)
        """
        # Standard training: no geometry guidance
        if not train_with_guidance or not self.use_geometry_guidance:
            pred = diffuser(matcher, data)
        else:
            # Experimental: train with geometry guidance for alignment
            # Estimate E from GT matches to create guidance
            K0, K1 = self.extract_camera_intrinsics(data)
            kpts0 = data.get("keypoints0")
            kpts1 = data.get("keypoints1")
            
            geometry_guidance_fn = None
            if kpts0 is not None and kpts1 is not None and "gt_assignment" in data:
                gt_matches = data["gt_assignment"].float()
                gt_matches = F.softmax(gt_matches, dim=-1)
                E_gt = self.estimate_essential_matrix(gt_matches, kpts0, kpts1, K0, K1)
                
                # Create guidance from GT essential matrix
                G = self.compute_sampson_map(E_gt, kpts0, kpts1, K0, K1)
                scale = getattr(diffuser.conf, 'scale', 1.0)
                geometry_guidance_fn = self.create_geometry_guidance_fn(
                    G=G, kpts0=kpts0, kpts1=kpts1, E=E_gt, K0=K0, K1=K1, scale=scale
                )
            
            # Training with geometry guidance (experimental)
            pred = diffuser(
                matcher, data,
                geometry_guidance_fn=geometry_guidance_fn,
                geometry_weight=self.geometry_guidance_weight * 0.1,  # Reduced weight for training
            )
        
        # Optionally store estimated E for auxiliary losses
        if self.use_geometry_guidance and "gt_assignment" in data:
            K0, K1 = self.extract_camera_intrinsics(data)
            kpts0 = data.get("keypoints0")
            kpts1 = data.get("keypoints1")
            
            if kpts0 is not None and kpts1 is not None:
                gt_matches = data["gt_assignment"].float()
                gt_matches = F.softmax(gt_matches, dim=-1)
                E = self.estimate_essential_matrix(gt_matches, kpts0, kpts1, K0, K1)
                pred["estimated_E"] = E
        
        return pred


def create_alternating_refinement(conf: dict) -> AlternatingRefinement:
    """Factory function to create AlternatingRefinement from config."""
    ar = AlternatingRefinement(
        num_refinement_iters=conf.get("num_refinement_iters", 1),
        use_geometry_guidance=conf.get("use_geometry_guidance", False),
        geometry_guidance_weight=conf.get("geometry_guidance_weight", 0.1),
        feedback_to_loftr=conf.get("feedback_to_loftr", False),
        feedback_scale=conf.get("feedback_scale", 0.5),
        geometry_error_type=conf.get("geometry_error_type", "sampson"),
        min_matches_for_E=conf.get("min_matches_for_E", 8),
        top_k_candidates=conf.get("top_k_candidates", 500),
        reliability_threshold=conf.get("reliability_threshold", 0.1),
        init_from_previous=conf.get("init_from_previous", True),
        # Unrolled training options
        unrolled_training=conf.get("unrolled_training", False),
        unrolled_k=conf.get("unrolled_k", 2),
        unrolled_t=conf.get("unrolled_t", 4),
        loss_match_weight=conf.get("loss_match_weight", 1.0),
        loss_pose_weight=conf.get("loss_pose_weight", 0.1),
    )
    # Set warmup epochs
    ar.set_warmup_epochs(conf.get("warmup_epochs", 0))
    return ar
