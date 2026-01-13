"""
Geometry guidance utilities for diffusion-based feature matching refinement.

This module provides:
- Weighted 8-point algorithm for essential matrix estimation from soft correspondences
- Epipolar error computation (symmetric and Sampson)
- Epipolar gradient computation for diffusion guidance
- Fundamental matrix utilities

These utilities enable geometry-guided diffusion refinement where the reverse
diffusion step incorporates gradients that push matches toward epipolar consistency.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def build_K_matrix(f: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Construct batched camera intrinsic matrices from focal lengths and principal points.

    Args:
        f: [B, 2] focal lengths (fx, fy)
        c: [B, 2] principal points (cx, cy)

    Returns:
        K: [B, 3, 3] camera intrinsic matrices
    """
    B = f.shape[0]
    K = torch.zeros(B, 3, 3, device=f.device, dtype=f.dtype)
    K[:, 0, 0] = f[:, 0]  # fx
    K[:, 1, 1] = f[:, 1]  # fy
    K[:, 0, 2] = c[:, 0]  # cx
    K[:, 1, 2] = c[:, 1]  # cy
    K[:, 2, 2] = 1.0
    return K


def normalize_keypoints_for_E(
    kpts: torch.Tensor, 
    K: torch.Tensor
) -> torch.Tensor:
    """
    Normalize pixel keypoints to normalized camera coordinates.
    
    Args:
        kpts: [B, N, 2] pixel coordinates
        K: [B, 3, 3] camera intrinsic matrices
    
    Returns:
        kpts_norm: [B, N, 2] normalized coordinates (x/fx - cx/fx, y/fy - cy/fy)
    """
    B, N, _ = kpts.shape
    
    # Extract intrinsics
    fx = K[:, 0, 0].unsqueeze(1)  # [B, 1]
    fy = K[:, 1, 1].unsqueeze(1)  # [B, 1]
    cx = K[:, 0, 2].unsqueeze(1)  # [B, 1]
    cy = K[:, 1, 2].unsqueeze(1)  # [B, 1]
    
    # Normalize
    x_norm = (kpts[..., 0] - cx) / fx  # [B, N]
    y_norm = (kpts[..., 1] - cy) / fy  # [B, N]
    
    return torch.stack([x_norm, y_norm], dim=-1)  # [B, N, 2]


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Create skew-symmetric matrix from 3D vector.
    
    Args:
        v: [B, 3] or [3] vector
    
    Returns:
        [B, 3, 3] or [3, 3] skew-symmetric matrix
    """
    if v.dim() == 1:
        return torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], device=v.device, dtype=v.dtype)
    
    B = v.shape[0]
    zero = torch.zeros(B, device=v.device, dtype=v.dtype)
    
    skew = torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=1)
    ], dim=1)
    
    return skew


def weighted_eight_point(
    kpts0: torch.Tensor,
    kpts1: torch.Tensor,
    weights: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    normalize: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Weighted 8-point algorithm for essential matrix estimation from soft correspondences.
    
    This estimates E using weighted least squares where weights come from a soft
    correspondence matrix (e.g., from diffusion output or LoFTR confidence).
    
    Args:
        kpts0: [B, N, 2] keypoints in image 0 (pixel coordinates)
        kpts1: [B, M, 2] keypoints in image 1 (pixel coordinates)
        weights: [B, N, M] soft correspondence weights (sum should be ~1 per row/col)
        K0: [B, 3, 3] camera intrinsic matrix for image 0
        K1: [B, 3, 3] camera intrinsic matrix for image 1
        normalize: whether to apply Hartley normalization for numerical stability
    
    Returns:
        E: [B, 3, 3] essential matrix (E such that x1^T E x0 = 0)
        F: [B, 3, 3] fundamental matrix (optional, F = K1^{-T} E K0^{-1})
    """
    B, N, _ = kpts0.shape
    M = kpts1.shape[1]
    device = kpts0.device
    dtype = kpts0.dtype
    
    # Normalize keypoints to camera coordinates
    kpts0_norm = normalize_keypoints_for_E(kpts0, K0)  # [B, N, 2]
    kpts1_norm = normalize_keypoints_for_E(kpts1, K1)  # [B, M, 2]
    
    # For weighted 8-point, we create weighted correspondences
    # Each (i, j) pair contributes weight[i,j] to the system
    
    # Expand for pairwise computation
    x0 = kpts0_norm.unsqueeze(2)  # [B, N, 1, 2]
    x1 = kpts1_norm.unsqueeze(1)  # [B, 1, M, 2]
    
    # Create homogeneous coordinates
    ones = torch.ones(B, N, M, 1, device=device, dtype=dtype)
    x0_h = torch.cat([x0.expand(-1, -1, M, -1), ones], dim=-1)  # [B, N, M, 3]
    x1_h = torch.cat([x1.expand(-1, N, -1, -1), ones], dim=-1)  # [B, N, M, 3]
    
    # Build the constraint matrix A for each correspondence pair
    # For essential matrix: x1^T E x0 = 0
    # This gives us a linear system in the 9 entries of E
    # A[i,j] = kron(x0[i], x1[j]) = [x1_0*x0_0, x1_0*x0_1, x1_0, x1_1*x0_0, x1_1*x0_1, x1_1, x0_0, x0_1, 1]
    
    # Compute outer products: each row of A is vec(x1 * x0^T)
    # A_ij = [x1_x*x0_x, x1_x*x0_y, x1_x, x1_y*x0_x, x1_y*x0_y, x1_y, x0_x, x0_y, 1]
    
    A = torch.zeros(B, N, M, 9, device=device, dtype=dtype)
    A[..., 0] = x1_h[..., 0] * x0_h[..., 0]  # x1_x * x0_x
    A[..., 1] = x1_h[..., 0] * x0_h[..., 1]  # x1_x * x0_y
    A[..., 2] = x1_h[..., 0]                  # x1_x
    A[..., 3] = x1_h[..., 1] * x0_h[..., 0]  # x1_y * x0_x
    A[..., 4] = x1_h[..., 1] * x0_h[..., 1]  # x1_y * x0_y
    A[..., 5] = x1_h[..., 1]                  # x1_y
    A[..., 6] = x0_h[..., 0]                  # x0_x
    A[..., 7] = x0_h[..., 1]                  # x0_y
    A[..., 8] = 1.0                           # 1
    
    # Apply weights: weighted least squares
    # W = diag(weights.flatten())
    # Solve (A^T W A) e = 0 via SVD of sqrt(W) A
    
    # Flatten spatial dimensions
    A_flat = A.view(B, N * M, 9)  # [B, N*M, 9]
    w_flat = weights.view(B, N * M)  # [B, N*M]
    
    # Apply sqrt weights for weighted SVD
    w_sqrt = torch.sqrt(w_flat.clamp(min=1e-8)).unsqueeze(-1)  # [B, N*M, 1]
    A_weighted = A_flat * w_sqrt  # [B, N*M, 9]
    
    # SVD to find null space
    # E is the right singular vector corresponding to smallest singular value
    try:
        U, S, Vh = torch.linalg.svd(A_weighted, full_matrices=False)
        e = Vh[:, -1, :]  # [B, 9] - last row of Vh
    except RuntimeError:
        # Fallback for numerical issues
        e = torch.zeros(B, 9, device=device, dtype=dtype)
        e[:, -1] = 1.0
    
    # Reshape to 3x3 matrix
    E = e.view(B, 3, 3)
    
    # Enforce essential matrix constraint: E should have two equal singular values
    # and one zero singular value. Project E onto the essential matrix manifold.
    E = project_to_essential_manifold(E)
    
    # Compute fundamental matrix: F = K1^{-T} E K0^{-1}
    K0_inv = torch.inverse(K0)
    K1_inv = torch.inverse(K1)
    F = K1_inv.transpose(-1, -2) @ E @ K0_inv
    
    return E, F


def project_to_essential_manifold(E: torch.Tensor) -> torch.Tensor:
    """
    Project a 3x3 matrix onto the essential matrix manifold.
    
    Essential matrices have the form E = [t]_x R where R is rotation and t is translation.
    This means E has singular values (σ, σ, 0) for some σ > 0.
    
    Args:
        E: [B, 3, 3] input matrices
    
    Returns:
        E_proj: [B, 3, 3] projected essential matrices
    """
    U, S, Vh = torch.linalg.svd(E)
    
    # Set singular values to (1, 1, 0) and reconstruct
    S_new = torch.zeros_like(S)
    S_new[:, 0] = 1.0
    S_new[:, 1] = 1.0
    # S_new[:, 2] = 0.0  # already zero
    
    E_proj = U @ torch.diag_embed(S_new) @ Vh
    
    return E_proj


def compute_epipolar_error_pairwise(
    kpts0: torch.Tensor,
    kpts1: torch.Tensor,
    E: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    error_type: str = "sampson"
) -> torch.Tensor:
    """
    Compute pairwise epipolar errors between all keypoint pairs.
    
    Args:
        kpts0: [B, N, 2] keypoints in image 0 (pixel coordinates)
        kpts1: [B, M, 2] keypoints in image 1 (pixel coordinates)
        E: [B, 3, 3] essential matrix
        K0: [B, 3, 3] camera intrinsic matrix for image 0
        K1: [B, 3, 3] camera intrinsic matrix for image 1
        error_type: "symmetric" or "sampson"
    
    Returns:
        errors: [B, N, M] pairwise epipolar errors
    """
    B, N, _ = kpts0.shape
    M = kpts1.shape[1]
    
    # Normalize to camera coordinates
    kpts0_norm = normalize_keypoints_for_E(kpts0, K0)  # [B, N, 2]
    kpts1_norm = normalize_keypoints_for_E(kpts1, K1)  # [B, M, 2]
    
    # Homogeneous coordinates
    ones_0 = torch.ones(B, N, 1, device=kpts0.device, dtype=kpts0.dtype)
    ones_1 = torch.ones(B, M, 1, device=kpts1.device, dtype=kpts1.dtype)
    x0_h = torch.cat([kpts0_norm, ones_0], dim=-1)  # [B, N, 3]
    x1_h = torch.cat([kpts1_norm, ones_1], dim=-1)  # [B, M, 3]
    
    # Compute E @ x0 for all x0: [B, N, 3]
    Ex0 = torch.einsum('bij,bnj->bni', E, x0_h)  # [B, N, 3]
    
    # Compute E^T @ x1 for all x1: [B, M, 3]
    Etx1 = torch.einsum('bji,bmj->bmi', E, x1_h)  # [B, M, 3]
    
    # Compute x1^T E x0 for all pairs: [B, N, M]
    # numerator = (x1^T E x0)^2
    x1_E_x0 = torch.einsum('bmi,bni->bnm', x1_h, Ex0)  # [B, N, M]
    
    if error_type == "symmetric":
        # Symmetric epipolar error: (x1^T E x0)^2
        errors = x1_E_x0 ** 2
    
    elif error_type == "sampson":
        # Sampson error: (x1^T E x0)^2 / (Ex0_x^2 + Ex0_y^2 + Etx1_x^2 + Etx1_y^2)
        numerator = x1_E_x0 ** 2  # [B, N, M]
        
        # Denominator terms
        Ex0_sq = Ex0[..., 0:2] ** 2  # [B, N, 2]
        Etx1_sq = Etx1[..., 0:2] ** 2  # [B, M, 2]
        
        # Sum for each pair
        Ex0_sum = Ex0_sq.sum(dim=-1).unsqueeze(2)  # [B, N, 1]
        Etx1_sum = Etx1_sq.sum(dim=-1).unsqueeze(1)  # [B, 1, M]
        denom = Ex0_sum + Etx1_sum  # [B, N, M]
        
        errors = numerator / (denom + 1e-8)
    
    else:
        raise ValueError(f"Unknown error_type: {error_type}")
    
    return errors


def epipolar_gradient(
    soft_matches: torch.Tensor,
    kpts0: torch.Tensor,
    kpts1: torch.Tensor,
    E: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    error_type: str = "sampson",
    include_dustbin: bool = True
) -> torch.Tensor:
    """
    Compute gradient of epipolar constraint w.r.t. soft match matrix.
    
    The epipolar loss for soft matches is:
        L_epi = Σ_{i,j} M_{ij} * epipolar_error(i, j)
    
    The gradient w.r.t. M is simply the pairwise epipolar errors:
        ∂L_epi/∂M_{ij} = epipolar_error(i, j)
    
    This gradient can be used to guide diffusion:
        x_{t-1} = ReverseStep(x_t, score - λ * ∇L_epi)
    
    Args:
        soft_matches: [B, N, M] or [B, 1, N+1, M+1] soft correspondence matrix
        kpts0: [B, N, 2] keypoints in image 0
        kpts1: [B, M, 2] keypoints in image 1
        E: [B, 3, 3] essential matrix
        K0: [B, 3, 3] camera intrinsic matrix for image 0
        K1: [B, 3, 3] camera intrinsic matrix for image 1
        error_type: "symmetric" or "sampson"
        include_dustbin: if True, output includes dustbin rows/cols (for diffusion format)
    
    Returns:
        gradient: [B, 1, N+1, M+1] or [B, N, M] gradient tensor
    """
    B = kpts0.shape[0]
    N = kpts0.shape[1]
    M = kpts1.shape[1]
    device = kpts0.device
    dtype = kpts0.dtype
    
    # Compute pairwise epipolar errors
    errors = compute_epipolar_error_pairwise(
        kpts0, kpts1, E, K0, K1, error_type
    )  # [B, N, M]
    
    if include_dustbin:
        # Add dustbin rows/cols (zero gradient for dustbin entries)
        grad = torch.zeros(B, 1, N + 1, M + 1, device=device, dtype=dtype)
        grad[:, 0, :N, :M] = errors
        # Dustbin gradients are zero (no epipolar constraint on unmatched points)
        return grad
    else:
        return errors


def decompose_essential_matrix(E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose essential matrix into rotation and translation.
    
    E = [t]_x R has four possible decompositions. This returns one of them.
    In practice, you'd use cheirality check to pick the correct one.
    
    Args:
        E: [B, 3, 3] essential matrix
    
    Returns:
        R: [B, 3, 3] rotation matrix (one of four possibilities)
        t: [B, 3] translation vector (up to scale)
    """
    U, S, Vh = torch.linalg.svd(E)
    
    # W matrix for decomposition
    W = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], device=E.device, dtype=E.dtype)
    
    # Two possible rotations: R = U W V^T or R = U W^T V^T
    # Two possible translations: t = U[:, 2] or t = -U[:, 2]
    
    # Return one decomposition (the correct one needs cheirality check)
    R = U @ W @ Vh
    t = U[:, :, 2]  # [B, 3]
    
    # Ensure R is a proper rotation (det = 1)
    det_R = torch.linalg.det(R)
    R = R * det_R.unsqueeze(-1).unsqueeze(-1).sign()
    
    return R, t


def soft_matches_to_hard_matches(
    soft_matches: torch.Tensor,
    threshold: float = 0.0,
    mutual: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert soft match matrix to hard matches.
    
    Args:
        soft_matches: [B, N, M] soft correspondence probabilities
        threshold: minimum confidence threshold
        mutual: if True, require mutual nearest neighbors
    
    Returns:
        matches0: [B, N] indices into kpts1 (-1 for unmatched)
        match_confidence: [B, N] confidence scores
    """
    B, N, M = soft_matches.shape
    
    # Find best match for each point in image 0
    max_scores_0, matches_0to1 = soft_matches.max(dim=2)  # [B, N]
    
    if mutual:
        # Find best match for each point in image 1
        max_scores_1, matches_1to0 = soft_matches.max(dim=1)  # [B, M]
        
        # Check mutual consistency
        indices_0 = torch.arange(N, device=soft_matches.device).unsqueeze(0).expand(B, -1)
        mutual_mask = matches_1to0.gather(1, matches_0to1) == indices_0
        
        # Apply mutual constraint
        matches_0to1 = torch.where(mutual_mask, matches_0to1, torch.tensor(-1, device=soft_matches.device))
        max_scores_0 = torch.where(mutual_mask, max_scores_0, torch.tensor(0.0, device=soft_matches.device))
    
    # Apply threshold
    matches_0to1 = torch.where(max_scores_0 > threshold, matches_0to1, torch.tensor(-1, device=soft_matches.device))
    
    return matches_0to1, max_scores_0


class GeometryGuidance:
    """
    Helper class for geometry-guided diffusion refinement.
    
    This class manages the iterative estimation of geometry (essential matrix)
    and computation of epipolar gradients for guiding diffusion.
    """
    
    def __init__(
        self,
        error_type: str = "sampson",
        guidance_weight: float = 0.1,
        min_matches_for_E: int = 8
    ):
        """
        Args:
            error_type: "symmetric" or "sampson" for epipolar error
            guidance_weight: λ weight for gradient injection
            min_matches_for_E: minimum matches required for E estimation
        """
        self.error_type = error_type
        self.guidance_weight = guidance_weight
        self.min_matches_for_E = min_matches_for_E
    
    def estimate_geometry(
        self,
        soft_matches: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        K0: torch.Tensor,
        K1: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate essential matrix from soft matches.
        
        Args:
            soft_matches: [B, N, M] soft correspondence matrix
            kpts0: [B, N, 2] keypoints in image 0
            kpts1: [B, M, 2] keypoints in image 1
            K0, K1: [B, 3, 3] camera intrinsics
        
        Returns:
            E: [B, 3, 3] essential matrix
        """
        E, _ = weighted_eight_point(kpts0, kpts1, soft_matches, K0, K1)
        return E
    
    def compute_guidance_gradient(
        self,
        soft_matches: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        E: torch.Tensor,
        K0: torch.Tensor,
        K1: torch.Tensor,
        include_dustbin: bool = True
    ) -> torch.Tensor:
        """
        Compute geometry guidance gradient for diffusion.
        
        Args:
            soft_matches: [B, N, M] or [B, 1, N+1, M+1] soft matches
            kpts0: [B, N, 2] keypoints
            kpts1: [B, M, 2] keypoints
            E: [B, 3, 3] essential matrix
            K0, K1: [B, 3, 3] camera intrinsics
            include_dustbin: whether to include dustbin in output
        
        Returns:
            gradient: [B, 1, N+1, M+1] or [B, N, M] weighted gradient
        """
        grad = epipolar_gradient(
            soft_matches, kpts0, kpts1, E, K0, K1,
            error_type=self.error_type,
            include_dustbin=include_dustbin
        )
        return self.guidance_weight * grad
    
    def refine_step(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        E: torch.Tensor,
        K0: torch.Tensor,
        K1: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply geometry-guided refinement to diffusion score.
        
        This implements: score_guided = score - λ * ∇L_epi
        
        Args:
            x_t: [B, 1, N+1, M+1] current noisy sample
            score: [B, 1, N+1, M+1] predicted score/noise
            kpts0, kpts1: keypoints
            E: essential matrix
            K0, K1: camera intrinsics
        
        Returns:
            guided_score: [B, 1, N+1, M+1] geometry-guided score
        """
        # Extract match matrix dimensions
        B, _, Np1, Mp1 = x_t.shape
        N, M = Np1 - 1, Mp1 - 1
        
        # Compute epipolar gradient
        grad = self.compute_guidance_gradient(
            x_t, kpts0[:, :N], kpts1[:, :M], E, K0, K1, include_dustbin=True
        )
        
        # Apply guidance: score - λ * gradient
        guided_score = score - grad
        
        return guided_score

