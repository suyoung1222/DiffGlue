import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F

def select_topk_from_dense(coords, desc, K: int):
    """
    coords: [B, N_all, 2]  (e.g., coarse or image coords)
    desc:   [B, N_all, C]
    K:      int, number of keypoints to keep

    Returns:
        coords_k: [B, K, 2]
        desc_k:   [B, K, C]
        scores_k: [B, K]  (selection score, optional)
    """
    B, N_all, C = desc.shape

    if N_all <= K:
        # not enough points, just return all
        scores = desc.norm(dim=-1)
        return coords, desc, scores

    # simple saliency: L2 norm of descriptor (you can swap this with a learned score later)
    scores = desc.norm(dim=-1)  # [B,N_all]

    topk_scores, topk_idx = scores.topk(K, dim=1)  # [B,K]

    # gather coords: expand indices to match last dimension
    idx_coords = topk_idx.unsqueeze(-1).expand(-1, -1, 2)   # [B,K,2]
    coords_k = torch.gather(coords, 1, idx_coords)          # [B,K,2]

    # gather desc: expand indices to match descriptor dim
    idx_desc = topk_idx.unsqueeze(-1).expand(-1, -1, C)     # [B,K,C]
    desc_k = torch.gather(desc, 1, idx_desc)                # [B,K,C]

    return coords_k, desc_k, topk_scores  # [B,K,2], [B,K,C], [B,K]

def sample_descriptors(desc_map, coords, patch_size=1):
    """
    desc_map: [B,D,h,w], coords: [B,K,2] in [-1,1].
    Returns descriptors [B,K,D] via bilinear sampling (optionally average small patches).
    """
    B, D, h, w = desc_map.shape
    K = coords.shape[1]
    grid = coords.view(B, K, 1, 2)  # [B,K,1,2]
    sampled = F.grid_sample(desc_map, grid, align_corners=True)  # [B,D,K,1]
    descs = sampled.squeeze(-1).permute(0,2,1).contiguous()      # [B,K,D]
    return F.normalize(descs, dim=-1)

def dense_positions_and_descriptors(feat):
    """
    Convert a dense feature map into coarse-grid coords + descriptors.

    feat: [B, C, Hc, Wc] (coarse feature map, e.g. from LocalFeatureEncoder)

    returns:
        coords_coarse: [B, N, 2]  with N = Hc * Wc, coords in (x,y) index space
                       x in [0..Wc-1], y in [0..Hc-1]
        desc:          [B, N, C]
    """
    B, C, Hc, Wc = feat.shape

    # make grid of (x,y) in coarse feature index coordinates
    ys, xs = torch.meshgrid(
        torch.arange(Hc, device=feat.device),
        torch.arange(Wc, device=feat.device),
        indexing="ij",
    )  # each [Hc,Wc]

    coords = torch.stack([xs, ys], dim=-1).float()  # [Hc,Wc,2] in (x,y)
    coords = coords.view(1, Hc * Wc, 2).repeat(B, 1, 1)  # [B,N,2]

    # flatten descriptors
    desc = feat.permute(0, 2, 3, 1).contiguous().view(B, Hc * Wc, C)  # [B,N,C]

    return coords, desc

def upsample_coords_to_image(coords_coarse, downsample=8, offset=0.5):
    """
    Convert coarse-grid coordinates to full-resolution image pixel coordinates.

    coords_coarse: [B,N,2], in (x,y) index space on the coarse grid,
                   where x ∈ [0..Wc-1], y ∈ [0..Hc-1].

    downsample:    int, the total stride between image and coarse grid (e.g. 8)
    offset:        float, sub-cell offset; 0.5 puts you at the *center* of each cell.

    returns:
        coords_img: [B,N,2], pixel coordinates (x_img, y_img) in original image space.
    """
    # (x_coarse, y_coarse) -> (x_img, y_img)
    coords_img = (coords_coarse + offset) * downsample  # [B,N,2]
    return coords_img
