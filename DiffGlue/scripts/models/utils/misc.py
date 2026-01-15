import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F


def to_sequence(map):
    return map.flatten(-2).transpose(-1, -2)


def to_map(sequence):
    n = sequence.shape[-2]
    e = math.isqrt(n)
    assert e * e == n
    assert e * e == n
    sequence.transpose(-1, -2).unflatten(-1, [e, e])


def pad_to_length(
    x,
    length: int,
    pad_dim: int = -2,
    mode: str = "zeros",  # zeros, ones, random, random_c
    bounds: Tuple[int] = (None, None),
):
    shape = list(x.shape)
    d = x.shape[pad_dim]
    assert d <= length
    if d == length:
        return x
    shape[pad_dim] = length - d

    low, high = bounds

    if mode == "zeros":
        xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
    elif mode == "ones":
        xn = torch.ones(*shape, device=x.device, dtype=x.dtype)
    elif mode == "random":
        low = low if low is not None else x.min()
        high = high if high is not None else x.max()
        xn = torch.empty(*shape, device=x.device).uniform_(low, high)
    elif mode == "random_c":
        low, high = bounds  # we use the bounds as fallback for empty seq.
        xn = torch.cat(
            [
                torch.empty(*shape[:-1], 1, device=x.device).uniform_(
                    x[..., i].min() if d > 0 else low,
                    x[..., i].max() if d > 0 else high,
                )
                for i in range(shape[-1])
            ],
            dim=-1,
        )
    else:
        raise ValueError(mode)
    return torch.cat([x, xn], dim=pad_dim)


def pad_and_stack(
    sequences: List[torch.Tensor],
    length: Optional[int] = None,
    pad_dim: int = -2,
    **kwargs,
):
    if length is None:
        length = max([x.shape[pad_dim] for x in sequences])

    y = torch.stack([pad_to_length(x, length, pad_dim, **kwargs) for x in sequences], 0)
    return y


def align_image_pair_sizes(
    img0: torch.Tensor, 
    img1: torch.Tensor,
    hw0: Tuple[int, int],
    hw1: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Align two images to the same size by padding smaller image(s) to match the maximum size.
    This prevents FPN dimension mismatches when processing images of different sizes.
    
    Args:
        img0: First image tensor [B, C, H0, W0]
        img1: Second image tensor [B, C, H1, W1]
        hw0: Height and width of first image (H0, W0)
        hw1: Height and width of second image (H1, W1)
    
    Returns:
        img0_aligned: Padded first image [B, C, max_H, max_W]
        img1_aligned: Padded second image [B, C, max_H, max_W]
        aligned_hw: Aligned height and width (max_H, max_W)
    """
    if hw0 == hw1:
        return img0, img1, hw0
    
    max_h = max(hw0[0], hw1[0])
    max_w = max(hw0[1], hw1[1])
    
    # Pad img0 if needed
    if img0.shape[2] < max_h or img0.shape[3] < max_w:
        pad_h0 = max_h - img0.shape[2]
        pad_w0 = max_w - img0.shape[3]
        img0 = F.pad(img0, (0, pad_w0, 0, pad_h0), mode='constant', value=0)
    
    # Pad img1 if needed
    if img1.shape[2] < max_h or img1.shape[3] < max_w:
        pad_h1 = max_h - img1.shape[2]
        pad_w1 = max_w - img1.shape[3]
        img1 = F.pad(img1, (0, pad_w1, 0, pad_h1), mode='constant', value=0)
    
    return img0, img1, (max_h, max_w)
