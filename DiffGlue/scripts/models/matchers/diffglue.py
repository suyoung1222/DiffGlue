import warnings
from pathlib import Path
from typing import Callable, List, Optional
import pdb
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from ...settings import DATA_PATH
from ..utils.losses import NLLLoss
from ..utils.metrics import matcher_metrics
from ..utils.net import timestep_embedding
from ..utils.pose_utils import epipolar_loss, solve_pnp_ransac, sampson_epipolar_loss
from ..utils.local_feature import dense_positions_and_descriptors, upsample_coords_to_image, select_topk_from_dense
from ..utils.misc import align_image_pair_sizes

from .LoFTR.src.loftr import default_cfg
from .LoFTR.src.loftr.backbone import build_backbone
from .LoFTR.src.loftr.utils.position_encoding import PositionEncodingSine
from .LoFTR.src.loftr.utils.coarse_matching import CoarseMatching
from .LoFTR.src.loftr.loftr_module import LocalFeatureTransformer
from einops.einops import rearrange
import pdb
import matplotlib.pyplot as plt
import numpy as np

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True


# PyTorch 2.2+ compatible custom_fwd
try:
    _custom_fwd = torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
except (AttributeError, TypeError):
    _custom_fwd = torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)

@_custom_fwd
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, adj_mat: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                if adj_mat is not None:
                    v_add = torch.einsum("...ij,...jd->...id", F.softmax(adj_mat, -1).unsqueeze(1), v)
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                if adj_mat is not None:
                    v += v_add
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            if adj_mat is not None:
                v_add = torch.einsum("...ij,...jd->...id", F.softmax(adj_mat, -1).unsqueeze(1), v)
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            if adj_mat is not None:
                v += v_add
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            if adj_mat is not None:
                attn = F.softmax(sim, -1) + F.softmax(adj_mat, -1).unsqueeze(1)
            else:
                attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, encoding0: torch.Tensor = None, encoding1: torch.Tensor = None, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if encoding0 is not None and encoding1 is not None:
            qk0 = apply_cached_rotary_emb(encoding0, qk0)
            qk1 = apply_cached_rotary_emb(encoding1, qk1)
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask=mask)
            m1 = self.flash(
                qk1, qk0, v0, mask=mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class AdjBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int = 1, bias: bool = True #, slope: float = 0.1
    ) -> None:
        super().__init__()
        self.heads = num_heads
        # self.slope = slope
        dim_head = embed_dim // num_heads
        inner_dim = dim_head * num_heads
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, adj_mat: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        v0, v1 = self.map_(self.to_v, x0, x1)
        v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (v0, v1),
        )
        sim = adj_mat.unsqueeze(1)
        if mask is not None:
            sim = sim.masked_fill(~mask, -float("inf"))
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
        m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
        if mask is not None:
            m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.atten_layers = 1
        self.embd_layers = nn.ModuleList(
            [nn.Sequential(
                nn.ReLU(),
                nn.Linear(args[2], args[0]),
            ) for _ in range(self.atten_layers)]
        )
        self_attn_args = (args[0], args[1], args[-1])
        self_attn_kwargs = {}
        self.self_attn = nn.ModuleList(
            [SelfBlock(*self_attn_args, **self_attn_kwargs) for _ in range(self.atten_layers)]
        )
        adj_attn_args = (args[0],)
        adj_attn_kwargs = {}
        self.adj_attn = nn.ModuleList(
            [AdjBlock(*adj_attn_args, **adj_attn_kwargs) for _ in range(self.atten_layers)]
        )
        cross_attn_args = (args[0], args[1], args[-1])
        cross_attn_kwargs = {}
        self.cross_attn = nn.ModuleList(
            [CrossBlock(*cross_attn_args, **cross_attn_kwargs) for _ in range(self.atten_layers)]
        )
        self.desc_compact = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(2*args[0], args[0]),
            ) for _ in range(self.atten_layers)]
        )

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        embd,
        adj_mat: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, embd, adj_mat, mask0, mask1)
        else:
            for i in range(self.atten_layers):
                desc0 = self.self_attn[i](desc0, encoding0)
                desc1 = self.self_attn[i](desc1, encoding1)
                cross_embd = self.embd_layers[i](embd).type(desc0.dtype)
                desc0_adj = desc0 + cross_embd.unsqueeze(1)
                desc1_adj = desc1 + cross_embd.unsqueeze(1)
                desc0_adj, desc1_adj = self.adj_attn[i](desc0_adj, desc1_adj, adj_mat)
                desc0, desc1 = self.desc_compact[i](torch.cat([desc0, desc0_adj], dim=-1)), self.desc_compact[i](torch.cat([desc1, desc1_adj], dim=-1))
                desc0, desc1 = self.cross_attn[i](desc0, desc1)
            return desc0, desc1

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, embd, adj_mat, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        for i in range(self.atten_layers):
            desc0 = self.self_attn[i](desc0, encoding0)
            desc1 = self.self_attn[i](desc1, encoding1)
            cross_embd = self.embd_layers[i](embd).type(desc0.dtype)
            desc0_adj = desc0 + cross_embd.unsqueeze(1)
            desc1_adj = desc1 + cross_embd.unsqueeze(1)
            desc0_adj, desc1_adj = self.adj_attn[i](desc0_adj, desc1_adj, adj_mat, mask)
            desc0, desc1 = self.desc_compact[i](torch.cat([desc0, desc0_adj], dim=-1)), self.desc_compact[i](torch.cat([desc1, desc1_adj], dim=-1))
            desc0, desc1 = self.cross_attn[i](desc0, desc1, mask)
        return desc0, desc1


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float): # TODO: how is this threshold is different from ransac? 
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1



class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class LocalFeatureEncoder(nn.Module):
    """
    Simple local feature encoder:

      Input:  [B,3,H,W]  (RGB image)
      Output: feat_coarse: [B, C, H/8, W/8]  (L2-normalized along channel)

    You can set C via `desc_dim`. This will be the descriptor dimension
    you feed into your matcher.
    """
    def __init__(self, desc_dim=256):
        super().__init__()
        self.desc_dim = desc_dim

        # Stem: downsample to 1/2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Downsample to 1/4
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # /4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
        )

        # Downsample to 1/8
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # /8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
        )

        # Project to descriptor dimension
        self.head = nn.Conv2d(256, desc_dim, kernel_size=1, bias=True)

    def forward(self, img):
        """
        img: [B,3,H,W]
        returns:
            feat_coarse: [B, desc_dim, H/8, W/8]  (L2-normalized)
        """
        x = self.stem(img)
        x = self.layer1(x)
        x = self.layer2(x)          # [B,256,H/8,W/8]
        feat = self.head(x)         # [B,desc_dim,H/8,W/8]
        feat = F.normalize(feat, dim=1)  # normalize descriptors per location
        return feat

class DiffGlue_old(nn.Module):
    default_conf = {
        "name": "diffglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": -1,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "weights": None,  # either a path or the name of pretrained weights (disk, ...)
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
    }

    required_data_keys = ["view0", "view1"]
    # required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"] 
    # required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1", "T_0to1"] 

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        self.time_embed_channels = conf.descriptor_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(conf.descriptor_dim, self.time_embed_channels), 
            nn.ReLU(), 
            nn.Linear(self.time_embed_channels, self.time_embed_channels), 
        )

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()
            
        # local keypoint encoder
        self.keypoint_encoder = LocalFeatureEncoder(
            desc_dim=conf.input_dim
        )

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, self.time_embed_channels, conf.flash) for layer_index in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            # [TokenConfidence(d) for _ in range(n - 1)]
            [TokenConfidence(d) for _ in range(n)]
        )

        self.loss_fn = NLLLoss(conf.loss)

        state_dict = None
        if conf.weights is not None:
            # weights can be either a path or an existing file from official LG
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                state_dict = torch.load(
                    str(DATA_PATH / conf.weights), map_location="cpu"
                )
            else:
                assert FileExistsError

        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, adj_mat_fore, timesteps, data: dict) -> dict:
        adj_mat_fore[...,:-1,:-1] = adj_mat_fore[...,:-1,:-1]/self.conf.scale+0.5
        adj_mat_fore[...,:-1,-1] = adj_mat_fore[...,:-1,-1]/self.conf.scale+0.5
        adj_mat_fore[...,-1,:-1] = adj_mat_fore[...,-1,:-1]/self.conf.scale+0.5
        adj_mat_fore = adj_mat_fore.squeeze(1)
        time_embd = self.time_embed(timestep_embedding(timesteps, self.conf.descriptor_dim))

        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
            
        # # Off the shelf (superpoint)
        # kpts0_old, kpts1_old = data["keypoints0"], data["keypoints1"]
        # desc0_old = data["descriptors0"].contiguous()
        # desc1_old = data["descriptors1"].contiguous()
        # print("desc0:", desc0_old.shape, 'kpts0:', kpts0_old.shape) 
        
        # # detector free
        # pdb.set_trace()
        # kpts0, desc0, conf0 = self.keypoint_encoder(data['view0']['image'])
        # kpts1, desc1, conf1 = self.keypoint_encoder(data['view1']['image'])
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
            assert "image" in data[key], f"Missing 'image' in data['{key}']"

        img0 = data["view0"]["image"]  # [B,3,H,W]
        img1 = data["view1"]["image"]
        feat0 = self.keypoint_encoder(img0)  # [B,C,H/8,W/8]
        feat1 = self.keypoint_encoder(img1)
        coords0_coarse, desc0 = dense_positions_and_descriptors(feat0)  # [B,N,2], [B,N,C]
        coords1_coarse, desc1 = dense_positions_and_descriptors(feat1)
        coords0_img = upsample_coords_to_image(coords0_coarse, downsample=8)  # [B,N,2]
        coords1_img = upsample_coords_to_image(coords1_coarse, downsample=8)
        kpts0 = coords0_img
        kpts1 = coords1_img
        # mconf0 = conf0.unsqueeze(-1)
        # mconf1 = conf1.unsqueeze(-1)
        
        K = 2048
        kpts0, desc0, score0 = select_topk_from_dense(coords0_img, desc0, K)
        kpts1, desc1, score1 = select_topk_from_dense(coords1_img, desc1, K)
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        mconf0 = score0.unsqueeze(-1)   # [B,K,1]
        mconf1 = score1.unsqueeze(-1)
        data['keypoints0'] = kpts0
        data['keypoints1'] = kpts1
        data['descriptors0'] = desc0
        data['descriptors1'] = desc1
        device = kpts0.device
        
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        # kpts0 = normalize_keypoints(kpts0, size0).clone()
        # kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )


        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)
        if torch.isnan(encoding0).any() or torch.isnan(encoding1).any():
            encoding0 = encoding0
            encoding1 = encoding1
            assert 1==2

        # GNN + final_proj + assignment
        all_desc0, all_desc1 = [], []

        for i in range(self.conf.n_layers):
            # NOTE: Checkpointing works for DiffGlue_old because it uses a simpler
            # LocalFeatureEncoder backbone instead of LoFTR. The computation graph
            # is shallow enough that checkpointing doesn't cause issues.
            if self.conf.checkpointed and self.training:
                desc0, desc1 = checkpoint(
                    self.transformers[i], desc0, desc1, encoding0, encoding1, time_embd, adj_mat_fore[...,:-1,:-1],
                    use_reentrant=False  # Required for DDP compatibility
                )
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1, time_embd, adj_mat_fore[...,:-1,:-1])
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        adj_mat = scores.unsqueeze(1).clone()
        adj_mat[...,:-1,:-1] = (adj_mat[...,:-1,:-1].exp()-0.5)*self.conf.scale
        adj_mat[...,:-1,-1] = (adj_mat[...,:-1,-1].exp()-0.5)*self.conf.scale
        adj_mat[...,-1,:-1] = (adj_mat[...,-1,:-1].exp()-0.5)*self.conf.scale

        ### TODO: Relative Pose Estimation (PnP vs W/o depth)
        # if "depth" in data['view0']:
        #     Esti_T_0to1 = solve_pnp_ransac(data["keypoints0"],
        #                                     data["keypoints1"],
        #                                     m0,
        #                                     data['view0']['camera'],
        #                                     data['view1']['camera'],
        #                                     data['view0']['depth']) # kpts0, kpts1, matches0, cam0, cam1, depth0
        # else:
        #     Esti_T_0to1 = None

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "adj_mat": adj_mat,
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "descriptors0": desc0,
            "descriptors1": desc1,
            "keypoint_scores0": mconf0,
            "keypoint_scores1": mconf1,
            # "Esti_T_0to1": Esti_T_0to1
        }

        return pred

    def loss(self, pred, data): # L_match loss and transformer related loss??
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {
                "log_assignment": la,
            }

        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"matcher_total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            # Initialize as tensor on the same device as nll to avoid mixing Python floats with tensors
            losses["confidence"] = torch.zeros_like(nll)

        # row_norm is only for logging, detach to avoid unnecessary computation graph
        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1).detach()

        if self.training:
            #L_match
            # Accumulate losses in a list to avoid in-place operations that can cause issues with checkpointing
            loss_terms = [nll]
            confidence_terms = []
            
            for i in range(N):
                params_i = loss_params(pred, i)
                nll_i, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

                if self.conf.loss.gamma > 0.0:
                    weight = self.conf.loss.gamma ** (N - i)
                else:
                    weight = i + 1
                sum_weights += weight
                loss_terms.append(nll_i * weight)

                confidence_terms.append(self.token_confidence[i].loss(
                    pred["ref_descriptors0"][:, i],
                    pred["ref_descriptors1"][:, i],
                    params_i["log_assignment"],
                    pred["log_assignment"],
                ) / (N))

                del params_i

            # Sum all loss terms at once (more efficient and avoids in-place ops that can hang with checkpointing)
            losses["matcher_total"] = torch.stack(loss_terms).sum(0)
            losses["confidence"] = torch.stack(confidence_terms).sum(0) if confidence_terms else torch.zeros_like(nll)

            #L_epipolar
            if "T_0to1" in data:
                L_epi = sampson_epipolar_loss( #32개 이미지 쌍이 들어감!
                    data["keypoints0"],
                    data["keypoints1"],
                    pred["matches0"],
                    data["T_0to1"],
                    data['view0']['camera'],
                    data['view1']['camera'],
                    weight=1.0  # or some tunable value
                ) # kpts0, kpts1, matches0, T0to1, cam0, cam1, weight=1.0
                losses["geometry"] = L_epi # * self.conf.epi_weight
            else:
                # Use tensor zero instead of Python float to avoid type mixing issues
                losses["geometry"] = torch.zeros_like(nll)

        losses["matcher_total"] /= sum_weights
        # confidences
        if self.training:
            losses["matcher_total"] = losses["matcher_total"]*self.conf.matcher_loss_weight + losses["confidence"] + self.conf.epi_weight * losses["geometry"] # TODO: lambda??weight??

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics
    
class DiffGlue(nn.Module):
    default_conf = {
        "name": "diffglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": -1,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "weights": None,  # either a path or the name of pretrained weights (disk, ...)
        "weights_from_version": "v0.1_arxiv",
        # LoFTR pretrained weights - for initializing backbone and coarse matching
        "loftr_pretrained": None,  # path to LoFTR checkpoint (e.g., "outdoor_ds.ckpt")
        "freeze_loftr_backbone": False,  # freeze ResNetFPN backbone
        "freeze_loftr_coarse": False,  # freeze LocalFeatureTransformer + CoarseMatching
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
    }

    required_data_keys = ["view0", "view1"]
    # required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"] 
    # required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1", "T_0to1"] 

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        # Misc
        self.config_loftr = default_cfg # from loftr module
        
        self.time_embed_channels = conf.descriptor_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(conf.descriptor_dim, self.time_embed_channels), 
            nn.ReLU(), 
            nn.Linear(self.time_embed_channels, self.time_embed_channels), 
        )

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()
            
        # local keypoint encoder
        self.backbone = build_backbone(self.config_loftr)
        self.pos_encoding = PositionEncodingSine(
            self.config_loftr['coarse']['d_model'],
            temp_bug_fix=self.config_loftr['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(self.config_loftr['coarse'])
        self.coarse_matching = CoarseMatching(self.config_loftr['match_coarse'])

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, self.time_embed_channels, conf.flash) for layer_index in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            # [TokenConfidence(d) for _ in range(n - 1)]
            [TokenConfidence(d) for _ in range(n)]
        )

        self.loss_fn = NLLLoss(conf.loss)

        # Load pretrained LoFTR weights for backbone and coarse matching components
        if conf.loftr_pretrained is not None:
            self._load_loftr_pretrained(conf.loftr_pretrained)
        
        # Freeze LoFTR components if requested
        if conf.freeze_loftr_backbone:
            self._freeze_module(self.backbone)
            print("[DiffGlue] Froze backbone (ResNetFPN)")
        
        if conf.freeze_loftr_coarse:
            self._freeze_module(self.pos_encoding)
            self._freeze_module(self.loftr_coarse)
            self._freeze_module(self.coarse_matching)
            print("[DiffGlue] Froze pos_encoding, loftr_coarse, coarse_matching")

        state_dict = None
        if conf.weights is not None:
            # weights can be either a path or an existing file from official LG
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                state_dict = torch.load(
                    str(DATA_PATH / conf.weights), map_location="cpu"
                )
            else:
                assert FileExistsError

        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)
    
    def _load_loftr_pretrained(self, ckpt_path: str):
        """
        Load pretrained LoFTR weights into DiffGlue's LoFTR-derived components.
        
        This loads weights for:
        - backbone (ResNetFPN)
        - pos_encoding (PositionEncodingSine)
        - loftr_coarse (LocalFeatureTransformer)
        - coarse_matching (CoarseMatching)
        
        Args:
            ckpt_path: Path to LoFTR checkpoint (e.g., "outdoor_ds.ckpt")
        """
        # Try multiple possible paths
        possible_paths = [
            Path(ckpt_path),
            Path(DATA_PATH) / ckpt_path,
            Path(__file__).parent / "LoFTR" / "weights" / ckpt_path,
        ]
        
        loftr_ckpt = None
        for p in possible_paths:
            if p.exists():
                loftr_ckpt = torch.load(str(p), map_location="cpu")
                print(f"[DiffGlue] Loading LoFTR weights from: {p}")
                break
        
        if loftr_ckpt is None:
            # Try to auto-download
            loftr_ckpt = self._auto_download_loftr(ckpt_path)
            if loftr_ckpt is None:
                print(f"[DiffGlue] Warning: Could not find or download LoFTR checkpoint '{ckpt_path}'")
                print(f"[DiffGlue] Run: python scripts/download_loftr_weights.py")
                return
        
        # LoFTR checkpoints have 'state_dict' key
        if "state_dict" in loftr_ckpt:
            loftr_state = loftr_ckpt["state_dict"]
        else:
            loftr_state = loftr_ckpt
        
        # Map LoFTR keys to DiffGlue keys
        # LoFTR uses: matcher.backbone.*, matcher.pos_encoding.*, etc.
        # We need: backbone.*, pos_encoding.*, etc.
        
        loaded_components = {
            "backbone": 0,
            "pos_encoding": 0,
            "loftr_coarse": 0,
            "coarse_matching": 0,
        }
        
        diffglue_state = self.state_dict()
        
        for loftr_key, loftr_param in loftr_state.items():
            # Remove common prefixes from LoFTR checkpoint
            key = loftr_key
            for prefix in ["matcher.", "model."]:
                if key.startswith(prefix):
                    key = key[len(prefix):]
            
            # Check if this key belongs to our target components
            for component in loaded_components:
                if key.startswith(component + "."):
                    if key in diffglue_state:
                        if diffglue_state[key].shape == loftr_param.shape:
                            diffglue_state[key] = loftr_param
                            loaded_components[component] += 1
                        else:
                            print(f"[DiffGlue] Shape mismatch for {key}: "
                                  f"{diffglue_state[key].shape} vs {loftr_param.shape}")
                    break
        
        # Load the updated state dict
        self.load_state_dict(diffglue_state, strict=False)
        
        # Print summary
        print(f"[DiffGlue] Loaded LoFTR pretrained weights:")
        for comp, count in loaded_components.items():
            if count > 0:
                print(f"  - {comp}: {count} parameters")
    
    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False
    
    def _auto_download_loftr(self, ckpt_name: str):
        """
        Auto-download LoFTR weights if not found locally.
        
        Args:
            ckpt_name: Name of the checkpoint file (e.g., "outdoor_ds.ckpt")
        
        Returns:
            Loaded checkpoint dict, or None if download failed
        """
        # Google Drive file IDs for LoFTR weights
        gdrive_ids = {
            "outdoor_ds.ckpt": "1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY",
            "indoor_ds.ckpt": "1w1Qhea3WLRMS81Vod_k5rxS_GNRgIi-O",
        }
        
        if ckpt_name not in gdrive_ids:
            print(f"[DiffGlue] Unknown checkpoint '{ckpt_name}', cannot auto-download")
            return None
        
        # Determine output path
        output_dir = Path(__file__).parent / "LoFTR" / "weights"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / ckpt_name
        
        print(f"[DiffGlue] Auto-downloading LoFTR weights '{ckpt_name}'...")
        
        try:
            import gdown
        except ImportError:
            print("[DiffGlue] Installing gdown for auto-download...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
            import gdown
        
        try:
            file_id = gdrive_ids[ckpt_name]
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=False)
            
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"[DiffGlue] ✓ Downloaded {ckpt_name} ({size_mb:.1f} MB)")
                return torch.load(str(output_path), map_location="cpu")
            else:
                print(f"[DiffGlue] ✗ Download failed for {ckpt_name}")
                return None
        except Exception as e:
            print(f"[DiffGlue] ✗ Auto-download failed: {e}")
            return None

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def _visualize_coarse_matching(self, data: dict, desc0: torch.Tensor, desc1: torch.Tensor, 
                                   kpts0: torch.Tensor, kpts1: torch.Tensor, iteration: int):
        """Visualize coarse matching results during refinement iterations."""
        batch_idx = 0
        img0 = data["view0"]["image"][batch_idx, 0].cpu().numpy()
        img1 = data["view1"]["image"][batch_idx, 0].cpu().numpy()
        kpts0_orig = data["keypoints0"][batch_idx].cpu().numpy()
        kpts1_orig = data["keypoints1"][batch_idx].cpu().numpy()
        num_kpts0, num_kpts1 = len(kpts0_orig), len(kpts1_orig)
        desc0_curr = desc0[batch_idx, :num_kpts0].cpu().numpy() if num_kpts0 > 0 else np.array([]).reshape(0, desc0.shape[-1])
        desc1_curr = desc1[batch_idx, :num_kpts1].cpu().numpy() if num_kpts1 > 0 else np.array([]).reshape(0, desc1.shape[-1])
        kpts0_norm = kpts0[batch_idx, :num_kpts0, :2].cpu().numpy() if num_kpts0 > 0 else np.array([]).reshape(0, 2)
        kpts1_norm = kpts1[batch_idx, :num_kpts1, :2].cpu().numpy() if num_kpts1 > 0 else np.array([]).reshape(0, 2)
        print("=" * 80)
        print(f"VISUALIZATION - Iteration {iteration} (after coarse matching)")
        print("=" * 80)
        print(f"\nImage 0: Shape={img0.shape}, Keypoints={num_kpts0}, Descriptors={desc0_curr.shape}")
        print(f"Image 1: Shape={img1.shape}, Keypoints={num_kpts1}, Descriptors={desc1_curr.shape}")
        if num_kpts0 > 0:
            print(f"  First 5 kpts0 (img coords):\n{kpts0_orig[:min(5, num_kpts0)]}")
            print(f"  First 5 kpts0 (normalized):\n{kpts0_norm[:min(5, num_kpts0)]}")
            print(f"  desc0 stats: mean={desc0_curr.mean():.4f}, std={desc0_curr.std():.4f}")
        if num_kpts1 > 0:
            print(f"  First 5 kpts1 (img coords):\n{kpts1_orig[:min(5, num_kpts1)]}")
            print(f"  First 5 kpts1 (normalized):\n{kpts1_norm[:min(5, num_kpts1)]}")
            print(f"  desc1 stats: mean={desc1_curr.mean():.4f}, std={desc1_curr.std():.4f}")
        print("=" * 80)
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes[0, 0].imshow(img0, cmap='gray')
        axes[0, 0].set_title(f'Original Input Image 0\nShape: {img0.shape}')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(img1, cmap='gray')
        axes[0, 1].set_title(f'Original Input Image 1\nShape: {img1.shape}')
        axes[0, 1].axis('off')
        axes[1, 0].imshow(img0, cmap='gray')
        if num_kpts0 > 0:
            axes[1, 0].scatter(kpts0_orig[:, 0], kpts0_orig[:, 1], c='red', s=10, alpha=0.6, marker='x')
        axes[1, 0].set_title(f'Image 0 After Coarse Matching\n{num_kpts0} keypoints')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(img1, cmap='gray')
        if num_kpts1 > 0:
            axes[1, 1].scatter(kpts1_orig[:, 0], kpts1_orig[:, 1], c='red', s=10, alpha=0.6, marker='x')
        axes[1, 1].set_title(f'Image 1 After Coarse Matching\n{num_kpts1} keypoints')
        axes[1, 1].axis('off')
        plt.tight_layout()
        plt.show()
        pdb.set_trace()

    def _visualize_fine_matching(self, data: dict, m0: torch.Tensor, m1: torch.Tensor, 
                                 mscores0: torch.Tensor, mscores1: torch.Tensor):
        """Visualize final fine matching results."""
        batch_idx = 0
        img0 = data["view0"]["image"][batch_idx, 0].cpu().numpy()
        img1 = data["view1"]["image"][batch_idx, 0].cpu().numpy()
        kpts0_orig = data["keypoints0"][batch_idx].cpu().numpy()
        kpts1_orig = data["keypoints1"][batch_idx].cpu().numpy()
        m0_final = m0[batch_idx].cpu().numpy()
        m1_final = m1[batch_idx].cpu().numpy()
        mscores0_final = mscores0[batch_idx].cpu().numpy()
        mscores1_final = mscores1[batch_idx].cpu().numpy()
        valid_matches = m0_final >= 0
        num_valid_matches = valid_matches.sum()
        print("\n" + "=" * 80)
        print("FINE MATCHING RESULTS (Final)")
        print("=" * 80)
        print(f"Keypoints: Image0={len(kpts0_orig)}, Image1={len(kpts1_orig)}")
        print(f"Valid matches: {num_valid_matches}")
        if len(kpts0_orig) > 0:
            print(f"Match rate (img0): {num_valid_matches / len(kpts0_orig) * 100:.2f}%")
        if len(kpts1_orig) > 0:
            print(f"Match rate (img1): {num_valid_matches / len(kpts1_orig) * 100:.2f}%")
        if num_valid_matches > 0:
            print(f"Score stats - mscores0: mean={mscores0_final[valid_matches].mean():.4f}, std={mscores0_final[valid_matches].std():.4f}")
            print(f"Score stats - mscores1: mean={mscores1_final[valid_matches].mean():.4f}, std={mscores1_final[valid_matches].std():.4f}")
            print(f"First 10 matches:")
            for idx in np.where(valid_matches)[0][:10]:
                print(f"  {idx}: {kpts0_orig[idx]} <-> {kpts1_orig[m0_final[idx]]}, scores=({mscores0_final[idx]:.4f}, {mscores1_final[m0_final[idx]]:.4f})")
        print("=" * 80)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        h0, w0, h1, w1 = img0.shape[0], img0.shape[1], img1.shape[0], img1.shape[1]
        max_h, total_w = max(h0, h1), w0 + w1
        combined_img = np.zeros((max_h, total_w), dtype=img0.dtype)
        combined_img[:h0, :w0] = img0
        combined_img[:h1, w0:w0+w1] = img1
        axes[0].imshow(combined_img, cmap='gray')
        axes[0].set_title(f'Fine Matching Results\n{num_valid_matches} valid matches out of {len(kpts0_orig)} keypoints')
        axes[0].axis('off')
        if num_valid_matches > 0:
            axes[0].scatter(kpts0_orig[:, 0], kpts0_orig[:, 1], c='red', s=20, alpha=0.7, marker='o', label='Image 0', zorder=3)
            kpts1_offset = kpts1_orig.copy()
            kpts1_offset[:, 0] += w0
            axes[0].scatter(kpts1_offset[:, 0], kpts1_offset[:, 1], c='blue', s=20, alpha=0.7, marker='o', label='Image 1', zorder=3)
            valid_indices = np.where(valid_matches)[0]
            all_scores = np.concatenate([mscores0_final[valid_matches], mscores1_final[valid_matches]])
            score_min, score_max = (all_scores.min(), all_scores.max()) if len(all_scores) > 0 else (0.0, 1.0)
            score_range = score_max - score_min if score_max > score_min else 1.0
            for idx in valid_indices:
                kpt0, kpt1 = kpts0_orig[idx], kpts1_orig[m0_final[idx]]
                kpt1_offset = kpt1.copy()
                kpt1_offset[0] += w0
                score = (mscores0_final[idx] + mscores1_final[m0_final[idx]]) / 2.0
                score_norm = np.clip((score - score_min) / score_range, 0.0, 1.0)
                axes[0].plot([kpt0[0], kpt1_offset[0]], [kpt0[1], kpt1_offset[1]], 
                            color=plt.cm.viridis(score_norm), alpha=0.6, linewidth=1.5, zorder=2)
            axes[0].legend(loc='upper right')
        if num_valid_matches > 0:
            axes[1].hist(mscores0_final[valid_matches], bins=50, alpha=0.7, label='mscores0', color='red')
            axes[1].hist(mscores1_final[valid_matches], bins=50, alpha=0.7, label='mscores1', color='blue')
            axes[1].set_xlabel('Match Score')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Match Score Distribution\n({num_valid_matches} valid matches)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No valid matches', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Match Score Distribution\n(No matches)')
            axes[1].axis('off')
        plt.tight_layout()
        plt.show()
        pdb.set_trace()

    def forward(self, adj_mat_fore, timesteps, data: dict) -> dict:
        adj_mat_fore[...,:-1,:-1] = adj_mat_fore[...,:-1,:-1]/self.conf.scale+0.5
        adj_mat_fore[...,:-1,-1] = adj_mat_fore[...,:-1,-1]/self.conf.scale+0.5
        adj_mat_fore[...,-1,:-1] = adj_mat_fore[...,-1,:-1]/self.conf.scale+0.5
        adj_mat_fore = adj_mat_fore.squeeze(1)
        time_embd = self.time_embed(timestep_embedding(timesteps, self.conf.descriptor_dim))

        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        
        # Get attention bias from alternating refinement loop (if provided)
        # This allows feeding refined match distributions back to LoFTR
        attention_bias = data.get("_attention_bias", None)
        # pdb.set_trace()
        # # Off the shelf (superpoint)
        # kpts0_old, kpts1_old = data["keypoints0"], data["keypoints1"]
        # desc0_old = data["descriptors0"].contiguous()
        # desc1_old = data["descriptors1"].contiguous()
        # print("desc0:", desc0_old.shape, 'kpts0:', kpts0_old.shape) 
        
        # # detector free
        # pdb.set_trace()
        # kpts0, desc0, conf0 = self.keypoint_encoder(data['view0']['image'])
        # kpts1, desc1, conf1 = self.keypoint_encoder(data['view1']['image'])
        
        
        # 1. Local Feature CNN
        data.update({
            'bs': data["view0"]["image"].size(0), # B, [B,3,H,W]???
            'hw0_i': data["view0"]["image"].shape[2:], 'hw1_i': data["view1"]["image"].shape[2:]
        })
 
        # to grayscale       
        data["view0"]["image"] = TF.rgb_to_grayscale(
            data["view0"]["image"], num_output_channels=1
        )
        data["view1"]["image"] = TF.rgb_to_grayscale(
            data["view1"]["image"], num_output_channels=1
        )
        
        # img = data["view0"]["image"]
        # save_image(
        #     img[0],               # shape: 1 x H x W
        #     "debug_view0.png",
        #     normalize=True
        # )

        # Align image sizes by padding to avoid FPN dimension mismatches
        data["view0"]["image"], data["view1"]["image"], aligned_hw = align_image_pair_sizes(
            data["view0"]["image"], 
            data["view1"]["image"], 
            data['hw0_i'], 
            data['hw1_i']
        )
        data['hw0_i'] = data['hw1_i'] = aligned_hw

        # Process images separately to maintain batch size consistency
        # Even when sizes match, process separately to avoid batch dimension confusion
        (feat_c0, feat_f0) = self.backbone(data['view0']['image'])
        (feat_c1, feat_f1) = self.backbone(data['view1']['image'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level with optional attention bias from refinement loop
        # This adds to data: keypoints0, keypoints1, descriptors0, descriptors1
        # - keypoints0/1: [B, L, 2] dense coarse grid positions in image coordinates
        # - descriptors0/1: [B, L, C] LoFTR coarse features after transformer (feat_c0, feat_c1)
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1, attention_bias=attention_bias)     

        # Extract keypoints and descriptors from data (set by coarse_matching.get_contextual_match)
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        # Extract descriptors from data (LoFTR coarse features set by coarse_matching.get_contextual_match)
        desc0 = data["descriptors0"].contiguous()  # [B, L, C] - original LoFTR coarse descriptors
        desc1 = data["descriptors1"].contiguous()  # [B, S, C] - original LoFTR coarse descriptors

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)
        if torch.isnan(encoding0).any() or torch.isnan(encoding1).any():
            encoding0 = encoding0
            encoding1 = encoding1
            assert 1==2

        # GNN + final_proj + assignment
        all_desc0, all_desc1 = [], []
        
        # Get num_refinement_iters from data dict (passed by alternating_refinement)
        # or fall back to n_layers if not available
        if self.training:
            n_layers_to_use = self.conf.n_layers
        else:
            # Try to get from data dict (set by alternating_refinement module)
            num_refinement_iters = data.get("_num_refinement_iters", None)
            if num_refinement_iters is not None:
                n_layers_to_use = num_refinement_iters
            else:
                # Fallback: use n_layers if refinement iters not provided
                n_layers_to_use = self.conf.n_layers
        for i in range(n_layers_to_use):  # Iteration Start
            # NOTE: Checkpointing is disabled because it causes backward pass to hang
            # when matcher is called from diffuser. The diffuser's training_losses() calls
            # matcher.forward() which creates nested autograd contexts that deadlock.
            # To re-enable, set use_checkpoint = self.conf.checkpointed and self.training
            # and ensure checkpointing works with your diffuser setup.
            desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1, time_embd, adj_mat_fore[...,:-1,:-1])
            if self.training or i == n_layers_to_use - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

            # for debug and visualization
            # self._visualize_coarse_matching(data, desc0, desc1, kpts0, kpts1, i)
            print("layer", i, "desc0:", desc0[0][1][:10], 'desc1:', desc1[0][1][:10])

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)
        
        # Fine matching visualization
        # self._visualize_fine_matching(data, m0, m1, mscores0, mscores1)
        pdb.set_trace()
        adj_mat = scores.unsqueeze(1).clone()
        adj_mat[...,:-1,:-1] = (adj_mat[...,:-1,:-1].exp()-0.5)*self.conf.scale
        adj_mat[...,:-1,-1] = (adj_mat[...,:-1,-1].exp()-0.5)*self.conf.scale
        adj_mat[...,-1,:-1] = (adj_mat[...,-1,:-1].exp()-0.5)*self.conf.scale

        ### TODO: Relative Pose Estimation (PnP vs W/o depth)
        # if "depth" in data['view0']:
        #     Esti_T_0to1 = solve_pnp_ransac(data["keypoints0"],
        #                                     data["keypoints1"],
        #                                     m0,
        #                                     data['view0']['camera'],
        #                                     data['view1']['camera'],
        #                                     data['view0']['depth']) # kpts0, kpts1, matches0, cam0, cam1, depth0
        # else:
        #     Esti_T_0to1 = None

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "adj_mat": adj_mat,
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "descriptors0": desc0,
            "descriptors1": desc1,
            # "keypoint_scores0": mconf0,
            # "keypoint_scores1": mconf1,
            # "Esti_T_0to1": Esti_T_0to1
        }

        return pred

    def loss(self, pred, data): # L_match loss and transformer related loss??
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {
                "log_assignment": la,
            }

        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"matcher_total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            # Initialize as tensor on the same device as nll to avoid mixing Python floats with tensors
            losses["confidence"] = torch.zeros_like(nll)

        # row_norm is only for logging, detach to avoid unnecessary computation graph
        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1).detach()

        if self.training:
            #L_match
            # Accumulate losses in a list to avoid in-place operations that can cause issues with checkpointing
            loss_terms = [nll]
            confidence_terms = []
            
            for i in range(N):
                params_i = loss_params(pred, i)
                nll_i, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

                if self.conf.loss.gamma > 0.0:
                    weight = self.conf.loss.gamma ** (N - i)
                else:
                    weight = i + 1
                sum_weights += weight
                loss_terms.append(nll_i * weight)

                confidence_terms.append(self.token_confidence[i].loss(
                    pred["ref_descriptors0"][:, i],
                    pred["ref_descriptors1"][:, i],
                    params_i["log_assignment"],
                    pred["log_assignment"],
                ) / (N))

                del params_i

            # Sum all loss terms at once (more efficient and avoids in-place ops that can hang with checkpointing)
            losses["matcher_total"] = torch.stack(loss_terms).sum(0)
            losses["confidence"] = torch.stack(confidence_terms).sum(0) if confidence_terms else torch.zeros_like(nll)

            #L_epipolar
            if "T_0to1" in data:
                L_epi = sampson_epipolar_loss( #32개 이미지 쌍이 들어감!
                    data["keypoints0"],
                    data["keypoints1"],
                    pred["matches0"],
                    data["T_0to1"],
                    data['view0']['camera'],
                    data['view1']['camera'],
                    weight=1.0  # or some tunable value
                ) # kpts0, kpts1, matches0, T0to1, cam0, cam1, weight=1.0
                losses["geometry"] = L_epi # * self.conf.epi_weight
            else:
                # Use tensor zero instead of Python float to avoid type mixing issues
                losses["geometry"] = torch.zeros_like(nll)

        losses["matcher_total"] /= sum_weights
        # confidences
        if self.training:
            losses["matcher_total"] = losses["matcher_total"]*self.conf.matcher_loss_weight + losses["confidence"] + self.conf.epi_weight * losses["geometry"] # TODO: lambda??weight??

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics

__main_model__ = DiffGlue
