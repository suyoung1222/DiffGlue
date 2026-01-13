import warnings
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from .diffglue import (
    AdjBlock,
    CrossBlock,
    DiffGlue,
    LearnableFourierPositionalEncoding,
    MatchAssignment,
    NLLLoss,
    SelfBlock,
    TokenConfidence,
    filter_matches,
    normalize_keypoints,
    timestep_embedding,
)

torch.backends.cudnn.deterministic = True


class TransformerLayerEnhanced(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.atten_layers = 1
        self.embd_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(args[2], args[0]),
                )
                for _ in range(self.atten_layers)
            ]
        )
        self.feature_embd_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(kwargs["feature_dim"], args[0]),
                )
                for _ in range(self.atten_layers)
            ]
        )
        self_attn_args = (args[0], args[1], args[-1])
        self_attn_kwargs = {}
        self.self_attn = nn.ModuleList(
            [
                SelfBlock(*self_attn_args, **self_attn_kwargs)
                for _ in range(self.atten_layers)
            ]
        )
        adj_attn_args = (args[0],)
        adj_attn_kwargs = {}
        self.adj_attn = nn.ModuleList(
            [
                AdjBlock(*adj_attn_args, **adj_attn_kwargs)
                for _ in range(self.atten_layers)
            ]
        )
        cross_attn_args = (args[0], args[1], args[-1])
        cross_attn_kwargs = {}
        self.cross_attn = nn.ModuleList(
            [
                CrossBlock(*cross_attn_args, **cross_attn_kwargs)
                for _ in range(self.atten_layers)
            ]
        )
        self.desc_compact = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * args[0], args[0]),
                )
                for _ in range(self.atten_layers)
            ]
        )

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        embd,
        feature0_embd,
        feature1_embd,
        adj_mat: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(
                desc0, desc1, encoding0, encoding1, embd, adj_mat, mask0, mask1
            )
        else:
            for i in range(self.atten_layers):
                desc0 = self.self_attn[i](desc0, encoding0)
                desc1 = self.self_attn[i](desc1, encoding1)
                cross_embd = self.embd_layers[i](embd).type(desc0.dtype)
                feature0_embd = self.feature_embd_layers[i](
                    feature0_embd
                ).type(desc0.dtype)
                feature1_embd = self.feature_embd_layers[i](
                    feature1_embd
                ).type(desc0.dtype)
                desc0_adj = (
                    desc0
                    + cross_embd.unsqueeze(1)
                    + feature0_embd.unsqueeze(1)
                )
                desc1_adj = (
                    desc1
                    + cross_embd.unsqueeze(1)
                    + feature1_embd.unsqueeze(1)
                )
                desc0_adj, desc1_adj = self.adj_attn[i](
                    desc0_adj, desc1_adj, adj_mat
                )
                desc0, desc1 = self.desc_compact[i](
                    torch.cat([desc0, desc0_adj], dim=-1)
                ), self.desc_compact[i](torch.cat([desc1, desc1_adj], dim=-1))
                desc0, desc1 = self.cross_attn[i](desc0, desc1)
            return desc0, desc1

    # This part is compiled and allows padding inputs
    def masked_forward(
        self, desc0, desc1, encoding0, encoding1, embd, adj_mat, mask0, mask1
    ):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        for i in range(self.atten_layers):
            desc0 = self.self_attn[i](desc0, encoding0)
            desc1 = self.self_attn[i](desc1, encoding1)
            cross_embd = self.embd_layers[i](embd).type(desc0.dtype)
            desc0_adj = desc0 + cross_embd.unsqueeze(1)
            desc1_adj = desc1 + cross_embd.unsqueeze(1)
            desc0_adj, desc1_adj = self.adj_attn[i](
                desc0_adj, desc1_adj, adj_mat, mask
            )
            desc0, desc1 = self.desc_compact[i](
                torch.cat([desc0, desc0_adj], dim=-1)
            ), self.desc_compact[i](torch.cat([desc1, desc1_adj], dim=-1))
            desc0, desc1 = self.cross_attn[i](desc0, desc1, mask)
        return desc0, desc1


class DiffGlueEnhanced(DiffGlue):
    required_data_keys = [
        "features0",
        "features1",
        "keypoints0",
        "keypoints1",
        "descriptors0",
        "descriptors1",
    ]

    def __init__(self, conf) -> None:
        super().__init__(conf)
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        self.time_embed_channels = conf.descriptor_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(conf.descriptor_dim, self.time_embed_channels),
            nn.ReLU(),
            nn.Linear(self.time_embed_channels, self.time_embed_channels),
        )

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(
                conf.input_dim, conf.descriptor_dim, bias=True
            )
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [
                TransformerLayerEnhanced(
                    d,
                    h,
                    self.time_embed_channels,
                    conf.flash,
                    feature_dim=conf.feature_dim,
                )
                for layer_index in range(n)
            ]
        )

        self.log_assignment = nn.ModuleList(
            [MatchAssignment(d) for _ in range(n)]
        )
        self.token_confidence = nn.ModuleList(
            # [TokenConfidence(d) for _ in range(n - 1)]
            [TokenConfidence(d) for _ in range(n)]
        )

        self.loss_fn = NLLLoss(conf.loss)

    def forward(self, adj_mat_fore, timesteps, data: dict) -> dict:
        adj_mat_fore[..., :-1, :-1] = (
            adj_mat_fore[..., :-1, :-1] / self.conf.scale + 0.5
        )
        adj_mat_fore[..., :-1, -1] = (
            adj_mat_fore[..., :-1, -1] / self.conf.scale + 0.5
        )
        adj_mat_fore[..., -1, :-1] = (
            adj_mat_fore[..., -1, :-1] / self.conf.scale + 0.5
        )
        adj_mat_fore = adj_mat_fore.squeeze(1)
        time_embd = self.time_embed(
            timestep_embedding(timesteps, self.conf.descriptor_dim)
        )

        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

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

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

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
            assert 1 == 2

        # GNN + final_proj + assignment
        all_desc0, all_desc1 = [], []

        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = checkpoint(
                    self.transformers[i],
                    desc0,
                    desc1,
                    encoding0,
                    encoding1,
                    time_embd,
                    data["features0"],
                    data["features1"],
                    adj_mat_fore[..., :-1, :-1],
                )
            else:
                desc0, desc1 = self.transformers[i](
                    desc0,
                    desc1,
                    encoding0,
                    encoding1,
                    time_embd,
                    data["features0"],
                    data["features1"],
                    adj_mat_fore[..., :-1, :-1],
                )
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(
            scores, self.conf.filter_threshold
        )

        adj_mat = scores.unsqueeze(1).clone()
        adj_mat[..., :-1, :-1] = (
            adj_mat[..., :-1, :-1].exp() - 0.5
        ) * self.conf.scale
        adj_mat[..., :-1, -1] = (
            adj_mat[..., :-1, -1].exp() - 0.5
        ) * self.conf.scale
        adj_mat[..., -1, :-1] = (
            adj_mat[..., -1, :-1].exp() - 0.5
        ) * self.conf.scale

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "adj_mat": adj_mat,
        }

        return pred


__main_model__ = DiffGlueEnhanced
