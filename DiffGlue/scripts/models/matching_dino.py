from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision
from omegaconf import OmegaConf

from .diffglue_pipeline import DiffGluePipeline
from .superpoint import SuperPoint


def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = desc_ii.squeeze(0).transpose(0, 1), desc_jj.squeeze(
        0
    ).transpose(0, 1)
    d1 = (desc_ii**2).sum(1)
    d2 = (desc_jj**2).sum(1)
    distmat = (
        d1.unsqueeze(1)
        + d2.unsqueeze(0)
        - 2 * torch.matmul(desc_ii, desc_jj.transpose(0, 1))
    ).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:, 0]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)
    nnIdx2 = nnIdx2.squeeze()
    mutual_nearest = nnIdx2[nnIdx1] == torch.arange(nnIdx1.shape[0]).cuda()
    ratio_test = distVals[:, 0] / distVals[:, 1].clamp(min=1e-10)
    idx_sort = [torch.arange(nnIdx1.shape[0]), nnIdx1]
    return idx_sort, ratio_test, mutual_nearest


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MultiScaleImageFeatureExtractor(nn.Module):
    def __init__(
        self,
        modelname: str = "dino_vits16",
        freeze: bool = True,
        scale_factors: list = [1, 1 / 2, 1 / 3],
    ):
        super().__init__()
        self.freeze = freeze
        self.scale_factors = scale_factors

        self.modelname = modelname

        if "res" in modelname:
            self._net = getattr(torchvision.models, modelname)(pretrained=True)
            self._output_dim = self._net.fc.weight.shape[1]
            self._net.fc = nn.Identity()
        elif "dinov2" in modelname:
            self._net = torch.hub.load("facebookresearch/dinov2", modelname)
            self._output_dim = self._net.norm.weight.shape[0]
        elif "dino" in modelname:
            self._net = torch.hub.load("facebookresearch/dino:main", modelname)
            self._output_dim = self._net.norm.weight.shape[0]
        else:
            raise ValueError(f"Unknown model name {modelname}")

        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 3, 1, 1),
                persistent=False,
            )

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def get_output_dim(self):
        return self._output_dim

    def forward(self, image_rgb: torch.Tensor) -> torch.Tensor:
        img_normed = self._resnet_normalize_image(image_rgb)
        features = self._compute_multiscale_features(img_normed)
        return features

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std

    def _compute_multiscale_features(
        self, img_normed: torch.Tensor
    ) -> torch.Tensor:
        multiscale_features = None

        if len(self.scale_factors) <= 0:
            raise ValueError(
                f"Wrong format of self.scale_factors: {self.scale_factors}"
            )

        for scale_factor in self.scale_factors:
            if scale_factor == 1:
                inp = img_normed
            else:
                inp = self._resize_image(img_normed, scale_factor)

            if "dinov2" in self.modelname:
                # resize to mutiple of 14
                h, w = inp.shape[2:]
                h = h // 14 * 14
                w = w // 14 * 14

                net_inp = nn.functional.interpolate(
                    inp, size=(h, w), mode="bilinear", align_corners=False
                )
            else:
                net_inp = inp

            if multiscale_features is None:
                multiscale_features = self._net(net_inp)
            else:
                multiscale_features += self._net(net_inp)

        averaged_features = multiscale_features / len(self.scale_factors)
        return averaged_features

    def _resize_image(
        self, image: torch.Tensor, scale_factor: float
    ) -> torch.Tensor:
        return nn.functional.interpolate(
            image,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )


class BatchImageFeatureExtractor(nn.Module):
    def __init__(self, extractor_modelname):
        super().__init__()
        self.extractor = MultiScaleImageFeatureExtractor(
            modelname=extractor_modelname
        )
        self.extractor.eval()
        self.extractor.train = disabled_train

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # img_0 = data["view0"]["image"]
        # img_1 = data["view1"]["image"]
        img_0 = data["image0"]
        img_1 = data["image1"]

        with torch.no_grad():
            features_0 = self.extractor(img_0)
            features_1 = self.extractor(img_1)

        return {"features0": features_0, "features1": features_1}


class Matching(torch.nn.Module):
    """Image Matching Frontend (SuperPoint + DiffGlue)"""

    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get("superpoint", {}))
        self.encoder = BatchImageFeatureExtractor(
            config.get("encoder", {}).get("name", "dino_vits16")
        )

        default_conf = OmegaConf.create(DiffGluePipeline.default_conf)
        self.diffglue = (
            DiffGluePipeline(default_conf).eval().cuda()
        )  # load the matcher

        print("Loaded DiffGlue model")
        ckpt = self.superpoint.config["ckpt"]
        ckpt = torch.load(str(ckpt), map_location="cpu")

        state_dict = ckpt["model"]
        dict_params = set(state_dict.keys())
        model_params = set(
            map(lambda n: n[0], self.diffglue.named_parameters())
        )
        diff = model_params - dict_params
        if len(diff) > 0:
            state_dict = {
                k.replace("matcher.", "matcher.net."): v
                for k, v in state_dict.items()
            }
        self.diffglue.load_state_dict(state_dict, strict=False)

    def forward(self, data):
        """Run SuperPoint and DiffGlue"""
        pred = {}
        pred = {**pred, **self.encoder({**data, **pred})}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if "keypoints0" not in data:
            pred0 = self.superpoint({"image": data["image0"]})
            pred = {**pred, **{k + "0": v for k, v in pred0.items()}}
        if "keypoints1" not in data:
            pred1 = self.superpoint({"image": data["image1"]})
            pred = {**pred, **{k + "1": v for k, v in pred1.items()}}

        # Batch all features
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**self.diffglue(data)}
        pred = {**data, **pred}

        return pred
