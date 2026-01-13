#!/usr/bin/env python
import random
from pathlib import Path

import torch

from estimators import estimate_relative_pose as estimator
from models.matching import Matching

random.seed(10)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matcher_config = {
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.0005,  # 0.005,
        "max_keypoints": 1024,
        "ckpt": Path(__file__).parent.parent / "models/weights/SP_DiffGlue_e0.tar",
    },
}
matcher = Matching(matcher_config).eval().to(device)


def estimate_relative_pose(*args, **kwargs):
    return estimator(matcher, *args, **kwargs)
