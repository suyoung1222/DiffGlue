#!/usr/bin/env python
import random
from pathlib import Path

import torch
from estimators import estimate_relative_pose as estimator
from models.matching_dino import Matching

random.seed(10)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matcher_config = {
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.0005,  # 0.005,
        "max_keypoints": 1024,
        "ckpt": Path(__file__).parent.parent
        / "models/weights/SP_DiffGlue_dinov2_vits14_e0.tar",
        # "ckpt" : Path("/home/suyoung/Documents/limo/agilex_open_class/limo/limo_gazebo_sim/scripts/models/weights/SP_DiffGlue.tar")
    },
    "encoder": {
        # "name": "dino_vits16",
        "name": "dinov2_vits14",
    },
}
matcher = Matching(matcher_config).eval().to(device)


def estimate_relative_pose(*args, **kwargs):
    return estimator(matcher, *args, **kwargs)
