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
        "ckpt": Path(__file__).parent.parent
        / "/project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/diffglue_data/outputs/training/SP+DiffGlue_megadepth_detecterfree_ggdm_80_2/checkpoint_best.tar",
        # "ckpt" : Path("/home/suyoung/Documents/limo/agilex_open_class/limo/limo_gazebo_sim/scripts/models/weights/SP_DiffGlue.tar")
    },
}
matcher = Matching(matcher_config).eval().to(device)


def estimate_relative_pose(*args, **kwargs):
    return estimator(matcher, *args, **kwargs)
