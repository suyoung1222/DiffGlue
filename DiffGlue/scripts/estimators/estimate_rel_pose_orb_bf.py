#!/usr/bin/env python
import random
from pathlib import Path

import torch
import cv2 as cv

from estimators import estimate_relative_pose as estimator
from models.classic_matcher.bfmatching import BFMatching
random.seed(10)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matcher_config = {
    'nfeatures' : 1024,        # The maximum number of features to retain, usually 500
    'scaleFactor' : 1.2,      # Pyramid decimation ratio, greater values will decrease the number of features
    'nlevels' : 8,            # The number of pyramid levels (levels in the scale space)
    'edgeThreshold' : 31,     # The size of the border where features are not detected (default is 31)
    'firstLevel' : 0,         # The level of the pyramid to start with
    'WTA_K' : 2,              # Number of points in the oriented BRIEF descriptor (default is 2)
    'scoreType' : cv.ORB_HARRIS_SCORE,  # The method used to assign scores to keypoints (default is Harris score)
    'patchSize' : 31,         # Size of the patch used for BRIEF descriptor computation (default is 31)
    'fastThreshold' : 20      # Threshold used for the FAST feature detector (default is 20)
}

matcher = BFMatching(matcher_config).eval().to(device)


def estimate_relative_pose(*args, **kwargs):
    return estimator(matcher, *args, **kwargs)
