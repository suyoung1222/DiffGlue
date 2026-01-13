from dataclasses import dataclass

import numpy as np
from utils.eval_utils import read_txt_poses


@dataclass
class RelativePoseEstimatorOutput:
    R: np.ndarray = np.zeros((3, 3), dtype=np.float32)
    t: np.ndarray = np.zeros(3, dtype=np.float32)
    valid: int = 0
    matched_ransac_kpts_l: list = None
    matched_ransac_kpts_f: list = None
    matched_kpts_l: list = None
    matched_kpts_f: list = None
    score: list = None
    valid_E: int = 0
    R_E: np.ndarray = None
    t_E: np.ndarray = None
    E: np.ndarray = None
    valid_madpose: int = 0
    R_madpose: np.ndarray = None
    t_madpose: np.ndarray = None
    pred_time: float = 0.0

    @property
    def num_matches(self):
        return (
            len(self.matched_kpts_f) if self.matched_kpts_f is not None else 0
        )

    @property
    def num_ransac_matches(self):
        return (
            len(self.matched_ransac_kpts_f)
            if self.matched_ransac_kpts_f is not None
            else 0
        )


@dataclass
class ImageConfig:
    K: np.ndarray
    resize: list


@dataclass
class RobotArgs:
    name: str
    images_dir: str
    odoms_txt: str

    image_config_raw: object

    def __post_init__(self):
        self.poses = read_txt_poses(self.odoms_txt)
        self.n = len(self.poses)

        self.image_config = ImageConfig(
            K=np.array(
                [
                    [
                        self.image_config_raw.intrinsics.fx,
                        0.0,
                        self.image_config_raw.intrinsics.cx,
                    ],
                    [
                        0.0,
                        self.image_config_raw.intrinsics.fy,
                        self.image_config_raw.intrinsics.cy,
                    ],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            resize=[
                self.image_config_raw.size.width,
                self.image_config_raw.size.height,
            ],
        )


@dataclass
class LeaderRobotArgs(RobotArgs):
    depths_dir: str = None
    depth_config: object = None


@dataclass
class FollowerRobotArgs(RobotArgs):
    depths_dir: str
    depth_config: object
