import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from utils.eval_utils import quat_to_rotmat
from schema import RelativePoseEstimatorOutput
import time

model_base = "0kc5po4ee18"
weights_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "covisnet", "weights"
)

enc = torch.jit.load(
    os.path.join(weights_dir, f"{model_base}_float32_jit_cpu_enc.ts")
)
msg = torch.jit.load(
    os.path.join(weights_dir, f"{model_base}_float32_jit_cpu_msg.ts")
)
post = torch.jit.load(
    os.path.join(weights_dir, f"{model_base}_float32_jit_cpu_post.ts")
)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            224,
            antialias=True,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),
    ]
)


def estimate_relative_pose(follower_img, leader_img, *_args, **_kwargs):
    if follower_img.ndim == 2 or (follower_img.ndim == 3 and follower_img.shape[2] == 1):
        follower_img = cv2.cvtColor(follower_img, cv2.COLOR_GRAY2RGB)
        leader_img = cv2.cvtColor(leader_img, cv2.COLOR_GRAY2RGB)
    with torch.no_grad():
        start_time = time.time()
        enc_out_l = enc(transform(leader_img).unsqueeze(0))
        enc_out_f = enc(transform(follower_img).unsqueeze(0))
        m = msg(enc_out_f, enc_out_l)
        pos, _, heading, _ = post(m)
        pos = pos.squeeze(0).cpu().numpy()
        heading = heading.squeeze(0).cpu().numpy()
        end_time = time.time()

    R = quat_to_rotmat(heading[:4].astype(np.float32))
    t = pos[:3].astype(np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    T_inv = np.linalg.inv(T)
    R_inv = T_inv[:3, :3]
    t_inv = T_inv[:3, 3]
    

    return RelativePoseEstimatorOutput(
        R=R_inv,
        t=t_inv,
        valid=True,
        matched_ransac_kpts_l=[],
        matched_ransac_kpts_f=[],
        matched_kpts_l=[],
        matched_kpts_f=[],
        valid_E=False,
        R_E=np.eye(3, dtype=np.float32),
        t_E=np.zeros(3, dtype=np.float32),
        E=np.zeros((3, 3), dtype=np.float32),
        pred_time=(end_time - start_time)*1000.0,  # ms
    )
