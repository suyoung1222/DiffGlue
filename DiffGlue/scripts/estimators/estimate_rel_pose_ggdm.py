#!/usr/bin/env python
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf

from estimators import estimate_relative_pose as estimator
from models.diffglue_pipeline import DiffGluePipeline

random.seed(10)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detector-free GGDM config (no extractor)
default_conf = {
    "matcher": {
        "name": "diffglue.diffglue",
        "filter_threshold": 0.1,
        "flash": True,
        "checkpointed": True,
        "n_layers": 9,
        "scale": 2,
        "epipolar": "enabled",
        "epi_weight": 1.0,
        "matcher_loss_weight": 0.1,
        "loftr_pretrained": "outdoor_ds.ckpt",
        "freeze_loftr_backbone": True,
        "freeze_loftr_coarse": False,
    },
    "diffuser": {
        "name": "diffglue.diffuser",
        "steps": 4096,
        "learn_sigma": False,
        "sigma_small": False,
        "noise_schedule": "linear",
        "use_kl": False,
        "predict_xstart": True,
        "rescale_timesteps": True,
        "rescale_learned_sigmas": True,
        "timestep_respacing": "",
        "ddim_steps": 2,
        "schedule_sampler": "uniform",
        "use_ddim": True,
        "clip_denoised": True,
        "diffuser_loss_weight": 1,
        "scale": 2,
    },
}

# Checkpoint path - fix the path construction
checkpoint_path = Path("/home/suyoung/mydata/Diffglue/weights/checkpoint_best_ggdm_80_2.tar")

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint file not found: {checkpoint_path}\n"
        f"Please ensure the checkpoint file exists at the specified path."
    )

# Create detector-free matcher (no SuperPoint) using DiffGluePipeline
conf = OmegaConf.create(default_conf)
matcher = DiffGluePipeline(conf).eval().to(device)

# Load checkpoint
print(f"Loading GGDM checkpoint from: {checkpoint_path}")
ckpt = torch.load(str(checkpoint_path), map_location="cpu")

state_dict = ckpt["model"]
dict_params = set(state_dict.keys())
model_params = set(map(lambda n: n[0], matcher.named_parameters()))
diff = model_params - dict_params
if len(diff) > 0:
    state_dict = {
        k.replace("matcher.", "matcher.net."): v for k, v in state_dict.items()
    }
matcher.load_state_dict(state_dict, strict=False)
print("GGDM detector-free matcher loaded successfully!")


def estimate_relative_pose(*args, **kwargs):
    return estimator(matcher, *args, **kwargs)
