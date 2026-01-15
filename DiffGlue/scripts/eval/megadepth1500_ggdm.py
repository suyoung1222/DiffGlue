import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import pdb

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH, EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_matches_epipolar, eval_poses, eval_relative_pose_robust, eval_relative_pose_dlt

logger = logging.getLogger(__name__)


class MegaDepth1500GGDMPipeline(EvalPipeline):
    """
    MegaDepth-1500 evaluation pipeline for detector-free GGDM model.
    
    This pipeline is specifically designed for evaluating the Geometry-Guided
    Diffusion Matching (GGDM) model in detector-free mode, where no keypoint
    extractor (e.g., SuperPoint) is used. The model operates directly on raw images.
    """
    default_conf = {
        "data": {
            "name": "image_pairs",
            "pairs": "megadepth1500/pairs_calibrated.txt",
            "root": "megadepth1500/images/",
            "extra_data": "relative_pose",
            "preprocessing": {
                "side": "long",
            },
        },
        "model": {
            "extractor": {
                "name": None,  # Detector-free: no keypoint extractor
            },
            "ground_truth": {
                "name": None,  # remove gt matches
            },
            # Ensure GGDM refinement is enabled
            "refinement": {
                "num_refinement_iters": 3,
                "use_geometry_guidance": True,
                "geometry_guidance_weight": 0.1,
                "feedback_to_loftr": True,
                "feedback_scale": 0.5,
            },
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0,  # -1 runs a bunch of thresholds and selects the best
        },
    }

    # Export keys for detector-free model (no keypoint scores since no detector)
    export_keys = [
        "keypoints0",
        "keypoints1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
        # GGDM-specific outputs
        "estimated_E",
        "estimated_R",
        "estimated_t",
    ]
    optional_export_keys = [
        "keypoint_scores0",
        "keypoint_scores1",
    ]

    def _init(self, conf):
        """Initialize the pipeline and download dataset if needed."""
        if not (DATA_PATH / "megadepth1500").exists():
            logger.info("Downloading the MegaDepth-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip"
            zip_path = DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(DATA_PATH)
            zip_path.unlink()
        
        # Ensure detector-free mode is enabled
        if conf.model.get("extractor", {}).get("name") is not None:
            logger.warning(
                "Detector-free mode: Overriding extractor.name to None. "
                "GGDM detector-free model does not use keypoint extractors."
            )
            if "extractor" not in conf.model:
                conf.model["extractor"] = {}
            conf.model["extractor"]["name"] = None

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
                print("checkpoint: ")
                print(self.conf.checkpoint)
            
            # Ensure model is in eval mode and detector-free
            model.eval()
            
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = defaultdict(list)
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        pose_results_dlt = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            
            # Evaluate matches (epipolar error)
            results_i = eval_matches_epipolar(data, pred)
            
            # Evaluate pose estimation with different RANSAC thresholds
            for th in test_thresholds:
                pose_results_i = eval_relative_pose_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

                pose_results_i_dlt = eval_relative_pose_dlt(
                    data,
                    pred,
                    {"estimator": conf.estimator, "options": {"confidence": 0.99999, "method": "dlt"}},
                )
                [pose_results_dlt[th][k].append(v) for k, v in pose_results_i_dlt.items()]

            # Store metadata
            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        # Summarize results as a dict[str, float]
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        # Find best threshold and compute pose metrics
        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        best_pose_results_dlt, best_th_dlt = eval_poses(
            pose_results_dlt, auc_ths=[5, 10, 20], key="rel_pose_error_dlt"
        )

        results = {**results, **pose_results[best_th]}
        results = {**results, **pose_results_dlt[best_th_dlt]}
        summaries = {
            **summaries,
            **best_pose_results,
            **best_pose_results_dlt,
        }

        # Generate visualization figures
        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            ),
            "pose_recall_dlt": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error_dlt"]},
                [0, 30],
                unit="°",
                title="Pose_dlt ",
            )
        }

        return summaries, figures, results


import random

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(MegaDepth1500GGDMPipeline.default_conf)

    # Setup output directory
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    # Force detector-free mode for GGDM
    if "model" not in conf:
        conf["model"] = {}
    if "extractor" not in conf["model"]:
        conf["model"]["extractor"] = {}
    conf["model"]["extractor"]["name"] = None
    print("GGDM Detector-free mode: Using raw images directly (no keypoint extractor)")

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    set_seed(0)

    pipeline = MegaDepth1500GGDMPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()


# Example usage:
# python -m scripts.eval.megadepth1500_ggdm --conf ggdm-official --checkpoint /path/to/checkpoint.tar --overwrite
# 
# Or with custom config:
# python -m scripts.eval.megadepth1500_ggdm \
#     --conf superpoint+diffglue_megadepth_ggdm \
#     --checkpoint /path/to/checkpoint.tar \
#     --overwrite \
#     model.refinement.num_refinement_iters=3 \
#     model.refinement.use_geometry_guidance=true
