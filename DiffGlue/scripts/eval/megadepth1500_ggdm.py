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
from ..visualization.viz2d import plot_cumulative, plot_images, plot_matches
from ..geometry.epipolar import relative_pose_error
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_matches_epipolar, eval_poses, eval_relative_pose_robust, eval_relative_pose_dlt, get_matches_scores

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
                "resize": 1024,  # Match training preprocessing for consistency
                "side": "long",
                "square_pad": True,  # Match training: pad to square (1024x1024) with black padding
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

    def run_eval(self, loader, pred_file, viz=False):
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
            pose_results_i_best = None
            pose_results_i_dlt_best = None
            est_best = None  # Store estimator output for visualization
            est_dlt_best = None
            
            for th in test_thresholds:
                pose_results_i = eval_relative_pose_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]
                # Store the first threshold results for visualization
                if pose_results_i_best is None:
                    pose_results_i_best = pose_results_i
                    # Also store the actual estimator output for R/t extraction
                    if viz:
                        from ..robust_estimators import load_estimator
                        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
                        m0, scores0 = pred["matches0"], pred["matching_scores0"]
                        pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
                        estimator = load_estimator("relative_pose", conf.estimator)({"estimator": conf.estimator, "ransac_th": th})
                        data_ = {
                            "m_kpts0": pts0,
                            "m_kpts1": pts1,
                            "camera0": data["view0"]["camera"][0],
                            "camera1": data["view1"]["camera"][0],
                        }
                        est_best = estimator(data_)

                pose_results_i_dlt = eval_relative_pose_dlt(
                    data,
                    pred,
                    {"estimator": conf.estimator, "options": {"confidence": 0.99999, "method": "dlt"}},
                )
                [pose_results_dlt[th][k].append(v) for k, v in pose_results_i_dlt.items()]
                # Store the first threshold results for visualization
                if pose_results_i_dlt_best is None:
                    pose_results_i_dlt_best = pose_results_i_dlt
                    # Also store the actual estimator output for R/t extraction
                    if viz:
                        from ..robust_estimators import load_estimator
                        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
                        m0, scores0 = pred["matches0"], pred["matching_scores0"]
                        pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
                        estimator = load_estimator("relative_pose", conf.estimator)({"estimator": conf.estimator, "options": {"confidence": 0.99999, "method": "dlt"}})
                        data_ = {
                            "m_kpts0": pts0,
                            "m_kpts1": pts1,
                            "camera0": data["view0"]["camera"][0],
                            "camera1": data["view1"]["camera"][0],
                        }
                        est_dlt_best = estimator(data_)

            # Visualize if requested
            if viz:
                self.visualize_matches(data, pred, pose_results_i_best, pose_results_i_dlt_best, est_best, est_dlt_best)
                plt.show()

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

    def run(self, experiment_dir, model=None, overwrite=False, overwrite_eval=False, viz=False):
        """Run export+eval loop with optional visualization"""
        # Import here to avoid circular imports
        from .eval_pipeline import exists_eval, save_eval, load_eval
        
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )
        pred_file = self.get_predictions(
            experiment_dir, model=model, overwrite=overwrite
        )

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            s, f, r = self.run_eval(self.get_dataloader(), pred_file, viz=viz)
            if not viz:  # Only save if not visualizing (viz might modify figures)
                save_eval(experiment_dir, s, f, r)
        else:
            s, r = load_eval(experiment_dir)
        return s, f, r

    def compute_RMD_error(self, R_gt, R_est):
        """Compute Rotation Matrix Distance error (same as two_view_pipeline.py).
        
        Args:
            R_gt: Ground truth rotation matrix [3, 3]
            R_est: Estimated rotation matrix [3, 3]
            
        Returns:
            Rotation error in degrees
        """
        R_rel = R_est @ R_gt.T
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_angle = (trace - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1, 1)
        rot_error_rad = torch.acos(cos_angle)
        rot_error_deg = rot_error_rad * 180 / 3.14159265
        return rot_error_deg

    def compute_ATE_error(self, t_gt, t_est):
        """Compute Absolute Translation Error (same as two_view_pipeline.py).
        
        Args:
            t_gt: Ground truth translation vector [3]
            t_est: Estimated translation vector [3]
            
        Returns:
            Translation error (Euclidean distance)
        """
        # Compute Euclidean distance: ||t_gt - t_est||_2
        ate = torch.norm(t_gt - t_est, dim=-1)
        return ate

    def visualize_matches(self, data, pred, pose_results_i=None, pose_results_i_dlt=None, est=None, est_dlt=None):
        """Visualize feature matches with colored lines based on scores and pose errors.
        
        Args:
            data: Dictionary containing view0, view1, and T_0to1 (ground truth)
            pred: Dictionary containing keypoints, matches, and matching scores
            pose_results_i: Optional pose estimation results from robust estimator
            pose_results_i_dlt: Optional pose estimation results from DLT estimator
            est: Optional estimator output containing M_0to1 (R, t) for robust estimator
            est_dlt: Optional estimator output containing M_0to1 (R, t) for DLT estimator
        """
        # Extract images
        img0 = data["view0"]["image"][0]  # [C, H, W]
        img1 = data["view1"]["image"][0]  # [C, H, W]
        
        # Convert to numpy and handle grayscale/RGB
        if img0.shape[0] == 1:
            img0_np = img0[0].cpu().numpy()
            img1_np = img1[0].cpu().numpy()
            cmap = "gray"
        else:
            img0_np = img0.permute(1, 2, 0).cpu().numpy()
            img1_np = img1.permute(1, 2, 0).cpu().numpy()
            cmap = None
            # Normalize if needed (assuming images are in [0, 1] range)
            if img0_np.max() <= 1.0:
                img0_np = (img0_np * 255).astype(np.uint8)
                img1_np = (img1_np * 255).astype(np.uint8)
        
        # Get matches and scores
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0, scores0 = pred["matches0"], pred["matching_scores0"]
        pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
        
        # Convert to numpy
        pts0_np = pts0.cpu().numpy()
        pts1_np = pts1.cpu().numpy()
        scores_np = scores.cpu().numpy() if scores is not None else None
        
        # Determine colors based on scores
        if scores_np is not None and len(scores_np) > 0:
            # Normalize scores to [0, 1] for color mapping
            scores_normalized = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min() + 1e-8)
            # Use green colormap: higher scores = greener
            colors = plt.cm.Greens(scores_normalized)[:, :3]  # RGB in [0, 1]
        else:
            # Yellow color when scores are not available
            colors = np.array([[1.0, 1.0, 0.0]] * len(pts0_np))  # Yellow RGB
        
        # Plot images side by side
        plot_images([img0_np, img1_np], cmaps=[cmap, cmap] if cmap else None)
        
        # Plot matches with colors
        plot_matches(pts0_np, pts1_np, color=colors.tolist(), lw=1.5, ps=3, a=0.7)
        
        # Compute and display pose errors
        T_gt = data["T_0to1"]
        
        # Compute RMD and ATE errors if estimated poses are available
        error_text = []
        
        # From robust estimator (if available)
        if pose_results_i is not None and "rel_pose_error" in pose_results_i:
            rel_pose_err = pose_results_i["rel_pose_error"]
            if rel_pose_err != float("inf"):
                error_text.append(f"Robust Pose Error: {rel_pose_err:.2f}°")
                
                # Get estimated R and t from estimator output or pred
                R_est = None
                t_est = None
                
                if est is not None and est.get("success", False) and "M_0to1" in est:
                    M = est["M_0to1"]
                    R_est = M.R
                    t_est = M.t
                elif "estimated_R" in pred and "estimated_t" in pred:
                    R_est = pred["estimated_R"]
                    t_est = pred["estimated_t"]
                
                if R_est is not None and t_est is not None:
                    # Convert to CPU tensors if needed
                    if isinstance(R_est, torch.Tensor):
                        R_est = R_est.cpu()
                    if isinstance(t_est, torch.Tensor):
                        t_est = t_est.cpu()
                    
                    # Get ground truth R and t
                    if hasattr(T_gt, 'R'):
                        R_gt = T_gt.R
                        t_gt = T_gt.t
                    elif hasattr(T_gt, '__getitem__'):
                        R_gt = T_gt[:3, :3]
                        t_gt = T_gt[:3, 3]
                    else:
                        R_gt = T_gt.R if hasattr(T_gt, 'R') else None
                        t_gt = T_gt.t if hasattr(T_gt, 't') else None
                    
                    if R_gt is not None and t_gt is not None:
                        # Convert to torch tensors if needed
                        if not isinstance(R_gt, torch.Tensor):
                            R_gt = torch.tensor(R_gt, dtype=torch.float32)
                        if not isinstance(t_gt, torch.Tensor):
                            t_gt = torch.tensor(t_gt, dtype=torch.float32)
                        if not isinstance(R_est, torch.Tensor):
                            R_est = torch.tensor(R_est, dtype=torch.float32)
                        if not isinstance(t_est, torch.Tensor):
                            t_est = torch.tensor(t_est, dtype=torch.float32)
                        
                        # Ensure they're on CPU
                        R_gt = R_gt.cpu()
                        t_gt = t_gt.cpu()
                        R_est = R_est.cpu()
                        t_est = t_est.cpu()
                        
                        # Ensure correct shapes: R should be [3, 3], t should be [3]
                        # Remove extra dimensions if any
                        while R_gt.dim() > 2:
                            R_gt = R_gt.squeeze(0)
                        while R_est.dim() > 2:
                            R_est = R_est.squeeze(0)
                        while t_gt.dim() > 1:
                            t_gt = t_gt.squeeze(0)
                        while t_est.dim() > 1:
                            t_est = t_est.squeeze(0)
                        
                        # Reshape if needed
                        if R_gt.shape != (3, 3):
                            R_gt = R_gt.view(3, 3)
                        if R_est.shape != (3, 3):
                            R_est = R_est.view(3, 3)
                        if t_gt.shape != (3,):
                            t_gt = t_gt.view(3)
                        if t_est.shape != (3,):
                            t_est = t_est.view(3)
                        
                        # Compute RMD error using method from two_view_pipeline.py
                        rmd_error = self.compute_RMD_error(R_gt, R_est)
                        rmd_error_deg = rmd_error.item() if isinstance(rmd_error, torch.Tensor) else rmd_error
                        
                        # Compute ATE error using method from two_view_pipeline.py
                        ate_error = self.compute_ATE_error(t_gt, t_est)
                        ate_error_val = ate_error.item() if isinstance(ate_error, torch.Tensor) else ate_error
                        
                        error_text.append(f"RMD Error: {rmd_error_deg:.3f}°")
                        error_text.append(f"ATE Error: {ate_error_val:.4f}")
        
        # From DLT estimator (if available)
        if pose_results_i_dlt is not None and "rel_pose_error_dlt" in pose_results_i_dlt:
            rel_pose_err_dlt = pose_results_i_dlt["rel_pose_error_dlt"]
            if rel_pose_err_dlt != float("inf"):
                error_text.append(f"DLT Pose Error: {rel_pose_err_dlt:.2f}°")
        
        # Add text to figure
        fig = plt.gcf()
        ax = fig.axes[0]
        text_str = "\n".join(error_text) if error_text else "No pose estimation available"
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        return fig


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
    parser.add_argument("--viz", action="store_true", 
                       help="Visualize matches and pose estimation errors for each pair")
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
        viz=args.viz,
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
#     --visualize \
#     model.refinement.num_refinement_iters=3 \
#     model.refinement.use_geometry_guidance=true
