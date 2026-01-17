"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""
import torch
from omegaconf import OmegaConf

from . import get_model
from .base_model import BaseModel
from .utils.alternating_refinement import create_alternating_refinement
import pdb
to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class TwoViewPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "diffuser": {"name": None},
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
        # Alternating refinement options (Phase 4)
        "refinement": {
            "num_refinement_iters": 1,  # K=1 means no loop (original behavior)
            "use_geometry_guidance": False,  # Enable epipolar gradient in diffusion
            "geometry_guidance_weight": 0.1,  # λ for geometry gradient
            "feedback_to_loftr": False,  # Feed refined matches back to LoFTR
            "feedback_scale": 0.5,  # γ for attention bias
            # Unrolled training options (Section 3.7)
            "unrolled_training": False,  # If True, backprop through K iterations
            "unrolled_k": 2,  # Number of iterations for unrolled training
            "unrolled_t": 4,  # Number of diffusion steps for unrolled training
            "loss_match_weight": 1.0,  # λ_match for L_match
            "loss_pose_weight": 0.1,  # λ_pose for L_pose
            # Warmup: gradual transition from DSM to unrolled GGDM
            "warmup_epochs": 0,  # Epochs of standard DSM before unrolled (0 = no warmup)
            # Gradual unfreezing: unfreeze backbone after warmup for e2e fine-tuning
            "unfreeze_backbone_after_warmup": False,
            "backbone_lr_factor": 0.1,  # Backbone LR = main_lr * factor
        },
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "diffuser",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.diffuser.name:
            self.diffuser = get_model(conf.diffuser.name)(to_ctr(conf.diffuser))
            self.eval_diffuser = get_model(conf.diffuser.name)({**to_ctr(conf.diffuser), **{"timestep_respacing": str(conf.diffuser.ddim_steps)}})

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )
        
        # Initialize alternating refinement if configured
        refinement_conf = to_ctr(conf.get("refinement", {}))
        self.use_alternating_refinement = (
            refinement_conf.get("num_refinement_iters", 1) > 1 or
            refinement_conf.get("use_geometry_guidance", False)
        )
        self.use_unrolled_training = refinement_conf.get("unrolled_training", False)
        
        # Create alternating refinement module if needed for inference OR unrolled training
        if self.use_alternating_refinement or self.use_unrolled_training:
            self.alternating_refinement = create_alternating_refinement(refinement_conf)
    
    def set_epoch(self, epoch: int):
        """
        Set current epoch for warmup-based training mode switching.
        
        Call this at the start of each epoch to enable warmup behavior:
        - For epochs < warmup_epochs: uses standard DSM training
        - For epochs >= warmup_epochs: uses unrolled GGDM training
        
        Args:
            epoch: Current training epoch (0-indexed)
        """
        if hasattr(self, 'alternating_refinement'):
            self.alternating_refinement.set_epoch(epoch)
            # Log mode switch
            if self.use_unrolled_training:
                warmup = self.alternating_refinement._warmup_epochs
                if epoch == warmup and warmup > 0:
                    print(f"[TwoViewPipeline] Epoch {epoch}: Switching from DSM to unrolled GGDM training")

    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        if self.conf.extractor.name and not skip_extract:
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        return pred_i

    def _forward(self, data):
        # pdb.set_trace()
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }

        if self.conf.ground_truth.name and self.training:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        if self.conf.diffuser.name and self.conf.matcher.name:
            merged_data = {**data, **pred}
            
            if self.training:
                if self.use_unrolled_training:
                    # Unrolled GGDM training: backprop through K iterations
                    # Uses L_DSM + L_match + L_pose (Section 3.7)
                    pred = {**pred, **self.alternating_refinement(
                        self.diffuser,
                        self.matcher,
                        merged_data
                    )}
                else:
                    # Standard DSM training: single forward pass
                    pred = {**pred, **self.diffuser(self.matcher, merged_data)}
            elif self.use_alternating_refinement:
                # Inference with alternating refinement loop
                pred = {**pred, **self.alternating_refinement(
                    self.eval_diffuser, 
                    self.matcher, 
                    merged_data
                )}
            else:
                # Inference without alternating refinement (original behavior)
                pred = {**pred, **self.eval_diffuser(self.matcher, merged_data)}
                
        elif self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}

        # pdb.set_trace()
# pred.keys()
# dict_keys(['matches0', 'matches1', 'matching_scores0', 'matching_scores1', 
# 'ref_descriptors0', 'ref_descriptors1', 'log_assignment', 'keypoints0', 'keypoints1', 
# 'descriptors0', 'descriptors1', 'mean', 'variance', 'log_variance', 'pred_xstart', 'sample', 
# 'estimated_E', 'estimated_R', 'estimated_t', 'refined_matches'])
        return pred

    def loss(self, pred, data):
        """
        Compute losses and metrics for the two-view pipeline.
        
        Aggregates losses from all components (extractor, matcher, diffuser, etc.)
        and adds geometry metrics when alternating refinement is used.
        """
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name and not self.training:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    if k=="diffuser" and not self.training:
                        k = "eval_diffuser"
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                if "matcher_total" in losses_.keys():
                    losses_["total"] = losses_["matcher_total"]
                elif "diffuser_total" in losses_.keys():
                    losses_["total"] = losses_["diffuser_total"] * self.conf.diffuser.diffuser_loss_weight
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        
        # Add unrolled training losses (L_match and L_pose) if present
        if self.use_unrolled_training and self.training:
            if "loss_match" in pred:
                losses["loss_match"] = pred["loss_match"]
                total = total + pred["loss_match"]
                metrics["loss_match"] = pred["loss_match"].item()
            
            if "loss_pose" in pred:
                losses["loss_pose"] = pred["loss_pose"]
                total = total + pred["loss_pose"]
                metrics["loss_pose"] = pred["loss_pose"].item()
        
        # Add geometry metrics when alternating refinement provides estimated E
        if "estimated_E" in pred and not self.training:
            geometry_metrics = self._compute_geometry_metrics(pred, data)
            metrics = {**metrics, **geometry_metrics}
        
        return {**losses, "total": total}, metrics
    
    def _compute_geometry_metrics(self, pred, data):
        """
        Compute metrics comparing estimated essential matrix with ground truth.
        
        Returns metrics like rotation error and translation error when GT is available.
        """
        import torch
        
        metrics = {}
        
        estimated_E = pred.get("estimated_E")
        if estimated_E is None:
            return metrics
        
        # Check if ground truth pose is available
        if "T_0to1" not in data:
            return metrics
        
        try:
            from .utils.geometry_guidance import decompose_essential_matrix
            
            # Get ground truth transformation
            T_0to1 = data["T_0to1"]
            
            # Extract GT rotation and translation
            if hasattr(T_0to1, 'R') and hasattr(T_0to1, 't'):
                # Pose object with R and t attributes
                R_gt = T_0to1.R
                t_gt = T_0to1.t
            elif isinstance(T_0to1, torch.Tensor) and T_0to1.shape[-2:] == (4, 4):
                # 4x4 transformation matrix
                R_gt = T_0to1[..., :3, :3]
                t_gt = T_0to1[..., :3, 3]
            else:
                # Cannot extract GT pose
                return metrics
            
            # Decompose estimated E to get R, t
            R_est, t_est = decompose_essential_matrix(estimated_E)
            
            # Compute rotation error (in degrees)
            # Using trace formula: cos(θ) = (trace(R_rel) - 1) / 2
            metrics["rotation_error_deg"] = self.compute_RMD_error(R_gt, R_est)
            
            # Compute translation distance error 
            metrics["translation_error_m"] = self.compute_ATE_error(t_gt, t_est)
            
        except Exception as e:
            # Geometry metrics are optional, don't fail on errors
            pass
        
        return metrics


    def compute_RMD_error(self, R_gt, R_est):
        R_rel = R_est @ R_gt.T
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_angle = (trace - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1, 1)
        rot_error_rad = torch.acos(cos_angle)
        rot_error_deg = rot_error_rad * 180 / 3.14159265
        return rot_error_deg

    def compute_ATE_error(self, t_gt, t_est):
        # Compute Euclidean distance: ||t_gt - t_est||_2
        ate = torch.norm(t_gt - t_est, dim=-1)
        return ate
       