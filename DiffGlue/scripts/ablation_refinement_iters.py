#!/usr/bin/env python3
"""
Ablation study for optimal number of GGDM refinement iterations.

Usage:
    python scripts/ablation_refinement_iters.py --conf <config> --checkpoint <ckpt> --max_k 5

This evaluates different values of K (1, 2, 3, ..., max_k) and reports:
- Pose accuracy (rotation/translation error)
- Matching accuracy (precision, recall)  
- Sampson error convergence
- Inference time per K
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_ablation(conf_path: str, checkpoint: str = None, max_k: int = 5, num_samples: int = 100):
    """
    Run ablation study for different K values.
    
    Args:
        conf_path: Path to config file
        checkpoint: Path to model checkpoint
        max_k: Maximum K to test (tests 1, 2, ..., max_k)
        num_samples: Number of samples to evaluate
    """
    from scripts.models import get_model
    from scripts.datasets import get_dataset
    
    # Load base config
    conf = OmegaConf.load(conf_path)
    
    # Results storage
    results = {k: {
        'rot_errors': [],
        'trans_errors': [],
        'sampson_errors': [],
        'num_matches': [],
        'inference_times': [],
    } for k in range(1, max_k + 1)}
    
    print("=" * 60)
    print(f"GGDM Refinement Iterations Ablation Study")
    print(f"Testing K = 1 to {max_k}")
    print(f"Samples: {num_samples}")
    print("=" * 60)
    
    # Load dataset
    dataset_conf = OmegaConf.to_container(conf.data)
    dataset = get_dataset(dataset_conf['name'])(dataset_conf)
    val_loader = dataset.get_data_loader('val')
    
    # Test each K value
    for k in range(1, max_k + 1):
        print(f"\n[K={k}] Evaluating...")
        
        # Update config for this K
        test_conf = OmegaConf.merge(conf, {
            'model': {
                'refinement': {
                    'num_refinement_iters': k,
                    'use_geometry_guidance': k > 1,  # Enable for K > 1
                    'feedback_to_loftr': k > 1,
                }
            }
        })
        
        # Load model
        model = get_model(test_conf.model.name)(OmegaConf.to_container(test_conf.model))
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
        
        model = model.cuda().eval()
        
        # Evaluate
        sample_count = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader, desc=f"K={k}")):
                if sample_count >= num_samples:
                    break
                
                # Move to GPU
                data = {k: v.cuda() if torch.is_tensor(v) else v 
                        for k, v in data.items()}
                
                # Time inference
                torch.cuda.synchronize()
                start_time = time.time()
                
                pred = model(data)
                
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                results[k]['inference_times'].append(elapsed)
                
                # Compute metrics if GT available
                if 'estimated_R' in pred and 'T_0to1' in data:
                    # Get GT pose
                    T_0to1 = data['T_0to1']
                    R_gt = T_0to1[..., :3, :3] if T_0to1.dim() == 3 else T_0to1.R
                    t_gt = T_0to1[..., :3, 3] if T_0to1.dim() == 3 else T_0to1.t
                    
                    R_est = pred['estimated_R']
                    t_est = pred['estimated_t']
                    
                    # Rotation error
                    R_rel = torch.bmm(R_est, R_gt.transpose(-1, -2))
                    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
                    cos_angle = ((trace - 1) / 2).clamp(-1, 1)
                    rot_err = torch.acos(cos_angle) * 180 / np.pi
                    results[k]['rot_errors'].extend(rot_err.cpu().numpy().tolist())
                    
                    # Translation error
                    t_est_n = t_est / (t_est.norm(dim=-1, keepdim=True) + 1e-8)
                    t_gt_n = t_gt / (t_gt.norm(dim=-1, keepdim=True) + 1e-8)
                    cos_t = (t_est_n * t_gt_n).sum(-1).abs().clamp(0, 1)
                    trans_err = torch.acos(cos_t) * 180 / np.pi
                    results[k]['trans_errors'].extend(trans_err.cpu().numpy().tolist())
                
                # Count matches
                if 'matches0' in pred:
                    num_matches = (pred['matches0'] >= 0).sum().item()
                    results[k]['num_matches'].append(num_matches)
                
                sample_count += data['view0']['image'].shape[0]
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'K':<4} {'Rot Err°':<12} {'Trans Err°':<12} {'Time (s)':<12} {'#Matches':<10}")
    print("-" * 60)
    
    best_k = 1
    best_score = float('inf')
    
    for k in range(1, max_k + 1):
        rot_err = np.mean(results[k]['rot_errors']) if results[k]['rot_errors'] else float('nan')
        trans_err = np.mean(results[k]['trans_errors']) if results[k]['trans_errors'] else float('nan')
        time_avg = np.mean(results[k]['inference_times']) if results[k]['inference_times'] else float('nan')
        matches = np.mean(results[k]['num_matches']) if results[k]['num_matches'] else float('nan')
        
        print(f"{k:<4} {rot_err:<12.2f} {trans_err:<12.2f} {time_avg:<12.4f} {matches:<10.1f}")
        
        # Track best (lowest combined pose error)
        combined = rot_err + trans_err
        if not np.isnan(combined) and combined < best_score:
            best_score = combined
            best_k = k
    
    print("-" * 60)
    print(f"RECOMMENDED K = {best_k} (lowest combined pose error)")
    print("=" * 60)
    
    # Convergence analysis
    print("\nCONVERGENCE ANALYSIS:")
    print("(Check if error plateaus - indicates optimal K)")
    
    rot_errs = [np.mean(results[k]['rot_errors']) for k in range(1, max_k + 1)]
    for k in range(2, max_k + 1):
        delta = rot_errs[k-1] - rot_errs[k-2]
        if abs(delta) < 0.5:  # Less than 0.5° improvement
            print(f"  K={k}: Δ={delta:.2f}° (plateau reached)")
        else:
            print(f"  K={k}: Δ={delta:.2f}° (still improving)")
    
    return results, best_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for GGDM refinement iterations")
    parser.add_argument("--conf", type=str, required=True, help="Config file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--max_k", type=int, default=5, help="Maximum K to test")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    
    args = parser.parse_args()
    
    run_ablation(args.conf, args.checkpoint, args.max_k, args.num_samples)

