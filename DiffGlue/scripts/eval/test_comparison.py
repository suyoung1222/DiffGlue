#!/usr/bin/env python3
"""
Test script to compare DiffGlue, SuperGlue, and LoFTR
with real-time visualization and metrics (ATE, RMD, number of matches)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add paths for imports - scripts folder structure
script_dir = Path(__file__).parent.parent  # scripts folder  
repo_root = script_dir.parent.parent  # repository root
demo_models_path = repo_root / "demo" / "models"

# Import demo models by adding demo/models to path and importing as modules
import importlib.util
matching_spec = importlib.util.spec_from_file_location("demo_matching", demo_models_path / "matching.py")
demo_matching = importlib.util.module_from_spec(matching_spec)
matching_spec.loader.exec_module(demo_matching)

superpoint_spec = importlib.util.spec_from_file_location("demo_superpoint", demo_models_path / "superpoint.py")
demo_superpoint = importlib.util.module_from_spec(superpoint_spec)
superpoint_spec.loader.exec_module(demo_superpoint)

DiffGlueMatching = demo_matching.Matching
SuperPoint = demo_superpoint.SuperPoint

# Try to import LoFTR - use try/except with different import methods
LOFTR_AVAILABLE = False
try:
    # Try relative import first (when run as module)
    from ..models.matchers.LoFTR.src.loftr import LoFTR
    from ..models.matchers.LoFTR.src.config.default import get_cfg_defaults
    LOFTR_AVAILABLE = True
except (ImportError, ValueError):
    try:
        # Try absolute import
        script_dir = Path(__file__).parent.parent
        loftr_path = script_dir / "models" / "matchers" / "LoFTR" / "src" / "loftr.py"
        config_path = script_dir / "models" / "matchers" / "LoFTR" / "src" / "config" / "default.py"
        if loftr_path.exists() and config_path.exists():
            loftr_spec = importlib.util.spec_from_file_location("loftr", loftr_path)
            loftr_module = importlib.util.module_from_spec(loftr_spec)
            loftr_spec.loader.exec_module(loftr_module)
            LoFTR = loftr_module.LoFTR
            
            config_spec = importlib.util.spec_from_file_location("loftr_config", config_path)
            config_module = importlib.util.module_from_spec(config_spec)
            config_spec.loader.exec_module(config_module)
            get_cfg_defaults = config_module.get_cfg_defaults
            LOFTR_AVAILABLE = True
    except Exception:
        print("Warning: LoFTR not available")
        LOFTR_AVAILABLE = False

# Try to import SuperGlue (via kornia or provide alternative)
try:
    import kornia as K
    from kornia.feature import SuperGlue
    SUPERGLUE_AVAILABLE = True
except ImportError:
    print("Warning: SuperGlue (kornia) not available. Install with: pip install kornia")
    SUPERGLUE_AVAILABLE = False


# ==================== Metrics Calculation ====================

def calculate_ATE(T_rel, T_est):
    """
    Calculate Absolute Trajectory Error (ATE) in meters.
    ATE = ||translation(T_rel) - translation(T_est)||
    """
    if T_rel is None or T_est is None:
        return None
    
    t_rel = T_rel[:3, 3]
    t_est = T_est[:3, 3]
    
    # Normalize translations to have unit scale (since relative pose is up to scale)
    t_rel_norm = t_rel / (np.linalg.norm(t_rel) + 1e-8)
    t_est_norm = t_est / (np.linalg.norm(t_est) + 1e-8)
    
    # For ATE, we compute the difference in translation directions
    # In practice, ATE is often computed with aligned scales
    ate = np.linalg.norm(t_rel_norm - t_est_norm)
    return float(ate)


def calculate_RMD(T_rel, T_est):
    """
    Calculate Relative Motion Distance (RMD) in meters.
    RMD = ||translation(T_rel) - translation(T_est)||_2
    Alternative: distance between translation vectors
    """
    if T_rel is None or T_est is None:
        return None
    
    t_rel = T_rel[:3, 3]
    t_est = T_est[:3, 3]
    
    # For relative motion, compute distance between normalized translations
    t_rel_norm = t_rel / (np.linalg.norm(t_rel) + 1e-8)
    t_est_norm = t_est / (np.linalg.norm(t_est) + 1e-8)
    
    rmd = np.linalg.norm(t_rel_norm - t_est_norm)
    return float(rmd)


def compute_rotation_error(R_rel, R_est):
    """Compute rotation error in degrees"""
    R_err = R_est @ R_rel.T
    trace = np.trace(R_err)
    cos_angle = np.clip((trace - 1) / 2, -1, 1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.rad2deg(angle_rad)
    return float(angle_deg)


# ==================== Utility Functions ====================

def frame2tensor(frame, device):
    """Convert numpy image to tensor"""
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def process_resize(w, h, resize):
    """Resize image maintaining aspect ratio"""
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:
        w_new, h_new = resize[0], resize[1]
    return w_new, h_new


def read_image(path, device, resize=[1600]):
    """Read and preprocess image"""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    image = cv2.resize(image.astype('float32'), (w_new, h_new))
    inp = frame2tensor(image, device)
    return image, inp, scales


def error_colormap(x):
    """Error colormap: red (low) -> yellow (mid) -> green (high)"""
    x = np.asarray(x, dtype=np.float32)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)


def draw_matches_with_confidence(img1, img2, mkpts1, mkpts2, 
                                  ransac_kpts1=None, ransac_kpts2=None,
                                  scores=None, margin=20):
    """Draw matches with confidence coloring and RANSAC inliers"""
    # Ensure 3-channel BGR inputs
    if img1.ndim == 2: 
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2: 
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    H0, W0 = img1.shape[:2]
    H1, W1 = img2.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = img1
    out[:H1, W0 + margin:W0 + margin + W1] = img2

    if mkpts1 is None or mkpts2 is None or len(mkpts1) == 0:
        return out

    mkpts1 = np.round(mkpts1).astype(int)
    mkpts2 = np.round(mkpts2).astype(int)

    # Build per-match BGR colors
    if scores is None:
        color_bgr = np.full((len(mkpts1), 3), 255, dtype=np.uint8)
    else:
        s = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
        s = np.clip(s, 0.0, 1.0)
        rgba = error_colormap(s.squeeze())[:, :3]
        rgb_255 = (rgba * 255.0).astype(np.uint8)
        color_bgr = rgb_255[:, ::-1]  # RGB to BGR

        if len(color_bgr) != len(mkpts1):
            n = min(len(color_bgr), len(mkpts1))
            mkpts1, mkpts2, color_bgr = mkpts1[:n], mkpts2[:n], color_bgr[:n]

    # Draw all matches
    for (x0, y0), (x1, y1), c in zip(mkpts1, mkpts2, color_bgr):
        x1_shift = x1 + margin + W0
        cv2.line(out, (x0, y0), (x1_shift, y1), 
                color=tuple(int(v) for v in c), thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0), 2, tuple(int(v) for v in c), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1_shift, y1), 2, tuple(int(v) for v in c), -1, lineType=cv2.LINE_AA)

    # Draw RANSAC inliers in green
    if ransac_kpts1 is not None and ransac_kpts2 is not None:
        rk1 = np.round(ransac_kpts1).astype(int)
        rk2 = np.round(ransac_kpts2).astype(int)
        n = min(len(rk1), len(rk2))
        rk1, rk2 = rk1[:n], rk2[:n]

        inlier_color = (0, 255, 0)  # green
        for (x0, y0), (x1, y1) in zip(rk1, rk2):
            x1_shift = x1 + margin + W0
            cv2.line(out, (x0, y0), (x1_shift, y1), 
                    color=inlier_color, thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(out, (x0, y0), 3, inlier_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1_shift, y1), 3, inlier_color, -1, lineType=cv2.LINE_AA)

    return out


def add_text_to_image(image, text, position="top-left"):
    """Add text overlay to image"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # Fallback to OpenCV if PIL not available
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        lines = text.split('\n')
        y_offset = 20
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(image, (10, y_offset - text_height - 5), 
                         (10 + text_width + 5, y_offset + baseline), bg_color, -1)
            cv2.putText(image, line, (10, y_offset), font, font_scale, color, thickness)
            y_offset += text_height + 10
        return image
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    
    # Get text size (handle multiline)
    lines = text.split('\n')
    line_heights = [draw.textsize(line, font=font)[1] for line in lines]
    total_height = sum(line_heights)
    max_width = max(draw.textsize(line, font=font)[0] for line in lines)
    
    # Calculate position
    w, h = pil_image.size
    if position == "top-left":
        text_position = (10, 10)
    elif position == "top-right":
        text_position = (w - max_width - 10, 10)
    elif position == "bottom-left":
        text_position = (10, h - total_height - 10)
    elif position == "bottom-right":
        text_position = (w - max_width - 10, h - total_height - 10)
    else:
        text_position = (10, 10)
    
    # Draw text with background
    y_offset = text_position[1]
    for line in lines:
        bbox = draw.textbbox((text_position[0], y_offset), line, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 128))
        draw.text((text_position[0], y_offset), line, fill=(255, 255, 255), font=font)
        y_offset += draw.textsize(line, font=font)[1]
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ==================== Matching Methods ====================

class MatcherResult:
    """Container for matcher results"""
    def __init__(self):
        self.mkpts0 = None
        self.mkpts1 = None
        self.scores = None
        self.num_matches = 0
        self.ransac_mkpts0 = None
        self.ransac_mkpts1 = None
        self.num_ransac_matches = 0
        self.R = None
        self.t = None
        self.T = None
        self.pred_time = 0


def run_diffglue(image0, image1, inp0, inp1, device, config):
    """Run DiffGlue matching"""
    result = MatcherResult()
    
    try:
        matcher = DiffGlueMatching(config).eval().to(device)
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            pred = matcher({'image0': inp0, 'image1': inp1})
        
        torch.cuda.synchronize() if device == 'cuda' else None
        result.pred_time = time.time() - start_time
        
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        
        valid = matches > -1
        result.mkpts0 = kpts0[valid]
        result.mkpts1 = kpts1[matches[valid]]
        result.scores = confidence[valid]
        result.num_matches = len(result.mkpts0)
        
    except Exception as e:
        print(f"DiffGlue error: {e}")
        import traceback
        traceback.print_exc()
        result.num_matches = 0
    
    return result


def run_loftr(image0, image1, inp0, inp1, device):
    """Run LoFTR matching"""
    result = MatcherResult()
    
    if not LOFTR_AVAILABLE:
        return result
    
    try:
        # Initialize LoFTR
        _default_cfg = get_cfg_defaults()
        matcher = LoFTR(config=_default_cfg).eval().to(device)
        
        # Load pretrained weights (adjust path as needed)
        script_dir = Path(__file__).parent.parent  # scripts folder
        ckpt_path = script_dir / "models" / "matchers" / "LoFTR" / "weights" / "outdoor_ds.ckpt"
        if not ckpt_path.exists():
            # Try alternative path from repo root
            repo_root = script_dir.parent.parent
            ckpt_path = repo_root / "DiffGlue" / "scripts" / "models" / "matchers" / "LoFTR" / "weights" / "outdoor_ds.ckpt"
        
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=device)
            matcher.load_state_dict(ckpt['state_dict'])
        else:
            print(f"Warning: LoFTR weights not found at {ckpt_path}")
        
        # Prepare input
        data = {'image0': inp0, 'image1': inp1}
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            matcher(data)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        result.pred_time = time.time() - start_time
        
        # Extract matches
        if 'mkpts0_f' in data and 'mkpts1_f' in data:
            result.mkpts0 = data['mkpts0_f'].cpu().numpy()
            result.mkpts1 = data['mkpts1_f'].cpu().numpy()
            result.scores = data.get('mconf', torch.ones(len(result.mkpts0))).cpu().numpy()
            result.num_matches = len(result.mkpts0)
        
    except Exception as e:
        print(f"LoFTR error: {e}")
        import traceback
        traceback.print_exc()
        result.num_matches = 0
    
    return result


def run_superglue(image0, image1, inp0, inp1, device):
    """Run SuperGlue matching (via kornia)"""
    result = MatcherResult()
    
    if not SUPERGLUE_AVAILABLE:
        return result
    
    try:
        # Initialize SuperGlue
        matcher = SuperGlue('outdoor').eval().to(device)
        
        # Prepare input (kornia expects [B, C, H, W] format)
        # SuperGlue uses SuperPoint for detection, so we need to extract features first
        superpoint = SuperPoint({}).eval().to(device)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            # Extract SuperPoint features
            feat0 = superpoint({'image': inp0})
            feat1 = superpoint({'image': inp1})
            
            # Run SuperGlue
            input_dict = {
                'keypoints0': feat0['keypoints'][None],
                'keypoints1': feat1['keypoints'][None],
                'descriptors0': feat0['descriptors'][None],
                'descriptors1': feat1['descriptors'][None],
                'image0': inp0,
                'image1': inp1,
            }
            pred = matcher(input_dict)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        result.pred_time = time.time() - start_time
        
        # Extract matches
        matches = pred['matches0'][0].cpu().numpy()
        match_confidence = pred['matching_scores0'][0].cpu().numpy()
        kpts0 = feat0['keypoints'][0].cpu().numpy()
        kpts1 = feat1['keypoints'][0].cpu().numpy()
        
        valid = matches > -1
        result.mkpts0 = kpts0[valid]
        result.mkpts1 = kpts1[matches[valid]]
        result.scores = match_confidence[valid]
        result.num_matches = len(result.mkpts0)
        
    except Exception as e:
        print(f"SuperGlue error: {e}")
        import traceback
        traceback.print_exc()
        result.num_matches = 0
    
    return result


# ==================== Pose Estimation ====================

def estimate_pose_from_matches(mkpts0, mkpts1, K, ransac_threshold=1e-3):
    """Estimate relative pose from matches using RANSAC"""
    if len(mkpts0) < 8:
        return None, None, None, None, None
    
    # Normalize keypoints
    mkpts0_norm = cv2.undistortPoints(mkpts0.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    mkpts1_norm = cv2.undistortPoints(mkpts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    
    # Estimate Essential matrix
    E, mask = cv2.findEssentialMat(
        mkpts0_norm, mkpts1_norm, 
        focal=1.0, pp=(0., 0.),
        method=cv2.RANSAC, prob=0.999, threshold=ransac_threshold
    )
    
    if E is None:
        return None, None, None, None, None
    
    # Recover pose
    points, R, t, mask_pose = cv2.recoverPose(E, mkpts0_norm, mkpts1_norm)
    
    # Build transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    
    # Get inliers
    inliers = mask_pose.ravel().astype(bool)
    ransac_mkpts0 = mkpts0[inliers]
    ransac_mkpts1 = mkpts1[inliers]
    
    return R, t, T, ransac_mkpts0, ransac_mkpts1


# ==================== Dataset Loading Functions ====================

def parse_pairs_line(line):
    """Parse a line from pairs_calibrated file format
    Format: image0_path image1_path K0[9] K1[9] R[9] t[3]
    Returns: (img0_path, img1_path, K0, K1, R, t)
    """
    parts = line.strip().split()
    if len(parts) < 32:
        raise ValueError(f"Invalid pairs line format, expected 32 values, got {len(parts)}")
    
    img0_path = parts[0]
    img1_path = parts[1]
    
    # Parse K0 (9 values: fx, 0, cx, 0, fy, cy, 0, 0, 1)
    K0_flat = [float(x) for x in parts[2:11]]
    K0 = np.array(K0_flat).reshape(3, 3).astype(np.float32)
    
    # Parse K1 (9 values)
    K1_flat = [float(x) for x in parts[11:20]]
    K1 = np.array(K1_flat).reshape(3, 3).astype(np.float32)
    
    # Parse R (9 values for 3x3 rotation matrix)
    R_flat = [float(x) for x in parts[20:29]]
    R = np.array(R_flat).reshape(3, 3).astype(np.float32)
    
    # Parse t (3 values for translation vector)
    t_flat = [float(x) for x in parts[29:32]]
    t = np.array(t_flat).astype(np.float32)
    
    return img0_path, img1_path, K0, K1, R, t


def load_pair_from_dataset(pairs_file, pair_index, dataset_root):
    """Load a pair from the megadepth1500 dataset format"""
    pairs_file = Path(pairs_file)
    if not pairs_file.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    
    with open(pairs_file, 'r') as f:
        lines = f.readlines()
    
    if pair_index >= len(lines):
        raise IndexError(f"Pair index {pair_index} out of range (total pairs: {len(lines)})")
    
    line = lines[pair_index]
    img0_path, img1_path, K0, K1, R, t = parse_pairs_line(line)
    
    # Build full image paths
    dataset_root = Path(dataset_root)
    img0_full = dataset_root / img0_path
    img1_full = dataset_root / img1_path
    
    # Build ground truth transformation matrix
    T_gt = np.eye(4, dtype=np.float32)
    T_gt[:3, :3] = R
    T_gt[:3, 3] = t
    
    return {
        'img0_path': str(img0_full),
        'img1_path': str(img1_full),
        'K0': K0,
        'K1': K1,
        'R': R,
        't': t,
        'T_gt': T_gt
    }


# ==================== Main Evaluation Function ====================

def evaluate_method(img0, img1, inp0, inp1, method_name, device, config=None, K=None):
    """Evaluate a single matching method"""
    # Run matching
    if method_name == 'DiffGlue':
        result = run_diffglue(img0, img1, inp0, inp1, device, config or {})
    elif method_name == 'LoFTR':
        result = run_loftr(img0, img1, inp0, inp1, device)
    elif method_name == 'SuperGlue':
        result = run_superglue(img0, img1, inp0, inp1, device)
    else:
        return None
    
    # Estimate pose if matches available
    if result.num_matches > 0 and K is not None:
        R, t, T, ransac_mkpts0, ransac_mkpts1 = estimate_pose_from_matches(
            result.mkpts0, result.mkpts1, K
        )
        if R is not None:
            result.R = R
            result.t = t
            result.T = T
            result.ransac_mkpts0 = ransac_mkpts0
            result.ransac_mkpts1 = ransac_mkpts1
            result.num_ransac_matches = len(ransac_mkpts0)
    
    return result


def create_visualization(img0, img1, result, method_name, errors=None):
    """Create visualization for a single method"""
    if result.num_matches == 0:
        # Create empty visualization
        if img0.ndim == 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        if img1.ndim == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        vis = draw_matches_with_confidence(img0, img1, None, None)
    else:
        vis = draw_matches_with_confidence(
            img0, img1,
            result.mkpts0, result.mkpts1,
            result.ransac_mkpts0, result.ransac_mkpts1,
            scores=result.scores
        )
    
    # Add text overlay
    text_lines = [
        f"{method_name}",
        f"Matches: {result.num_matches}",
        f"Inliers: {result.num_ransac_matches}",
        f"Time: {result.pred_time*1000:.1f}ms"
    ]
    
    if errors:
        if errors.get('ATE') is not None:
            text_lines.append(f"ATE: {errors['ATE']:.4f}")
        if errors.get('RMD') is not None:
            text_lines.append(f"RMD: {errors['RMD']:.4f}")
        if errors.get('Rot_Err') is not None:
            text_lines.append(f"Rot Err: {errors['Rot_Err']:.2f}°")
    
    text = '\n'.join(text_lines)
    vis = add_text_to_image(vis, text, position="top-right")
    
    return vis


def main():
    parser = argparse.ArgumentParser(
        description='Compare DiffGlue, SuperGlue, and LoFTR with visualization'
    )
    
    # Image inputs (either use pairs_file OR image0/image1)
    parser.add_argument('--pairs_file', type=str, default=None,
                       help='Path to pairs_calibrated.txt file (uses megadepth1500 format by default)')
    parser.add_argument('--pair_index', type=int, default=0,
                       help='Index of pair to use from pairs_file (default: 0)')
    parser.add_argument('--dataset_root', type=str, default=None,
                       help='Root directory for dataset images (default: auto-detect from repo root)')
    parser.add_argument('--image0', type=str, default=None,
                       help='Path to first image (required if not using --pairs_file)')
    parser.add_argument('--image1', type=str, default=None,
                       help='Path to second image (required if not using --pairs_file)')
    
    parser.add_argument('--resize', type=int, nargs='+', default=[1600],
                       help='Resize images (max dimension)')
    parser.add_argument('--K', type=str, default=None,
                       help='Camera intrinsic matrix (fx,fy,cx,cy) or path to .npy file (not used if --pairs_file provided)')
    parser.add_argument('--gt_pose', type=str, default=None,
                       help='Ground truth relative pose (4x4 matrix .npy file) (not used if --pairs_file provided)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save visualization to file')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable real-time display')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['DiffGlue', 'SuperGlue', 'LoFTR'],
                       help='Methods to compare')
    
    # DiffGlue config
    parser.add_argument('--max_keypoints', type=int, default=2048)
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)
    parser.add_argument('--nms_radius', type=int, default=3)
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent.parent  # scripts folder
    repo_root = script_dir.parent.parent  # repository root
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load images and metadata
    T_gt = None
    K = None
    
    if args.pairs_file:
        # Use dataset format
        print(f"Loading pair {args.pair_index} from {args.pairs_file}")
        # Auto-detect dataset root if not provided
        if args.dataset_root is None:
            # Try to find from pairs_file location
            pairs_path = Path(args.pairs_file)
            if 'megadepth1500' in str(pairs_path):
                # Assume pairs file is in data_su/megadepth1500/
                dataset_root = pairs_path.parent / "images"
            else:
                # Default to repo structure
                dataset_root = repo_root / "DiffGlue" / "data_su" / "megadepth1500" / "images"
            args.dataset_root = str(dataset_root)
        pair_data = load_pair_from_dataset(args.pairs_file, args.pair_index, args.dataset_root)
        img0_path = pair_data['img0_path']
        img1_path = pair_data['img1_path']
        K0 = pair_data['K0']
        K1 = pair_data['K1']
        T_gt = pair_data['T_gt']
        
        print(f"Image0: {img0_path}")
        print(f"Image1: {img1_path}")
        print(f"K0:\n{K0}")
        print(f"K1:\n{K1}")
        print(f"Ground truth pose:\n{T_gt}")
        
        # Use K0 for pose estimation (or average of K0 and K1)
        K = K0  # Using first camera's intrinsics
        
        # Load images
        img0, inp0, scales0 = read_image(img0_path, device, args.resize)
        img1, inp1, scales1 = read_image(img1_path, device, args.resize)
        
    else:
        # Use individual image paths
        if args.image0 is None or args.image1 is None:
            parser.error("Either --pairs_file or both --image0 and --image1 must be provided")
        
        img0_path = args.image0
        img1_path = args.image1
        img0, inp0, scales0 = read_image(img0_path, device, args.resize)
        img1, inp1, scales1 = read_image(img1_path, device, args.resize)
    
    if img0 is None or img1 is None:
        print("Error: Failed to load images")
        return
    
    print(f"Image0 shape: {img0.shape}, Image1 shape: {img1.shape}")
    
    # Setup camera intrinsics (only if not using dataset format)
    if not args.pairs_file:
        if args.K:
            if os.path.exists(args.K):
                K = np.load(args.K)
            else:
                # Parse as fx,fy,cx,cy
                params = [float(x) for x in args.K.split(',')]
                if len(params) == 4:
                    K = np.array([[params[0], 0, params[2]],
                                 [0, params[1], params[3]],
                                 [0, 0, 1]])
            print(f"Using camera matrix K:\n{K}")
        else:
            # Default camera matrix
            h, w = img0.shape[:2]
            fx = fy = max(h, w)
            cx, cy = w / 2, h / 2
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            print(f"Using default camera matrix K:\n{K}")
        
        # Load ground truth pose if provided
        if args.gt_pose and os.path.exists(args.gt_pose):
            T_gt = np.load(args.gt_pose)
            print(f"Loaded ground truth pose:\n{T_gt}")
    
    # DiffGlue config
    config = {
        'superpoint': {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': args.max_keypoints
        },
    }
    
    # Run all methods
    results = {}
    methods_to_run = [m for m in args.methods if m in ['DiffGlue', 'SuperGlue', 'LoFTR']]
    
    print(f"\nRunning methods: {methods_to_run}")
    for method_name in methods_to_run:
        print(f"\n--- Running {method_name} ---")
        result = evaluate_method(img0, img1, inp0, inp1, method_name, device, config, K)
        results[method_name] = result
        
        if result:
            print(f"  Matches: {result.num_matches}")
            print(f"  RANSAC inliers: {result.num_ransac_matches}")
            print(f"  Time: {result.pred_time*1000:.1f}ms")
    
    # Calculate metrics if GT available
    all_errors = {}
    if T_gt is not None:
        print("\n--- Calculating Metrics ---")
        for method_name, result in results.items():
            if result and result.T is not None:
                errors = {}
                errors['ATE'] = calculate_ATE(T_gt, result.T)
                errors['RMD'] = calculate_RMD(T_gt, result.T)
                if result.R is not None:
                    R_gt = T_gt[:3, :3]
                    errors['Rot_Err'] = compute_rotation_error(R_gt, result.R)
                all_errors[method_name] = errors
                print(f"\n{method_name}:")
                for key, val in errors.items():
                    if val is not None:
                        print(f"  {key}: {val:.4f}")
    
    # Create visualizations
    print("\n--- Creating Visualizations ---")
    vis_frames = []
    for method_name in methods_to_run:
        if method_name in results and results[method_name]:
            errors = all_errors.get(method_name, {})
            vis = create_visualization(img0, img1, results[method_name], method_name, errors)
            vis_frames.append(vis)
    
    # Combine visualizations
    if vis_frames:
        combined_vis = cv2.hconcat(vis_frames) if len(vis_frames) > 1 else vis_frames[0]
        
        # Display
        if not args.no_display:
            cv2.namedWindow('Comparison', cv2.WINDOW_NORMAL)
            cv2.imshow('Comparison', combined_vis)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save
        if args.save:
            cv2.imwrite(args.save, combined_vis)
            print(f"\nSaved visualization to: {args.save}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for method_name in methods_to_run:
        if method_name in results and results[method_name]:
            result = results[method_name]
            print(f"\n{method_name}:")
            print(f"  Total matches: {result.num_matches}")
            print(f"  RANSAC inliers: {result.num_ransac_matches}")
            print(f"  Inference time: {result.pred_time*1000:.1f}ms")
            if method_name in all_errors:
                for key, val in all_errors[method_name].items():
                    if val is not None:
                        print(f"  {key}: {val:.4f}")


if __name__ == '__main__':
    main()
