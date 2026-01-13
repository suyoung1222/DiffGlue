import os
from typing import Literal
import pdb
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Optional ROS dependencies - only needed for ROS bag file reading
try:
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from sensor_msgs.msg import Image
    from tf_transformations import quaternion_from_matrix
    ROS_AVAILABLE = True
    bridge = CvBridge()
except ImportError:
    ROS_AVAILABLE = False
    bridge = None
    # Create dummy classes to avoid errors if ROS functions are called
    class CvBridge:
        def imgmsg_to_cv2(self, *args, **kwargs):
            raise ImportError("cv_bridge not available. Install ROS2 cv_bridge package for ROS bag support.")
    
    # Fallback for quaternion_from_matrix using scipy
    def quaternion_from_matrix(matrix):
        """Convert rotation matrix to quaternion using scipy (fallback when tf_transformations not available)"""
        from scipy.spatial.transform import Rotation as R
        rot = R.from_matrix(matrix[:3, :3])
        return rot.as_quat()  # Returns [x, y, z, w]


def add_text_to_image(
    image,
    text,
    position: Literal[
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
        "top-center",
        "bottom-center",
    ],
):
    # Add multiline text to image using PIL
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    # Convert OpenCV image to PIL image
    pil_image = PILImage.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    # Load a default font
    font = ImageFont.load_default()
    # Get text size
    text_size = draw.textsize(text, font=font)
    # Calculate position based on the specified position
    if position == "top-left":
        text_position = (10, 10)
    elif position == "top-right":
        text_position = (pil_image.width - text_size[0] - 10, 10)
    elif position == "bottom-left":
        text_position = (10, pil_image.height - text_size[1] - 10)
    elif position == "bottom-right":
        text_position = (
            pil_image.width - text_size[0] - 10,
            pil_image.height - text_size[1] - 10,
        )
    elif position == "top-center":
        text_position = ((pil_image.width - text_size[0]) // 2, 10)
    elif position == "bottom-center":
        text_position = (
            (pil_image.width - text_size[0]) // 2,
            pil_image.height - text_size[1] - 10,
        )
    else:
        raise ValueError(
            "Invalid position specified. Use one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'top-center', 'bottom-center'."
        )

    # Draw text on the image
    draw.text(text_position, text, fill=(255, 255, 255), font=font)
    # Convert PIL image back to OpenCV format
    return np.asarray(pil_image)


def get_rosbag_reader(bag_path):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader.open(storage_options, converter_options)
    return reader


def read_rosbag2_images(bag_path, topic_list):
    reader = get_rosbag_reader(bag_path)

    type_map = {}
    msgs = []
    # bridge = CvBridge()

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in topic_list:
            continue

        if topic not in type_map:
            type_map[topic] = Image

        msg = deserialize_message(data, type_map[topic])
        msgs.append((topic, msg, t))

    return msgs


def read_image(msg, encoding="mono8"):
    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
    except Exception as e:
        print(f"Image decode failed: {e}")
        return None


def read_depth(msg, encoding="32FC1", convert_to_meters=False):
    try:
        depth = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
        if convert_to_meters:
            depth = depth.astype(np.float32) / 1000.0
        return depth
    except Exception as e:
        print(f"Depth decode failed: {e}")
        return None


def calculate_pose(R, t):
    T_mat = np.eye(4)
    T_mat[:3, :3] = R
    T_mat[:3, 3] = t.ravel()

    T_conv = np.array(
        [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    )
    T_mat = T_conv.T @ T_mat @ T_conv

    pos = T_mat[:3, 3]
    quat = quaternion_from_matrix(T_mat)

    return pos, quat


def quat_to_rotmat(q):
    return R.from_quat(q).as_matrix()


def calculate_Rt_rel_est(R_est, t_est, q1, q2, t1, t2):
    T_est = np.eye(4)
    T_est[:3, :3] = R_est
    T_est[:3, 3] = t_est.ravel()

    # comment out T_conv when TUM
    # T_conv = np.array(
    #     [
    #         [0, -1, 0, 0],
    #         [0, 0, -1, 0],
    #         [1, 0, 0, 0],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # T_est = T_conv.T @ T_est @ T_conv

    R_est = T_est[:3, :3]

    # Ground-truth Essential Matrix
    R1, R2 = quat_to_rotmat(q1), quat_to_rotmat(q2)

    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[:3, 3] = t1

    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, 3] = t2

    T_rel = np.linalg.inv(T1) @ T2
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]

    return R_rel, t_rel, T_rel, R_est, T_est


def draw_matches_from_keypoints_all(
    img1, img2, ransac_kpts1=None, ransac_kpts2=None, kpts1=None, kpts2=None
):
    concat = cv2.hconcat([img1, img2])
    h1, w1 = img1.shape[:2]

    if len(concat.shape) == 2 or concat.shape[2] == 1:
        concat = cv2.cvtColor(concat, cv2.COLOR_GRAY2BGR)

    if kpts1 is not None and kpts2 is not None:
        for pt1, pt2 in zip(kpts1, kpts2):  # all matches
            pt1 = tuple(map(int, pt1))
            pt2 = (int(pt2[0] + w1), int(pt2[1]))
            # color = tuple([random.randint(100, 255) for _ in range(3)])
            color = [255, 255, 255]
            cv2.line(concat, pt1, pt2, color, 2)
            cv2.circle(concat, pt1, 4, color, -1)
            cv2.circle(concat, pt2, 4, color, -1)

    if ransac_kpts1 is not None and ransac_kpts2 is not None:
        for pt1, pt2 in zip(ransac_kpts1, ransac_kpts2):  # inliers
            pt1 = tuple(map(int, pt1))
            pt2 = (int(pt2[0] + w1), int(pt2[1]))
            # color = tuple([random.randint(100, 255) for _ in range(3)])
            color = [0, 0, 255]
            cv2.line(concat, pt1, pt2, color, 2)
            cv2.circle(concat, pt1, 4, color, -1)
            cv2.circle(concat, pt2, 4, color, -1)

    return concat

def draw_matches_from_keypoints_all_with_confidence(
    img1, img2, ransac_kpts1=None, ransac_kpts2=None, mkpts1=None, mkpts2=None, score=None, margin=20
):
    # ensure 3-channel BGR inputs
    if img1.ndim == 2: img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    H0, W0 = img1.shape[:2]
    H1, W1 = img2.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = img1
    out[:H1, W0 + margin : W0 + margin + W1] = img2

    # guard: no matches
    if mkpts1 is None or mkpts2 is None or len(mkpts1) == 0:
        return out

    mkpts1 = np.round(mkpts1).astype(int)
    mkpts2 = np.round(mkpts2).astype(int)
  
    # build per-match BGR colors in uint8
    if score is None:
        color_bgr = np.full((len(mkpts1), 3), 255, dtype=np.uint8)  # white
    else:
        # expect score in [0,1]; clip just in case
        s = np.asarray(score, dtype=np.float32).reshape(-1, 1)
        s = np.clip(s, 0.0, 1.0)
        # your error_colormap returns [R, G, B, A] in 0..1
        rgba = error_colormap(s.squeeze())[:, :3]  # RGB
        rgb_255 = (rgba * 255.0).astype(np.uint8)
        color_bgr = rgb_255[:, ::-1]  # to BGR

        # in case #scores != #matches, align lengths
        if len(color_bgr) != len(mkpts1):
            n = min(len(color_bgr), len(mkpts1))
            mkpts1, mkpts2, color_bgr = mkpts1[:n], mkpts2[:n], color_bgr[:n]

    for (x0, y0), (x1, y1), c in zip(mkpts1, mkpts2, color_bgr):
        x1_shift = x1 + margin + W0
        cv2.line(out, (x0, y0), (x1_shift, y1), color=tuple(int(v) for v in c), thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0), 2, tuple(int(v) for v in c), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1_shift, y1), 2, tuple(int(v) for v in c), -1, lineType=cv2.LINE_AA)

    # Inliers all green
    if ransac_kpts1 is not None and ransac_kpts2 is not None:
        rk1 = np.round(ransac_kpts1).astype(int)
        rk2 = np.round(ransac_kpts2).astype(int)
        n = min(len(rk1), len(rk2))
        rk1, rk2 = rk1[:n], rk2[:n]

        inlier_color = [0, 255, 0]  # green
        g = tuple(int(v) for v in inlier_color)
        for (x0, y0), (x1, y1) in zip(rk1, rk2):
            x1_shift = x1 + margin + W0
            cv2.line(out, (x0, y0), (x1_shift, y1), color=g, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x0, y0), 2, g, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1_shift, y1), 2, g, -1, lineType=cv2.LINE_AA)

    return out

def error_colormap(x):
    x = np.asarray(x, dtype=np.float32)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)


def read_txt_poses(txt_file):
    poses = []
    with open(txt_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            timestamp = parts[0]
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q = np.array(
                [
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                ]
            )
            poses.append(
                (timestamp, t, q)
            )  # No dictionary, just append to list
    return poses


def align_timestamps(gt1, gt2, max_diff_ns=5e3):  # 5 milliseconds tolerance
    matched_ts = []
    gt2_keys = sorted(gt2.keys())
    for ts1 in sorted(gt1.keys()):
        nearest_ts2 = min(gt2_keys, key=lambda ts2: abs(ts2 - ts1))
        if abs(nearest_ts2 - ts1) <= max_diff_ns:
            matched_ts.append((ts1, nearest_ts2))
    return matched_ts


def read_data(ts1, ts2, image1_dir, image2_dir, depth1_dir, depth2_dir):
    img1_path = os.path.join(image1_dir, f"{ts1}.png")
    img2_path = os.path.join(image2_dir, f"{ts2}.png")
    depth1_path = os.path.join(depth1_dir, f"{ts1}.npy")
    depth2_path = os.path.join(depth2_dir, f"{ts2}.npy")

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    depth1 = np.load(depth1_path) if os.path.exists(depth1_path) else None
    depth2 = np.load(depth2_path) if os.path.exists(depth2_path) else None

    return img1, img2, depth1, depth2


def import_func_from_module(module_name, func_name):
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, func_name)


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

