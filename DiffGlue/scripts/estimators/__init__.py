#!/usr/bin/env python
import random
import pdb
import cv2
# import madpose  # install independently (follow models/external/madpose/README.md)
import numpy as np
import torch
# from madpose.utils import get_depths
from models.classic_matcher.bfmatching import BFMatching
# from models.detector_free_matcher.detectorfreematching import DetectorFreeMatching
import time

# LightGlue + SuperPoint
from schema import RelativePoseEstimatorOutput
from torchvision.transforms import ToTensor

random.seed(10)

reproj_pix_thres = 8.0
epipolar_pix_thres = 2.0

epipolar_weight = 1.0

# MADPOSE configurations
# madpose_options = madpose.HybridLORansacOptions()
# madpose_options.min_num_iterations = 100
# madpose_options.max_num_iterations = 1000
# madpose_options.success_probability = 0.9999
# madpose_options.random_seed = 10  # for reproducibility
# madpose_options.final_least_squares = True
# madpose_options.threshold_multiplier = 5.0
# madpose_options.num_lo_steps = 4
# # squared px thresholds for reprojection error and epipolar error
# madpose_options.squared_inlier_thresholds = [
#     reproj_pix_thres**2,
#     epipolar_pix_thres**2,
# ]
# # weight when scoring for the two types of errors
# madpose_options.data_type_weights = [1.0, epipolar_weight]

# madpose_est_config = madpose.EstimatorConfig()
# # if enabled, the input min_depth values are guaranteed to be positive with the estimated depth offsets (shifts), default: True
# madpose_est_config.min_depth_constraint = True
# # if disabled, will model the depth with only scale (only applicable to the calibrated camera case)
# madpose_est_config.use_shift = True
# # best set to the number of PHYSICAL CPU cores
# madpose_est_config.ceres_num_threads = 8


def frame2tensor(image, device):
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    # Normalize to [0,1] if it is in [0..255]
    if image.max() > 1.0:
        image /= 255.0
    return torch.unsqueeze(ToTensor()(image), 0).to(device)


def process_resize(w, h, resize):
    if len(resize) == 1 and resize[0] > 0:
        scale = resize[0] / float(max(h, w))
        w_new = int(round(w * scale))
        h_new = int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        # No resizing
        w_new, h_new = w, h
    elif len(resize) == 2:
        w_new, h_new = resize[0], resize[1]
    else:
        raise ValueError("Invalid resize specification.")
    return w_new, h_new


def backproject_pixel_to_3d(x, y, depth_map, K, scale):
    h, w = depth_map.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    z = depth_map[y, x] / scale  # Convert depth to meters
    # Skip invalid or missing depth
    if z <= 0.0 or z >= 10.0:
        return None
    # Homogeneous pixel coords * depth
    pt_h = np.array([x * z, y * z, z], dtype=np.float32)
    K_inv = np.linalg.inv(K)
    pt_cam = K_inv @ pt_h  # 3D point in the follower camera
    return pt_cam


def read_image(path, device, resize):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    image = cv2.resize(image.astype("float32"), (w_new, h_new))

    inp = frame2tensor(image, device)
    return image, inp, scales


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_matching_plot_fast(
    image0,
    image1,
    kpts0,
    kpts1,
    mkpts0,
    mkpts1,
    text,
    path=None,
    show_keypoints=False,
    margin=10,
    opencv_display=False,
    opencv_title="",
    small_text=[],
):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin :] = image1
    out = np.stack([out] * 3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(
                out, (x + margin + W0, y), 2, black, -1, lineType=cv2.LINE_AA
            )
            cv2.circle(
                out, (x + margin + W0, y), 1, white, -1, lineType=cv2.LINE_AA
            )

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    # color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        c = [1, 0, 0] if random.random() < 0.5 else [0, 1, 0]
        cv2.line(
            out,
            (x0, y0),
            (x1 + margin + W0, y1),
            color=c,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640.0, 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def compute_essential_matrix_based_pose(
    estimator_output: RelativePoseEstimatorOutput,
    K_follower: np.ndarray,
    matched_kpts_f,
    matched_kpts_l,
):
    E = None
    R_f2l_E = None
    t_f2l_E = None
    valid_E = 0

    # Estimate Essential Matrix
    if len(matched_kpts_f) >= 8:
        E, inlier_mask = cv2.findEssentialMat(
            matched_kpts_f,
            matched_kpts_l,
            cameraMatrix=K_follower,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if (
            E is not None
            and inlier_mask is not None
            and inlier_mask.sum() >= 6
        ):
            # Recover pose (follower → leader)
            _, R_f2l_E, t_f2l_E, _ = cv2.recoverPose(
                E,
                matched_kpts_f,
                matched_kpts_l,
                cameraMatrix=K_follower,
                mask=inlier_mask,
            )
            valid_E = 1

    estimator_output.E = E
    estimator_output.R_E = R_f2l_E
    estimator_output.t_E = t_f2l_E
    estimator_output.valid_E = valid_E


def compute_pnp_solver_based_pose(
    estimator_output: RelativePoseEstimatorOutput,
    matched_kpts_f,
    matched_kpts_l,
    matching_score,
    follower_depth: np.ndarray,
    K_leader: np.ndarray,
    K_follower: np.ndarray,
    scale_f_x: float,
    scale_f_y: float,
    scale_l_x: float,
    scale_l_y: float,
    depth_scale: float,
):
    pts_3d_follower = []
    pts_2d_leader = []
    valid_depth_pixel = []
    matched_valid_depth_kpts_l = []
    matched_valid_depth_kpts_f = []
    matched_ransac_kpts_l = []
    matched_ransac_kpts_f = []
    matching_score_valid_depth = []
    for i in range(len(matched_kpts_f)):
        # Follower pixel (resized -> original)
        x_f_res, y_f_res = matched_kpts_f[i]
        x_f = x_f_res * scale_f_x
        y_f = y_f_res * scale_f_y
        x_f_int = int(round(x_f))
        y_f_int = int(round(y_f))

        # Depth-based backprojection in the follower frame
        pt3d = backproject_pixel_to_3d(
            x_f_int, y_f_int, follower_depth, K_follower, scale=depth_scale
        )
        if pt3d is None:
            continue
        valid_depth_pixel.append(i)
        matched_valid_depth_kpts_l.append(matched_kpts_l[i])
        matched_valid_depth_kpts_f.append(matched_kpts_f[i])
        matching_score_valid_depth.append(matching_score[i])

        # Leader pixel (resized -> original)
        x_l_res, y_l_res = matched_kpts_l[i]
        x_l = x_l_res * scale_l_x
        y_l = y_l_res * scale_l_y

        pts_3d_follower.append(pt3d)
        pts_2d_leader.append([x_l, y_l])

    if len(pts_3d_follower) < 6:
        estimator_output.R = None
        estimator_output.t = None
        estimator_output.valid = 0
        estimator_output.matched_kpts_l = matched_kpts_l
        estimator_output.matched_kpts_f = matched_kpts_f
        estimator_output.score = matching_score
        estimator_output.matched_ransac_kpts_l = None
        estimator_output.matched_ransac_kpts_f = None
        return

    print(
        f"Number of feature matches after filtering: {len(pts_3d_follower)}."
    )

    pts_3d_follower = np.array(pts_3d_follower, dtype=np.float32)
    pts_2d_leader = np.array(pts_2d_leader, dtype=np.float32)

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_follower,
            pts_2d_leader,
            K_leader,
            distCoeffs=None,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount = 10, # TODO: default 100, test for lower number
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        inliers = inliers.flatten()
        matched_valid_depth_kpts_l = np.array(matched_valid_depth_kpts_l)
        matched_valid_depth_kpts_f = np.array(matched_valid_depth_kpts_f)
        matched_ransac_kpts_l = matched_valid_depth_kpts_l[inliers]
        matched_ransac_kpts_f = matched_valid_depth_kpts_f[inliers]

        if success and len(inliers) >= 15:
            # Convert rotation vector to a 3x3 rotation matrix
            R_f2l, _ = cv2.Rodrigues(rvec)  # follower->leader
            t_f2l = tvec.reshape(3)

            estimator_output.R = R_f2l
            estimator_output.t = t_f2l
            estimator_output.valid = 1
            estimator_output.matched_kpts_l = matched_valid_depth_kpts_l
            estimator_output.matched_kpts_f = matched_valid_depth_kpts_f
            estimator_output.score = matching_score_valid_depth
            estimator_output.matched_ransac_kpts_l = matched_ransac_kpts_l
            estimator_output.matched_ransac_kpts_f = matched_ransac_kpts_f
    except:
        return


# def compute_madpose_based_pose(
#     estimator_output: RelativePoseEstimatorOutput,
#     image0: np.ndarray,
#     image1: np.ndarray,
#     mkpts0: np.ndarray,
#     mkpts1: np.ndarray,
#     depth0: np.ndarray,
#     depth1: np.ndarray,
#     K0: np.ndarray,
#     K1: np.ndarray,
# ):
#     """
#     madpose argument shapes:
#     x0: (N, 2),
#     x1: (N, 2),
#     depth0: (N,),
#     depth1: (N,),
#     min_depth: [2,],
#     K0: (3, 3),
#     K1: (3, 3),
#     options: madpose.madpose.HybridLORansacOptions,
#     est_config: madpose.madpose.EstimatorConfig = <madpose.madpose.EstimatorConfig object at 0x7dc5c6fda930>)

#     madpose returns:
#     tuple[madpose.madpose.PoseScaleOffset, madpose.madpose.HybridRansacStatistics]
#     """
#     if depth1 is None:
#         return

#     pose, _ = madpose.HybridEstimatePoseScaleOffset(
#         mkpts0,
#         mkpts1,
#         get_depths(image0, depth0, mkpts0),
#         get_depths(image1, depth1, mkpts1),
#         [depth0.min(), depth1.min()],
#         K0,
#         K1,
#         madpose_options,
#         madpose_est_config,
#     )

#     # rotation and translation of the estimated pose
#     R_est, t_est = pose.R(), pose.t()

#     estimator_output.R_madpose = R_est
#     estimator_output.t_madpose = t_est
#     estimator_output.valid_madpose = 1


def estimate_relative_pose(
    matcher,
    follower_img: np.ndarray,  # Grayscale image from follower (H x W)
    leader_img: np.ndarray,  # Grayscale image from leader   (H x W)
    follower_depth: np.ndarray,  # Depth image from follower, same size as follower_img, in meters
    K_follower: np.ndarray,
    K_leader: np.ndarray,
    resize: list = [640, 480],
    max_keypoints: int = 1024,
    depth_scale: float = 1.0,
    leader_depth: np.ndarray = None,
) -> RelativePoseEstimatorOutput:
    """
    Estimate the relative pose (R, t) of follower camera with respect to the leader camera

    Args:
        follower_img (np.ndarray): Grayscale image from the follower camera, shape (H, W).
        leader_img   (np.ndarray): Grayscale image from the leader camera,   shape (H, W).
        follower_depth (np.ndarray): Depth image aligned with follower_img, same shape (H, W), in meters.
        K_follower (np.ndarray): 3x3 intrinsics for the follower camera.
                                 If None, uses default placeholders.
        K_leader   (np.ndarray): 3x3 intrinsics for the leader camera.
                                 If None, uses default placeholders.
        resize (list): Desired resize. E.g. [320, 240], or [600], or [-1] for no resize.
        max_keypoints (int): Maximum number of keypoints for SuperPoint.

    Returns:
        R (np.ndarray): 3x3 rotation matrix for the transform (follower -> leader).
        t (np.ndarray): 3x1 translation vector for the transform (follower -> leader).

    Raises:
        ValueError: If not enough 3D-2D correspondences are found or PnP fails.
    """

    h_f, w_f = follower_img.shape[:2]
    h_l, w_l = leader_img.shape[:2]
    w_new_f, h_new_f = process_resize(w_f, h_f, resize)
    w_new_l, h_new_l = process_resize(w_l, h_l, resize)

    # Resize the images
    follower_img_resized = cv2.resize(follower_img, (w_new_f, h_new_f)).astype(
        np.float32
    )
    leader_img_resized = cv2.resize(leader_img, (w_new_l, h_new_l)).astype(
        np.float32
    )

    # Keep track of scaling factors for backprojection or 2D correspondences
    scale_f_x = float(w_f) / float(w_new_f)
    scale_f_y = float(h_f) / float(h_new_f)
    scale_l_x = float(w_l) / float(w_new_l)
    scale_l_y = float(h_l) / float(h_new_l)

    # Convert to torch
    inp_f = frame2tensor(follower_img_resized, device)  # Follower
    inp_l = frame2tensor(leader_img_resized, device)  # Leader

    # ----------------------------------------------------------------
    # Extract and Match keypoints with Some-Glue
    # ----------------------------------------------------------------
    if isinstance(matcher, BFMatching):  # classic matcher (ORB + BFMatcher)
        start_time = time.time()
        pred = matcher({"image0": follower_img, "image1": leader_img})
        # end_time = time.time()
        kpts_f = pred["keypoints0"]  # (N,2)
        kpts_l = pred["keypoints1"]  # (N,2)
        matches = pred["matches0"]
        
        valid_mask = matches > -1
        matched_kpts_f = kpts_f[valid_mask]
        matched_kpts_l = kpts_l[valid_mask]
        matching_scores = pred['matching_scores0'][valid_mask]
        print(f"number of Valid matches: {len(matched_kpts_f)}")

    elif isinstance(matcher, DetectorFreeMatching):  # classic matcher (ORB + BFMatcher)
        start_time = time.time()
        pred = matcher({"image0": follower_img, "image1": leader_img})
        # end_time = time.time()
        kpts_f = pred["keypoints0"]  # (N,2)
        kpts_l = pred["keypoints1"]  # (N,2)
        matches = pred["matches0"]
        
        valid_mask = matches > -1
        matched_kpts_f = kpts_f[valid_mask]
        matched_kpts_l = kpts_l[valid_mask]
        matching_scores = pred['matching_scores0'][valid_mask]
        print(f"number of Valid matches: {len(matched_kpts_f)}")

    else:  # learning based matcher (glue)
        start_time = time.time()
        pred = matcher({"image0": inp_f, "image1": inp_l})
        # end_time = time.time()
        kpts_f = pred["keypoints0"][0].detach().cpu().numpy()  # (N,2)
        kpts_l = pred["keypoints1"][0].detach().cpu().numpy()  # (N,2)
        matches = pred["matches0"][0].detach().cpu().numpy()

        valid_mask = matches > -1
        matched_kpts_f = kpts_f[valid_mask]
        matched_kpts_l = kpts_l[matches[valid_mask]]
        matching_scores = pred['matching_scores0'][0].detach().cpu().numpy()[valid_mask]
        print(f"number of Valid matches: {len(matched_kpts_f)}")

    # fully invalid output initialization
    estimator_output = RelativePoseEstimatorOutput()
    if kpts_f.shape[0] == 0 or kpts_l.shape[0] == 0:
        return estimator_output
        

    if len(matched_kpts_f) < 6:
        return estimator_output

    

    # ----------------------------------------------------
    # PNP solver based pose estimation
    # ----------------------------------------------------
    compute_pnp_solver_based_pose(
        estimator_output,
        matched_kpts_f,
        matched_kpts_l,
        matching_scores,
        follower_depth,
        K_leader,
        K_follower,
        scale_f_x,
        scale_f_y,
        scale_l_x,
        scale_l_y,
        depth_scale,
    )
    end_time = time.time()
    pred_time = (end_time - start_time)*1000.0 # ms
    estimator_output.pred_time = pred_time
    # ----------------------------------------------------
    # Essential Matrix based pose estimation
    # ----------------------------------------------------
    compute_essential_matrix_based_pose(
        estimator_output, K_follower, matched_kpts_f, matched_kpts_l
    )
    # ----------------------------------------------------
    # MADPose based pose estimation
    # ----------------------------------------------------
    # compute_madpose_based_pose(
    #     estimator_output,
    #     follower_img_resized,
    #     leader_img_resized,
    #     matched_kpts_f,
    #     matched_kpts_l,
    #     follower_depth,
    #     leader_depth,
    #     K_follower,
    #     K_leader,
    # )

    return estimator_output


if __name__ == "__main__":
    # Example usage / quick test (NOT a full demo):
    # You would replace these with actual images loaded from disk
    # or from ROS messages as np.ndarray (grayscale & depth).
    import sys

    # Dummy synthetic data for demonstration:
    # (In a real scenario, load real images + depth + intrinsics.)
    follower_gray = np.zeros((240, 320), dtype=np.uint8)
    leader_gray = np.zeros((240, 320), dtype=np.uint8)
    follower_depth = (
        np.ones((240, 320), dtype=np.float32) * 2.0
    )  # 2 meters everywhere

    try:
        R, t = estimate_relative_pose(
            follower_gray, leader_gray, follower_depth
        )
        print("Estimated rotation:\n", R)
        print("Estimated translation:\n", t)
    except ValueError as e:
        print("Estimation failed:", e)