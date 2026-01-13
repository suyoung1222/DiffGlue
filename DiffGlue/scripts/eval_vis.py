#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add scripts directory to path to handle imports when run directly
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import pdb
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils.metrics import (
    calculate_ATE,
    calculate_RMD,
    compute_essential_matrix,
    compute_rotation_error,
    sampson_error,
)
from omegaconf import OmegaConf
from schema import (
    FollowerRobotArgs,
    LeaderRobotArgs,
    RelativePoseEstimatorOutput,
)

from tabulate import tabulate
from utils.eval_utils import (
    add_text_to_image,
    calculate_Rt_rel_est,
    draw_matches_from_keypoints_all,
    import_func_from_module,
    read_data,
)
from scipy.spatial.transform import Rotation as R
from utils.eval_utils import quat_to_rotmat

def evaluate(t1, t2, q1, q2, estimator_output):
    rot_err_deg = None
    ate = None
    rmd = None
    T_rel = None
    T_est = None
    ate_madpose = None
    rmd_madpose = None
    T_rel_madpose = None
    T_est_madpose = None
    pred_time = None

    if estimator_output.valid_E:
        R_rel, _, _, R_est, _ = calculate_Rt_rel_est(
            estimator_output.R_E,
            estimator_output.t_E,
            q1,
            q2,
            t1,
            t2,
        )
        # E_gt = compute_essential_matrix(
        #     R_rel, t_rel / np.linalg.norm(t_rel)
        # )

        # from essential matrix
        rot_err_deg = compute_rotation_error(R_rel, R_est)

        # sampson_err = sampson_error(E_gt, matched_kpts1, matched_kpts2)
        # Sampson_err.append(sampson_err)

    if estimator_output.valid:
        _, _, T_rel, _, T_est = calculate_Rt_rel_est(
            estimator_output.R,
            estimator_output.t,
            q1,
            q2,
            t1,
            t2,
        )
        ate = calculate_ATE(T_rel, T_est)
        rmd = calculate_RMD(T_rel, T_est)

        # if ate is not None and ate > 50:
        #     ate = None
        # if rmd is not None and rmd > 100:
        #     rmd = None

    if estimator_output.valid_madpose:
        _, _, T_rel_madpose, _, T_est_madpose = calculate_Rt_rel_est(
            estimator_output.R_madpose,
            estimator_output.t_madpose,
            q1,
            q2,
            t1,
            t2,
        )
        ate_madpose = calculate_ATE(T_rel_madpose, T_est_madpose)
        rmd_madpose = calculate_RMD(T_rel_madpose, T_est_madpose)

    # TODO add estimated pose, and gt rel pose to return
    pred_time = estimator_output.pred_time
    return (
        rot_err_deg,
        ate,
        rmd,
        T_rel,
        T_est,
        ate_madpose,
        rmd_madpose,
        T_rel_madpose,
        T_est_madpose,
        pred_time
    )


def perform_evaluation(
    leader_img,
    leader_depth,
    follower_img,
    follower_depth,
    t1,
    t2,
    q1,
    q2,
    estimator,
    leader_img_config,
    follower_img_config,
    follower_depth_config,
):
    if leader_img is None or follower_img is None or follower_depth is None:
        return (None, None, False), None

    estimator_output: RelativePoseEstimatorOutput = estimator(
        follower_img=follower_img,
        leader_img=leader_img,
        follower_depth=follower_depth,
        leader_depth=leader_depth,
        K_follower=follower_img_config.K,
        K_leader=leader_img_config.K,
        resize=leader_img_config.resize,
        depth_scale=follower_depth_config.scale,
    )

    (
        E_rot_err,
        ate,
        rmd,
        T_rel,
        T_est,
        ate_madpose,
        rmd_madpose,
        T_rel_madpose,
        T_est_madpose,
        pred_time,
    ) = evaluate(t1, t2, q1, q2, estimator_output)

    error = {}
    est_pose = {}
    if E_rot_err is not None:
        error["ERE"] = E_rot_err
    if ate is not None:
        error["ATE"] = ate
    if rmd is not None:
        error["RMD"] = rmd
    if T_est is not None and T_rel is not None:
        est_pose["T_est"] = T_est  # TODO: add image name as well
        est_pose["T_gt"] = T_rel
        R1 = quat_to_rotmat(q1)
        T1 = np.eye(4)
        T1[:3, :3] = R1
        T1[:3, 3] = t1
        est_pose["T_gt_l"] = T1
    if ate_madpose is not None:
        error["ATE_madpose"] = ate_madpose
    if rmd_madpose is not None:
        error["RMD_madpose"] = rmd_madpose
    if pred_time is not None:
        error["pred_time"] = pred_time
    # pdb.set_trace()
    return error, estimator_output, est_pose


def create_leader_follower_viz(
    leader_img,
    follower_img,
    estimated_rel_pose,
    errors,
    estimator_name,
    leader_name,
    follower_name,
):
    text = (
        f"Matches: {estimated_rel_pose.num_matches}\n"
        + f"Inliers: {estimated_rel_pose.num_ransac_matches}\n"
        + f"E-Rot Err: {errors.get('ERE', 'N/A')} deg\n"
        + f"ATE: {errors.get('ATE', 'N/A')} m\n"
        + f"RMD: {errors.get('RMD', 'N/A')} m\n"
        + f"ATE_madpose: {errors.get('ATE_madpose', 'N/A')} m\n"
        + f"RMD_madpose: {errors.get('RMD_madpose', 'N/A')} m\n"
    )

    vis = draw_matches_from_keypoints_all(
        leader_img,
        follower_img,
        estimated_rel_pose.matched_ransac_kpts_l,
        estimated_rel_pose.matched_ransac_kpts_f,
        estimated_rel_pose.matched_kpts_l,
        estimated_rel_pose.matched_kpts_f,
    )

    vis = add_text_to_image(vis, text, position="top-right")
    vis = add_text_to_image(vis, f"{estimator_name=}", "bottom-center")
    vis = add_text_to_image(vis, f"{leader_name=}", "bottom-left")
    vis = add_text_to_image(vis, f"{follower_name=}", "bottom-right")

    return vis


def combine_frames(viz_frames):
    return cv2.vconcat(list(viz_frames.values()))


def visualize(args, viz_frames, count):
    if not args.viz:
        return

    combined_frame = combine_frames(viz_frames)
    cv2.imshow(
        "Relative Pose Estimation Visualization",
        combined_frame,
    )

    # log the frame
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        cv2.imwrite(
            os.path.join(args.save_frames, f"frame_{count}.png"),
            combined_frame,
        )

    cv2.waitKey(200)


def calculate_error_stats(error_bucket):
    error_bucket_stats = defaultdict(lambda: defaultdict(dict))
    # Collect and print errors
    for (l, f), est_errors in error_bucket.items():
        # print(f"Errors for Leader: {l}, Follower: {f}")
        total_frames = est_errors.pop("frames")
        for est_name, errors in est_errors.items():
            # print(f"  Estimator: {est_name}")
            for err_name, err_values in errors.items():
                err_values = np.array(err_values)
                mean_error = np.mean(err_values)
                std_error = np.std(err_values)
                # print(
                #     f"    {err_name}: {mean_error:.2f} ± {std_error:.3f} "
                #     + f"with success rate: {len(err_values) / total_frames:.2%}"
                # )
                error_bucket_stats[(l, f)][est_name][err_name] = {
                    "mean": mean_error,
                    "rmse": np.sqrt(np.mean(err_values**2)),
                    "std": std_error,
                    "success_rate": len(err_values) / total_frames,
                    "min": np.min(err_values),
                    "max": np.max(err_values),
                }
    return error_bucket_stats


def calculate_synced_error_stats(error_bucket):
    error_bucket_stats = defaultdict(lambda: defaultdict(dict))
    # Collect and print errors
    for (l, f), est_errors in error_bucket.items():
        # print(f"Errors for Leader: {l}, Follower: {f}")
        if "frames" in est_errors:
            est_errors.pop("frames")
        for est_name, errors in est_errors.items():
            # print(f"  Estimator: {est_name}")
            for err_name, err_values in errors.items():
                err_values = np.array(err_values)
                mean_error = np.mean(err_values)
                std_error = np.std(err_values)

                error_bucket_stats[(l, f)][est_name][err_name] = {
                    "mean": mean_error,
                    "rmse": np.sqrt(np.mean(err_values**2)),
                    "std": std_error,
                    "min": np.min(err_values),
                    "max": np.max(err_values),
                }
    return error_bucket_stats


def calculate_matches_stats(matches_bucket):
    matches_bucket_stats = defaultdict(lambda: defaultdict(dict))
    # Collect and print matches
    for (l, f), est_matches in matches_bucket.items():
        # print(f"Matches for Leader: {l}, Follower: {f}")
        for est_name, matches in est_matches.items():
            # print(f"  Estimator: {est_name}")
            for match_type, match_values in matches.items():
                match_values = np.array(match_values)
                mean_matches = np.mean(match_values)
                std_matches = np.std(match_values)
                # print(
                #     f"    {match_type}: {mean_matches:.2f} ± {std_matches:.3f}"
                # )
                matches_bucket_stats[(l, f)][est_name][match_type] = {
                    "mean": mean_matches,
                    "std": std_matches,
                    "min": np.min(match_values),
                    "max": np.max(match_values),
                }
    return matches_bucket_stats


def calculate_est_pose_stats(est_pose_bucket):
    est_pose_bucket_stats = defaultdict(lambda: defaultdict(dict))
    # Collect and print errors
    # pdb.set_trace()
    for (l, f), est_poses in est_pose_bucket.items():
        for est_name, poses in est_poses.items():
            # print(f"  Estimator: {est_name}")
            for pose_type, pose_values in poses.items():
                pose_values = np.array(
                    [p for p in pose_values if p is not None]
                )
                est_pose_bucket_stats[(l, f)][est_name][
                    pose_type
                ] = pose_values
    return est_pose_bucket_stats


def run_core(args):
    count = 0

    effective_numels = {
        follower.name: min(follower.n, args.leader.n)
        for follower in args.followers
    }
    if args.start_frame_id >= min(effective_numels.values()):
        raise ValueError(
            f"Start frame ID {args.start_frame_id} is out of bounds."
            + f" MAX VALID frame ID for this dataset is {min(effective_numels.values()) - 1}."
        )

    curr_frame_id = {
        follower.name: args.start_frame_id for follower in args.followers
    }
    viz_frames = {
        (args.leader.name, follower.name): None for follower in args.followers
    }

    matches_bucket = {}
    error_bucket = {}
    synced_error_bucket = {}
    est_pose_bucket = {}
    for follower in args.followers:
        key = (args.leader.name, follower.name)

        matches_bucket[key] = {
            est: defaultdict(list) for est in args.estimators
        }

        est_pose_bucket[key] = {
            est: defaultdict(list) for est in args.estimators
        }

        error_bucket[key] = {est: defaultdict(list) for est in args.estimators}
        error_bucket[key]["frames"] = 0

        synced_error_bucket[key] = {
            est: defaultdict(list) for est in args.estimators
        }

    cv2.namedWindow(
        "Relative Pose Estimation Visualization", cv2.WINDOW_NORMAL
    )
    not_updated = False

    while not not_updated:
        for follower in args.followers:
            if curr_frame_id[follower.name] >= effective_numels[follower.name]:
                not_updated = True
                continue

            not_updated = False
            count += 1

            i = curr_frame_id[follower.name]
            error_bucket[(args.leader.name, follower.name)]["frames"] += 1

            gt1 = args.leader.poses
            gt2 = follower.poses

            ts1, t1, q1 = gt1[i]
            ts2, t2, q2 = gt2[i]

            img1, img2, depth1, depth2 = read_data(
                ts1,
                ts2,
                args.leader.images_dir,
                follower.images_dir,
                args.leader.depths_dir,
                follower.depths_dir,
            )

            viz_est_frames = []
            tmp_synced_items = defaultdict(dict)
            for est_name, estimator in args.estimators.items():
                errors, estimated_rel_pose, est_poses = perform_evaluation(
                    leader_img=img1,
                    leader_depth=depth1,
                    follower_img=img2,
                    follower_depth=depth2,
                    t1=t1,
                    t2=t2,
                    q1=q1,
                    q2=q2,
                    estimator=estimator,
                    leader_img_config=args.leader.image_config,
                    follower_img_config=follower.image_config,
                    follower_depth_config=follower.depth_config,
                )
                matches_bucket[(args.leader.name, follower.name)][est_name][
                    "total matches"
                ].append(estimated_rel_pose.num_matches)
                matches_bucket[(args.leader.name, follower.name)][est_name][
                    "inlier matches"
                ].append(estimated_rel_pose.num_ransac_matches)

                T_est_val = est_poses.get("T_est")
                T_gt_val = est_poses.get("T_gt")
                T_gt_l_val = est_poses.get("T_gt_l")
                est_pose_bucket[(args.leader.name, follower.name)][est_name][
                    "T_est"
                ].append(T_est_val)
                est_pose_bucket[(args.leader.name, follower.name)][est_name][
                    "T_gt"
                ].append(T_gt_val)
                est_pose_bucket[(args.leader.name, follower.name)][est_name][
                    "T_gt_l"
                ].append(T_gt_l_val)
                

                for err_name, err_value in errors.items():
                    error_bucket[(args.leader.name, follower.name)][est_name][
                        err_name
                    ].append(err_value)
                    tmp_synced_items[err_name][est_name] = err_value

                if args.viz:
                    viz_est_frames.append(
                        create_leader_follower_viz(
                            leader_img=img1,
                            follower_img=img2,
                            estimated_rel_pose=estimated_rel_pose,
                            errors=errors,
                            estimator_name=est_name,
                            leader_name=args.leader.name,
                            follower_name=follower.name,
                        )
                    )

            viz_frames[(args.leader.name, follower.name)] = (
                cv2.hconcat(viz_est_frames) if viz_est_frames else None
            )

            for err_name, est_errors in tmp_synced_items.items():
                if len(est_errors) == len(args.estimators):
                    for est_name, err_value in est_errors.items():
                        synced_error_bucket[(args.leader.name, follower.name)][
                            est_name
                        ][f"synced_{err_name}"].append(err_value)

            curr_frame_id[follower.name] += 1

        visualize(args, viz_frames, count)

    cv2.destroyAllWindows()

    error_bucket_stats = calculate_error_stats(error_bucket)
    matches_bucket_stats = calculate_matches_stats(matches_bucket)
    est_pose_bucket_stats = calculate_est_pose_stats(est_pose_bucket)
    synced_error_bucket_stats = calculate_synced_error_stats(
        synced_error_bucket
    )

    return (
        error_bucket_stats,
        matches_bucket_stats,
        est_pose_bucket_stats,
        synced_error_bucket_stats,
    )


run_core.stat_keys = {
    "error": ["mean", "rmse", "std", "success_rate", "min", "max"],
    "matches": ["mean", "std", "min", "max"],
    "synced_error": ["mean", "rmse", "std", "min", "max"],
}


def run_experiment(args):
    # progress tracking with tqdm
    outputs = Parallel(n_jobs=args.n_jobs)(
        delayed(run_core)(args) for _ in range(args.n_runs)
    )
    (
        error_stats_list,
        matches_stats_list,
        est_pose_stats_list,
        synced_error_stats_list,
    ) = zip(*outputs)

    """
    error_stats item schema:
    {
        (leader_name, follower_name): {
            estimator_name: {
                "err_name": {
                    "stat_key": value,
                    ...
                }
    """

    summary_df = pd.DataFrame()

    print("\n ========= Final Error Statistics ========= ")
    for ((l, f), est_errors), (_, match_errors) in zip(
        error_stats_list[0].items(), matches_stats_list[0].items()
    ):
        rows = []
        print(f"Results for Leader: {l}, Follower: {f}")
        for err_name in est_errors[next(iter(est_errors))].keys():
            sub_rows = []
            for stat_key in run_core.stat_keys["error"]:
                curr_row = [err_name] if not sub_rows else [""]
                curr_row.append(stat_key.capitalize())
                for est_name in args.estimators:
                    values = [
                        error_stats[(l, f)][est_name][err_name][stat_key]
                        for error_stats in error_stats_list
                        if err_name in error_stats[(l, f)][est_name]
                    ]
                    if not values:
                        curr_row.append("N/A")
                    else:
                        mean_value = np.mean(values)
                        std_value = np.std(values)
                        curr_row.append(f"{mean_value:.2f} ± {std_value:.3f}")
                sub_rows.append(curr_row)
            rows.extend(sub_rows)
            rows.append(["----"] * (2 + len(est_errors)))
        for match_type in match_errors[next(iter(match_errors))].keys():
            sub_rows = []
            for stat_key in run_core.stat_keys["matches"]:
                curr_row = [match_type.upper()] if not sub_rows else [""]
                curr_row.append(stat_key.capitalize())
                for est_name in args.estimators:
                    values = [
                        matches_stats[(l, f)][est_name][match_type][stat_key]
                        for matches_stats in matches_stats_list
                        if match_type in matches_stats[(l, f)][est_name]
                    ]
                    if not values:
                        curr_row.append("N/A")
                    else:
                        mean_value = np.mean(values)
                        std_value = np.std(values)
                        curr_row.append(f"{mean_value:.2f} ± {std_value:.3f}")
                sub_rows.append(curr_row)
            rows.extend(sub_rows)
            rows.append(["----"] * (2 + len(est_errors)))
        for synced_err_name in synced_error_stats_list[0][(l, f)][
            next(iter(synced_error_stats_list[0][(l, f)]))
        ].keys():
            sub_rows = []
            for stat_key in run_core.stat_keys["synced_error"]:
                curr_row = [synced_err_name] if not sub_rows else [""]
                curr_row.append(stat_key.capitalize())
                for est_name in args.estimators:
                    values = [
                        synced_error_stats[(l, f)][est_name][synced_err_name][
                            stat_key
                        ]
                        for synced_error_stats in synced_error_stats_list
                        if synced_err_name
                        in synced_error_stats[(l, f)][est_name]
                    ]
                    if not values:
                        curr_row.append("N/A")
                    else:
                        mean_value = np.mean(values)
                        std_value = np.std(values)
                        curr_row.append(f"{mean_value:.2f} ± {std_value:.3f}")
                sub_rows.append(curr_row)
            rows.extend(sub_rows)
            rows.append(["----"] * (2 + len(est_errors)))

        # Print the table using tabulate
        print(
            tabulate(
                rows,
                headers=["Metric", "Dimension"] + list(est_errors.keys()),
                tablefmt="orgtbl",
            )
        )
        print()

        # add rows to a dataframe for creating a csv file
        for row in rows:
            summary_df = summary_df._append(pd.Series(row), ignore_index=True)

    # Save the summary dataframe to a CSV file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    summary_df.to_csv(
        os.path.join(log_dir, f"summary_{np.datetime64('now')}.csv"),
        index=False,
    )

    ########################################################################
    ## Save pose estimation and GT frame by frame, per follower and method
    for (leader_name, follower_name), estimator_dict in est_pose_stats_list[0].items():
        for est_name, pose_dict in estimator_dict.items():
            est_result_df = pd.DataFrame()
            gt_result_df = pd.DataFrame()
            gt_l_result_df = pd.DataFrame()
            est_rows = []
            gt_rows = []
            gt_l_rows = []
            for pose_type, pose_list in pose_dict.items():
                for pose in pose_list:
                    row = pose.flatten().tolist() if pose is not None else [np.nan] * 12
                    if pose_type == "T_est":
                        est_rows.append(row)
                    elif pose_type == "T_gt":
                        gt_rows.append(row)
                    elif pose_type == "T_gt_l":
                        gt_l_rows.append(row)

            for row in est_rows:
                est_result_df = est_result_df._append(pd.Series(row), ignore_index=True)
            for row in gt_rows:
                gt_result_df = gt_result_df._append(pd.Series(row), ignore_index=True)
            for row in gt_l_rows:
                gt_l_result_df = gt_l_result_df._append(pd.Series(row), ignore_index=True)

            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            est_result_df.to_csv(
                os.path.join(log_dir, f"est_relpose_{follower_name}_{est_name}_{np.datetime64('now')}.csv"),
                index=False,
            )
            gt_result_df.to_csv(
                os.path.join(log_dir, f"gt_relpose_{follower_name}_{est_name}_{np.datetime64('now')}.csv"),
                index=False,
            )
            gt_l_result_df.to_csv(
                os.path.join(log_dir, f"gt_leaderpose_{follower_name}_{est_name}_{np.datetime64('now')}.csv"),
                index=False,
            )

    # ## save pose estimation and gt frame by frame TODO: need to figure out the format!!!
    # for follower in args.followers:
    #     est_result_df = pd.DataFrame()
    #     gt_result_df = pd.DataFrame()
    #     for T_gt, T_rel in est_pose_stats_list[0].items():
    #         est_rows = []
    #         gt_rows = []
    #         curr_row = []
    #         for est_name, poses in T_rel.items():
    #             for pose_type, pose_values in poses.items():
    #                 for t_rel in pose_values:
    #                     if pose_type == "T_est":
    #                         est_rows.append(
    #                             t_rel.flatten().tolist()
    #                             if t_rel is not None
    #                             else [np.nan] * 12
    #                         )
    #                     elif pose_type == "T_gt":
    #                         gt_rows.append(
    #                             t_rel.flatten().tolist()
    #                             if t_rel is not None
    #                             else [np.nan] * 12
    #                         )
    #                     else:
    #                         continue

    #     for row in est_rows:
    #         est_result_df = est_result_df._append(
    #             pd.Series(row), ignore_index=True
    #         )  # r11 r12 r13 x
    #     for row in gt_rows:
    #         gt_result_df = gt_result_df._append(pd.Series(row), ignore_index=True)

    #     log_dir = "logs"
    #     os.makedirs(log_dir, exist_ok=True)
    #     est_result_df.to_csv(
    #         os.path.join(log_dir, f"estimation_result_1_{np.datetime64('now')}.csv"),
    #         index=False,
    #     )
    #     gt_result_df.to_csv(
    #         os.path.join(log_dir, f"gt_result_1_{np.datetime64('now')}.csv"),
    #         index=False,
    #     )


def parse_args():
    parser = ArgumentParser(
        description="Visualize image matches from a ROS2 bag file."
    )
    parser.add_argument(
        "-b", "--bag", type=str, help="Path to the ROS2 bag file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/viz.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-s",
        "--save-frames",
        type=str,
        default="logs",
        help="Path to the directory where frames will be saved.",
    )
    parser.add_argument(
        "--n_runs", type=int, default=20, help="Number of runs to average over"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=8, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--viz", action="store_true", help="Enable visualization"
    )
    parser.add_argument(
        "-fid",
        "--start_frame_id",
        type=int,
        default=0,
        help="Start frame ID for evaluation",
    )
    args = parser.parse_args()

    if args.viz:
        print("Visualization enabled. Changing n_jobs and n_runs to 1.")
        args.n_jobs = 1
        args.n_runs = 1

    config = OmegaConf.load(args.config)
    OmegaConf.set_struct(config, True)  # Make config immutable

    args.leader = LeaderRobotArgs(
        name=config.leader.name,
        images_dir=config.leader.images,
        odoms_txt=config.leader.odoms,
        image_config_raw=config.leader.image,
        depths_dir=config.leader.depths,
        depth_config=config.leader.depth,
    )
    args.followers = [
        FollowerRobotArgs(
            name=follower.name,
            images_dir=follower.images,
            depths_dir=follower.depths,
            odoms_txt=follower.odoms,
            image_config_raw=follower.image,
            depth_config=follower.depth,
        )
        for follower in config.followers
    ]
    args.estimators = {
        est: import_func_from_module(
            module_name=est_cfg.module, func_name=est_cfg.func
        )
        for est, est_cfg in config.estimators.items()
    }

    return args


if __name__ == "__main__":
    run_experiment(parse_args())


"""
python3 eval_vis.py \
    --bag "/home/rishavdutta/repos/rpe_dl_ros2_sim/agilex_open_class/limo/limo_gazebo_sim/Dataset/straight_yaw_1/rosbag2_2025_06_24-17_50_10_0.db3" \
    --config "configs/viz.yaml" \
    --n_runs 5 \
    --n_jobs 4 \
    -fid 100

python3 eval_vis.py \
    --config "configs/viz_occ.yaml" \
    --save-frames "logs" \
    --viz \
    -fid 50


    /home/suyoung/mydata/icra2025_dataset/three_limo_test_1
"""