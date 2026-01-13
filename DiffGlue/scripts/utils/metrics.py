import numpy as np


def calculate_ATE(gt_pose, est_pose):
    return np.linalg.norm(np.array(gt_pose[:3]) - np.array(est_pose[:3]))


def calculate_RMD(gt_pose, est_pose):
    R_gt = gt_pose[:3, :3]  # R.from_quat(gt_pose[3:]).as_matrix()
    R_est = est_pose[:3, :3]  # R.from_quat(est_pose[3:]).as_matrix()
    diff_rot = R_gt.T @ R_est
    angle = np.arccos(np.clip((np.trace(diff_rot) - 1) / 2, -1.0, 1.0))
    return np.degrees(abs(angle))


def calculate_stats(values):
    return {
        "mean": np.mean(values),
        "rmse": np.sqrt(np.mean(np.square(values))),
        "max": np.max(values),
        "min": np.min(values),
    }


calculate_stats.keys = ["mean", "rmse", "max", "min"]


def calculate_mean_std(values_list):
    values = np.array(values_list)
    return {
        "mean": np.mean(values),
        "std": np.std(values),
    }


def summarize_metric(name, values_list):
    values = np.array(values_list)
    print(f"====== {name} ======")
    for key in calculate_stats.keys():
        stats = calculate_stats([v[key] for v in values])
        print(f"{key.capitalize()}: {stats['mean']:.3f} ± {stats['std']:.3f}")


def sampson_error(E, pts1, pts2):
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    Ex1 = E @ pts1_h.T
    Etx2 = E.T @ pts2_h.T
    x2tEx1 = np.sum(pts2_h * (E @ pts1_h.T).T, axis=1)
    dists = x2tEx1**2 / (
        Ex1[0] ** 2 + Ex1[1] ** 2 + Etx2[0] ** 2 + Etx2[1] ** 2 + 1e-8
    )
    return np.mean(dists)


def compute_essential_matrix(R, t):
    t_x = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return t_x @ R


def compute_rotation_error(R1, R2):
    R_err = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    return np.degrees(angle)
