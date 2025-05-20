import numpy as np
import torch
import torch.nn.functional as F
import cv2

def build_K_matrix(f, c):
    """
    Construct batched camera intrinsic matrices from fx, fy and cx, cy.

    Args:
        f: [B, 2] focal lengths (fx, fy)
        c: [B, 2] principal points (cx, cy)

    Returns:
        K: [B, 3, 3] camera intrinsic matrices
    """
    B = f.shape[0]
    K = torch.zeros(B, 3, 3, device=f.device, dtype=f.dtype)
    K[:, 0, 0] = f[:, 0]  # fx
    K[:, 1, 1] = f[:, 1]  # fy
    K[:, 0, 2] = c[:, 0]  # cx
    K[:, 1, 2] = c[:, 1]  # cy
    K[:, 2, 2] = 1.0
    return K

def backproject_to_3d(kpts, depth_map, K):
    """
    kpts: [N, 2] tensor of image coordinates
    depth_map: [H, W] numpy array
    K: [3, 3] numpy array (intrinsics)
    returns: [N, 3] tensor of 3D points
    """
    pts3d = []
    for pt in kpts:
        u, v = int(pt[0]), int(pt[1])
        if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
            z = depth_map[v, u]
            if z <= 0:
                continue
            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]
            pts3d.append([x, y, z])
    return torch.tensor(pts3d, dtype=torch.float32)


def solve_pnp_ransac(kpts2d, pts3d, K):
    """
    kpts2d: [N, 2] 2D keypoints in image 0
    pts3d: [N, 3] 3D keypoints from image 1
    K: [3, 3] numpy array
    returns: 4x4 SE(3) transform from frame1 to frame0
    """
    if len(kpts2d) < 6 or len(pts3d) < 6:
        return None

    pts2d_np = kpts2d.cpu().numpy().astype(np.float32)
    pts3d_np = pts3d.cpu().numpy().astype(np.float32)
    K_np = K.cpu().numpy()

    success, rvec, tvec, _ = cv2.solvePnPRansac(pts3d_np, pts2d_np, K_np, None)
    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return torch.tensor(T, dtype=torch.float32)


def epipolar_loss(kpts0, kpts1, matches0, T0to1, cam0, cam1, weight=1.0):
    """
    Computes mean squared epipolar residual for valid matches.

    kpts0, kpts1: [B=32, N, 2]
    matches0: [B, N] → indices in kpts1 or -1
    pose0, pose1: [B, 4, 4] ground truth poses
    K: [B, 3, 3]
    """
   
    K0 = build_K_matrix(cam0.f, cam0.c)
    K1 = build_K_matrix(cam1.f, cam1.c)

    B = kpts0.shape[0]
    losses = torch.zeros(B, device=kpts0.device, dtype=kpts0.dtype)
    for b in range(B):
        m0 = matches0[b]  # [N]
        valid = m0 > -1
        x0 = kpts0[b][valid]  # [M, 2]  # TODO: Epipolar calculation based on only valid matches or every???
        x1 = kpts1[b][m0[valid]]  # [M, 2]

        if x0.shape[0] < 8:
            continue

        # Normalize to camera coordinates
        K0_inv = torch.inverse(K0[b])
        K1_inv = torch.inverse(K1[b])
        x0_h = F.pad(x0, (0, 1), value=1.0)
        x1_h = F.pad(x1, (0, 1), value=1.0)

        x0_cam = (K0_inv @ x0_h.T).T  # [M, 3]
        x1_cam = (K1_inv @ x1_h.T).T  # [M, 3]

        # Relative pose: T = T1 * T0^-1
        T = T0to1[b]
        R = T0to1[b].R
        t = T0to1[b].t

        # Essential matrix
        tx = torch.tensor([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], device=t.device)

        E = tx @ R

        # Epipolar residuals
        r = torch.einsum('bi,ij,bj->b', x1_cam, E, x0_cam)  # [M]
        losses[b] = weight * (r ** 2).mean()

    return losses

def sampson_epipolar_loss(kpts0, kpts1, matches0, T0to1, cam0, cam1, weight=1.0):
    """
    Computes Sampson epipolar error per batch element.
    
    Returns:
        losses: [B] Sampson errors per batch sample
    """
    K0 = build_K_matrix(cam0.f, cam0.c)
    K1 = build_K_matrix(cam1.f, cam1.c)

    B = kpts0.shape[0]
    losses = torch.zeros(B, device=kpts0.device, dtype=kpts0.dtype)

    for b in range(B):
        m0 = matches0[b]  # [N]
        valid = m0 > -1
        if valid.sum() < 8:
            losses[b] = 0.0
            continue

        x0 = kpts0[b][valid]  # [M, 2]
        x1 = kpts1[b][m0[valid]]  # [M, 2]

        x0_h = F.pad(x0, (0, 1), value=1.0)  # [M, 3]
        x1_h = F.pad(x1, (0, 1), value=1.0)  # [M, 3]

        # Normalize to camera coordinates
        K0_inv = torch.inverse(K0[b])
        K1_inv = torch.inverse(K1[b])
        x0_cam = (K0_inv @ x0_h.T).T  # [M, 3]
        x1_cam = (K1_inv @ x1_h.T).T  # [M, 3]

        # Get relative pose
        if isinstance(T0to1, list) or hasattr(T0to1[b], 'R'):
            R = T0to1[b].R  # [3, 3]
            t = T0to1[b].t  # [3]
        else:
            T = T0to1[b]
            R = T[:3, :3]
            t = T[:3, 3]

        # Essential matrix E = [t]_x R
        tx = torch.tensor([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], device=t.device, dtype=t.dtype)
        E = tx @ R  # [3, 3]

        # Compute numerator: (x1^T E x0)^2
        Ex0 = (E @ x0_cam.T).T  # [M, 3]
        Etx1 = (E.T @ x1_cam.T).T  # [M, 3]
        x1_E_x0 = torch.einsum('ni,ni->n', x1_cam, Ex0)  # [M]
        numerator = x1_E_x0 ** 2  # [M]

        # Compute denominator: sum of squared derivatives
        denom = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2  # [M]
        sampson_error = numerator / (denom + 1e-8)  # [M]

        losses[b] = weight * sampson_error.mean()

    return losses  # Shape: [B]