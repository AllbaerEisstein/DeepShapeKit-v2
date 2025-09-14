"""
functions for multiview output from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""
# import trimesh
from typing import Optional
import yaml
import os
import numpy as np
import torch
import src.constants as c
import cv2

# from .renderer import Renderer
from .geometry import perspective_projection, perspective_projection_homo, perspective_projection_ref


def get_fullsize_masks(masks, bboxes, h=368, w=368):
    full_masks = []
    for i in range(len(masks)):
        box = bboxes[i]
        full_mask = torch.zeros([h, w], dtype=torch.bool)
        full_mask[box[1]:box[1] + box[3] + 1, box[0]:box[0] + box[2] + 1] = masks[i]
        full_masks.append(full_mask)
    full_masks = torch.stack(full_masks)

    return full_masks


def get_cam(device='cpu'):
    proj_m_set = torch.stack([c.proj_front, c.proj_bottom], 0).to(device)
    proj_m_set_homo = torch.cat([proj_m_set, torch.tensor([[[0,0,-1,0]], [[0,0,-1,0]]]).to(device)], 1)
    f1 = 3930.0
    f2 = 3930
    focal = torch.tensor([f1, f1]).to(device)
    center = torch.tensor([[1024.,520.],[1024.,520.]]).to(device)
    # K = torch.tensor([[f1/2048,0,0.],[0,f1/1040,0.],[0,0,1]]).to(device)
    K = torch.tensor([[f1, 0, 1024.], [0, f1, 520.], [0, 0, 1]]).to(device)
    # K = torch.tensor([[ 7.67578125,0.,-1.,0.],
    #                  [ 0.,15.11538462,1.,0.],
    #                  [ 0.,0.,-1.00010001,-0.100005],
    #                  [ 0.,0.,-1.,0.]])
    # K = torch.tensor([[3.83789062,0.,0.,0.],
    #                 [0.,7.55769231,0.,0.],
    #                 [0.,0.,- 1.00010001,- 0.100005],
    #                 [0.,0.,- 1.,0.]])
    H = torch.matmul(K.inverse(), proj_m_set)
    # H = torch.matmul(K.inverse(), proj_m_set_homo)

    distortion = torch.tensor(c.distortion).to(device)

    return  proj_m_set, focal, center, H[:,:,:-1], H[:,:,-1], distortion


def projection_loss(x, y):
    loss = (x.float() - y.float()).norm(p=2)
    return loss


def triangulation_LBFGS(
    points: torch.Tensor,    
    Ps: torch.Tensor, 
    Ks: torch.Tensor, 
    Rs: torch.Tensor, 
    Ts: torch.Tensor, 
    focals: torch.Tensor, 
    principal_points: torch.Tensor, 
    distortions: torch.Tensor,
):
    cam_params = (Ps, Ks, Rs, Ts, focals, principal_points, distortions)
    assert all(points.device == param.device for param in cam_params), "All inputs should be on the same device"

    device = points.device
    vn = points.shape[0]
    
    # Compute a better initial guess for X: mean of back-projected rays
    # Convert 2D points to homogeneous coordinates
    points_h = torch.cat([points, torch.ones(vn, 1, device=device)], dim=1)  # (n, 3)
    # Back-project using pseudo-inverse of projection matrices
    Xs = []
    for i in range(vn):
        P = Ps[i]  # (3, 4)
        # Least squares solution to PX ~ x (find X so that || PX - x || is minimal)
        X_h = torch.linalg.lstsq(P, points_h[i].unsqueeze(1)).solution  # (4, 1)
        X_cart = X_h[:3] / X_h[3]
        Xs.append(X_cart.squeeze())
    X_init = torch.stack(Xs).mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, 3)

    X = X_init.clone().detach().requires_grad_()

    losses = []
    optimizer = torch.optim.LBFGS([X], lr=1, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        projected_points = perspective_projection_ref(X.repeat(vn, 1, 1), Rs, Ts, focals, principal_points, distortions)
        loss = projection_loss(projected_points.squeeze(), points)

        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        projected_points = perspective_projection_ref(X.repeat(vn, 1, 1), Rs, Ts, focals, principal_points, distortions)
        loss = projection_loss(projected_points.squeeze(), points)
        losses.append(loss.detach().item())
    X = X.detach().squeeze()

    return X, losses


def triangulation(    
    points: torch.Tensor,    
    Ps: torch.Tensor, 
    Ks: torch.Tensor, 
    Rs: torch.Tensor, 
    Ts: torch.Tensor, 
    focals: torch.Tensor, 
    principal_points: torch.Tensor, 
    distortions: torch.Tensor
):
    cam_params = (Ps, Ks, Rs, Ts, focals, principal_points, distortions)
    assert all(points.device == param.device for param in cam_params), "All inputs should be on the same device"

    device = points.device
    vn = points.shape[0]

    # Compute a better initial guess for X: mean of back-projected rays
    # Convert 2D points to homogeneous coordinates
    points_h = torch.cat([points, torch.ones(vn, 1, device=device)], dim=1)  # (n, 3)
    # Back-project using pseudo-inverse of projection matrices
    Xs = []
    for i in range(vn):
        P = Ps[i]  # (3, 4)
        # Least squares solution to PX ~ x
        X_h = torch.linalg.lstsq(P, points_h[i].unsqueeze(1)).solution  # (4, 1)
        X_cart = X_h[:3] / X_h[3]
        Xs.append(X_cart.squeeze())
    X_init = torch.stack(Xs).mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, 3)

    X = X_init.clone().detach().requires_grad_()

    losses = []
    optimizer = torch.optim.Adam([X], lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 90], gamma=0.1)
    for i in range(100):
        projected_points = perspective_projection_ref(X.repeat(vn, 1, 1), Rs, Ts, focals, principal_points, distortions)
        loss = projection_loss(projected_points.squeeze(), points)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().item())

    X = X.detach().squeeze()

    return X, losses


def get_gt_3d(
    keypoints: torch.Tensor,     
    Ps: torch.Tensor, 
    Ks: torch.Tensor, 
    Rs: torch.Tensor, 
    Ts: torch.Tensor, 
    focals: torch.Tensor, 
    principal_points: torch.Tensor, 
    distortions: torch.Tensor,
    LBFGS: bool = True
):
    '''
    Triangulate 3D keypoints from multi-view 2D keypoints.
    !! This step sorts out low-confidence keypoints and only uses high-confidence ones for triangulation !!
    Input:
        keypoints (vn, kn, 2): 2D kpts from different views
        Ps (vn, 3, 4): camera projection matrices for each view
        distortions (vn, 5): distortion coefficients in the order k_1, k_2, k_3, p_1, p_2
    Output:
        kpts_3d (kn, 4): ground truth 3D kpts, with validility (not per-view anymore because triangulated)
    '''
    cam_params = (Ps, Ks, Rs, Ts, focals, principal_points, distortions)
    assert all(keypoints.device == param.device for param in cam_params), "All inputs should be on the same device"

    vn, kn, _ = keypoints.shape
    kpts_3d = torch.zeros([kn, 4])

    kpts_valid = [] # shape: (kn, ? , 2); second dimension is number of views where this kpt is valid
    views_valid_per_kpt = []
    for k in range(kn):
        # Check the confidence score for each keypoint at index i across all views.
        # (element in the last dimension: confidence score)
        # This produces a boolean array, valid, indicating which keypoints have a
        # confidence score greater than zero in all views.
        # Next, keypoints[valid, i, :2] selects the x and y coordinates (the first
        # two elements) of the keypoints at index i, but only for those views where
        # the confidence score is positive.
        views_where_this_kpt_is_valid = keypoints[:, k, -1] > 0
        kpts_valid.append(keypoints[views_where_this_kpt_is_valid, k, :2])
        views_valid_per_kpt.append(views_where_this_kpt_is_valid)

    for k in range(kn):
        x = kpts_valid[k]
        if len(x) >= 2: # need at least two views to triangulate
            cam_params_k = [param[views_valid_per_kpt[k]] for param in cam_params]
            
            if LBFGS:
                X, _ = triangulation_LBFGS(x, *cam_params_k) 
            else:
                X, _ = triangulation(x, *cam_params_k)

            kpts_3d[k, :3] = X
            kpts_3d[k, -1] = 1

    return kpts_3d


def Procrustes(X: torch.Tensor, Y: torch.Tensor):
    """
    Solve full Procrustes: Y = s*RX + t
    Given a set of 3D points X in one coordinate system and the same points Y in another system, 
    find the best similarity transform (rotation, translation, scale) that aligns them.
    Input:
        X (N,3): tensor of N points
        Y (N,3): tensor of N points in world coordinate
    Returns:
        R (3x3): tensor describing camera orientation in the world (R_wc)
        t (3,): tensor describing camera translation in the world (t_wc)
        s (1): scale
    """
    # Procrustes only works on cpu
    X = X.cpu()
    Y = Y.cpu()
    # remove translation
    A = (Y - Y.mean(dim=0, keepdim=True))
    B = (X - X.mean(dim=0, keepdim=True))

    # remove scale
    sA = (A * A).sum() / A.shape[0]
    sA = sA.sqrt()
    sB = (B * B).sum() / B.shape[0]
    sB = sB.sqrt()
    A = A / sA
    B = B / sB
    s = sA / sB

    # to numpy, then solve for R
    A = A.t().numpy()
    B = B.t().numpy()

    M = B @ A.T
    U, S, VT = np.linalg.svd(M)
    V = VT.T

    d = np.eye(3)
    d[-1, -1] = np.linalg.det(V @ U.T)
    R = V @ d @ U.T

    # back to tensor
    R = torch.tensor(R).float()
    t = Y.mean(dim=0) - R @ X.mean(dim=0) * s

    return R, t, s

def batch_render_reconstructions(
    imgs: torch.Tensor,  # (vn, h, w, 3)
    vertex_world_coors_reconstructed: torch.Tensor,  # (num_vertices, 3)
    Ps: torch.Tensor,
    Ks: torch.Tensor,
    Rs: torch.Tensor,
    Ts: torch.Tensor,
    focals: torch.Tensor,
    distortions: torch.Tensor,
    principal_points: torch.Tensor,
    kpts: Optional[torch.Tensor] = None,  # (num_keypoints, 3)
    bboxs: Optional[torch.Tensor] = None,  # (vn, 4)
) -> np.ndarray:
    """
    Overlay projected mesh vertices, keypoints, and bounding boxes onto a batch of images.

    Args:
        imgs (vn, h, w, 3): Batch of images.
        vertex_world_coors_reconstructed (1, num_vertices, 3): Mesh vertices in world coordinates.
        kpts (1, num_keypoints, 3): Keypoints in world space.
        bboxs (vn, 4): Bounding boxes per image.

    Returns:
        imgs_with_overlay (vn, h, w, 3): Images with overlays.
    """
    vn, h, w, _ = imgs.shape
    imgs_out = imgs.clone() if torch.is_tensor(imgs) else torch.tensor(imgs).clone()

    # Project vertices and keypoints for all views
    vertex_projections = perspective_projection_ref(
        vertex_world_coors_reconstructed.repeat(vn, 1, 1), Rs, Ts, focals, principal_points
    )  # (vn, num_vertices, 2)
    if kpts is not None:
        keypoint_projections = perspective_projection_ref(
            kpts.repeat(vn, 1, 1), Rs, Ts, focals, principal_points
        )  # (vn, num_keypoints, 2)

    for view_idx in range(vn):
        img = imgs_out[view_idx]

        # Draw projected vertices (green)
        verts_2d = vertex_projections[view_idx]  # (num_vertices, 2)
        ix = torch.clamp(verts_2d[:, 1].long(), 0, h - 1)
        iy = torch.clamp(verts_2d[:, 0].long(), 0, w - 1)
        img[ix, iy] = torch.tensor([0, 255, 0], dtype=img.dtype)

        # Draw keypoints (red)
        if kpts is not None:
            kpts_2d = keypoint_projections[view_idx]  # (num_keypoints, 2)
            kx = torch.clamp(kpts_2d[:, 0].long(), 0, w - 1)
            ky = torch.clamp(kpts_2d[:, 1].long(), 0, h - 1)
            img[ky, kx] = torch.tensor([255, 0, 0], dtype=img.dtype)

        # Draw bounding box (yellow)
        if bboxs is not None:
            x1, y1, x2, y2 = bboxs[view_idx].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img[y1:y2, x1] = torch.tensor([255, 255, 0], dtype=img.dtype)
            img[y1:y2, x2] = torch.tensor([255, 255, 0], dtype=img.dtype)
            img[y1, x1:x2] = torch.tensor([255, 255, 0], dtype=img.dtype)
            img[y2, x1:x2] = torch.tensor([255, 255, 0], dtype=img.dtype)

        imgs_out[view_idx] = img

    imgs_out = imgs_out.cpu().numpy().astype(np.uint8)
    return imgs_out


# def render_mesh(bird, pose_est, bone_est, scale_est=1, camera_t=torch.tensor([[2, -7, 35]]).float()):
#     # Background
#     background = torch.ones([368, 368, 3]).float()
#
#     # Camera parameters
#     # camera_t = torch.tensor([[2, -7, 35]]).float()
#     camera_center = torch.tensor([[368 // 2, 368 // 2]]).float()
#     focal_length = 1000.1
#
#     # Bird Mesh
#     bird_output = bird(pose_est[:, 0:3], pose_est[:, 3:], bone_est, scale_est)
#     vertex_posed = bird_output['vertices']
#     # vertex_posed += torch.tensor([[[0,10,8]]]).float()
#
#     # Rendering
#     renderer = Renderer(focal_length=focal_length, center=(184, 184), img_w=368, img_h=368, faces=bird.faces)
#     img_1, _ = renderer(vertex_posed[0].clone().numpy(), np.eye(3), camera_t[0].clone().numpy(),
#                         background.clone().numpy())
#
#     # Render: Second View
#     aroundy = cv2.Rodrigues(np.array([0, np.radians(45.), 0]))[0]
#     center = vertex_posed.numpy()[0].mean(axis=0)
#     rot_vertices = np.dot((vertex_posed.numpy()[0] - center), aroundy) + center
#     img_2, _ = renderer(rot_vertices, np.eye(3), camera_t[0].clone().numpy(), background.clone().numpy())
#
#     # Render: Third View
#     aroundy = cv2.Rodrigues(np.array([0, np.radians(-45.), 0]))[0]
#     center = vertex_posed.numpy()[0].mean(axis=0)
#     rot_vertices = np.dot((vertex_posed.numpy()[0] - center), aroundy) + center
#     img_3, _ = renderer(rot_vertices, np.eye(3), camera_t[0].clone().numpy(), background.clone().numpy())
#
#     return [img_1, img_2, img_3]