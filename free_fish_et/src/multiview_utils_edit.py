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
import cv2

# from .renderer import Renderer
from .geometry import perspective_projection, perspective_projection_homo
from .CameraGroups import CameraGroup



def get_fullsize_masks(masks, bboxes, h=368, w=368):
    full_masks = []
    for i in range(len(masks)):
        box = bboxes[i]
        full_mask = torch.zeros([h, w], dtype=torch.bool)
        full_mask[box[1]:box[1] + box[3] + 1, box[0]:box[0] + box[2] + 1] = masks[i]
        full_masks.append(full_mask)
    full_masks = torch.stack(full_masks)

    return full_masks


def projection_loss(x, y):
    loss = (x.float() - y.float()).norm(p=2)
    return loss


def triangulation_LBFGS(
    points: torch.Tensor,
    cameras: CameraGroup,
) -> tuple[torch.Tensor, list[float]]:
    camera_group = cameras.to(points.device)

    vn = points.shape[0]
    points_h = torch.cat([points, torch.ones(vn, 1, device=points.device)], dim=1)
    from_bl_inv = camera_group.from_blenderworld.transpose(1, 2)

    Xs = []
    for i in range(vn):
        P = camera_group.P[i]
        print(f"Triangulation LBFGS - View {i}, points_h: {points_h[i]}")
        X_h = torch.linalg.lstsq(P, points_h[i].unsqueeze(1)).solution
        X_custom_coord_conv = (X_h[:3] / X_h[3]).squeeze()
        print(f"Triangulation LBFGS - View {i}, X_custom_coord_conv: {X_custom_coord_conv}")
        X_bl = torch.matmul(from_bl_inv[i], X_custom_coord_conv)
        print(f"Triangulation LBFGS - View {i}, X_bl: {X_bl}")

        Xs.append(X_bl)
    X_init = torch.stack(Xs).mean(dim=0, keepdim=True).unsqueeze(0)

    X = X_init.clone().detach().requires_grad_()

    losses: list[float] = []
    optimizer = torch.optim.LBFGS([X], lr=1, max_iter=1000, line_search_fn='strong_wolfe')

    def closure() -> torch.Tensor:
        projected_points = camera_group.perspective_projection_from_blworld(
            X.repeat(vn, 1, 1)
        )
        loss = projection_loss(projected_points.squeeze(), points)
        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        projected_points = camera_group.perspective_projection_from_blworld(
            X.repeat(vn, 1, 1)
        )
        loss = projection_loss(projected_points.squeeze(), points)
        losses.append(loss.detach().item())

    return X.detach().squeeze(), losses


def triangulation(
    points: torch.Tensor,
    cameras: CameraGroup,
) -> tuple[torch.Tensor, list[float]]:
    camera_group = cameras.to(points.device)

    vn = points.shape[0]
    points_h = torch.cat([points, torch.ones(vn, 1, device=points.device)], dim=1)
    from_bl_inv = camera_group.from_blenderworld.transpose(1, 2)

    Xs = []
    for i in range(vn):
        P = camera_group.P[i]
        X_h = torch.linalg.lstsq(P, points_h[i].unsqueeze(1)).solution
        X_cv = (X_h[:3] / X_h[3]).squeeze()
        X_bl = torch.matmul(from_bl_inv[i], X_cv)
        Xs.append(X_bl)
    X_init = torch.stack(Xs).mean(dim=0, keepdim=True).unsqueeze(0)

    X = X_init.clone().detach().requires_grad_()

    losses: list[float] = []
    optimizer = torch.optim.Adam([X], lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 90], gamma=0.1)
    for _ in range(100):
        projected_points = camera_group.perspective_projection_from_blworld(
            X.repeat(vn, 1, 1)
        )
        loss = projection_loss(projected_points.squeeze(), points)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().item())

    return X.detach().squeeze(), losses


def get_gt_3d(
    keypoints: torch.Tensor,
    cameras: CameraGroup,
    LBFGS: bool = True,
) -> torch.Tensor:
    """Triangulate 3D keypoints from multi-view 2D keypoints."""
    camera_group = cameras.to(keypoints.device)

    vn, kn, _ = keypoints.shape
    kpts_3d = keypoints.new_zeros((kn, 4))

    for k in range(kn):
        # TODO: keypoint validity
        valid_views = keypoints[:, k, -1] > 0
        if valid_views.sum() < 2:
            continue
        obs = keypoints[valid_views, k, :2]
        print(f"Triangulating keypoint {obs}.")
        image_size = None
        if camera_group.original_image_size_wh is not None:
            img_size = camera_group.original_image_size_wh
            if img_size.dim() >= 2 and img_size.shape[0] == camera_group.batch_size:
                image_size = img_size[valid_views].clone()
            else:
                image_size = img_size.clone()
        sub_group = CameraGroup(
            P=camera_group.P[valid_views],
            K=camera_group.K[valid_views],
            R=camera_group.R[valid_views],
            t=camera_group.t[valid_views],
            from_blenderworld=camera_group.from_blenderworld[valid_views],
            original_image_size_wh=image_size,
        ).to(keypoints.device)

        if LBFGS:
            X, _ = triangulation_LBFGS(obs, sub_group)
        else:
            X, _ = triangulation(obs, sub_group)

        kpts_3d[k, :3] = X
        kpts_3d[k, -1] = 1
        print(f"Triangulated keypoint {k}: {X}")

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
    print(f"Procrustes X: {X}"
          f"\nProcrustes Y: {Y}")
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
    vertex_world_coors_reconstructed: torch.Tensor,  # (1, num_vertices, 3)
    cameras: CameraGroup,
    kpts: Optional[torch.Tensor] = None,  # (1, num_keypoints, 3)
    bboxs: Optional[torch.Tensor] = None,  # (vn, 4)
) -> np.ndarray:
    """Overlay projected mesh vertices, keypoints, and bounding boxes onto images."""
    device = vertex_world_coors_reconstructed.device
    camera_group = cameras.to(device)

    vn, h, w, _ = imgs.shape
    imgs_out = imgs.clone() if torch.is_tensor(imgs) else torch.tensor(imgs).clone()

    vertex_projections = camera_group.perspective_projection_from_blworld(
        vertex_world_coors_reconstructed.repeat(vn, 1, 1)
    )

    keypoint_projections = None
    if kpts is not None:
        keypoint_projections = camera_group.perspective_projection_from_blworld(
            kpts.repeat(vn, 1, 1)
        )

    for view_idx in range(vn):
        img = imgs_out[view_idx]

        verts_2d = vertex_projections[view_idx]
        ix = torch.clamp(verts_2d[:, 1].long(), 0, h - 1)
        iy = torch.clamp(verts_2d[:, 0].long(), 0, w - 1)
        img[ix, iy] = torch.tensor([0, 255, 0], dtype=img.dtype)

        if keypoint_projections is not None:
            kpts_2d = keypoint_projections[view_idx]
            kx = torch.clamp(kpts_2d[:, 0].long(), 0, w - 1)
            ky = torch.clamp(kpts_2d[:, 1].long(), 0, h - 1)
            img[ky, kx] = torch.tensor([255, 0, 0], dtype=img.dtype)

        if bboxs is not None:
            x1, y1, x2, y2 = bboxs[view_idx].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img[y1:y2, x1] = torch.tensor([255, 255, 0], dtype=img.dtype)
            img[y1:y2, x2] = torch.tensor([255, 255, 0], dtype=img.dtype)
            img[y1, x1:x2] = torch.tensor([255, 255, 0], dtype=img.dtype)
            img[y2, x1:x2] = torch.tensor([255, 255, 0], dtype=img.dtype)

        imgs_out[view_idx] = img

    return imgs_out.cpu().numpy().astype(np.uint8)


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
