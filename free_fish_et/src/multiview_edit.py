"""
functions for multiview modified from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""
from typing import Optional

import cv2
import numpy as np
import torch

from src.fish_model_edit import fish_model
from src.losses import camera_fitting_loss
from src.geometry import perspective_projection
import src.multiview_utils_edit as mutils
from src.pose_optimizer_edit import OptimizeMV
from src.Silhouette_Renderer_edit import Silhouette_Renderer
from src.CameraGroups import CameraGroup


def fit_geometry(
    fish: fish_model,
    keypoints: torch.Tensor,
    cameras: CameraGroup,
    init_pose: Optional[torch.Tensor] = None,
    init_body_bone_l: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initial fit with geometry: triangulation + Procrustes
    Input:
        fish: fish model
        keypoints (vn, kn, 3): 2d keypoints from each view with hard confidence
        cameras: camera parameters for each view
    """

    # 3D kpts on bird
    if init_pose == None and init_body_bone_l == None:
        # initial position: rest pose
        fish_mesh_kpts_local = torch.matmul(fish.vert2kpt, fish.V[0])
    else:
        # no initial guess for global position/orientation
        init_ori = torch.zeros([1, 3]).float().to(keypoints.device)
        fish_articulated = fish(init_ori, init_pose, init_body_bone_l)
        fish_mesh_kpts_local = fish_articulated["keypoints"][0]

    # Triangulation with LBFGS
    camera_group = cameras.to(keypoints.device)
    observed_kpts_3d = mutils.get_gt_3d(keypoints, camera_group, LBFGS=True)

    valid_kpts_3d_boolmask = observed_kpts_3d[:, -1] > 0
    valid_kpts_3d = observed_kpts_3d[valid_kpts_3d_boolmask, :3]
    fish_mesh_kpts_local = fish_mesh_kpts_local[valid_kpts_3d_boolmask, :]

    # Procrustes with available 3D kpts
    # procrustes 
    R, t, s = mutils.Procrustes(fish_mesh_kpts_local, valid_kpts_3d)
    aa, _ = cv2.Rodrigues(R.numpy())

    init_ori = torch.tensor(aa).reshape(1, 3).float().to(keypoints.device)
    init_t = t
    init_s = s

    return init_ori, init_t, init_s


def fit_mesh(
    fish: fish_model,
    optimizer: OptimizeMV,
    cameras: CameraGroup,
    keypoints: torch.Tensor,
    masks: torch.Tensor,
    renderer: Silhouette_Renderer,
    device: str,
    init_global_ori: Optional[torch.Tensor] = None,
    init_t: Optional[torch.Tensor] = None,
    init_s: Optional[torch.Tensor] = None,
    init_body_pose: Optional[torch.Tensor] = None,
    init_body_bone_length: Optional[torch.Tensor] = None,
    index: Optional[torch.Tensor] = None,
    bboxs: Optional[torch.Tensor] = None,
):
    """
    Only used in multiview and crossview fitting:
    Input:
        cameras: camera parameters for each view
        init_pose (vn, 4*3): body pose in axis-angle (exclude root joint orient)
        init_bone (vn, 4): bone length
    """
    # move to device
    camera_group = cameras.to(device)
    Ps = camera_group.P
    keypoints = keypoints.to(device)
    masks = masks.to(device)
    fish.to_device(device)
    assert keypoints.shape[0] == Ps.shape[0], "camera batch size must match keypoints"
    assert fish.device_active, "fish model must be on target device"

    if (
        init_global_ori != None
        or init_t != None
        or init_s != None
        or init_body_pose != None
        or init_body_bone_length != None
    ):
        has_prev = True
        # optimizer.prior_weight = 80
    else:
        has_prev = False

    ### Triangulation + Procrustes as initialization
    if init_global_ori == None and init_t == None and init_s == None:
        init_global_ori, init_t, init_s = fit_geometry(
            fish, keypoints, camera_group, init_pose=init_body_pose, init_body_bone_l=init_body_bone_length
        )

    ### If not provided (as in multiview), initialize with canonical
    if init_body_pose is None:
        init_body_pose = torch.zeros([1, fish.n_body_bones * 3], device=device)
    if init_body_bone_length is None:
        init_body_bone_length = torch.ones([1, fish.n_bones], device=device)

    #### Change suitable format for optimizer
    ###### particularly, combine orient and body pose
    init_ori_plus_pose = torch.cat([init_global_ori, init_body_pose], dim=1).to(device)
    init_body_pose = init_body_pose.float().to(device)
    init_body_bone_length = init_body_bone_length.float().to(device)
    init_s = init_s.float().to(device)
    init_t = init_t.float().to(device)

    assert all(
        tensor.device == keypoints.device
        for tensor in (masks, Ps, init_ori_plus_pose, init_body_bone_length, init_s, init_t)
    ), "All inputs must reside on the target device"

    ### Mesh fitting
    vertices, global_ori_plus_pose_est, body_bone_est, scale_est, t, losses = optimizer(
        init_ori_plus_pose,
        init_body_bone_length,
        init_t,
        init_s,
        Ps,
        keypoints,
        masks,
        renderer,
        has_prev,
        index,
        bboxs,
    )

    ### Generating mesh output
    fish_output = fish(global_ori_plus_pose_est[:, 0:3], global_ori_plus_pose_est[:, 3:], body_bone_est, scale_est)

    # NOTE:
    # things to check: Correct row-major, column major order always?
    # pixel/mm/m? E.g. in fish model there seems to be a conversion cm -> m
    # world/local coords for fish model?
    vertex_posed = fish_output["vertices"] + t
    mesh_keypoint = fish_output["keypoints"] + t

    return vertex_posed, mesh_keypoint, t, global_ori_plus_pose_est, body_bone_est, scale_est, losses


def multiview_rigid_alignment(
    fish, pose, bone, keypoints, frames, device="cpu", num_iters=100
):
    """
    Rigidly align single view reconstruction to multiview instance so we can
    check reconstruction accuracy across different views.
    1. First run general Procrustes for global alignment
    2. Because Procrustes can only use keypoints that are visible from at least two views,
    we run a short optimizition (rigidly, fixed pose and shape) afterward to improve alignment.
    Input:
        pose and bone are from singleview reconstruction;
        keypoints are ground truth for alignments
    """

    ### Camera parameter
    proj_m_set, focal, center, R, T, distortion = mutils.get_cam(device)

    ### Triangulation + Procrustes for global alignemnt
    global_orient, global_t, scale = fit_geometry(
        fish, keypoints, frames, init_pose=pose, init_body_bone_l=bone
    )

    ### Optimization to improve alignment
    pose = pose.detach().clone().to(device)
    bone = bone.detach().clone().to(device)
    keypoints = keypoints.clone().to(device)
    batch_size = len(frames)

    global_orient = global_orient.to(device)
    global_t = global_t.to(device)
    scale = scale.to(device)

    global_orient.requires_grad = True
    global_t.requires_grad = True
    scale.requires_grad = True

    global_params = [global_orient, global_t, scale]
    global_optimizer = torch.optim.Adam(global_params, lr=1e-2, betas=(0.9, 0.999))
    for i in range(num_iters):
        fish_output = fish(
            global_ori=global_orient, body_pose=pose, body_bone_length=bone, scale=scale
        )

        model_keypoints = fish_output["keypoints"] + global_t.repeat(1, 1, 1)
        model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

        loss = camera_fitting_loss(
            model_keypoints, proj_m_set, keypoints[:, :, :2], keypoints[:, :, -1]
        )

        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()

    # Output
    fish_output = fish(
        global_ori=global_orient, body_pose=pose, body_bone_length=bone, scale=scale
    )
    model_mesh = fish_output["vertices"] + global_t.repeat(1, 1, 1)
    model_keypoints = fish_output["keypoints"] + global_t.repeat(1, 1, 1)

    model_mesh = model_mesh.detach().to("cpu")
    model_keypoints = model_keypoints.detach().to("cpu")

    return model_mesh, model_keypoints


def reproject_masks(vertex_est, renderer_list, frames):
    # Transform vertex for each camera view
    proj_m_set, focal, center, R, T, distortion = mutils.get_cam()
    rotation = torch.stack([R, R], 0)
    translation = torch.stack([T, T], 0).unsqueeze(1)

    points = vertex_est.repeat([len(frames), 1, 1])
    points = torch.einsum("bij,bkj->bki", rotation, points) + translation

    # Render for each view
    img = torch.zeros([368, 368, 3])
    proj_masks = []
    for i in range(len(frames)):
        renderer = renderer_list[frames[i]]
        img_pose, depth_map = renderer(
            points[i].cpu().numpy(), np.eye(3), [0, 0, 0], img.clone().numpy()
        )
        mask = torch.tensor(depth_map > 0)
        proj_masks.append(mask)

    return proj_masks


def reproject_keypoints(mesh_keypoints, frames):
    proj_m_set, focal, center, R, T, distortion = mutils.get_cam()

    kpts = mesh_keypoints.repeat([len(frames), 1, 1])
    proj_kpts = perspective_projection(kpts, proj_m_set)

    return proj_kpts
