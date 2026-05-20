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

    # Filter keypoints for confidence > 0
    valid_kpts_3d_boolmask = observed_kpts_3d[:, -1] > 0
    valid_kpts_3d = observed_kpts_3d[valid_kpts_3d_boolmask, :3]
    fish_mesh_kpts_local = fish_mesh_kpts_local[valid_kpts_3d_boolmask, :]

    # Procrustes with available 3D kpts
    # Procrustes yields R, t, s so that if applying R,t,s to the fish mesh, 
    # the mesh keypoints align best with the triangulated observed keypoints
    # (Procrustes because the gt fish might be articulated but we try to fit 
    # the rest model to the keypoints anyway, just like Procrustes did)
    R_matrix, t, s = mutils.Procrustes(fish_mesh_kpts_local, valid_kpts_3d)
    R_exp_map, _ = cv2.Rodrigues(R_matrix.numpy())

    init_ori = torch.tensor(R_exp_map).reshape(1, 3).float().to(keypoints.device)
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
    init_body_pose: Optional[torch.Tensor] = None,
    init_body_bone_length: Optional[torch.Tensor] = None,
    init_s: Optional[torch.Tensor] = None,
    init_t: Optional[torch.Tensor] = None,
    index: Optional[torch.Tensor] = None,
    bboxs: Optional[torch.Tensor] = None,
):
    """
    Only used in multiview and crossview fitting:
    Input:
        cameras: camera parameters for each view
    """
    # move to device
    camera_group = cameras.to(device)
    Ps = camera_group.projection_matrices(blender=True)
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

    nonfinite_init_names = []
    for name, tensor in [
        ("init_global_ori", init_global_ori),
        ("init_body_pose", init_body_pose),
        ("init_body_bone_length", init_body_bone_length),
        ("init_s", init_s),
        ("init_t", init_t),
    ]:
        if tensor is not None and not torch.isfinite(tensor).all():
            nonfinite_init_names.append(name)
    if nonfinite_init_names:
        print(
            f"Warning: non-finite initialization in {', '.join(nonfinite_init_names)}; "
            "falling back to geometry-based initialization."
        )
        init_global_ori = None
        init_body_pose = None
        init_body_bone_length = None
        init_s = None
        init_t = None
        has_prev = False

    ### Triangulation + Procrustes as initialization
    if init_global_ori == None and init_t == None and init_s == None:
        init_global_ori, init_t, init_s = fit_geometry(
            fish, keypoints, camera_group, init_pose=init_body_pose, init_body_bone_l=init_body_bone_length
        )
        if (
            (not torch.isfinite(init_global_ori).all())
            or (not torch.isfinite(init_t).all())
            or (not torch.isfinite(init_s).all())
        ):
            print(
                "Warning: geometry initialization produced non-finite values; "
                "using canonical defaults."
            )
            init_global_ori = torch.zeros([1, 3], device=device)
            init_t = torch.zeros([1, 3], device=device)
            init_s = torch.ones([1], device=device)

    ### If not provided (as in multiview), initialize with canonical
    if init_body_pose is None:
        init_body_pose = torch.zeros([1, fish.n_body_bones * 3], device=device)
    if init_body_bone_length is None:
        init_body_bone_length = torch.ones([1, fish.n_body_bones], device=device)

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
    vertices, global_ori_plus_pose_est, body_bone_est, scale_est, global_t_est, losses, global_ori_plus_body_pose_rest_bone_spaces, swing_twist = optimizer(
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
    fish_output = fish(global_ori_plus_pose_est[:, 0:3], global_ori_plus_pose_est[:, 3:], body_bone_est, scale_est, deform=True)

    vertices_world_est = fish_output["vertices"] + global_t_est
    keypoints_world_est = fish_output["keypoints"] + global_t_est

    return (
        vertices_world_est, 
        keypoints_world_est, 
        global_t_est, global_ori_plus_pose_est, 
        body_bone_est, scale_est, losses, 
        global_ori_plus_body_pose_rest_bone_spaces, 
        swing_twist
    )
    
    # # sanity-checking code; this is meant for skipping reconstruction
    # fish_output = fish(init_ori_plus_pose[:, 0:3], init_ori_plus_pose[:, 3:], init_body_bone_length, init_s, deform=True)
    # vertices_world_est = fish_output["vertices"] + init_t.cpu()
    # keypoints_world_est = fish_output["keypoints"] + init_t.cpu()
    # return vertices_world_est, keypoints_world_est, init_t, init_ori_plus_pose, init_body_bone_length, init_s, None
