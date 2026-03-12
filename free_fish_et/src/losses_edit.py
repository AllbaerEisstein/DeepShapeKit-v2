"""
loss functions, modified from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""

import torch
import torch.nn.functional as F

from src.geometry import perspective_projection
from typing import Optional

from torchmetrics.classification import BinaryJaccardIndex


def gmof(x, sigma):
    """
    Implementation of robust Geman-McClure function
    """
    x_squared = x**2 # this squares element-wise
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def keypoint_reprojection_loss_global(
    model_keypoints: torch.Tensor,
    proj_m: torch.Tensor,
    keypoints_2d: torch.Tensor,
    keypoints_conf: torch.Tensor,
):
    # Project model keypoints
    projected_keypoints = perspective_projection(model_keypoints, proj_m)

    # Weighted robust reprojection loss
    sigma = 50
    # this returns Geman-McClure rubustified x², y² for each kpt:
    reprojection_error = gmof(projected_keypoints - keypoints_2d, sigma) 
    # here, sum x² and y² and scale by conf -> squared L2-error per kpt:
    reprojection_loss = (keypoints_conf**2) * reprojection_error.sum(dim=-1) 

    # finally, sum error of each kpt -> we get a sum of squared, Geman-McClure robustified L2-errors per view
    total_loss = reprojection_loss.sum(dim=-1)

    # and, we return a scalar loss by summing over all views.
    # -> views contribute equally
    return total_loss.sum() 

def decompose_to_swing_twist(quats):
    """
    Args:
        quats: (BS, bn, 4) (w, x, y, z); y is the twist axis
    Returns:
        swing_twist: (BS, bn, 3) (swing_x, twist_y, swing_z)
    """
    device = quats.device
    singular_quats_mask = (quats[:, :, 0].abs() < 1e-6 and quats[:, :, 2].abs() < 1e-6).to(device) # if w and y are close to 0, we have a singularity in the swing-twist decomposition (twist axis is not well defined)
    twists = torch.zeros(quats.shape[0], quats.shape[1], 1, device=device) # (BS, bn, 1)
    twists[~singular_quats_mask] = 2*torch.atan2(quats[~singular_quats_mask][:, :, 2], quats[~singular_quats_mask][:, :, 0]).unsqueeze(-1) # atan2(y, w) gives the twist angle around the y-axis (twist axis) for non-singular quaternions
    
    beta = torch.atan2(torch.sqrt(quats[:, :, 1]**2 + quats[:, :, 3]**2), torch.sqrt(quats[:, :, 2]**2 + quats[:, :, 0]**2))
    gamma = twists[:, :, 0] / 2
    
    mtx = torch.stack([
        torch.stack([torch.cos(gamma), -torch.sin(gamma)], dim=-1),
        torch.stack([torch.sin(gamma), torch.cos(gamma)], dim=-1)
    ], dim=-2)

    def sinc(x):
        return torch.where(x.abs() < 1e-6, torch.ones_like(x), torch.sin(x) / x)
    
    swing_x_swing_z = (2/sinc(beta)).unsqueeze(-1).unsqueeze(-1) * mtx @ quats[:,:, [1, 3]].unsqueeze(-1) # (BS, bn, 2, 1)
    swing_x = swing_x_swing_z[:, :, 0, 0]
    swing_z = swing_x_swing_z[:, :, 1, 0]
    return torch.stack([swing_x, twists[:, :, 0], swing_z], dim=-1) # (BS, bn, 3) swing_x, twist_y, swing_z


def kpt_repr_plus_bone_pose_and_length_loss(
    model_keypoints: torch.Tensor,
    bone_angle_min: torch.Tensor,
    bone_angle_max: torch.Tensor,
    bone_length_min: torch.Tensor,
    bone_length_max: torch.Tensor,
    proj_m: torch.Tensor,
    keypoints_2d: torch.Tensor,
    keypoints_conf: torch.Tensor,
    body_pose: torch.Tensor,
    body_bone_ori_rest_head_spaces: torch.Tensor,
    bone_length: torch.Tensor,
    sigma=50,
    angle_constraint_weight: float = 1.0,
    smooth_weight: float = 1.0,
    bone_length_constraint_weight: float = 1.0,
    pose_init: Optional[torch.Tensor] = None,
    bone_init: Optional[torch.Tensor] = None,
):
    # Project model keypoints
    device = body_pose.device
    projected_keypoints = perspective_projection(model_keypoints, proj_m)

    # Weighted robust reprojection loss
    reprojection_error = gmof(projected_keypoints - keypoints_2d, sigma)
    reprojection_loss = (keypoints_conf**2) * reprojection_error.sum(dim=-1)

    # Joint angle limit loss
    swing_twist = decompose_to_swing_twist(body_bone_ori_rest_head_spaces).view(body_pose.shape) # (BS, bn, 3)
    angle_max_lim = bone_angle_max.repeat(1, 1).to(device)
    angle_min_lim = bone_angle_min.repeat(1, 1).to(device)
    # Add a loss if the angle is lower than min or higher than max
    lim_loss =   (swing_twist - angle_max_lim).clamp(0, float("Inf")) \
               + (angle_min_lim - swing_twist).clamp(0, float("Inf"))
    lim_loss = angle_constraint_weight * lim_loss

    # Prior Loss: difference to initialization paramaters (either from prior frame or from prior optimization stage)
    if pose_init == None or bone_init == None:
        prior_loss = body_pose.abs()
        prior_loss = smooth_weight * prior_loss
    else:
        prior_loss = (body_pose - pose_init).abs().sum() + (
            bone_length - bone_init
        ).abs().sum()
        prior_loss = smooth_weight * prior_loss

    # Bone Length Limit Loss
    max_bone = bone_length_max.repeat(1, 1).to(device)
    min_bone = bone_length_min.repeat(1, 1).to(device)
    # Add a loss if the length is lower than min or higher than max
    bone_loss =   (bone_length - max_bone).clamp(0, float("Inf")) \
                + (min_bone - bone_length).clamp(0, float("Inf"))
    bone_loss = bone_length_constraint_weight * bone_loss

    total_loss = (
        reprojection_loss.sum(dim=-1)
        + lim_loss.sum()
        + prior_loss.sum()
        + bone_loss.sum()
    )

    return total_loss.sum()


def mask_fitting_loss(proj_masks, masks, mask_weight):
    # L1 mask loss
    total_loss = F.smooth_l1_loss(proj_masks, masks, reduction="none").sum(dim=[1, 2])
    total_loss = mask_weight * total_loss

    return total_loss.sum()


def mask_jaccard_index(proj_masks, masks, mask_weight):
    metric = BinaryJaccardIndex()
    total_loss = mask_weight * metric(proj_masks, masks)
    return total_loss.sum()



def kpts_fitting_loss(
    model_keypoints,
    proj_m,
    keypoints_2d,
    keypoints_conf,
    body_pose,
    bone_length,
    prior_weight=1,
    pose_init=None,
    bone_init=None,
    sigma=100,
):
    device = body_pose.device

    # Project model keypoints
    # projected_keypoints = perspective_projection_ref(model_keypoints, rotation, camera_t, focal_length, camera_center, distortion)
    projected_keypoints = perspective_projection(model_keypoints, proj_m)

    # Weighted robust reprojection loss
    reprojection_error = gmof(projected_keypoints - keypoints_2d, sigma)
    reprojection_loss = (keypoints_conf**2) * reprojection_error.sum(dim=-1)

    # If provided pose/bone initialization, constraint objective from deviation from it
    if pose_init == None or bone_init == None:
        total_loss = reprojection_loss.sum(dim=-1)

    else:
        init_loss = (body_pose - pose_init).abs().sum() + (
            bone_length - bone_init
        ).abs().sum()
        init_loss = init_loss * prior_weight
        total_loss = reprojection_loss.sum(dim=-1) + init_loss.sum()

    return total_loss.sum()


def prior_loss(p, mean, cov_in, weight):
    # Squared Mahalanobis distance
    pm = p - mean

    dis = pm @ cov_in @ pm.t()
    dis = weight * torch.diag(dis).sum()

    return dis
