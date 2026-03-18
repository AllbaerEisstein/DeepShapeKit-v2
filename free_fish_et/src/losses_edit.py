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
        quats: (BS, bn, 4) quaternion order (w, x, y, z); y is the twist axis
    Returns:
        swing_twist: (BS, bn, 3) = (swing_x, twist_y, swing_z)
    """
    # Extract quaternion components
    w = quats[:, :, 0]  # (BS, bn)
    x = quats[:, :, 1]  # (BS, bn)
    y = quats[:, :, 2]  # (BS, bn)
    z = quats[:, :, 3]  # (BS, bn)

    # print("swing-twist")
    # print("   quats:", quats)
    # Singular when both w and y are ~0
    singular_quats_mask = (w.abs() < 1e-6) & (y.abs() < 1e-6)  # (BS, bn)

    # atan2(y, w) gives the twist angle around the y-axis (twist axis) for non-singular quaternions
    twists = torch.where(
        singular_quats_mask,
        torch.zeros_like(w),
        2 * torch.atan2(y, w)
    )  # (BS, bn)
    # print("   twists:", twists)

    # sqrt(r^2) has undefined gradient at r=0; add epsilon to keep gradients finite
    # for identity/near-identity quaternions where x=z=0 (and similarly y=w=0 edge cases).
    eps = 1e-12
    xz_norm = torch.sqrt(x**2 + z**2 + eps)
    yw_norm = torch.sqrt(y**2 + w**2 + eps)
    beta = torch.atan2(xz_norm, yw_norm)  # (BS, bn)

    gamma = twists / 2  # (BS, bn)

    # Rotation matrix of shape (BS, bn, 2, 2)
    mtx = torch.stack([
        torch.stack([torch.cos(gamma), -torch.sin(gamma)], dim=-1),
        torch.stack([torch.sin(gamma),  torch.cos(gamma)], dim=-1)
    ], dim=-2)

    def sinc(x):
        # torch.sinc(u) = sin(pi*u)/(pi*u), so sinc(x) = torch.sinc(x/pi).
        # This is smooth and finite at x=0, avoiding NaN gradients from sin(x)/x.
        return torch.sinc(x / torch.pi)

    # shape: (BS, bn, 2, 1)
    vec = quats[:, :, [1, 3]].unsqueeze(-1)

    scale = (2 / sinc(beta)).unsqueeze(-1).unsqueeze(-1)  # (BS, bn, 1, 1)

    swing_x_swing_z = scale * (mtx @ vec)  # (BS, bn, 2, 1)

    swing_x = swing_x_swing_z[:, :, 0, 0]  # (BS, bn)
    swing_z = swing_x_swing_z[:, :, 1, 0]  # (BS, bn)

    return torch.stack([swing_x, twists, swing_z], dim=-1)  # (BS, bn, 3)


def kpt_repr_plus_bone_pose_and_length_loss(
    model_keypoints: torch.Tensor,
    bone_angle_priors: torch.Tensor,
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
    swing_twist = decompose_to_swing_twist(body_bone_ori_rest_head_spaces) # (BS, bn, 3)
    bone_angle_priors_batch = bone_angle_priors[:, 1:].repeat(body_pose.shape[0], 1, 1) # (BS, body_bones, 3) repeat priors for each batch element

    # Add a loss according to the amount of violation of the angle limits

    # 0 twist loss if within limits, otherwise proportional to the squared distance to the limit
    eps = 1e-8
    twist_loss = ((swing_twist[:, :, 1] - bone_angle_priors_batch[:, :, 1]).clamp_min(eps))**2 # (BS, bn)
    # print("Twist loss:", twist_loss.sum().item())
    
    # this function checks if swing_x,swing_z are within an ellipse with the priors as radii.
    # If a prior is exactly zero, that axis is treated as locked and penalized directly without division.
    swing_x = swing_twist[:, :, 0]
    swing_z = swing_twist[:, :, 2]
    swing_x_lim = bone_angle_priors_batch[:, :, 0].abs()
    swing_z_lim = bone_angle_priors_batch[:, :, 2].abs()

    x_locked = swing_x_lim <= eps
    z_locked = swing_z_lim <= eps
    both_free = (~x_locked) & (~z_locked)
    zero = torch.zeros_like(swing_x)

    # Standard ellipse penalty for the non-degenerate case.
    ellipse_violation = torch.where(
        both_free,
        (
            (swing_x / swing_x_lim.clamp_min(eps)) ** 2
            + (swing_z / swing_z_lim.clamp_min(eps)) ** 2
            - 1
        ).clamp(0, float("Inf"))
        , zero
    )

    # Degenerate cases when one/both priors are zero: 
    # squared swing for the locked axes, and rectangular bounds for the free axis if only one is locked.
    x_locked_penalty = torch.where(x_locked, swing_x**2, zero)
    z_locked_penalty = torch.where(z_locked, swing_z**2, zero)
    z_bound_penalty = torch.where(x_locked & (~z_locked), (swing_z.abs() - swing_z_lim).clamp(0, float("Inf"))**2, zero)
    x_bound_penalty = torch.where(z_locked & (~x_locked), (swing_x.abs() - swing_x_lim).clamp(0, float("Inf"))**2, zero)

    swing_loss = (
        ellipse_violation
        + x_locked_penalty
        + z_locked_penalty
        + z_bound_penalty
        + x_bound_penalty
    )
    # print("Swing loss:", swing_loss.sum().item())
    bone_angle_prior_loss = angle_constraint_weight * (swing_loss + twist_loss).sum(dim=-1) # sum swing and twist loss, then sum over bones -> (BS,)

    # Prior Loss: difference to initialization paramaters (either from prior frame or from prior optimization stage)
    if pose_init == None or bone_init == None:
        init_prior_loss = body_pose.abs()
        init_prior_loss = smooth_weight * init_prior_loss
    else:
        init_prior_loss = (body_pose - pose_init).abs().sum() + (
            bone_length - bone_init
        ).abs().sum()
        init_prior_loss = smooth_weight * init_prior_loss

    # Bone Length Limit Loss
    max_bone = bone_length_max.repeat(1, 1).to(device)
    min_bone = bone_length_min.repeat(1, 1).to(device)
    # Add a loss if the length is lower than min or higher than max
    bone_length_prior_loss =   (bone_length - max_bone).clamp(0, float("Inf")) \
                + (min_bone - bone_length).clamp(0, float("Inf"))
    bone_length_prior_loss = bone_length_constraint_weight * bone_length_prior_loss

    # sum over batches (views)
    total_loss = (
        reprojection_loss.sum(dim=-1)
        + bone_angle_prior_loss.sum()
        + init_prior_loss.sum()
        + bone_length_prior_loss.sum()
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
