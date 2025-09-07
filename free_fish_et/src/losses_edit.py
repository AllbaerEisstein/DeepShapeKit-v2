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

from src.geometry import perspective_projection, perspective_projection_ref
import src.constants as constants
from typing import Optional


def gmof(x, sigma):
    """
    Implementation of robust Geman-McClure function
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def camera_fitting_loss(
    model_keypoints: torch.Tensor,
    proj_m: torch.Tensor,  # rotation: torch.Tensor, camera_t: torch.Tensor, focal_length: torch.Tensor, camera_center: torch.Tensor,
    keypoints_2d: torch.Tensor,
    keypoints_conf: torch.Tensor,
    distortion: Optional[torch.Tensor] = None,
):
    # Project model keypoints
    # projected_keypoints = perspective_projection_ref(model_keypoints, rotation, camera_t, focal_length, camera_center, distortion)
    projected_keypoints = perspective_projection(model_keypoints, proj_m)

    # Weighted robust reprojection loss
    sigma = 50
    reprojection_error = gmof(projected_keypoints - keypoints_2d, sigma)
    reprojection_loss = (keypoints_conf**2) * reprojection_error.sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1)

    return total_loss.sum()


def body_fitting_loss(
    model_keypoints: torch.Tensor,
    bone_angle_min: torch.Tensor,
    bone_angle_max: torch.Tensor,
    bone_length_min: torch.Tensor,
    bone_length_max: torch.Tensor,
    proj_m: torch.Tensor,  # rotation: torch.Tensor, camera_t: torch.Tensor, focal_length: torch.Tensor, camera_center: torch.Tensor,
    keypoints_2d: torch.Tensor,
    keypoints_conf: torch.Tensor,
    body_pose: torch.Tensor,
    bone_length: torch.Tensor,
    sigma=50,
    lim_weight=1,
    prior_weight=1,
    bone_weight=1,
    pose_init: Optional[torch.Tensor] = None,
    bone_init: Optional[torch.Tensor] = None,
    distortion: Optional[torch.Tensor] = None,
):
    # Project model keypoints
    device = body_pose.device
    # projected_keypoints = perspective_projection_ref(model_keypoints, rotation, camera_t, focal_length, camera_center, distortion)
    projected_keypoints = perspective_projection(model_keypoints, proj_m)

    # Weighted robust reprojection loss
    reprojection_error = gmof(projected_keypoints - keypoints_2d, sigma)
    reprojection_loss = (keypoints_conf**2) * reprojection_error.sum(dim=-1)

    # Joint angle limit loss
    angle_max_lim = bone_angle_max.repeat(1, 1).to(device)
    angle_min_lim = bone_angle_min.repeat(1, 1).to(device)
    # Add a loss if the angle is lower than min or higher than max
    lim_loss =   (body_pose - angle_max_lim).clamp(0, float("Inf")) \
               + (angle_min_lim - body_pose).clamp(0, float("Inf"))
    lim_loss = lim_weight * lim_loss

    # Prior Loss
    if pose_init == None or bone_init == None:
        prior_loss = body_pose.abs()
        prior_loss = prior_weight * prior_loss
    else:
        prior_loss = (body_pose - pose_init).abs().sum() + (
            bone_length - bone_init
        ).abs().sum()
        prior_loss = prior_weight * prior_loss

    # Bone Length Limit Loss
    max_bone = bone_length_max.repeat(1, 1).to(device)
    min_bone = bone_length_min.repeat(1, 1).to(device)
    # Add a loss if the length is lower than min or higher than max
    bone_loss =   (bone_length - max_bone).clamp(0, float("Inf")) \
                + (min_bone - bone_length).clamp(0, float("Inf"))
    bone_loss = bone_weight * bone_loss

    total_loss = (
        reprojection_loss.sum(dim=-1)
        + lim_loss.sum()
        + prior_loss.sum()
        + bone_loss.sum()
    )

    return total_loss.sum()


def kpts_fitting_loss(
    model_keypoints,
    proj_m,
    keypoints_2d,
    keypoints_conf,
    # rotation, camera_t, focal_length, camera_center,
    body_pose,
    bone_length,
    prior_weight=1,
    pose_init=None,
    bone_init=None,
    sigma=100,
    distortion=None,
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


def mask_fitting_loss(proj_masks, masks, mask_weight):
    # L1 mask loss
    total_loss = F.smooth_l1_loss(proj_masks, masks, reduction="none").sum(dim=[1, 2])
    total_loss = mask_weight * total_loss

    return total_loss.sum()


def prior_loss(p, mean, cov_in, weight):
    # Squared Mahalanobis distance
    pm = p - mean

    dis = pm @ cov_in @ pm.t()
    dis = weight * torch.diag(dis).sum()

    return dis
