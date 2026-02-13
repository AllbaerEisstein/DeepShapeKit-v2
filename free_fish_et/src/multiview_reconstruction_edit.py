from collections import defaultdict
import json
import math
import pickle
import os
import argparse
from typing import Any, List, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import BinaryJaccardIndex


import src.multiview_edit as multiview
import src.multiview_utils_edit as mutil
from src.CameraGroups import CameraGroup

from tqdm import tqdm
from src.fish_model_edit import fish_model
from src.pose_optimizer_edit import OptimizeMV
from src.Silhouette_Renderer_edit import Silhouette_Renderer
from src.dataloaders_edit import Multiview_Dataset


def setup_device(seed: int) -> str:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_rendered_views(
    outdir: str,
    instance_number: int,
    sample_index: int,
    img_filenames: list,
    mesh_vertices_after_reconstruction: torch.Tensor,
    cameras: CameraGroup,
    keypoints,
    bboxes,
) -> None:
    instance_dir = os.path.join(outdir, f"reconstruction_images_{instance_number}")
    print(f"vertices size: {mesh_vertices_after_reconstruction.size()}")
    print(f"keypoints size: {keypoints.size()}")
    _ensure_dir(instance_dir)
    imgs = torch.stack([torch.tensor(plt.imread(img_path) * 255) for img_path in img_filenames])
    imgs_with_projection = mutil.batch_render_reconstructions(
        imgs,
        mesh_vertices_after_reconstruction,
        cameras,
        kpts=keypoints,
        bboxs=bboxes,
    )
    for view_idx, img_with_projection in enumerate(imgs_with_projection):
        view_dir = os.path.join(instance_dir, f"view_{view_idx}")
        _ensure_dir(view_dir)
        save_path = os.path.join(view_dir, f"frame_{sample_index}_view_{view_idx}.png")
        plt.imsave(save_path, img_with_projection)


def _save_reconstruction_images(
    orig_image_paths: List[str],
    outdir: str,
    renderer: Silhouette_Renderer,  # Silhouette_Renderer instance
    instance_number: int,
    cameras: CameraGroup,
    reconstructed_keypoints_local: torch.Tensor,  # (1, K, 3)
    reconstructed_vertices_local: torch.Tensor,  # (1, V, 3)
    faces_from_vert_indices: torch.Tensor,  # (1, N, 3)
    global_t: torch.Tensor,  # (3)
    keypoint_names: List[str],
    view_names: List[str],
    mask_predictions: Optional[torch.Tensor] = None, # for calculating mask-reprojection IoU
    keypoint_predictions: Optional[torch.Tensor] = None, # for calculating keypoint L2 distance
    silhouette_threshold: float = 0.01,  # tiny alpha cutoff
    blend_factor: float = 0.6,           # overlay opacity (60%)
    draw_verts: bool = False,
    draw_coordinate_axes: bool = False,
    annotate_global_t: bool = True,
    annotate_keypoints_with_coords: bool = False,
    pad_to_see_full_reprojection: bool = False,
    plot_other_cameras: bool = False,
):
    """
    Render silhouettes, pad originals (zero padding) to silhouette size (centered),
    optionally extend the canvas to show out-of-frame reprojections, overlay silhouette in
    red with given blend_factor, draw keypoints (blue), optionally project the world
    coordinate axes, optionally draw other camera poses as seen in each view, and annotate
    projected world-space locations.
    This function returns quality metrics of the 
    """

    def draw_circle(
        image: np.ndarray,
        center: tuple,
        radius: int,
        color: tuple,
        *,
        thickness: int = -1,
        line_type: int = cv2.LINE_AA,
    ) -> None:
        """Rounding wrapper for cv2.circle to keep styling consistent."""
        rounded_center = tuple(int(round(c)) for c in center)
        cv2.circle(image, rounded_center, radius, color, thickness=thickness, lineType=line_type)

    def draw_text(
        image: np.ndarray,
        text: str,
        position: tuple,
        *,
        font_scale: float = 0.2,
        color: tuple = (255, 255, 255),
        thickness: int = 1,
        line_type: int = cv2.LINE_AA,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
    ) -> None:
        """Rounding wrapper for cv2.putText to reduce boilerplate."""
        anchor = tuple(int(round(c)) for c in position)
        cv2.putText(image, text, anchor, font, font_scale, color, thickness, line_type)

    silhouettes = renderer(reconstructed_vertices_local, faces_from_vert_indices, global_t)
    silhouettes_np = silhouettes.detach().cpu().numpy()
    alpha = silhouettes_np  # (N, H, W)
    n_views, H, W = alpha.shape

    metrics = {
        "mask_detection_IoU": {
            view_name: None for view_name in view_names
        },
        "orig_IoU": {
            view_name: None for view_name in view_names
        },
        "mask_IoU": {
            view_name: None for view_name in view_names
        },
        "keypoint_L2_distance": {
            view_name: {
                keypoint_name: None for keypoint_name in keypoint_names
            } for view_name in view_names
        },
    }

    assert len(orig_image_paths) == n_views, "orig_image_paths length must match rendered views"
    assert cameras.batch_size == n_views, "Camera batch size must match number of views"
    assert len(view_names) == n_views, "Number of view names must match number of views"

    keypoints_world = (reconstructed_keypoints_local + global_t).squeeze(0)
    keypoints_proj = cameras.perspective_projection_from_blworld(keypoints_world.unsqueeze(0)).detach().cpu()
    keypoints_proj = keypoints_proj.squeeze(0)
    keypoints_world_np = keypoints_world.detach().cpu().numpy()

    verts_world = (reconstructed_vertices_local + global_t).squeeze(0)
    verts_proj = cameras.perspective_projection_from_blworld(verts_world.unsqueeze(0)).detach().cpu()
    verts_proj = verts_proj.squeeze(0)

    global_t_tensor = global_t.reshape(1, 1, 3)
    if global_t_tensor.device != verts_world.device:
        global_t_tensor = global_t_tensor.to(verts_world.device)
    global_t_proj = cameras.perspective_projection_from_blworld(global_t_tensor)
    global_t_proj_np = global_t_proj.detach().cpu().numpy()
    global_t_world_np = global_t_tensor.detach().cpu().numpy().reshape(3)

    axes_projection_np = None
    axes_metadata = None
    if draw_coordinate_axes:
        origin_world = torch.tensor([0.0, 0.0, 0.0], dtype=verts_world.dtype, device=verts_world.device)
        axis_length = 50.0
        axis_dirs = torch.eye(3, dtype=verts_world.dtype, device=verts_world.device)
        ticks_per_unit = 4
        tick_spacing = 1.0 / ticks_per_unit
        max_tick_count = int(np.ceil(axis_length / tick_spacing))
        tick_values_positive = [(i + 1) * tick_spacing for i in range(max_tick_count)]

        world_points = [origin_world]
        axes_metadata = {
            "origin_idx": 0,
            "axes": {},
            "axis_length": axis_length,
            "tick_spacing": tick_spacing,
        }

        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            axis_vec = axis_dirs[axis_idx]
            axes_metadata["axes"][axis_name] = {}

            for direction_name, direction_sign in (("positive", 1.0), ("negative", -1.0)):
                end_point = origin_world + axis_vec * axis_length * direction_sign
                world_points.append(end_point)
                end_idx = len(world_points) - 1

                tick_indices = []
                tick_values = []
                for tick_value in tick_values_positive:
                    signed_tick_value = tick_value * direction_sign
                    if abs(signed_tick_value) > axis_length + 1e-6:
                        continue
                    tick_point = origin_world + axis_vec * signed_tick_value
                    world_points.append(tick_point)
                    tick_indices.append(len(world_points) - 1)
                    tick_values.append(signed_tick_value)

                axes_metadata["axes"][axis_name][direction_name] = {
                    "end_idx": end_idx,
                    "tick_indices": tick_indices,
                    "tick_values": tick_values,
                }

        axis_points_tensor = torch.stack(world_points, dim=0).unsqueeze(0)
        axes_projection = cameras.perspective_projection_from_blworld(axis_points_tensor)
        axes_projection_np = axes_projection.detach().cpu().numpy()

    other_camera_markers_proj = None
    if plot_other_cameras:
        from_bl_t = cameras.from_blenderworld.transpose(1, 2)
        camera_centers_custom = cameras.camera_centers
        camera_centers_bl = torch.matmul(from_bl_t, camera_centers_custom.unsqueeze(-1)).squeeze(-1)

        axis_length_world = 1.0
        if n_views > 1:
            pairwise_dists = torch.cdist(camera_centers_bl, camera_centers_bl)
            valid_dists = pairwise_dists[pairwise_dists > 1e-6]
            if valid_dists.numel() > 0:
                axis_length_world = float(torch.median(valid_dists).item() * 0.1)
        axis_length_world = max(1e-3, min(axis_length_world, 1e3))

        camera_axes_custom = cameras.R.transpose(1, 2)
        camera_axes_bl = torch.matmul(from_bl_t, camera_axes_custom)
        camera_axis_endpoints_bl = (
            camera_centers_bl.unsqueeze(1) + camera_axes_bl.transpose(1, 2) * axis_length_world
        )
        # Marker order per source camera: center, +x, +y, +z.
        marker_points_bl = torch.cat(
            [camera_centers_bl.unsqueeze(1), camera_axis_endpoints_bl], dim=1
        ).reshape(-1, 3)
        marker_points_bl_for_views = marker_points_bl.unsqueeze(0).expand(n_views, -1, -1)
        other_camera_markers_proj = cameras.perspective_projection_from_blworld(
            marker_points_bl_for_views
        ).detach().cpu()

    max_canvas_size = 10000

    for view_idx in range(n_views):
        # Read original image (BGR)
        orig_bgr = cv2.imread(str(orig_image_paths[view_idx]), cv2.IMREAD_COLOR)
        if orig_bgr is None:
            raise FileNotFoundError(f"Could not read image {orig_image_paths[view_idx]}")

        # cv2 reads images in row-major order
        h0, w0 = orig_bgr.shape[:2]
        # Compute integer padding to center the resized image within (W,H)
        pad_left = (W - w0) // 2
        pad_right = W - w0 - pad_left
        pad_top = (H - h0) // 2
        pad_bottom = H - h0 - pad_top

        # Pad with zeros (black). cv2.copyMakeBorder expects ints
        padded = cv2.copyMakeBorder(
            orig_bgr,
            top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        extra_pad_left = 0
        extra_pad_right = 0
        extra_pad_top = 0
        extra_pad_bottom = 0
        offset_x = 0
        offset_y = 0

        if pad_to_see_full_reprojection:
            coords_parts = [
                keypoints_proj[view_idx, :, :2],
                verts_proj[view_idx, :, :2],
            ]
            if plot_other_cameras and other_camera_markers_proj is not None:
                marker_coords = other_camera_markers_proj[view_idx, :, :2].reshape(n_views, 4, 2)
                if n_views > 1:
                    marker_coords_other_views = torch.cat(
                        [marker_coords[:view_idx], marker_coords[view_idx + 1 :]], dim=0
                    ).reshape(-1, 2)
                    coords_parts.append(marker_coords_other_views)
            coords = torch.cat(coords_parts, dim=0)
            finite_mask = torch.isfinite(coords).all(dim=1)
            if torch.any(finite_mask):
                coords = torch.round(coords[finite_mask])
                min_xy = coords.min(dim=0).values
                max_xy = coords.max(dim=0).values
                min_x, min_y = int(min_xy[0].item()), int(min_xy[1].item())
                max_x, max_y = int(max_xy[0].item()), int(max_xy[1].item())

                extra_pad_left = max(0, -min_x)
                extra_pad_top = max(0, -min_y)
                extra_pad_right = max(0, max_x - (W - 1))
                extra_pad_bottom = max(0, max_y - (H - 1))

                max_extra_w = max(0, max_canvas_size - W)
                max_extra_h = max(0, max_canvas_size - H)
                required_extra_w = extra_pad_left + extra_pad_right
                required_extra_h = extra_pad_top + extra_pad_bottom

                if required_extra_w > max_extra_w and required_extra_w > 0:
                    scale = max_extra_w / required_extra_w
                    extra_pad_left = int(math.floor(extra_pad_left * scale))
                    extra_pad_right = max_extra_w - extra_pad_left

                if required_extra_h > max_extra_h and required_extra_h > 0:
                    scale = max_extra_h / required_extra_h
                    extra_pad_top = int(math.floor(extra_pad_top * scale))
                    extra_pad_bottom = max_extra_h - extra_pad_top

            if any((extra_pad_left, extra_pad_right, extra_pad_top, extra_pad_bottom)):
                padded = cv2.copyMakeBorder(
                    padded,
                    top=extra_pad_top,
                    bottom=extra_pad_bottom,
                    left=extra_pad_left,
                    right=extra_pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
                offset_x = extra_pad_left
                offset_y = extra_pad_top

        padded_gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        _, padded_binary = cv2.threshold(padded_gray, 0, 1, cv2.THRESH_BINARY)

        a = np.clip(alpha[view_idx], 0.0, 1.0)  # (H,W), float
        # threshold tiny values
        a[a < silhouette_threshold] = 0.0
        if any((extra_pad_left, extra_pad_right, extra_pad_top, extra_pad_bottom)):
            a = np.pad(
                a,
                ((extra_pad_top, extra_pad_bottom), (extra_pad_left, extra_pad_right)),
                mode="constant",
            )

        # calculate IoUs
        device="cuda" if torch.cuda.is_available() else "cpu"
        reprojection_mask_binary = torch.tensor(cv2.threshold(a, 0, 1, cv2.THRESH_BINARY)[1], device=device)
        metric = BinaryJaccardIndex().to(device=device)

        orig_iou = metric(
            torch.tensor(padded_binary, device=device),
            reprojection_mask_binary
        )
        if mask_predictions is not None:
            mask_pred_view = mask_predictions[view_idx].to(device=device)
            if any((extra_pad_left, extra_pad_right, extra_pad_top, extra_pad_bottom)):
                mask_pred_view = torch.nn.functional.pad(
                    mask_pred_view,
                    (extra_pad_left, extra_pad_right, extra_pad_top, extra_pad_bottom),
                )
            mask_detection_iou = metric(
                torch.tensor(padded_binary, device=device),
                mask_pred_view
            ) # to evaluate the quality of yolo mask detection, given we are working with synthetic data
            mask_iou = metric(
                mask_pred_view,
                reprojection_mask_binary
            )
            metrics["mask_detection_IoU"][view_names[view_idx]] = mask_detection_iou.item()
            metrics["mask_IoU"][view_names[view_idx]] = mask_iou.item()
        metrics["orig_IoU"][view_names[view_idx]] = orig_iou.item()


        # Build red overlay image (BGR) and blend: out = orig*(1 - bf * a) + red*(bf * a)
        red_img = np.zeros_like(padded)
        red_img[:, :, 2] = 255  # full red channel in BGR


        a_exp = a[..., None] # add empty axis/dimension to a
        blended = (padded.astype(np.float32) * (1.0 - blend_factor * a_exp) +
                   red_img.astype(np.float32) * (blend_factor * a_exp))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        blended_h, blended_w = blended.shape[:2]

        # Draw keypoints:
        for kp_idx, name in enumerate(keypoint_names):
            # u,v are pixel coords in the same coordinate system as the silhouette (W,H)
            kp_u = keypoints_proj[view_idx, kp_idx, 0]
            kp_v = keypoints_proj[view_idx, kp_idx, 1]
            if not (torch.isfinite(kp_u) and torch.isfinite(kp_v)):
                continue
            ui = int(round(kp_u.item())) + offset_x
            vi = int(round(kp_v.item())) + offset_y
            if 0 <= ui < blended_w and 0 <= vi < blended_h:
                draw_circle(blended, (ui, vi), radius=5, color=(255, 150, 0))
                draw_text(blended, name, (ui, vi + 15), color=(255, 150, 0))
                if annotate_keypoints_with_coords:
                    kp_coords = keypoints_world_np[kp_idx]
                    coord_text = f"({kp_coords[0]:.2f}, {kp_coords[1]:.2f}, {kp_coords[2]:.2f})"
                    draw_text(
                        blended,
                        coord_text,
                        (ui, vi + 30),
                        color=(255, 255, 255),
                    )
            if keypoint_predictions is not None:
                pred_u = keypoint_predictions[view_idx, kp_idx, 0]
                pred_v = keypoint_predictions[view_idx, kp_idx, 1]
                if not (torch.isfinite(pred_u) and torch.isfinite(pred_v)):
                    continue
                ui_pred = int(round(pred_u.item())) + offset_x
                vi_pred = int(round(pred_v.item())) + offset_y
                ci_pred = keypoint_predictions[view_idx, kp_idx, 2].item()
                if ci_pred > 0:
                    conf_scaled_annot_radius = int(ci_pred*10)//2
                    cv2.circle(blended, (ui_pred, vi_pred), radius=1, color=(255,0,0), lineType=-1)
                    cv2.circle(
                        blended,
                        (ui_pred, vi_pred),
                        radius=conf_scaled_annot_radius,
                        color=(255,0,0),
                        lineType=-1
                    )
                    namelen = len(name)
                    draw_text(
                        blended,
                        f"{name[:min(namelen-1,4)]}: {int(ci_pred*100)/100}",
                        (ui_pred, vi_pred + conf_scaled_annot_radius - 20),
                        color=(255,0,0),
                    )
                metrics["keypoint_L2_distance"][view_names[view_idx]][keypoint_names[kp_idx]] = (
                    ( (ui - ui_pred)**2 + (vi - vi_pred)**2 )**0.5
                )

        if draw_verts:
            for vert_idx in range(verts_proj.size(1)):
                ui = int(round(verts_proj[view_idx, vert_idx, 0].item())) + offset_x
                vi = int(round(verts_proj[view_idx, vert_idx, 1].item())) + offset_y
                if 0 <= ui < blended_w and 0 <= vi < blended_h:
                    draw_circle(blended, (ui, vi), radius=2, color=(0, 255, 0))

        if annotate_global_t:
            gt_pt = global_t_proj_np[view_idx, 0]
            if np.all(np.isfinite(gt_pt)):
                ui_gt = int(round(float(gt_pt[0]))) + offset_x
                vi_gt = int(round(float(gt_pt[1]))) + offset_y
                if 0 <= ui_gt < blended_w and 0 <= vi_gt < blended_h:
                    draw_circle(
                        blended,
                        (ui_gt, vi_gt),
                        radius=6,
                        color=(0, 255, 255),
                    )
                    draw_text(
                        blended,
                        "global_t",
                        (ui_gt + 6, vi_gt - 6),
                        color=(0, 255, 255),
                    )
                    coord_text = (
                        f"({global_t_world_np[0]:.2f}, {global_t_world_np[1]:.2f}, {global_t_world_np[2]:.2f})"
                    )
                    draw_text(
                        blended,
                        coord_text,
                        (ui_gt + 6, vi_gt + 10),
                    )

        if plot_other_cameras and other_camera_markers_proj is not None:
            axis_labels = ["x", "y", "z"]
            axis_colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0)]  # BGR
            for source_cam_idx in range(n_views):
                if source_cam_idx == view_idx:
                    continue
                base_idx = source_cam_idx * 4
                center_pt = other_camera_markers_proj[view_idx, base_idx]
                if not torch.isfinite(center_pt).all():
                    continue

                center_ui = int(round(float(center_pt[0]))) + offset_x
                center_vi = int(round(float(center_pt[1]))) + offset_y
                square_half_size = 4
                cv2.rectangle(
                    blended,
                    (center_ui - square_half_size, center_vi - square_half_size),
                    (center_ui + square_half_size, center_vi + square_half_size),
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

                for axis_idx, (axis_label, axis_color) in enumerate(zip(axis_labels, axis_colors)):
                    axis_pt = other_camera_markers_proj[view_idx, base_idx + 1 + axis_idx]
                    if not torch.isfinite(axis_pt).all():
                        continue
                    axis_ui = int(round(float(axis_pt[0]))) + offset_x
                    axis_vi = int(round(float(axis_pt[1]))) + offset_y
                    cv2.arrowedLine(
                        blended,
                        (center_ui, center_vi),
                        (axis_ui, axis_vi),
                        color=axis_color,
                        thickness=1,
                        line_type=cv2.LINE_AA,
                        tipLength=0.35,
                    )
                    draw_text(
                        blended,
                        axis_label,
                        (axis_ui + 2, axis_vi - 2),
                        font_scale=0.35,
                        color=axis_color,
                    )

        if draw_coordinate_axes and axes_projection_np is not None and axes_metadata is not None:
            origin_idx = axes_metadata["origin_idx"]
            origin_pt = axes_projection_np[view_idx, origin_idx]
            if np.all(np.isfinite(origin_pt)):
                offset_vec = np.array([offset_x, offset_y], dtype=np.float32)
                origin_pt = origin_pt + offset_vec
                overlay = blended.copy()
                axis_length_world = axes_metadata["axis_length"]

                for axis_name, axis_meta in axes_metadata["axes"].items():
                    for direction_name, direction_meta in axis_meta.items():
                        end_pt = axes_projection_np[view_idx, direction_meta["end_idx"]]
                        if not np.all(np.isfinite(end_pt)):
                            continue
                        end_pt = end_pt + offset_vec

                        p0 = origin_pt.astype(np.float32)
                        p1 = end_pt.astype(np.float32)
                        axis_vec_px = p1 - p0
                        axis_len_px = np.linalg.norm(axis_vec_px)
                        if axis_len_px < 1e-3:
                            continue

                        axis_dir_unit = axis_vec_px / axis_len_px
                        perp_dir_unit = np.array([-axis_dir_unit[1], axis_dir_unit[0]], dtype=np.float32)
                        tick_length_px = min(4.0, axis_len_px * 0.05)
                        axis_line_thickness = 2 if direction_name == "positive" else 1

                        cv2.line(
                            overlay,
                            tuple(np.round(p0).astype(int)),
                            tuple(np.round(p1).astype(int)),
                            color=(255, 255, 255),
                            thickness=axis_line_thickness,
                            lineType=cv2.LINE_AA,
                        )

                        for tick_value, tick_idx in zip(direction_meta["tick_values"], direction_meta["tick_indices"]):
                            tick_pt = axes_projection_np[view_idx, tick_idx]
                            if not np.all(np.isfinite(tick_pt)):
                                continue
                            tick_center = (tick_pt + offset_vec).astype(np.float32)
                            offset = perp_dir_unit * (tick_length_px * 0.5)
                            tick_start = tick_center - offset
                            tick_end = tick_center + offset
                            cv2.line(
                                overlay,
                                tuple(np.round(tick_start).astype(int)),
                                tuple(np.round(tick_end).astype(int)),
                                color=(255, 255, 255),
                                thickness=1,
                                lineType=cv2.LINE_AA,
                            )

                            text_value = f"{tick_value:.2f}".rstrip("0").rstrip(".")

                            if abs(tick_value) < 1e-6:
                                text_value = "0"

                            text_anchor = tick_center + perp_dir_unit * (tick_length_px + 6.0)
                            draw_text(
                                overlay,
                                text_value,
                                text_anchor,
                            )

                        label_direction = axis_dir_unit if direction_name == "positive" else -axis_dir_unit
                        label_anchor = p1 + label_direction * 12.0 + perp_dir_unit * 6.0
                        axis_label = axis_name.upper() if direction_name == "positive" else f"-{axis_name.upper()}"
                        draw_text(
                            overlay,
                            axis_label,
                            label_anchor,
                            font_scale=0.4,
                        )

                origin_label_pos = origin_pt.astype(np.float32) + np.array([6.0, -6.0], dtype=np.float32)
                draw_text(overlay, "0", origin_label_pos, font_scale=0.35)

                blended = cv2.addWeighted(overlay, 0.6, blended, 0.4, 0)

        # Prepare output dirs & filename
        view_output_dir = os.path.join(outdir, view_names[view_idx] + "_reconstruction_images")
        instance_dir = os.path.join(view_output_dir, f"instance_{instance_number}")
        _ensure_dir(instance_dir)
        base_name = os.path.splitext(os.path.basename(orig_image_paths[view_idx]))[0]
        out_path = os.path.join(instance_dir, base_name + "_reconstructed.png")

        # Save PNG (BGR)
        cv2.imwrite(out_path, blended)

    return metrics




def _save_obj_model(
    outdir: str, sample_index: int, fish_place: int, vertex_posed: torch.Tensor, fish
) -> None:
    model_dir = os.path.join(outdir, "models")
    _ensure_dir(model_dir)

    obj_path = os.path.join(model_dir, f"frame_{sample_index}_out_model_{fish_place}.obj")
    with open(obj_path, "w") as f:
        # vertices
        verts = vertex_posed[0]
        for x, y, z in verts:
            f.write(f"v {x} {y} {z}\n")
        # faces
        for a, b, c in fish.faces + 1:
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def _save_pose_pickle(
    outdir: str,
    start: int,
    end: int,
    fish_place: int,
    parameters: list,
    sample_data: list,
    mesh_file: str,
    index: list,
) -> None:
    pickle_dir = os.path.join(outdir, "pose_pickle")
    _ensure_dir(pickle_dir)
    fname = f"pose_result_{start}-{end+1}_({fish_place}).pickle"
    path = os.path.join(pickle_dir, fname)

    data = {
        "individual_fit_parameters": parameters,
        "sample_data": sample_data,
        "indices": index,
        "model_file": mesh_file,
    }
    with open(path, "wb") as pf:
        pickle.dump(data, pf, protocol=pickle.HIGHEST_PROTOCOL)


def _get_reconstruction_cache_paths(
    outdir: str,
    dataset_dir: str,
    instance_number: int,
) -> tuple[str, str]:
    cache_dir = os.path.join(outdir, "reconstruction_cache")
    dataset_name = os.path.basename(os.path.normpath(dataset_dir)) or "dataset"
    filename = f"cache_{dataset_name}_instance_{instance_number}.pickle"
    return cache_dir, os.path.join(cache_dir, filename)


def _load_reconstruction_cache(cache_path: str) -> Optional[dict]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        print(f"Failed to load reconstruction cache '{cache_path}': {exc}")
        return None


def _save_reconstruction_cache(
    cache_dir: str,
    cache_path: str,
    cache_payload: dict,
) -> None:
    _ensure_dir(cache_dir)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump(cache_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)


def _clear_reconstruction_cache(cache_path: str) -> None:
    try:
        if os.path.exists(cache_path):
            os.remove(cache_path)
    except Exception as exc:
        print(f"Warning: failed to remove reconstruction cache '{cache_path}': {exc}")


def _save_pose_time_series_json(
    outdir: str,
    dataset_dir: str,
    mesh_path: str,
    fish: fish_model,
    instance_number: int,
    frame_payloads: list[dict],
    dataset_meta: dict,
) -> None:
    if not frame_payloads:
        return

    pose_dir = os.path.join(outdir, "pose_time_series")
    _ensure_dir(pose_dir)

    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    filename = f"pose_time_series_{dataset_name}_instance_{instance_number}.json"
    path = os.path.join(pose_dir, filename)

    processed_frames = sorted(payload["frame"] for payload in frame_payloads)
    frame_start = processed_frames[0]
    frame_end = processed_frames[-1]

    fps = dataset_meta.get("fps")
    if fps is not None:
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = None

    meta = {
        "source": "multiview_reconstruction_edit.py",
        "dataset_dir": dataset_dir,
        "dataset_name": dataset_name,
        "instance": int(instance_number),
        "bone_order": fish.dd.get("bone_order", []),
        "virtual_bone_names": fish.dd.get("virtual_bone_names", []),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "frame_indices": processed_frames,
        "mesh_file": mesh_path,
    }

    if fps is not None:
        meta["fps"] = fps

    if fps is not None and fps > 0:
        def time_from_frame(frame_id: int) -> float:
            return (frame_id - frame_start) / fps
    else:
        def time_from_frame(frame_id: int):
            return None

    frames = []
    for payload in frame_payloads:
        frame_dict = {
            "frame": int(payload["frame"]),
            "global_t": payload["global_t"],
            "global_ori": payload["global_ori"],
            "body_pose": payload["body_pose"],
            "body_bone_length": payload["body_bone_length"],
        }
        if payload.get("scale") is not None:
            frame_dict["scale"] = payload["scale"]
        time_val = time_from_frame(frame_dict["frame"])
        if time_val is None:
            time_val = float(frame_dict["frame"] - frame_start)
        frame_dict["time"] = time_val
        frames.append(frame_dict)

    payload = {
        "meta": meta,
        "frames": frames,
    }

    with open(path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, indent=2)


def reconstruct(
    mesh_path: str,
    dataset_dir: str,
    outdir: str,
    frame_indices: list[int],
    instance_number: int,
    seed: int = 1,
    save_models: bool = False,
    video_names: Optional[List[str]] = None,
    pause_event: Optional[Any] = None,
    center_origin_on_camera_mean: bool = False,
) -> None:
    """
    Run multiview reconstruction for given frames.
    """

    _ensure_dir(outdir)

    # --------------------------
    # setup; instantiate classes
    device = setup_device(seed)

    print("Starting multiview reconstruction with the following settings:")
    print("Device:", device)

    dataset = Multiview_Dataset(root=dataset_dir, views=video_names)
    camera_group_uniform_size_cpu = dataset.cams.get_camera_group().with_intrinsics_adjusted_for_uniform_image_size()
    if center_origin_on_camera_mean:
        camera_centers_custom = camera_group_uniform_size_cpu.camera_centers
        from_bl_inv = camera_group_uniform_size_cpu.from_blenderworld.transpose(1, 2)
        camera_centers_bl = torch.matmul(from_bl_inv, camera_centers_custom.unsqueeze(-1)).squeeze(-1)
        mean_center_bl = camera_centers_bl.mean(dim=0)
        transform = torch.eye(4, dtype=camera_group_uniform_size_cpu.K.dtype)
        transform[:3, 3] = -mean_center_bl
        camera_group_uniform_size_cpu = camera_group_uniform_size_cpu.transform_blenderworld(transform)
    camera_group_uniform_size_device = camera_group_uniform_size_cpu.to(device)

    fish = fish_model(mesh_json_path=mesh_path)
    optimizer = OptimizeMV(
        num_iters=100,
        lim_weight=200,
        prior_weight=30,
        bone_weight=200,
        mask_weight=200,
        smooth_weights=[100, 100, 20],
        device=torch.device(device),
        fish_model_obj=fish,
    )
    renderer = Silhouette_Renderer(device, camera_group_uniform_size_device)

    # --------------------------
    # load cache, if available
    cache_dir, cache_path = _get_reconstruction_cache_paths(outdir, dataset_dir, instance_number)
    cache_data = _load_reconstruction_cache(cache_path)

    parameters = []
    sample_data = []
    pose_time_series_frames: list[dict] = []
    processed_frames: set[int] = set()
    cached_metrics: Optional[dict] = None

    if cache_data is not None:
        cache_dataset = cache_data.get("dataset_dir")
        cache_mesh = cache_data.get("mesh_path")
        cache_frames = cache_data.get("frame_indices")
        if (
            (cache_dataset and cache_dataset != dataset_dir)
            or (cache_mesh and cache_mesh != mesh_path)
            or (cache_frames and list(cache_frames) != list(frame_indices))
        ):
            print("Existing reconstruction cache does not match current configuration; starting a new reconstruction run.")
        else:
            print(f"Resuming reconstruction from cache '{cache_path}'.")
            parameters = cache_data.get("parameters", [])
            sample_data = cache_data.get("sample_data", [])
            pose_time_series_frames = cache_data.get("pose_time_series_frames", []) or []
            cached_metrics = cache_data.get("metrics")
            processed_frames = set(cache_data.get("processed_frames", []))

    def _fresh_metrics() -> dict:
        return {
            "mask_detection_IoU": {view_name: [] for view_name in dataset.views},
            "orig_IoU": {view_name: [] for view_name in dataset.views},
            "mask_IoU": {view_name: [] for view_name in dataset.views},
            "keypoint_L2_distance": {
                view_name: {kpt: [] for kpt in dataset.index_json["keypoint_list"]}
                for view_name in dataset.views
            },
        }

    metrics = cached_metrics if cached_metrics is not None else _fresh_metrics()

    if cached_metrics is not None:
        cached_views = set(cached_metrics.get("orig_IoU", {}).keys())
        if cached_views != set(dataset.views):
            print("Cached reconstruction metrics do not match current dataset views; restarting reconstruction run.")
            parameters = []
            sample_data = []
            pose_time_series_frames = []
            processed_frames = set()
            metrics = _fresh_metrics()

    pbar = tqdm(
        total=len(frame_indices),
        desc=f"{os.path.basename(dataset_dir)} reconstruction frame {frame_indices[0]}",
        initial=len(processed_frames),
    )

    # --------------------------
    # loop through frames and reconstruct
    for idx in frame_indices:
        if idx in processed_frames:
            continue
        
        # --------------------------
        # load from dataset
        try:
            instance_sample = dataset.__getitem__(idx, instance_number)
        except IndexError:
            print(f"Sample {idx} missing, skipping")
            pbar.update()
            continue

        kpt_present_mask = instance_sample['kpt_present_mask']
        seg_mask_present_mask = instance_sample['seg_mask_present_mask']

        # QUESTION: how to deal with not enough keypoints/segmasks especially in first frame?
        if len([
                view_with_seg_mask for view_with_seg_mask in seg_mask_present_mask 
                if view_with_seg_mask == True
            ]) < 2:
            print(f"Less than two views with segmentation masks in sample for frame {idx} -> skipping")
            pbar.update()
            continue
        if len([
                view_with_kpts for view_with_kpts in kpt_present_mask 
                if any(kpt_present == True for kpt_present in view_with_kpts)
            ]) < 2:
            print(f"Less than two views with keypoints in sample for frame {idx} -> skipping")
            pbar.update()
            continue

        views_indices, orig_img_paths = instance_sample['frames'], instance_sample['imgpaths']

        # mask, bboxes, keypoint are specified in uniform_image_size coordinates (adjustment happened in dataset creation)
        keypoints = instance_sample["keypoints"]
        masks = instance_sample["masks_full"]
        bboxes = instance_sample["bboxes"]
        # Normalize mask to [0,1] on appropriate device
        masks = masks / 255.0
        keypoints = keypoints
        bboxes = bboxes


        # --------------------------
        # reconstruct

        # initialize from previous solution if available
        init = parameters[-1] if parameters else None

        result = multiview.fit_mesh(
            fish,
            optimizer,
            camera_group_uniform_size_device,
            keypoints,
            masks,
            renderer,
            device,
            *([] if init is None else init),
            index=idx,
            bboxs=bboxes,
        )
        vertices_world_est, keypoints_world_est, global_t_est, global_ori_plus_pose_est, body_bone_est, scale_est, _ = result

        # --------------------------
        # cache results
        parameters.append([global_ori_plus_pose_est[:, :3], global_ori_plus_pose_est[:, 3:], body_bone_est, scale_est, global_t_est])
        sample_data.append([views_indices, orig_img_paths, keypoints_world_est, bboxes, idx])

        out_reconstructed = fish(global_ori_plus_pose_est[:, :3], global_ori_plus_pose_est[:, 3:], body_bone_est, scale_est)
        reconstructed_keypoints_local = out_reconstructed["keypoints"].to(device)
        reconstructed_vertices_local = out_reconstructed["vertices"].to(device)

        global_ori_cpu = global_ori_plus_pose_est[:, :3].detach().cpu().view(-1).tolist()
        body_pose_flat = global_ori_plus_pose_est[:, 3:].detach().cpu().view(-1).tolist()
        body_pose_triplets = [
            body_pose_flat[i : i + 3]
            for i in range(0, len(body_pose_flat), 3)
        ]
        body_bone_lengths = body_bone_est.detach().cpu().view(-1).tolist()
        global_t_list = global_t_est.detach().cpu().view(-1).tolist()
        scale_list = scale_est.detach().cpu().view(-1).tolist()
        pose_time_series_frames.append(
            {
                "frame": int(idx),
                "global_ori": [float(x) for x in global_ori_cpu],
                "body_pose": [[float(v) for v in triple] for triple in body_pose_triplets],
                "body_bone_length": [float(v) for v in body_bone_lengths],
                "global_t": [float(v) for v in global_t_list],
                "scale": float(scale_list[0]) if scale_list else None,
            }
        )

        frame_metrics = _save_reconstruction_images(
            orig_image_paths=orig_img_paths,
            outdir=outdir,
            renderer=renderer,
            instance_number=instance_number,
            cameras=camera_group_uniform_size_device,
            reconstructed_keypoints_local=reconstructed_keypoints_local,
            reconstructed_vertices_local=reconstructed_vertices_local,
            faces_from_vert_indices=fish.faces.unsqueeze(0).to(device),
            global_t=global_t_est.to(device),
            keypoint_names=dataset.index_json["keypoint_list"],
            view_names=dataset.views,
            mask_predictions=masks,
            keypoint_predictions=keypoints
        )

        if save_models:
            _save_obj_model(outdir, idx, instance_number, vertices_world_est, fish)
        

        # cache quality metrics for this frame
        for view in dataset.views:
            metrics["mask_detection_IoU"][view].append(frame_metrics["mask_detection_IoU"][view])
            metrics["orig_IoU"][view].append(frame_metrics["orig_IoU"][view])
            metrics["mask_IoU"][view].append(frame_metrics["mask_IoU"][view])
            for kpt in dataset.index_json["keypoint_list"]:
                metrics["keypoint_L2_distance"][view][kpt].append(frame_metrics["keypoint_L2_distance"][view][kpt])

        processed_frames.add(idx)

        cache_payload = {
            "parameters": parameters,
            "sample_data": sample_data,
            "metrics": metrics,
            "pose_time_series_frames": pose_time_series_frames,
            "processed_frames": sorted(processed_frames),
            "last_frame": idx,
            "frame_indices": list(frame_indices),
            "dataset_dir": dataset_dir,
            "mesh_path": mesh_path,
            "instance_number": instance_number,
        }
        _save_reconstruction_cache(cache_dir, cache_path, cache_payload)

        if pause_event is not None and pause_event.is_set():
            print("Pause requested; stopping after cache write.")
            break

        pbar.desc = f"{os.path.basename(dataset_dir)} reconstruction frame {idx}"
        pbar.update()

    # reconstruction done
    metrics['total_duration_min'] = str(pbar).split()[-2].split('<')[0].replace('[','')
    metrics['seconds_per_frame'] = str(pbar).split()[-1].replace(']','')
    pbar.close()

    _save_pose_pickle(
        outdir=outdir,
        start=frame_indices[0],
        end=frame_indices[-1],
        fish_place=instance_number,
        parameters=parameters,
        sample_data=sample_data,
        mesh_file=mesh_path,
        index=frame_indices,
    )

    _save_pose_time_series_json(
        outdir=outdir,
        dataset_dir=dataset_dir,
        mesh_path=mesh_path,
        fish=fish,
        instance_number=instance_number,
        frame_payloads=pose_time_series_frames,
        dataset_meta=dataset.index_json,
    )

    with open(os.path.join(outdir, f"metrics_instance_{instance_number}.json"), "w") as metrics_out_json:
        json.dump(metrics, metrics_out_json, indent=4)

    _clear_reconstruction_cache(cache_path)



def render_pose_time_series(    
    mesh_path: str,
    dataset_dir: str,
    pose_time_series_file_path: str,
    outdir: str,
    deform: bool = False,
    frame_range: Optional[List[int]] = None,
    offset_by_frame_range_start: bool = False,
    center_origin_on_camera_mean: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fish = fish_model(mesh_path)
    fish.to_device(device)

    dataset = Multiview_Dataset(root=dataset_dir)
    camera_group_cpu = dataset.cams.get_camera_group().with_intrinsics_adjusted_for_uniform_image_size()
    if center_origin_on_camera_mean:
        camera_centers_custom = camera_group_cpu.camera_centers
        from_bl_inv = camera_group_cpu.from_blenderworld.transpose(1, 2)
        camera_centers_bl = torch.matmul(from_bl_inv, camera_centers_custom.unsqueeze(-1)).squeeze(-1)
        mean_center_bl = camera_centers_bl.mean(dim=0)
        transform = torch.eye(4, dtype=camera_group_cpu.K.dtype)
        transform[:3, 3] = -mean_center_bl
        camera_group_cpu = camera_group_cpu.transform_blenderworld(transform)
    camera_group_device = camera_group_cpu.to(device)

    renderer = Silhouette_Renderer(device, camera_group_device)

    pose_time_series_outdir = os.path.join(outdir, "pose_time_series_rendered")
    _ensure_dir(pose_time_series_outdir)
    with open(pose_time_series_file_path) as jf:
        pose_time_series_json = json.load(jf)
    frames = pose_time_series_json["frames"]

    frame_range_list = frame_range or []
    range_start = min(frame_range_list) if frame_range_list else 0
    range_end = max(frame_range_list) if frame_range_list else None
    offset = range_start if (offset_by_frame_range_start and frame_range_list) else 0

    if frame_range_list:
        frame_set = set(frame_range_list)
        pose_end = max((int(frame.get("frame", 0)) for frame in frames), default=-1) + offset
        effective_end = min(range_end, pose_end) if range_end is not None else pose_end
        frames = [
            frame
            for frame in frames
            if (int(frame.get("frame", 0)) + offset) in frame_set
            and (int(frame.get("frame", 0)) + offset) <= effective_end
        ]

    for frame in frames:
        frame_id = int(frame.get("frame", 0)) + offset
        sample = dataset.__getitem__(frame_id)

        global_t = torch.tensor(frame["global_t"], device=device)
        global_ori = torch.tensor(frame["global_ori"], device=device)
        body_pose = torch.tensor(frame["body_pose"], device=device)
        bone_length = torch.tensor(frame["body_bone_length"], device=device)

        # articulated_verts_kpts = fish(global_ori.unsqueeze(0), body_pose.unsqueeze(0).flatten(1), bone_length.unsqueeze(0), deform=deform)
        # articulated_verts_kpts = fish(torch.zeros_like(global_ori.unsqueeze(0), device=device), body_pose.unsqueeze(0).flatten(1), bone_length.unsqueeze(0), deform=deform)
        articulated_verts_kpts = fish(global_ori.unsqueeze(0), torch.zeros_like(body_pose.unsqueeze(0).flatten(1), device=device), bone_length.unsqueeze(0), deform=deform)
        keypoints = articulated_verts_kpts["keypoints"].to(device)
        vertices = articulated_verts_kpts["vertices"].to(device)

        # silhouettes = renderer(vertices, fish.faces.unsqueeze(0), global_t)
        # silhouettes_np = silhouettes.detach().cpu().numpy()

        # for i in sample["frames"]:
        #     base_name = os.path.splitext(os.path.basename(sample["imgpaths"][i]))[0]
        #     out_path = os.path.join(pose_time_series_outdir, "just_silhouette_overlays")
        #     ensure_dir(out_path)
        #     a = np.clip(silhouettes_np[i], 0.0, 1.0)  # (H,W), float
        #     a_exp = a[..., None]
        #     a_exp = np.clip(a_exp, 0, 255).astype(np.uint8)
        #     cv2.imwrite(img=a_exp, filename=os.path.join(out_path, base_name+".png"))
        
        _save_reconstruction_images(
            orig_image_paths=sample["imgpaths"],
            outdir=pose_time_series_outdir,
            renderer=renderer,
            instance_number=0,
            cameras=camera_group_device,
            reconstructed_keypoints_local=keypoints,
            reconstructed_vertices_local=vertices,
            faces_from_vert_indices=fish.faces.unsqueeze(0),
            global_t=global_t,
            keypoint_names=dataset.index_json["keypoint_list"],
            view_names=dataset.views,
            draw_verts=True,
            pad_to_see_full_reprojection=True,
            plot_other_cameras=True,
        )
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", nargs="+", type=int, default=list(range(50, 100)))
    parser.add_argument("--mesh", type=str, default="goldfish_design_small.json")
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/output/GoldFish20171216_BL320/20171216_124610",
    )
    parser.add_argument(
        "--datadir", type=str, default="data/input/video_frames_20171215_101550"
    )
    parser.add_argument("--fish_place", type=int, choices=[1, 2], default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_models", action="store_true")
    args = parser.parse_args()

    reconstruct(
        mesh_path=args.mesh,
        dataset_dir=args.datadir,
        outdir=args.outdir,
        frame_indices=args.index,
        instance_number=args.fish_place,
        seed=args.seed,
        save_models=args.save_models,
    )
