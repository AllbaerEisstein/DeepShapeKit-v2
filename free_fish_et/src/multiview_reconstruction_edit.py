from collections import defaultdict
import json
import math
import pickle
import os
import argparse
import time
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
    mask_present_mask: Optional[List[bool]] = None,
    keypoint_predictions: Optional[torch.Tensor] = None, # for calculating keypoint L2 distance
    optimizer_losses: Optional[dict[str, Optional[float]]] = None,
    optimizer_loss_weights: Optional[dict[str, float]] = None,
    view_weights: Optional[List[float]] = None,
    silhouette_threshold: float = 0.01,  # tiny alpha cutoff
    blend_factor: float = 0.6,           # overlay opacity (60%)
    also_draw_mask_detection: bool = True,
    show_metrics: bool = True,
    draw_verts: bool = False,
    draw_coordinate_axes: bool = False,
    annotate_global_t: bool = True,
    annotate_keypoints_with_coords: bool = False,
    pad_to_see_full_reprojection: bool = False,
    plot_other_cameras: bool = False,
    # CLAUDE FIX: frames that were not optimized directly but filled in by interpolation across a
    # detection gap are rendered with a different silhouette colour so they are distinguishable
    # from directly-optimized frames at a glance.
    is_interpolated: bool = False,
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

    def _fmt_metric_value(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, str):
            return value
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(value_f):
            return "nan"
        return f"{value_f:.4f}"

    def _fmt_weighted_loss(
        loss_name: str,
        loss_value: Any,
        view_weight: Optional[float] = None,
    ) -> str:
        if loss_value is None:
            return "n/a"
        if optimizer_loss_weights is None:
            return _fmt_metric_value(loss_value)
        weight = optimizer_loss_weights.get(loss_name)
        if weight is None:
            return _fmt_metric_value(loss_value)
        try:
            loss_value_f = float(loss_value)
            weight_f = float(weight)
        except (TypeError, ValueError):
            return _fmt_metric_value(loss_value)
        if view_weight is None:
            if (not np.isfinite(loss_value_f)) or (not np.isfinite(weight_f)) or abs(weight_f) < 1e-12:
                return _fmt_metric_value(loss_value_f)
            return f"{weight_f:.5f} * {loss_value_f / weight_f:.5f}"

        try:
            view_weight_f = float(view_weight)
        except (TypeError, ValueError):
            view_weight_f = float("nan")
        combined_weight = view_weight_f * weight_f
        if (
            (not np.isfinite(loss_value_f))
            or (not np.isfinite(weight_f))
            or (not np.isfinite(view_weight_f))
            or abs(combined_weight) < 1e-12
        ):
            return _fmt_metric_value(loss_value_f)
        return f"{view_weight_f:.2f} * {weight_f:.2f} * {loss_value_f / combined_weight:.2f}"

    def _draw_metrics_table(image: np.ndarray, rows: list[tuple[str, Any]]) -> None:
        if not rows:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        header_scale = 0.45
        row_scale = 0.4
        thickness = 1
        margin = 8
        pad = 8
        row_height = 16
        col_gap = 12
        header = "Metrics"

        formatted_rows = [(name, _fmt_metric_value(value)) for name, value in rows]
        name_widths = [cv2.getTextSize(name, font, row_scale, thickness)[0][0] for name, _ in formatted_rows]
        value_widths = [cv2.getTextSize(value, font, row_scale, thickness)[0][0] for _, value in formatted_rows]
        header_width = cv2.getTextSize(header, font, header_scale, thickness)[0][0]

        name_col_width = max(name_widths) if name_widths else 0
        value_col_width = max(value_widths) if value_widths else 0
        panel_width = max(header_width, name_col_width + col_gap + value_col_width) + 2 * pad
        panel_height = pad + 18 + (row_height * len(formatted_rows)) + pad

        h, w = image.shape[:2]
        panel_width = min(panel_width, max(32, w - 2 * margin))
        panel_height = min(panel_height, max(32, h - 2 * margin))

        x0 = margin
        y0 = margin
        x1 = min(w - margin, x0 + panel_width)
        y1 = min(h - margin, y0 + panel_height)
        if x1 <= x0 or y1 <= y0:
            return

        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(10, 10, 10), thickness=-1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(210, 210, 210), thickness=1)
        image[:] = cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0)

        draw_text(
            image,
            header,
            (x0 + pad, y0 + pad + 10),
            font_scale=header_scale,
            color=(255, 255, 255),
            thickness=1,
            font=font,
        )

        current_y = y0 + pad + 26
        for name, value in formatted_rows:
            draw_text(
                image,
                name,
                (x0 + pad, current_y),
                font_scale=row_scale,
                color=(225, 225, 225),
                thickness=1,
                font=font,
            )
            draw_text(
                image,
                value,
                (x0 + pad + name_col_width + col_gap, current_y),
                font_scale=row_scale,
                color=(255, 255, 0),
                thickness=1,
                font=font,
            )
            current_y += row_height
    
    silhouettes = renderer(reconstructed_vertices_local, faces_from_vert_indices, global_t)
    silhouettes_np = silhouettes.detach().cpu().numpy()
    alpha = silhouettes_np  # (N, H, W)
    n_views, H, W = alpha.shape

    metrics = {
        "IoU_mask_detection_and_gt": {
            view_name: None for view_name in view_names
        },
        "IoU_reconstruction_and_gt": {
            view_name: None for view_name in view_names
        },
        "IoU_reconstruction_and_mask_detection": {
            view_name: None for view_name in view_names
        },
        "keypoint_L2_distance": {
            view_name: {
                keypoint_name: None for keypoint_name in keypoint_names
            } for view_name in view_names
        },
        "optimizer_losses": {
            "kpt_reprojection_loss": None,
            "mask_fitting_loss": None,
            "bone_angle_constraint_loss": None,
            "bone_length_constraint_loss": None,
            "final_deviation_from_prev_frame": None
        },
    }
    if optimizer_losses is not None:
        for name in metrics["optimizer_losses"].keys():
            value = optimizer_losses.get(name)
            metrics["optimizer_losses"][name] = None if value is None else float(value)

    assert len(orig_image_paths) == n_views, "orig_image_paths length must match rendered views"
    assert cameras.batch_size == n_views, "Camera batch size must match number of views"
    assert len(view_names) == n_views, "Number of view names must match number of views"
    if view_weights is not None:
        assert len(view_weights) == n_views, "view_weights length must match number of views"

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
        # Low-pass filter to counteract compression/codec noise from the lossy video source:
        # extracted frames come from cv2.VideoCapture on an .mp4, and H.264-style compression
        # routinely leaves isolated non-zero speckle in otherwise pure-black background regions.
        # A median blur removes this salt-and-pepper noise (without blurring the fish silhouette
        # edges the way a Gaussian blur would), and a small threshold margin (instead of >0)
        # absorbs any residual near-black noise that survives the blur.
        padded_gray_denoised = cv2.medianBlur(padded_gray, 3)
        _GT_MASK_NOISE_THRESHOLD = 10  # pixel values <= this are still treated as background
        _, padded_binary = cv2.threshold(padded_gray_denoised, _GT_MASK_NOISE_THRESHOLD, 1, cv2.THRESH_BINARY)

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

        IoU_reconstruction_and_gt = metric(
            torch.tensor(padded_binary, device=device),
            reprojection_mask_binary
        )
        if mask_predictions is not None:
            mask_is_present = True if mask_present_mask is None else bool(mask_present_mask[view_idx])
            if mask_is_present:
                mask_pred_view = mask_predictions[view_idx].to(device=device)
                if any((extra_pad_left, extra_pad_right, extra_pad_top, extra_pad_bottom)):
                    mask_pred_view = torch.nn.functional.pad(
                        mask_pred_view, (extra_pad_left, extra_pad_right, extra_pad_top, extra_pad_bottom),
                    )
                IoU_mask_detection_and_gt = metric(torch.tensor(padded_binary, device=device), mask_pred_view)
                IoU_reconstruction_and_mask_detection = metric(mask_pred_view, reprojection_mask_binary)
                metrics["IoU_mask_detection_and_gt"][view_names[view_idx]] = IoU_mask_detection_and_gt.item()
                metrics["IoU_reconstruction_and_mask_detection"][view_names[view_idx]] = IoU_reconstruction_and_mask_detection.item()
            else:
                metrics["IoU_mask_detection_and_gt"][view_names[view_idx]] = float("nan")
                metrics["IoU_reconstruction_and_mask_detection"][view_names[view_idx]] = float("nan")
        metrics["IoU_reconstruction_and_gt"][view_names[view_idx]] = IoU_reconstruction_and_gt.item()


        # Start from the padded original image.
        blended = padded.astype(np.float32).copy()
        # Optional mask-detection overlay
        # White where mask_full is present, transparent elsewhere.
        if also_draw_mask_detection and mask_predictions is not None:
            mask_is_present = True if mask_present_mask is None else bool(mask_present_mask[view_idx])
            if mask_is_present:
                mask_view = mask_predictions[view_idx]
                if isinstance(mask_view, torch.Tensor):
                    mask_view = mask_view.detach().cpu().numpy()
                mask_view = np.asarray(mask_view)
                if mask_view.max(initial=0) > 1.0:
                    mask_view = mask_view / 255.0
                mask_view = np.clip(mask_view, 0.0, 1.0)
                # Pad exactly like the reconstruction silhouette.
                if any((extra_pad_left, extra_pad_right, extra_pad_top, extra_pad_bottom)):
                    mask_view = np.pad(
                        mask_view,
                        ((extra_pad_top, extra_pad_bottom), (extra_pad_left, extra_pad_right),),
                        mode="constant",
                    )
                # Resize/validate shape if necessary.
                if mask_view.shape != padded.shape[:2]:
                    raise ValueError(
                        f"mask_full[{view_idx}] has shape {mask_view.shape}, "
                        f"but expected {padded.shape[:2]}"
                    )
                # Blue overlay.
                blue_img = np.full_like(blended, (255.0,0.0,0.0))
                # Only mask pixels contribute.
                mask_alpha = mask_view[..., None]
                # Alpha for mask visualization.
                mask_overlay_alpha = 0.35
                blended = (
                    blended * (1.0 - mask_overlay_alpha * mask_alpha)
                    + blue_img * (mask_overlay_alpha * mask_alpha)
                )
        # Reconstruction silhouette overlay
        # CLAUDE FIX: colour is BGR (images are written with cv2.imwrite). Directly-optimized
        # frames keep the original pure red; interpolated frames get orange so a gap fill is
        # obvious in the saved images without reading the metrics table.
        overlay_color_bgr = (0, 140, 255) if is_interpolated else (0, 0, 255)
        red_img = np.zeros_like(blended)
        red_img[:, :, 0] = overlay_color_bgr[0]
        red_img[:, :, 1] = overlay_color_bgr[1]
        red_img[:, :, 2] = overlay_color_bgr[2]
        a_exp = a[..., None]
        blended = (
            blended * (1.0 - blend_factor * a_exp)
            + red_img * (blend_factor * a_exp)
        )
        blended = np.nan_to_num(
            blended,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )
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
                draw_circle(blended, (ui, vi), radius=5, color=(0, 150, 255))
                draw_text(blended, name, (ui, vi + 15), color=(0, 150, 255))
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
                ci_pred = keypoint_predictions[view_idx, kp_idx, 2].item()
                if ci_pred > 0 and torch.isfinite(pred_u) and torch.isfinite(pred_v):
                    ui_pred = int(round(pred_u.item())) + offset_x
                    vi_pred = int(round(pred_v.item())) + offset_y
                    conf_scaled_annot_radius = int(ci_pred*10)//2
                    cv2.circle(blended, (ui_pred, vi_pred), radius=1, color=(255,0,0), lineType=-1)
                    cv2.circle(blended, (ui_pred, vi_pred), radius=conf_scaled_annot_radius, color=(255,0,0), lineType=-1)
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
                else:
                    metrics["keypoint_L2_distance"][view_names[view_idx]][keypoint_names[kp_idx]] = float("nan")

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

        if show_metrics:
            view_name = view_names[view_idx]
            current_view_weight = (
                float(view_weights[view_idx]) if view_weights is not None else None
            )
            view_kpt_errors = [
                distance
                for distance in metrics["keypoint_L2_distance"][view_name].values()
                if distance is not None and np.isfinite(distance)
            ]
            keypoint_mean_error = (
                float(np.mean(view_kpt_errors)) if len(view_kpt_errors) > 0 else None
            )
            metric_rows = [
                ("view", view_name),
            ]
            if is_interpolated:
                # CLAUDE FIX: only added for gap fills, so normal frames keep their original table.
                metric_rows.append(("frame_source", "INTERPOLATED"))
            metric_rows += [
                ("IoU_reconstruction_and_gt", metrics["IoU_reconstruction_and_gt"][view_name]),
                ("IoU_mask_detection_and_gt", metrics["IoU_mask_detection_and_gt"][view_name]),
                ("IoU_reconstruction_and_mask_detection", metrics["IoU_reconstruction_and_mask_detection"][view_name]),
                ("kpt_L2_mean", keypoint_mean_error),
                ("kpt_valid", f"{len(view_kpt_errors)}/{len(keypoint_names)}"),
            ]
            for keypoint_name in keypoint_names:
                metric_rows.append(
                    (
                        f"kpt_L2_{keypoint_name}",
                        metrics["keypoint_L2_distance"][view_name][keypoint_name],
                    )
                )
            for loss_name, loss_value in metrics["optimizer_losses"].items():
                metric_rows.append((loss_name, _fmt_weighted_loss(loss_name, loss_value, current_view_weight)))
            _draw_metrics_table(blended, metric_rows)

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


def interpolate_pose_pair(
    pose_a: torch.Tensor,
    pose_b: torch.Tensor,
    interpolate_size: int,
) -> torch.Tensor:
    """
    Reproduce the original np.interp behaviour for one pose pair.
    The original code sampled x = 0 .. interpolate_size-1 while the two
    endpoint poses were located at x = 0 and x = interpolate_size.
    Consequently, the final endpoint itself is not emitted.
    """
    if interpolate_size <= 0:
        return torch.empty(
            (0, pose_a.shape[1]),
            dtype=pose_a.dtype,
            device=pose_a.device,
        )
    samples = torch.arange(
        interpolate_size,
        dtype=pose_a.dtype,
        device=pose_a.device,
    )
    alpha = samples / interpolate_size
    return torch.lerp(
        pose_a.expand(interpolate_size, -1),
        pose_b.expand(interpolate_size, -1),
        alpha[:, None],
    )


def _interpolate_parameter_entries(
    entry_a: list,
    entry_b: list,
    interpolate_size: int,
) -> list[list[torch.Tensor]]:
    """
    CLAUDE FIX: linearly interpolate one full `parameters` entry
    ([global_ori, body_pose, body_bone_length, scale, global_t]) between two reconstructed
    frames. Each tensor is flattened to (1, D) so `interpolate_pose_pair` can be applied
    unchanged, then restored to its original shape so the produced entries are drop-in
    replacements for optimizer output (including the `*init` unpacking in `multiview.fit_mesh`).

    Sample i corresponds to the i-th deferred frame: sample 0 is exactly `entry_a`, and
    `entry_b` itself is never emitted (it is stored as its own frame).
    """
    interpolated_entries: list[list[torch.Tensor]] = [[] for _ in range(max(interpolate_size, 0))]
    for tensor_a, tensor_b in zip(entry_a, entry_b):
        tensor_a = tensor_a.detach()
        tensor_b = tensor_b.detach().to(dtype=tensor_a.dtype, device=tensor_a.device)
        original_shape = tensor_a.shape
        samples = interpolate_pose_pair(
            tensor_a.reshape(1, -1),
            tensor_b.reshape(1, -1),
            interpolate_size,
        )
        for sample_idx in range(interpolate_size):
            interpolated_entries[sample_idx].append(
                samples[sample_idx].reshape(original_shape).clone()
            )
    return interpolated_entries


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

    # CLAUDE FIX (pose_time_series schema v2): declare the schema and the orderings explicitly so
    # that the Blender importer can validate instead of guessing. `global_ori` and `body_pose` as
    # produced by the optimizer are already delta-from-rest rotations in template axes -- exactly
    # what v2 specifies -- so only the meta block needed to change here.
    bone_order = fish.template.get("bone_order", [])
    meta = {
        "schema": "pose_time_series/2",
        "producer": "multiview_reconstruction_edit.py",
        "source": "multiview_reconstruction_edit.py",
        "dataset_dir": dataset_dir,
        "dataset_name": dataset_name,
        "instance": int(instance_number),
        "bone_order": bone_order,
        "body_pose_bone_order": bone_order[1:],
        "virtual_bone_names": fish.template.get("virtual_bone_names", []),
        "virtual_bone_mask": fish.template.get("virtual_bone_mask", []),
        "rotation": "axis_angle_exponential_map",
        "space": ("global_t/global_ori in world; body_pose in template axes, delta from rest"),
        "units": "meters",
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "frame_indices": processed_frames,
        "mesh_file": mesh_path,
    }

    for payload in frame_payloads:
        if len(payload["body_pose"]) != len(bone_order) - 1:
            raise ValueError(
                f"pose_time_series frame {payload['frame']}: body_pose has "
                f"{len(payload['body_pose'])} entries but bone_order has {len(bone_order)} bones. "
                f"body_pose must be parallel to bone_order[1:]."
            )

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
        # CLAUDE FIX: gap-filled frames are flagged so consumers (e.g. the Blender importer) can
        # tell interpolated poses from directly-optimized ones. The key is omitted entirely for
        # normal frames, so output for a run without gaps is byte-identical to before.
        if payload.get("interpolated"):
            frame_dict["interpolated"] = True
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
    num_iters: int = 100,
    angle_constraint_weight: float = 1.0,
    smooth_weight: float = 1.0,
    big_artic_weight: float = 1.0,
    bone_length_constraint_weight: float = 1.0,
    mask_weight: float = 1.0,
    keypoints_weight: float = 1.0,
    view_weights: Optional[List[float]] = None,
    render_scale: float = 1.0,
) -> None:
    """
    Run multiview reconstruction for given frames.

    Args:
        render_scale: CLAUDE FIX. Linear resolution factor for the differentiable silhouette
            renderer used during fitting, relative to the calibrated image size (1.0 = full
            resolution). Lowering it reduces rasterization time and GPU memory roughly with its
            square. Images saved for inspection are always rendered at full resolution.
    """

    _ensure_dir(outdir)

    # --------------------------
    # setup; instantiate classes
    device = setup_device(seed)

    print("\n" + "="*40)
    print("Starting multiview reconstruction with the following settings:")
    dataset = Multiview_Dataset(root=dataset_dir, views=video_names)
    n_views = len(dataset.views)
    if view_weights is None or len(view_weights) == 0:
        view_weights = [1.0] * n_views
    elif len(view_weights) != n_views:
        raise ValueError(
            "view_weights length mismatch: "
            f"got {len(view_weights)} weight(s) for {n_views} view(s) {dataset.views}. "
            "Provide a comma-separated list with one weight per view index."
        )
    print("   Device:", device)
    print("   Contributions to the loss function:")
    print("     Mask reprojection error:          ", mask_weight)
    print("     Keypoint reprojection error:      ", keypoints_weight)
    print("     Smoothness violation:             ", smooth_weight)
    print("     Bone angle constraint violation:  ", angle_constraint_weight)
    print("     Bone length constraint violation: ", bone_length_constraint_weight)
    print("     Bone group deviation:             ", big_artic_weight)
    print("   Silhouette render scale:            ", render_scale)
    print("     Views:")
    for view_name, weight in zip(dataset.views, view_weights):
        print(f"          {view_name}: {weight}")
    print("="*40 + "\n")
    optimizer_loss_weight_map = {
        "kpt_reprojection_loss": float(keypoints_weight),
        "mask_fitting_loss": float(mask_weight),
        "bone_angle_constraint_loss": float(angle_constraint_weight),
        "bone_length_constraint_loss": float(bone_length_constraint_weight),
        "final_deviation_from_prev_frame": float(smooth_weight)
    }


    camera_group_uniform_size_cpu = dataset.cams.get_camera_group().with_intrinsics_adjusted_for_uniform_image_size()
    camera_group_uniform_size_device = camera_group_uniform_size_cpu.to(device)

    fish = fish_model(mesh_json_path=mesh_path)
    # CLAUDE FIX: the optimizer is told how large the capture volume is, in the calibration's own
    # world units, so it can size its translation steps relative to the rig instead of using a
    # fixed step that is meaningful only for one particular choice of units.
    scene_scale = float(camera_group_uniform_size_cpu.scene_scale)
    print(f"   Camera rig scale (mean baseline): {scene_scale:.4f} world units")
    optimizer = OptimizeMV(
        scene_scale=scene_scale,
        num_iters=num_iters,
        angle_constraint_weight=angle_constraint_weight,
        smooth_weight=smooth_weight,
        big_artic_weight=big_artic_weight,
        bone_length_constraint_weight=bone_length_constraint_weight,
        mask_weight=mask_weight,
        keypoints_weight=keypoints_weight,
        view_weights=view_weights,
        device=torch.device(device),
        fish_model_obj=fish,
    )
    # CLAUDE FIX: the fitting renderer runs at the configured render_scale; the renderer used only
    # for the saved inspection images stays at full resolution so the output is unaffected.
    renderer_for_reconstrcution = Silhouette_Renderer(
        device, camera_group_uniform_size_device, render_scale=render_scale
    )
    renderer_for_saving_images = Silhouette_Renderer(device, camera_group_uniform_size_device, sigma=0, gamma=0)

    # CLAUDE FIX: verify once, before any frame is processed, that the renderer's camera model and
    # the projection matrices driving the keypoint loss put the same 3D point in the same place.
    # If they disagree, the silhouette term and the keypoint term pull the mesh towards different
    # image positions and the fit quietly settles on a compromise between two inconsistent camera
    # models -- a failure mode that produces no error and no obviously broken output.
    renderer_for_reconstrcution.check_camera_consistency(tolerance_px=1.0)

    # --------------------------
    # load cache, if available
    cache_dir, cache_path = _get_reconstruction_cache_paths(outdir, dataset_dir, instance_number)
    cache_data = _load_reconstruction_cache(cache_path)

    parameters = []
    sample_data = []
    pose_time_series_frames: list[dict] = []
    processed_frames: set[int] = set()
    cached_metrics: Optional[dict] = None
    cached_elapsed_wall_time_sec = 0.0
    resumed_from_cache = False
    optimizer_loss_names = [
        "kpt_reprojection_loss",
        "mask_fitting_loss",
        "bone_angle_constraint_loss",
        "bone_length_constraint_loss",
        "final_deviation_from_prev_frame"
    ]

    def _normalize_path(path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if str(path).strip() == "":
            return None
        return os.path.abspath(os.path.normpath(str(path)))

    def _normalize_frame_indices(indices: Any) -> Optional[list[int]]:
        if not isinstance(indices, (list, tuple)):
            return None
        try:
            return sorted(int(i) for i in indices)
        except (TypeError, ValueError):
            return None

    def _normalize_weights(weights: Any) -> Optional[list[float]]:
        if not isinstance(weights, (list, tuple)):
            return None
        try:
            return [float(w) for w in weights]
        except (TypeError, ValueError):
            return None

    if cache_data is not None:
        cache_dataset = cache_data.get("dataset_dir")
        cache_mesh = cache_data.get("mesh_path")
        cache_frames = cache_data.get("frame_indices")
        cache_view_weights = _normalize_weights(cache_data.get("view_weights"))
        requested_view_weights = _normalize_weights(view_weights)
        requested_frame_indices = _normalize_frame_indices(frame_indices)
        cached_frame_indices = _normalize_frame_indices(cache_frames)
        if (
            (_normalize_path(cache_dataset) and _normalize_path(cache_dataset) != _normalize_path(dataset_dir))
            or (_normalize_path(cache_mesh) and _normalize_path(cache_mesh) != _normalize_path(mesh_path))
            or (cached_frame_indices and requested_frame_indices and cached_frame_indices != requested_frame_indices)
            or (cache_view_weights and requested_view_weights and cache_view_weights != requested_view_weights)
        ):
            print("Existing reconstruction cache does not match current configuration; starting a new reconstruction run.")
        else:
            print(f"Resuming reconstruction from cache '{cache_path}'.")
            parameters = cache_data.get("parameters", [])
            sample_data = cache_data.get("sample_data", [])
            pose_time_series_frames = cache_data.get("pose_time_series_frames", []) or []
            cached_metrics = cache_data.get("metrics")
            processed_frames = {int(i) for i in cache_data.get("processed_frames", []) if isinstance(i, (int, float))}
            processed_frames = processed_frames.intersection(set(frame_indices))
            try:
                cached_elapsed_wall_time_sec = float(cache_data.get("elapsed_wall_time_sec", 0.0) or 0.0)
            except (TypeError, ValueError):
                cached_elapsed_wall_time_sec = 0.0
            resumed_from_cache = True

    def _fresh_metrics() -> dict:
        return {
            "IoU_mask_detection_and_gt": {view_name: [] for view_name in dataset.views},
            "IoU_reconstruction_and_gt": {view_name: [] for view_name in dataset.views},
            "IoU_reconstruction_and_mask_detection": {view_name: [] for view_name in dataset.views},
            "keypoint_L2_distance": {
                view_name: {kpt: [] for kpt in dataset.index_json["keypoint_list"]}
                for view_name in dataset.views
            },
            "optimizer_losses": {name: [] for name in optimizer_loss_names},
            # CLAUDE FIX: frame indices that were filled in by interpolation across a
            # detection gap rather than optimized directly.
            "interpolated_frames": [],
        }

    def _ensure_metrics_schema(metrics_dict: dict) -> dict:
        metrics_dict.setdefault("interpolated_frames", [])
        metrics_dict.setdefault("IoU_mask_detection_and_gt", {})
        metrics_dict.setdefault("IoU_reconstruction_and_gt", {})
        metrics_dict.setdefault("IoU_reconstruction_and_mask_detection", {})
        metrics_dict.setdefault("keypoint_L2_distance", {})
        for view_name in dataset.views:
            metrics_dict["IoU_mask_detection_and_gt"].setdefault(view_name, [])
            metrics_dict["IoU_reconstruction_and_gt"].setdefault(view_name, [])
            metrics_dict["IoU_reconstruction_and_mask_detection"].setdefault(view_name, [])
            metrics_dict["keypoint_L2_distance"].setdefault(view_name, {})
            for kpt_name in dataset.index_json["keypoint_list"]:
                metrics_dict["keypoint_L2_distance"][view_name].setdefault(kpt_name, [])

        metrics_dict.setdefault("optimizer_losses", {})
        target_len = max(
            [len(metrics_dict["IoU_reconstruction_and_gt"].get(view_name, [])) for view_name in dataset.views] or [0]
        )
        for loss_name in optimizer_loss_names:
            if isinstance(metrics_dict["optimizer_losses"].get(loss_name), list):
                loss_values = metrics_dict["optimizer_losses"][loss_name]
            else:
                loss_values = []
            if len(loss_values) < target_len:
                loss_values.extend([None] * (target_len - len(loss_values)))
            metrics_dict["optimizer_losses"][loss_name] = loss_values
        return metrics_dict

    metrics = cached_metrics if cached_metrics is not None else _fresh_metrics()

    if cached_metrics is not None:
        cached_views = set(cached_metrics.get("IoU_reconstruction_and_gt", {}).keys())
        if cached_views != set(dataset.views):
            print("Cached reconstruction metrics do not match current dataset views; restarting reconstruction run.")
            parameters = []
            sample_data = []
            pose_time_series_frames = []
            processed_frames = set()
            metrics = _fresh_metrics()
            cached_elapsed_wall_time_sec = 0.0
            resumed_from_cache = False
    metrics = _ensure_metrics_schema(metrics)

    def _append_frame_metrics(frame_metrics: Optional[dict]) -> None:
        """CLAUDE FIX: keep every metric list the same length as the number of emitted frames,
        including gap-filled frames whose images could not be rendered."""
        for view in dataset.views:
            for metric_name in (
                "IoU_mask_detection_and_gt",
                "IoU_reconstruction_and_gt",
                "IoU_reconstruction_and_mask_detection",
            ):
                metrics[metric_name][view].append(
                    None if frame_metrics is None else frame_metrics[metric_name][view]
                )
            for kpt in dataset.index_json["keypoint_list"]:
                metrics["keypoint_L2_distance"][view][kpt].append(
                    None if frame_metrics is None else frame_metrics["keypoint_L2_distance"][view][kpt]
                )
        for loss_name in optimizer_loss_names:
            metrics["optimizer_losses"][loss_name].append(
                None if frame_metrics is None else frame_metrics.get("optimizer_losses", {}).get(loss_name)
            )

    def _emit_frame(
        frame_idx: int,
        entry: list,
        instance_sample: Optional[dict],
        final_losses: Optional[dict],
        world_outputs: Optional[tuple] = None,
        is_interpolated: bool = False,
    ) -> None:
        """
        CLAUDE FIX: single place where a reconstructed *or* interpolated frame is turned into all
        of its per-frame outputs (parameters/sample_data caching, deformation, pose time series
        entry, saved images, metrics, processed_frames bookkeeping). Directly-optimized frames go
        through exactly the same code path as before; interpolated frames differ only in having no
        optimizer losses and in the overlay colour / flags.

        Args:
            entry: [global_ori, body_pose, body_bone_length, scale, global_t], i.e. the layout
                stored in `parameters` and unpacked as `*init` into `multiview.fit_mesh`.
            world_outputs: optional (vertices_world, keypoints_world) already computed by
                `fit_mesh`; recomputed from the deformation when omitted.
        """
        global_ori_est, body_pose_est, body_bone_est, scale_est, global_t_est = entry

        out_reconstructed = fish(global_ori_est, body_pose_est, body_bone_est, scale_est)
        reconstructed_keypoints_local = out_reconstructed["keypoints"].to(device)
        reconstructed_vertices_local = out_reconstructed["vertices"].to(device)

        if world_outputs is None:
            global_t_cpu = global_t_est.detach().cpu()
            vertices_world_est = out_reconstructed["vertices"].detach().cpu() + global_t_cpu
            keypoints_world_est = out_reconstructed["keypoints"].detach().cpu() + global_t_cpu
        else:
            vertices_world_est, keypoints_world_est = world_outputs

        parameters.append([global_ori_est, body_pose_est, body_bone_est, scale_est, global_t_est])
        if instance_sample is not None:
            sample_data.append(
                [
                    instance_sample["frames"],
                    instance_sample["imgpaths"],
                    keypoints_world_est,
                    instance_sample["bboxes"],
                    frame_idx,
                ]
            )
        else:
            sample_data.append([None, None, keypoints_world_est, None, frame_idx])

        global_ori_cpu = global_ori_est.detach().cpu().view(-1).tolist()
        body_pose_flat = body_pose_est.detach().cpu().view(-1).tolist()
        body_pose_triplets = [
            body_pose_flat[i : i + 3]
            for i in range(0, len(body_pose_flat), 3)
        ]
        body_bone_lengths = body_bone_est.detach().cpu().view(-1).tolist()
        global_t_list = global_t_est.detach().cpu().view(-1).tolist()
        scale_list = scale_est.detach().cpu().view(-1).tolist()
        frame_payload = {
            "frame": int(frame_idx),
            "global_ori": [float(x) for x in global_ori_cpu],
            "body_pose": [[float(v) for v in triple] for triple in body_pose_triplets],
            "body_bone_length": [float(v) for v in body_bone_lengths],
            "global_t": [float(v) for v in global_t_list],
            "scale": float(scale_list[0]) if scale_list else None,
        }
        if is_interpolated:
            # Only set for gap fills; omitted for normal frames so their payload is unchanged.
            frame_payload["interpolated"] = True
        pose_time_series_frames.append(frame_payload)

        frame_metrics = None
        if instance_sample is not None:
            frame_metrics = _save_reconstruction_images(
                orig_image_paths=instance_sample["imgpaths"],
                outdir=outdir,
                renderer=renderer_for_saving_images,
                instance_number=instance_number,
                cameras=camera_group_uniform_size_device,
                reconstructed_keypoints_local=reconstructed_keypoints_local,
                reconstructed_vertices_local=reconstructed_vertices_local,
                faces_from_vert_indices=fish.faces.unsqueeze(0).to(device),
                global_t=global_t_est.to(device),
                keypoint_names=dataset.index_json["keypoint_list"],
                view_names=dataset.views,
                mask_predictions=instance_sample["masks_full"] / 255.0,
                mask_present_mask=instance_sample["seg_mask_present_mask"],
                keypoint_predictions=instance_sample["keypoints"],
                optimizer_losses=final_losses,
                optimizer_loss_weights=optimizer_loss_weight_map,
                view_weights=view_weights,
                is_interpolated=is_interpolated,
            )

        if save_models:
            _save_obj_model(outdir, frame_idx, instance_number, vertices_world_est, fish)

        # cache quality metrics for this frame
        _append_frame_metrics(frame_metrics)

        if is_interpolated and int(frame_idx) not in metrics["interpolated_frames"]:
            metrics["interpolated_frames"].append(int(frame_idx))

        processed_frames.add(frame_idx)

    pbar = tqdm(
        total=len(frame_indices),
        desc=f"{os.path.basename(dataset_dir)} reconstruction frame {frame_indices[0]}",
        initial=len(processed_frames),
    )
    run_start_wall_time_sec = time.perf_counter()
    paused = False
    # CLAUDE FIX: frame indices skipped for insufficient detections, awaiting retroactive
    # interpolation once a frame with sufficient detections is reconstructed again.
    pending_gap_frames: list[int] = []

    # --------------------------
    # loop through frames and reconstruct
    for idx in frame_indices:
        if idx in processed_frames:
            continue
        
        # --------------------------
        # load from dataset
        try:
            instance_sample = dataset.__getitem__(idx, instance_number)
        except (IndexError, KeyError) as exc:
            # CLAUDE FIX: a frame that is absent from one view's index surfaces as a lookup failure
            # rather than an out-of-range error, so both are treated as "this frame is unavailable"
            # and skipped, with the reason reported instead of aborting the whole run.
            print(f"Sample {idx} unavailable ({type(exc).__name__}: {exc}), skipping")
            pbar.update()
            continue

        kpt_present_mask = instance_sample['kpt_present_mask']
        seg_mask_present_mask = instance_sample['seg_mask_present_mask']

        # CLAUDE FIX: frames with insufficient detections are no longer discarded. They are
        # deferred ("pending") and retroactively filled in by interpolation once a later frame
        # with sufficient detections is reconstructed (see the gap handling below). The
        # sufficiency criteria themselves are unchanged.
        if len([
                view_with_seg_mask for view_with_seg_mask in seg_mask_present_mask 
                if view_with_seg_mask == True
            ]) < 2:
            print(
                f"Less than two views with segmentation masks in sample for frame {idx} -> "
                f"deferring frame for interpolation (presence={seg_mask_present_mask})"
            )
            pending_gap_frames.append(idx)
            pbar.update()
            continue
        if len([
                view_with_kpts for view_with_kpts in kpt_present_mask 
                if any(kpt_present == True for kpt_present in view_with_kpts)
            ]) < 2:
            print(
                f"Less than two views with keypoints in sample for frame {idx} -> "
                f"deferring frame for interpolation"
            )
            pending_gap_frames.append(idx)
            pbar.update()
            continue

        views_indices, orig_img_paths = instance_sample['frames'], instance_sample['imgpaths']

        # mask, bboxes, keypoint are specified in uniform_image_size coordinates (adjustment happened in dataset creation)
        keypoints = instance_sample["keypoints"]
        masks = instance_sample["masks_full"]
        bboxes = instance_sample["bboxes"]
        # Normalize mask to [0,1] on appropriate device
        masks = masks / 255.0


        # --------------------------
        # reconstruct

        # initialize from previous solution if available
        # CLAUDE FIX: after a run of deferred frames the last stored pose is separated from this
        # frame by an unknown amount of motion, which makes it a poor warm start. Force
        # `init = None` so this frame goes through the same triangulation+Procrustes
        # initialization as the very first frame of a run.
        if pending_gap_frames and parameters:
            print(
                f"Frame {idx}: gap of {len(pending_gap_frames)} frame(s) detected, "
                f"re-initializing with Procrustes instead of previous-frame warm start"
            )
            init = None
        else:
            init = parameters[-1] if parameters else None

        result = multiview.fit_mesh(
            fish,
            optimizer,
            camera_group_uniform_size_device,
            keypoints,
            masks,
            renderer_for_reconstrcution,
            device,
            *([] if init is None else init),
            index=idx,
            bboxs=bboxes,
            seg_mask_present_mask=seg_mask_present_mask
        )
        (
            vertices_world_est,
            keypoints_world_est,
            global_t_est,
            global_ori_plus_pose_est,
            body_bone_est,
            scale_est,
            final_losses,
        ) = result

        # --------------------------
        # cache results
        frame_b_entry = [
            global_ori_plus_pose_est[:, :3],
            global_ori_plus_pose_est[:, 3:],
            body_bone_est,
            scale_est,
            global_t_est,
        ]

        # --------------------------
        # CLAUDE FIX: retroactively fill the detection gap that preceded this frame, if any.
        # The gap is emitted BEFORE this frame's own entry so `parameters`, `sample_data`,
        # `pose_time_series_frames` and the metric lists all stay in chronological order and
        # `init = parameters[-1]` for the next frame still refers to this frame.
        if pending_gap_frames:
            if parameters:
                frame_a = sample_data[-1][4]
                frame_a_entry = parameters[-1]
                print(
                    f"Backfilling frames {pending_gap_frames} via interpolation between "
                    f"frame {frame_a} and frame {idx} ({len(pending_gap_frames)} frame(s))"
                )
                interpolated_entries = _interpolate_parameter_entries(
                    frame_a_entry, frame_b_entry, len(pending_gap_frames)
                )
                for gap_frame_idx, gap_entry in zip(pending_gap_frames, interpolated_entries):
                    try:
                        gap_sample = dataset.__getitem__(gap_frame_idx, instance_number)
                    except (IndexError, KeyError) as exc:
                        print(
                            f"Interpolated frame {gap_frame_idx}: sample could not be reloaded "
                            f"({type(exc).__name__}: {exc}); storing pose without saved images"
                        )
                        gap_sample = None
                    _emit_frame(
                        frame_idx=gap_frame_idx,
                        entry=gap_entry,
                        instance_sample=gap_sample,
                        # interpolated frames were never optimized, so there are no losses
                        final_losses=None,
                        is_interpolated=True,
                    )
            else:
                print(
                    f"Frames {pending_gap_frames} precede the first reconstructed frame "
                    f"({idx}); there is no earlier pose to interpolate from, so they cannot be "
                    f"reconstructed or interpolated."
                )
            pending_gap_frames = []

        _emit_frame(
            frame_idx=idx,
            entry=frame_b_entry,
            instance_sample=instance_sample,
            final_losses=final_losses,
            world_outputs=(vertices_world_est, keypoints_world_est),
            is_interpolated=False,
        )
        elapsed_wall_time_sec = cached_elapsed_wall_time_sec + (time.perf_counter() - run_start_wall_time_sec)

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
            "elapsed_wall_time_sec": elapsed_wall_time_sec,
            "view_weights": [float(v) for v in view_weights],
        }
        _save_reconstruction_cache(cache_dir, cache_path, cache_payload)

        pbar.desc = f"{os.path.basename(dataset_dir)} reconstruction frame {idx}"
        pbar.update()

        if pause_event is not None and pause_event.is_set():
            print("Pause requested; stopping after cache write.")
            paused = True
            break

    # reconstruction done (or paused)
    # CLAUDE FIX: a gap that is still open when the frame list runs out (or when the run is
    # paused) has no right-hand endpoint to interpolate towards, so those frames are left
    # unfilled and reported instead of silently disappearing.
    if pending_gap_frames:
        print(
            f"{len(pending_gap_frames)} trailing frame(s) {pending_gap_frames} had insufficient "
            f"detections and were never followed by a frame with sufficient detections; they "
            f"could not be reconstructed or interpolated."
        )

    elapsed_wall_time_sec = cached_elapsed_wall_time_sec + (time.perf_counter() - run_start_wall_time_sec)
    processed_frame_count = len(processed_frames)
    seconds_per_processed_frame = (
        elapsed_wall_time_sec / processed_frame_count if processed_frame_count > 0 else None
    )

    metrics["timing"] = {
        "elapsed_wall_time_sec": elapsed_wall_time_sec,
        "elapsed_wall_time_min": elapsed_wall_time_sec / 60.0,
        "seconds_per_processed_frame": seconds_per_processed_frame,
        "processed_frame_count": processed_frame_count,
        "target_frame_count": len(frame_indices),
        "resumed_from_cache": resumed_from_cache,
        "paused": paused,
    }
    # Keep legacy fields for backward compatibility.
    metrics["total_duration_min"] = elapsed_wall_time_sec / 60.0
    metrics["seconds_per_frame"] = seconds_per_processed_frame
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

    if paused:
        print(
            f"Reconstruction paused after {processed_frame_count}/{len(frame_indices)} frame(s). "
            f"Cache kept at '{cache_path}'."
        )
    else:
        _clear_reconstruction_cache(cache_path)



def render_pose_time_series(    
    mesh_path: str,
    dataset_dir: str,
    pose_time_series_file_path: str,
    outdir: str,
    # CLAUDE FIX (BUGREPORT A12): defaulted to False, which skipped linear blend skinning entirely
    # and rendered the rest-pose template for every frame.
    deform: bool = True,
    frame_range: Optional[List[int]] = None,
    offset_by_frame_range_start: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fish = fish_model(mesh_path)
    fish.to_device(device)

    dataset = Multiview_Dataset(root=dataset_dir)
    camera_group_cpu = dataset.cams.get_camera_group().with_intrinsics_adjusted_for_uniform_image_size()
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
        # CLAUDE FIX (BUGREPORT A12): `scale` is written by _save_pose_time_series_json but was
        # never read back, so every series was rendered at scale 1.
        scale = torch.tensor(float(frame.get("scale") or 1.0), device=device)

        # CLAUDE FIX (BUGREPORT A12): this call used to pass
        # `torch.zeros_like(body_pose.unsqueeze(0).flatten(1))` in place of the loaded body pose --
        # leftover debug code that discarded ALL articulation, so every pose_time_series rendered
        # as the rigidly-rotated rest template and looked plausible while showing nothing.
        articulated_verts_kpts = fish(
            global_ori.unsqueeze(0),
            body_pose.unsqueeze(0).flatten(1),
            bone_length.unsqueeze(0),
            scale=scale,
            deform=deform,
        )
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
        