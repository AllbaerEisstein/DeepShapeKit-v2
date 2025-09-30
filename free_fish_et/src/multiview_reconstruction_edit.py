import json
import pickle
import os
import argparse
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


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


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_multiview_dataset(root: str) -> Multiview_Dataset:
    return Multiview_Dataset(root=root)


def initialize_model(
    mesh_file: str, device: str, cameras: CameraGroup
) -> tuple:
    fish = fish_model(mesh_json_path=mesh_file)
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
    renderer = Silhouette_Renderer(device, cameras)
    return fish, optimizer, renderer


def get_tensors_from_instance_sample(sample: dict):
    """
    Just read a dict and return relevant contents.
    """
    keypoints = sample["keypoints"]
    masks = sample["masks_full"]
    bboxes = sample["bboxes"]

    # Normalize mask to [0,1] on appropriate device
    masks = masks / 255.0
    keypoints = keypoints
    bboxes = bboxes
    return keypoints, masks, bboxes


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
    ensure_dir(instance_dir)
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
        ensure_dir(view_dir)
        save_path = os.path.join(view_dir, f"frame_{sample_index}_view_{view_idx}.png")
        plt.imsave(save_path, img_with_projection)


def save_reconstruction_images(
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
    silhouette_threshold: float = 0.01,  # tiny alpha cutoff
    blend_factor: float = 0.6,           # overlay opacity (60%)
    draw_verts: bool = False,
    draw_coordinate_axes: bool = True,
    annotate_global_t: bool = True,
    annotate_keypoints_with_coords: bool = True,
):
    """
    Render silhouettes, pad originals (zero padding) to silhouette size (centered),
    overlay silhouette in red with given blend_factor, draw keypoints (blue), optionally
    project the world coordinate axes, and annotate projected world-space locations.
    """

    silhouettes = renderer(reconstructed_vertices_local, faces_from_vert_indices, global_t)

    silhouettes_np = silhouettes.detach().cpu().numpy()

    alpha = silhouettes_np  # (N, H, W)
    n_views, H, W = alpha.shape

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
        origin_world = torch.tensor([0.0,0.0,0.0], dtype=verts_world.dtype, device=verts_world.device)
        axis_length = 50
        axis_dirs = torch.eye(3, dtype=verts_world.dtype, device=verts_world.device)
        tick_fracs = [n / 4.0 for n in range(1, axis_length * 4 + 1)]
        world_points = [origin_world]
        axes_metadata = {"origin_idx": 0, "axes": {}, "axis_length": axis_length}

        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            axis_vec = axis_dirs[axis_idx] * axis_length
            end_point = origin_world + axis_vec
            world_points.append(end_point)
            end_idx = len(world_points) - 1

            tick_indices = []
            for frac in tick_fracs:
                tick_point = origin_world + axis_vec * frac
                world_points.append(tick_point)
                tick_indices.append(len(world_points) - 1)

            axes_metadata["axes"][axis_name] = {
                "end_idx": end_idx,
                "tick_indices": tick_indices,
                "tick_fracs": tick_fracs,
            }

        axis_points_tensor = torch.stack(world_points, dim=0).unsqueeze(0)
        axes_projection = cameras.perspective_projection_from_blworld(axis_points_tensor)
        axes_projection_np = axes_projection.detach().cpu().numpy()

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

        a = np.clip(alpha[view_idx], 0.0, 1.0)  # (H,W), float

        # threshold tiny values
        a[a < silhouette_threshold] = 0.0

        # Build red overlay image (BGR) and blend: out = orig*(1 - bf * a) + red*(bf * a)
        red_img = np.zeros_like(padded)
        red_img[:, :, 2] = 255  # full red channel in BGR


        a_exp = a[..., None]
        blended = (padded.astype(np.float32) * (1.0 - blend_factor * a_exp) +
                   red_img.astype(np.float32) * (blend_factor * a_exp))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Draw keypoints:
        for kp_idx, name in enumerate(keypoint_names):
            # u,v are pixel coords in the same coordinate system as the silhouette (W,H)
            ui = int(round(keypoints_proj[view_idx, kp_idx, 0].item()))
            vi = int(round(keypoints_proj[view_idx, kp_idx, 1].item()))
            if 0 <= ui < W and 0 <= vi < H:
                cv2.circle(blended, (ui, vi), radius=5, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(blended, name, (ui, vi + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.35, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                if annotate_keypoints_with_coords:
                    kp_coords = keypoints_world_np[kp_idx]
                    coord_text = f"({kp_coords[0]:.2f}, {kp_coords[1]:.2f}, {kp_coords[2]:.2f})"
                    cv2.putText(
                        blended,
                        coord_text,
                        (ui, vi + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        if draw_verts:
            for vert_idx in range(verts_proj.size(1)):
                ui = int(round(verts_proj[view_idx, vert_idx, 0].item()))
                vi = int(round(verts_proj[view_idx, vert_idx, 1].item()))
                if 0 <= ui < W and 0 <= vi < H:
                    cv2.circle(blended, (ui, vi), radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

        if annotate_global_t:
            gt_pt = global_t_proj_np[view_idx, 0]
            if np.all(np.isfinite(gt_pt)):
                ui_gt = int(round(float(gt_pt[0])))
                vi_gt = int(round(float(gt_pt[1])))
                if 0 <= ui_gt < W and 0 <= vi_gt < H:
                    cv2.circle(
                        blended,
                        (ui_gt, vi_gt),
                        radius=6,
                        color=(0, 255, 255),
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        blended,
                        "global_t",
                        (ui_gt + 6, vi_gt - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    coord_text = (
                        f"({global_t_world_np[0]:.2f}, {global_t_world_np[1]:.2f}, {global_t_world_np[2]:.2f})"
                    )
                    cv2.putText(
                        blended,
                        coord_text,
                        (ui_gt + 6, vi_gt + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        if draw_coordinate_axes and axes_projection_np is not None and axes_metadata is not None:
            origin_idx = axes_metadata["origin_idx"]
            origin_pt = axes_projection_np[view_idx, origin_idx]
            if np.all(np.isfinite(origin_pt)):
                overlay = blended.copy()
                axis_length_world = axes_metadata["axis_length"]

                for axis_name, meta in axes_metadata["axes"].items():
                    end_pt = axes_projection_np[view_idx, meta["end_idx"]]
                    if not np.all(np.isfinite(end_pt)):
                        continue

                    p0 = origin_pt.astype(np.float32)
                    p1 = end_pt.astype(np.float32)
                    axis_vec_px = p1 - p0
                    axis_len_px = np.linalg.norm(axis_vec_px)
                    if axis_len_px < 1e-3:
                        continue

                    axis_dir_unit = axis_vec_px / axis_len_px
                    perp_dir_unit = np.array([-axis_dir_unit[1], axis_dir_unit[0]], dtype=np.float32)
                    tick_length_px = min(4.0, axis_len_px * 0.05)
                    axis_line_thickness = 2

                    cv2.line(
                        overlay,
                        tuple(np.round(p0).astype(int)),
                        tuple(np.round(p1).astype(int)),
                        color=(255, 255, 255),
                        thickness=axis_line_thickness,
                        lineType=cv2.LINE_AA,
                    )

                    for frac, tick_idx in zip(meta["tick_fracs"], meta["tick_indices"]):
                        tick_pt = axes_projection_np[view_idx, tick_idx]
                        if not np.all(np.isfinite(tick_pt)):
                            continue

                        tick_center = tick_pt.astype(np.float32)
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

                        value = frac * axis_length_world
                        if axis_length_world >= 10:
                            text_value = f"{value:.0f}"
                        elif axis_length_world >= 1:
                            text_value = f"{value:.1f}".rstrip("0").rstrip(".")
                        else:
                            text_value = f"{value:.2f}".rstrip("0").rstrip(".")

                        text_anchor = tick_center + perp_dir_unit * (tick_length_px + 6.0)
                        cv2.putText(
                            overlay,
                            text_value,
                            tuple(np.round(text_anchor).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    label_anchor = p1 + axis_dir_unit * 12.0 + perp_dir_unit * 6.0
                    cv2.putText(
                        overlay,
                        axis_name.upper(),
                        tuple(np.round(label_anchor).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                origin_label_pos = origin_pt.astype(np.float32) + np.array([6.0, -6.0], dtype=np.float32)
                cv2.putText(
                    overlay,
                    "0",
                    tuple(np.round(origin_label_pos).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                blended = cv2.addWeighted(overlay, 0.6, blended, 0.4, 0)

        # Prepare output dirs & filename
        view_output_dir = os.path.join(outdir, view_names[view_idx] + "_reconstruction_images")
        instance_dir = os.path.join(view_output_dir, f"instance_{instance_number}")
        ensure_dir(instance_dir)
        base_name = os.path.splitext(os.path.basename(orig_image_paths[view_idx]))[0]
        out_path = os.path.join(instance_dir, base_name + "_reconstructed.png")

        # Save PNG (BGR)
        cv2.imwrite(out_path, blended)


def save_obj_model(
    outdir: str, sample_index: int, fish_place: int, vertex_posed: torch.Tensor, fish
) -> None:
    model_dir = os.path.join(outdir, "models")
    ensure_dir(model_dir)

    obj_path = os.path.join(model_dir, f"{sample_index}_out_model_{fish_place - 1}.obj")
    with open(obj_path, "w") as f:
        # vertices
        verts = vertex_posed[0]
        for x, y, z in verts:
            f.write(f"v {x} {y} {z}\n")
        # faces
        for a, b, c in fish.faces + 1:
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def save_pose_pickle(
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
    ensure_dir(pickle_dir)
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


def reconstruct(
    mesh_path: str,
    dataset_dir: str,
    outdir: str,
    frame_indices: list[int],
    instance_number: int,
    seed: int = 1,
    save_models: bool = False,
) -> None:
    """
    Run multiview reconstruction for given frames.
    """
    global device
    device = setup_device(seed)
    print("Device:", device)

    dataset = load_multiview_dataset(dataset_dir)

    camera_group_cpu = dataset.cams.get_camera_group()
    camera_group_device = camera_group_cpu.to(device)

    ensure_dir(outdir)
    fish, optimizer, renderer = initialize_model(
        mesh_path, device, camera_group_device
    )

    parameters = []
    sample_data = []
    start_idx = frame_indices[0]

    pbar = tqdm(total=len(frame_indices), desc=f"video {os.path.basename(dataset_dir)}")

    for idx in frame_indices:
        try:
            instance_sample = dataset.__getitem__(idx, instance_number)
        except IndexError:
            print(f"Sample {idx} missing, skipping")
            continue

        kpt_present_mask = instance_sample['kpt_present_mask']
        seg_mask_present_mask = instance_sample['seg_mask_present_mask']

        # QUESTION: how to deal with not enough keypoints/segmasks especially in first frame?
        # TODO: fallback to previous segmasks, keypoints
        if len([
                view_with_seg_mask for view_with_seg_mask in seg_mask_present_mask 
                if view_with_seg_mask == True
            ]) < 2:
            print(f"Less than two views with segmentation masks in sample for frame {idx} -> skipping")
            continue
        if len([
                view_with_kpts for view_with_kpts in kpt_present_mask 
                if any(kpt_present == True for kpt_present in view_with_kpts)
            ]) < 2:
            print(f"Less than two views with keypoints in sample for frame {idx} -> skipping")
            continue
        views_indices, orig_img_paths = instance_sample['frames'], instance_sample['imgpaths']
        keypoints, masks, bboxes = get_tensors_from_instance_sample(instance_sample)


        # initialize from previous solution if available
        init = parameters[-1] if parameters else None

        result = multiview.fit_mesh(
            fish,
            optimizer,
            camera_group_device,
            keypoints,
            masks,
            renderer,
            device,
            *([] if init is None else init),
            index=idx,
            bboxs=bboxes,
        )
        vertices_world_est, keypoints_world_est, global_t_est, global_ori_plus_pose_est, body_bone_est, scale_est, _ = result

        parameters += [global_ori_plus_pose_est[:, :3], global_ori_plus_pose_est[:, 3:], body_bone_est, scale_est, global_t_est]
        sample_data.append([views_indices, orig_img_paths, keypoints_world_est, bboxes, idx])

        out_reconstructed = fish(global_ori_plus_pose_est[:, :3], global_ori_plus_pose_est[:, 3:], body_bone_est, scale_est)
        reconstructed_keypoints_local = out_reconstructed["keypoints"].to(device)
        reconstructed_vertices_local = out_reconstructed["vertices"].to(device)

        save_reconstruction_images(
            orig_image_paths=orig_img_paths,
            outdir=outdir,
            renderer=renderer,
            instance_number=instance_number,
            cameras=camera_group_device,
            reconstructed_keypoints_local=reconstructed_keypoints_local,
            reconstructed_vertices_local=reconstructed_vertices_local,
            faces_from_vert_indices=fish.faces.unsqueeze(0).to(device),
            global_t=global_t_est.to(device),
            keypoint_names=dataset.index_json["keypoint_list"],
            view_names=dataset.views
        )

        if save_models:
            save_obj_model(outdir, idx, instance_number, vertices_world_est, fish)

        pbar.update(1)

    pbar.close()
    save_pose_pickle(
        outdir=outdir,
        start=frame_indices[0],
        end=frame_indices[-1],
        fish_place=instance_number,
        parameters=parameters,
        sample_data=sample_data,
        mesh_file=mesh_path,
        index=frame_indices,
    )


def render_pose_time_series(    
    mesh_path: str,
    dataset_dir: str,
    pose_time_series_file_path: str,
    outdir: str,
    deform: bool = False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fish = fish_model(mesh_path)
    fish.to_device(device)

    dataset = load_multiview_dataset(dataset_dir)
    camera_group_cpu = dataset.cams.get_camera_group()
    camera_group_device = camera_group_cpu.to(device)

    renderer = Silhouette_Renderer(device, camera_group_device)

    pose_time_series_outdir = os.path.join(outdir, "pose_time_series_rendered")
    ensure_dir(pose_time_series_outdir)
    with open(pose_time_series_file_path) as jf:
        pose_time_series_json = json.load(jf)
    frames = pose_time_series_json["frames"]

    for index, frame in enumerate(frames):
        sample = dataset.__getitem__(index)

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
        
        save_reconstruction_images(
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
            draw_verts=True
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
