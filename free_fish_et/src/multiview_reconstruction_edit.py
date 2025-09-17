import json
import pickle
import os
import argparse
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


from src.constants_edit import BLENDERCAM_2_CV
import src.multiview_edit as multiview
import src.multiview_utils_edit as mutil

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
    mesh_file: str, device: str, image_size, Ks, Rs, Ts, focals
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
    renderer = Silhouette_Renderer(device, image_size, Ks, Rs, Ts, focals)
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
    Ps,
    Ks,
    Rs,
    Ts,
    focals,
    distortions,
    principal_points,
    keypoints,
    bboxes,
) -> None:
    instance_dir = os.path.join(outdir, f"reconstruction_images_{instance_number}")
    print(f"vertices size: {mesh_vertices_after_reconstruction.size()}")
    print(f"keypoints size: {keypoints.size()}")
    ensure_dir(instance_dir)
    imgs = torch.stack([torch.tensor(plt.imread(img_path)*255) for img_path in img_filenames])
    imgs_with_projection = mutil.batch_render_reconstructions(
        imgs,
        mesh_vertices_after_reconstruction,
        Ps,
        Ks,
        Rs,
        Ts,
        focals,
        distortions,
        principal_points,
        keypoints,
        bboxes,
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
    Ps: torch.Tensor,  # (N,3,4)
    reconstructed_keypoints_local: torch.Tensor,  # (1, K, 3)
    reconstructed_vertices_local: torch.Tensor,  # (1, V, 3)
    faces_from_vert_indices: torch.Tensor,  # (1, N, 3)
    global_t: torch.Tensor, # (3)
    keypoint_names: List[str],
    view_names: List[str],
    silhouette_threshold: float = 0.01,  # tiny alpha cutoff
    blend_factor: float = 0.4,           # overlay opacity (40%)
):
    """
    Render silhouettes, pad originals (zero padding) to silhouette size (centered),
    overlay silhouette in red with given blend_factor, draw keypoints (blue), and save.
    """

    silhouettes = renderer(reconstructed_vertices_local, faces_from_vert_indices, global_t)

    silhouettes_np = silhouettes.detach().cpu().numpy()

    alpha = silhouettes_np  # (N, W, H)
    n_views, W, H = alpha.shape

    assert len(orig_image_paths) == n_views, "orig_image_paths length must match rendered views"
    assert Ps.shape[0] == n_views, "Ps batch size must match number of views"
    assert len(view_names) == n_views, "Number of view names must match number of views"

    for i in range(n_views):
        # Read original image (BGR)
        orig_bgr = cv2.imread(str(orig_image_paths[i]), cv2.IMREAD_COLOR)
        if orig_bgr is None:
            raise FileNotFoundError(f"Could not read image {orig_image_paths[i]}")

        # cv2 reads images in row-major order
        h0, w0 = orig_bgr.shape[:2]
        # Compute integer padding to center the resized image within (W,H)
        pad_left = (W - w0) // 2
        pad_right = W - w0 - pad_left
        pad_top = (H - h0) // 2
        pad_bottom = H - h0 - pad_top
        
        # print(f"alpha: {alpha.shape}")
        # print(f"left, right, top, bottom: {pad_left}, {pad_right}, {pad_top}, {pad_bottom}")
        # print(f"orig_bgr: {orig_bgr.shape}")


        # Pad with zeros (black). cv2.copyMakeBorder expects ints
        padded = cv2.copyMakeBorder(
            orig_bgr,
            top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        a = np.clip(alpha[i], 0.0, 1.0)  # (H,W), float

        # threshold tiny values
        a[a < silhouette_threshold] = 0.0

        # Build red overlay image (BGR) and blend: out = orig*(1 - bf * a) + red*(bf * a)
        red_img = np.zeros_like(padded)
        red_img[:, :, 2] = 255  # full red channel in BGR


        a_exp = a[..., None].transpose(1,0,2)  # (H,W,1)
        blended = (padded.astype(np.float32) * (1.0 - blend_factor * a_exp) +
                   red_img.astype(np.float32) * (blend_factor * a_exp))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Draw keypoints:
        # print(f"reconstructed_keypoints_world: {reconstructed_keypoints_local}")
        reconstructed_keypoints_local = (reconstructed_keypoints_local + global_t).squeeze(0)
        P_i = Ps[i].detach().cpu().numpy()  # (3,4)
        for kp_idx, name in enumerate(keypoint_names):
            coords3d = reconstructed_keypoints_local[kp_idx]
            # print(f"coords3d: {coords3d}")
            # print(f"reconstructed_keypoints_world: {reconstructed_keypoints_local}")
            coords_np = coords3d.detach().cpu().numpy()
            Xh = np.concatenate([coords_np, [1.0]])  # (4,)
            ph = P_i @ Xh  # (3,)
            z = ph[2]
            if abs(z) < 1e-6:
                continue
            u = ph[0] / z
            v = ph[1] / z

            # u,v are pixel coords in the same coordinate system as the silhouette (W,H)
            ui = int(round(u))
            vi = int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                cv2.circle(blended, (ui, vi), radius=5, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(blended, name, (ui, vi + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.35, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # Prepare output dirs & filename
        view_output_dir = os.path.join(outdir, view_names[i] + "_reconstruction_images")
        instance_dir = os.path.join(view_output_dir, f"instance_{instance_number}")
        ensure_dir(instance_dir)
        base_name = os.path.splitext(os.path.basename(orig_image_paths[i]))[0]
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

    cam_params = dataset.cams.get_camera_matrices()
    Ks = cam_params[1]
    principal_points = torch.stack((Ks[:, 0, 2], Ks[:, 1, 2]), dim=1)
    image_size = torch.tensor(dataset.uniform_img_size)

    ensure_dir(outdir)
    fish, optimizer, renderer = initialize_model(
        mesh_path, device, image_size, *cam_params[1:5]
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


        # if not sample["full_kpts"]:
        #     print(f"Frame {idx} missing keypoints, using last valid params")
        #     # replicate last params
        #     parameters.extend(parameters[-4:])
        #     sample_data.append(sample_data[-1] if sample_data else [0])
        #     pbar.update(1)
        #     continue


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
        init = (
            None  # (ori, pose, bone, scale, trans) unpacked inside multiview.fit_mesh
        )

        result = multiview.fit_mesh(
            fish,
            optimizer,
            *cam_params,
            principal_points,
            keypoints,
            masks,
            renderer,
            device,
            *([] if init is None else init),
            index=idx,
            bboxs=bboxes,
        )
        reconstructed_vertices_local, reconstructed_keypoints_local, global_t_est, global_ori_plus_pose_est, body_bone_est, scale_est, _ = result

        parameters += [global_ori_plus_pose_est, body_bone_est, scale_est, global_t_est]
        sample_data.append([views_indices, orig_img_paths, reconstructed_keypoints_local, bboxes, idx])

        # save_rendered_views(
        #     outdir,
        #     instance_number,
        #     idx,
        #     img_paths,
        #     mesh_vertices_after_reconstruction,
        #     *cam_params,
        #     principal_points,
        #     keypoints_after_reconstruction,
        #     bboxes,
        # )

        print(f"keypoints_after_reconstruction: {reconstructed_keypoints_local}")


        save_reconstruction_images(
            orig_image_paths=orig_img_paths,
            outdir=outdir,
            renderer=renderer,
            instance_number=instance_number,
            Ps=cam_params[0],
            reconstructed_keypoints_local=reconstructed_keypoints_local,
            reconstructed_vertices_local=reconstructed_vertices_local,
            faces_from_vert_indices=fish.faces.unsqueeze(0),
            global_t=global_t_est,
            keypoint_names=dataset.index_json["keypoint_list"],
            view_names=dataset.views
        )

        if save_models:
            save_obj_model(outdir, idx, instance_number, reconstructed_vertices_local, fish)

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
    cam_params = dataset.cams.get_camera_matrices()
    Ps, Ks, Rs, ts, focals, distortions = cam_params
    print(f"before: {Ps}")
    Ks = Ks.to(device)
    Rt = torch.cat([cam_params[2], cam_params[3].reshape(-1, 3, 1)], dim=2)
    Rt = Rt.to(device)
    Ps = Ks @ BLENDERCAM_2_CV.to(device=device, dtype=torch.float32).unsqueeze(0).expand(Rs.size(0),-1,-1) @ Rt
    Ps = Ks @ Rt
    print(f"after: {Ps}")
    image_size = torch.tensor(dataset.uniform_img_size, device=device)

    renderer = Silhouette_Renderer(device, image_size, *[param.to(device) for param in cam_params[1:-1]])

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

        articulated_verts_kpts = fish(global_ori.unsqueeze(0), body_pose.unsqueeze(0).flatten(1), bone_length.unsqueeze(0), deform=deform)
        keypoints = articulated_verts_kpts["keypoints"].to(device)
        vertices = articulated_verts_kpts["vertices"].to(device)

        silhouettes = renderer(vertices, fish.faces.unsqueeze(0), global_t)
        silhouettes_np = silhouettes.detach().cpu().numpy()

        for i in sample["frames"]:
            base_name = os.path.splitext(os.path.basename(sample["imgpaths"][i]))[0]
            out_path = os.path.join(pose_time_series_outdir, "just_silhouette_overlays")
            ensure_dir(out_path)
            a = np.clip(silhouettes_np[i], 0.0, 1.0)  # (H,W), float
            a_exp = a[..., None]
            cv2.imwrite(img=a_exp, filename=os.path.join(out_path, base_name+".png"))
        
        save_reconstruction_images(
            orig_image_paths=sample["imgpaths"],
            outdir=pose_time_series_outdir,
            renderer=renderer,
            instance_number=0,
            Ps=Ps,
            reconstructed_keypoints_local=keypoints,
            reconstructed_vertices_local=vertices,
            faces_from_vert_indices=fish.faces.unsqueeze(0),
            global_t=global_t,
            keypoint_names=dataset.index_json["keypoint_list"],
            view_names=dataset.views
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
