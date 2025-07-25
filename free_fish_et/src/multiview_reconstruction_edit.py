import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch


import src.multiview as multiview
import src.multiview_utils as mutil

from tqdm import tqdm
from src.fish_model_edit import fish_model
from src.pose_optimizer_edit import OptimizeMV
from src.Silhouette_Renderer import Silhouette_Renderer
from src.dataloaders_edit import Multiview_Dataset


def setup_device(seed: int) -> str:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_multiview_dataset(root: str) -> Multiview_Dataset:
    return Multiview_Dataset(root=root)


def initialize_model(mesh_file: str, device: str) -> tuple:
    fish = fish_model(mesh=mesh_file)
    optimizer = OptimizeMV(
        num_iters=100,
        lim_weight=200,
        prior_weight=30,
        bone_weight=200,
        mask_weight=200,
        smooth_weights=[100, 100, 20],
        device=device,
        fish_model_obj=fish
    )
    renderer = Silhouette_Renderer(device=device)
    return fish, optimizer, renderer


def process_frame(sample: dict, instance_number: int):
    """
        Just read a dict and return relevant contents.
    """
    if instance_number <= sample['instances']:
        keypoints = sample['keypoints'][instance_number]
        masks = sample['masks_full'][instance_number]
        bboxes = sample['bboxes'][instance_number]
    else:
        raise ValueError(f'Invalid fish_place: {instance_number}')

    # Normalize mask to [0,1] on appropriate device
    masks = masks.to(device) / 255.
    return sample['frames'], sample['imgpaths'], keypoints, masks, bboxes


def save_rendered_views(outdir: str, fish_place: int, sample_index: int,
                        img_filenames: list, vertex_posed: torch.Tensor,
                        fish, frames: list, keypoints, bboxes) -> None:
    view_dir = os.path.join(outdir, f'images_fish_{fish_place}')
    ensure_dir(view_dir)

    for view_idx, img_path in enumerate(img_filenames):
        img = plt.imread(img_path) * 255
        img_pose, _ = mutil.render_vertex_on_frame(
            img, [vertex_posed], fish, [frames[view_idx]], [keypoints], bboxes[view_idx]
        )
        save_path = os.path.join(view_dir, f'{sample_index}_view_{view_idx}.png')
        plt.imsave(save_path, img_pose)


def save_obj_model(outdir: str, sample_index: int, fish_place: int,
                   vertex_posed: torch.Tensor, fish) -> None:
    model_dir = os.path.join(outdir, 'models')
    ensure_dir(model_dir)

    obj_path = os.path.join(
        model_dir,
        f'{sample_index}_out_model_{fish_place - 1}.obj'
    )
    with open(obj_path, 'w') as f:
        # vertices
        verts = vertex_posed[0]
        for x, y, z in verts:
            f.write(f'v {x} {y} {z}\n')
        # faces
        for a, b, c in fish.faces + 1:
            f.write(f'f {a}//{a} {b}//{b} {c}//{c}\n')


def save_pose_pickle(
        outdir: str, 
        start: int, 
        end: int, 
        fish_place: int,
        parameters: list, 
        sample_data: list, 
        mesh_file: str,
        index: list
) -> None:
    pickle_dir = os.path.join(outdir, 'pose_pickle')
    ensure_dir(pickle_dir)
    fname = f'pose_result_{start}-{end+1}_({fish_place}).pickle'
    path = os.path.join(pickle_dir, fname)

    data = {
        'individual_fit_parameters': parameters,
        'sample_data': sample_data,
        'indices': index,
        'model_file': mesh_file
    }
    with open(path, 'wb') as pf:
        pickle.dump(data, pf, protocol=pickle.HIGHEST_PROTOCOL)


def reconstruct(    
        mesh_path: str,
        dataset_dir: str,
        outdir: str,
        index: list[int],
        instance_number: int,
        seed: int = 1,
        save_models: bool = False
) -> None:
    """
    Run multiview reconstruction for given frames.
    """
    global device 
    device = setup_device(seed)
    print('Device:', device)

    ensure_dir(outdir)
    fish, optimizer, renderer = initialize_model(mesh_path, device)
    dataset = load_multiview_dataset(dataset_dir)

    parameters = []
    sample_data = []
    start_idx = index[0]

    pbar = tqdm(total=len(index), desc=f'video {os.path.basename(dataset_dir)}')

    for idx in index:
        try:
            sample = dataset[idx]
        except IndexError:
            print(f'Sample {idx} missing, skipping')
            continue

        if not sample['full_kpts']:
            print(f"Frame {idx} missing keypoints, using last valid params")
            # replicate last params
            parameters.extend(parameters[-4:])
            sample_data.append(sample_data[-1] if sample_data else [0])
            pbar.update(1)
            continue

        frames, img_paths, keypoints, masks, bboxes = process_frame(sample, instance_number)
        # initialize from previous solution if available
        init = None  # (ori, pose, bone, scale, trans) unpacked inside multiview.fit_mesh

        result = multiview.fit_mesh(
            fish, optimizer, keypoints, frames, masks,
            renderer, device, *([] if init is None else init),
            img_filenames=img_paths, index=idx, bboxs=bboxes
        )
        vertex_posed, _, t, body_pose, bone, scale, _ = result

        parameters += [body_pose, bone, scale, t]
        sample_data.append([frames, img_paths, keypoints, bboxes, idx])

        save_rendered_views(
            outdir, instance_number, idx,
            img_paths, vertex_posed, fish, frames, keypoints, bboxes
        )

        if save_models:
            save_obj_model(outdir, idx, instance_number, vertex_posed, fish)

        pbar.update(1)

    pbar.close()
    save_pose_pickle(
        outdir      = outdir, 
        start       = index[0], 
        end         = index[-1],
        fish_place  = instance_number, 
        parameters  = parameters, 
        sample_data = sample_data, 
        mesh_file   = mesh_path, 
        index       = index
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', nargs='+', type=int, default=list(range(50, 100)))
    parser.add_argument('--mesh', type=str, default='goldfish_design_small.json')
    parser.add_argument('--outdir', type=str, default='data/output/GoldFish20171216_BL320/20171216_124610')
    parser.add_argument('--datadir', type=str, default='data/input/video_frames_20171215_101550')
    parser.add_argument('--fish_place', type=int, choices=[1,2], default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_models', action='store_true')
    args = parser.parse_args()

    reconstruct(
        mesh_path=args.mesh,
        dataset_dir=args.datadir,
        outdir=args.outdir,
        index=args.index,
        instance_number=args.fish_place,
        seed=args.seed,
        save_models=args.save_models
    )
