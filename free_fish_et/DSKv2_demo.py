import os
import argparse
from pathlib import Path
from src.extract_frames_edit import *
from src.multiview_reconstruction_edit import reconstruct

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', nargs='+', default=['extract', 'masks', 'keypoints', 'reconstruct'])
    args = parser.parse_args()
    steps = args.steps

    videos = [
        Path(path) for path in 
            [
                '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data/videos/05142025_4-cam-1_0s-3s_rotcc90.mp4',
                '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data/videos/05142025_4-cam-2_0s-3s_rotc90.mp4',
                '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data/videos/05142025_4-cam-4_0s-3s.mp4',
            ]
    ]

    segmentation_model_path = Path('src/DSKv2/segment_bluegill.pt')
    pose_model_path         = Path('src/DSKv2/bluegill_pose.pt')
    out_path                = Path('src/results')
    dataset_folder_name     = 'dataset'
    dataset_folder_path     = out_path / dataset_folder_name
    final_output_folder     = 'src/results/output/'


    # ====== preprocessing and dataset creation ======

    if 'extract' in steps:
        print('extracting frames from video...')
        extract_from_video(
            videos, 
            out_path, 
            dataset_folder_name, 
            also_create_frame2video_csv=True
        )

    if 'masks' in steps:
        print('detecting fish masks...')
        predict_masks_yolo(
            dataset_path    = dataset_folder_path, 
            model_path      = segmentation_model_path, 
            conf_threshold  = 0.8
        )

    if 'keypoints' in steps:
        print('detecting fish keypoints...')
        detect_keypoints_yolo(
            dataset_path    = dataset_folder_path,
            model_path      = pose_model_path,
            yolo_env_name   = 'yolo'
        )

    with open(os.path.join(dataset_folder_path, 'index.json')) as jf:
        video_meta = json.load(jf)

    # ==============================


    print('reconstructing mesh model sequence...')

    mesh_path       = 'bluegill_mesh.json'
    instance_number = 0
    seed            = 700
    save_models     = True

    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder, exist_ok=True)


    n_frames = video_meta['image_count']
    index = list(range(n_frames))

    if 'reconstruct' in steps:
        reconstruct(    
            mesh_path       = mesh_path,
            dataset_dir     = str(dataset_folder_path),
            outdir          = final_output_folder,
            index           = index,
            instance_number = instance_number,
            seed            = seed,
            save_models     = save_models
        )

    # ==============================