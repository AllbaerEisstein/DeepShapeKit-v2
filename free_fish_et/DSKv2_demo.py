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
                # '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data/videos/bluegill_renders/left.mp4',
                # '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data/videos/bluegill_renders/bottom.mp4',
                # '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data/videos/bluegill_renders/top_right.mp4',
                '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data/videos/bluegill_renders/006_Positive Z (Fish Ventral Side).mp4',
                '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data/videos/bluegill_renders/004a_Positive X (Fish Front).mp4',
                '/home/jonathan/Documents/fish_reconstruction/deepshapekit-v2/bluegill_data/videos/bluegill_renders/003_Fish Top R.mp4',
            ]
    ]

    # segmentation_model_path = Path('src/DSKv2/segment_bluegill.pt')
    # pose_model_path         = Path('src/DSKv2/bluegill_pose.pt')
    segmentation_model_path = Path('src/DSKv2/cygill_seg.pt')
    pose_model_path         = Path('src/DSKv2/cygill_pose.pt')
    mesh_path               = 'DSKv2/Bluegill_Body.json'
    cam_matrices_path       = 'src/DSKv2/cam_matrices.json'
    out_path                = Path('src/results/cygill')
    dataset_folder_name     = 'dataset'
    dataset_folder_path     = (out_path / dataset_folder_name).absolute()
    final_output_folder     = 'src/results/output/'

    keypoint_list = [
        'mouth tip', 
        'gill', 
        'root of pelvic fin', 
        'caudal peduncle', 
        'middle of caudal fin', 
        'lower tip of caudal fin'
    ]
    
    kpt_name_dict = {
        index: kpt_name 
        for index, kpt_name in enumerate(keypoint_list)
    }


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
            kpt_names_dict  = kpt_name_dict
        )

    with open(os.path.join(dataset_folder_path, 'index.json')) as jf:
        video_meta = json.load(jf)

    # ==============================

    instance_number = 0
    seed            = 700
    save_models     = True

    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder, exist_ok=True)


    n_frames = video_meta['image_count']
    index = list(range(n_frames))

    if 'reconstruct' in steps:
        print('reconstructing mesh model sequence...')
        reconstruct(    
            mesh_path       = mesh_path,
            dataset_dir     = str(dataset_folder_path),
            outdir          = final_output_folder,
            frame_indices           = index,
            instance_number = instance_number,
            seed            = seed,
            save_models     = save_models
        )

    # ==============================