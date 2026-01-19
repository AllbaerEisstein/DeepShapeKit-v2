from collections import defaultdict
from typing import Optional
import cv2
import csv
import json
import os
import numpy as np
from tqdm import trange
from tqdm import tqdm
import pickle
from pathlib import Path
from PIL import Image, ImageDraw

import src.infer_mask as infer_mask
import src.infer_keypoints as infer_keypoints
from src.parse_cams_json import (
    KEY_ALIASES,
    _find_any,
    _parse_distortion,
    #_parse_focal,
    _parse_K,
)
from src.types import *


def dynamic_print(msg):
    length = len(dynamic_print.last_msg) if hasattr(dynamic_print, 'last_msg') else 0
    print(' ' * length, end='\r')  # Clear the line
    print(msg, end='\r')
    dynamic_print.last_msg = msg  # Save for next overwrite



def extract_from_video(
        videos: list[Path], 
        cam_matrices_json_path: Path, 
        out_dir: Path, dataset_folder_name: str = 'dataset', 
        also_create_frame2video_csv: bool = False, 
        undistort: bool = False, 
        frame_indices: Optional[List[int]] = None
    ):
    """
    📂 Expected/Assumed Directory Structure Before Execution:
        - out_dir exists. If not, the function raises an exception.
        - The videos exist at the paths listed in videos.
    
    Creates the following directory structure and contents:
        out_dir/
        └── dataset/
            ├── index.json  ← Summary file
            └── <video_name_1>/
                ├── origin/
                │   ├── <video_name_1>_0.png
                │   ├── ...
                └── files.csv
            └── <video_name_2>/
                ...

    files.csv:
        Keeps track of the extracted frames.
        One row per extracted frame.
        Columns: 
            - 'frame' - frame number
            - 'file_loc' - relative path to the saved file ("{video_name}_{frame_number}.png"
            - 'category' - field is always set to 'origin'
            - 'sub_index' - field is always set to 0
            - 'folder' - video name
    """

    if not out_dir.exists():
        raise Exception('Output dir does not exist')

    destination = out_dir / dataset_folder_name
    destination.mkdir(exist_ok=True)
    max_frame_number = max(frame_indices) if frame_indices is not None else -1

    json_out_path = destination / 'index.json'
    json_index = {
        'frame_folders': [],
        'index_files': {},
        'image_sizes': {}
    }

    with open(cam_matrices_json_path) as jf:
        cam_matrices = json.load(jf)

    for video_path in videos:
        if not video_path.exists():
            raise Exception(f'Input video does not exist: {video_path}')

        print(f"processing video: {video_path}")

        # read one frame to get vheight and vwidth
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"couldn't open video: {video_path}")
            success, frame = capture.read()
            if not success:
                raise ValueError(f"video {video_path} couldn't be read.")
            vheight, vwidth = frame.shape[:2]
        finally:
            capture.release()

        base_video_name = video_path.stem
        video_name = base_video_name

        needs_undistortion = False
        K = None
        distortions = None
        newK = None
        new_focal_mm = None
        undistort_coeffs = None
        if undistort:
            matrices_json = cam_matrices.get(base_video_name, None)
            if matrices_json is None:
                raise ValueError(f'No camera matrices found for view "{base_video_name}" in {cam_matrices_json_path}')

            K_raw = _find_any(matrices_json, KEY_ALIASES['K'])
            dist_raw = _find_any(matrices_json, KEY_ALIASES['distortion'])
            #focal_raw = _find_any(matrices_json, KEY_ALIASES['f'])
            try:
                K = _parse_K(K_raw)
            except Exception as exc:
                raise ValueError(f'Intrinsic matrix for view "{base_video_name}" could not be parsed: {exc}')
            if K is None:
                raise ValueError(f'No intrinsic matrix found for view "{base_video_name}" in {cam_matrices_json_path} (checked aliases: {KEY_ALIASES["K"]})')

            try:
                distortions = _parse_distortion(dist_raw)
            except Exception as exc:
                raise ValueError(f'Distortion coefficients for view "{base_video_name}" could not be parsed: {exc}')

            # try:
            #     focal_pair = _parse_focal(focal_raw) if focal_raw is not None else None
            # except Exception as exc:
            #     raise ValueError(f'Focal length for view "{base_video_name}" could not be parsed: {exc}')

            undistort_coeffs = distortions
            if distortions == (0.0,) * 5:
                print(f"No undistortion possible or necessary for video {base_video_name}. Distortion coefficients not specified or all 0.0.")
            else:
                # TODO: Which camera parameters are actually changed by getOptimalNewCameraMatrix?
                newK, _ = cv2.getOptimalNewCameraMatrix(
                    np.array(K, dtype=float),
                    np.array(distortions, dtype=float),
                    [int(vwidth), int(vheight)],
                    1,
                    [int(vwidth), int(vheight)],
                    False
                )

                # fx_px = K[0][0]
                # fy_px = K[1][1] if K.shape[0] > 1 else K[0][0]
                # fx_mm = newK[0][0] * (focal_pair[0] / fx_px) if fx_px != 0 else None
                # fy_mm = newK[1][1] * (focal_pair[1] / fy_px) if fy_px != 0 else None
                # if fx_mm is not None and fy_mm is not None:
                #     new_focal_mm = [fx_mm, fy_mm]
                # elif fx_mm is not None:
                #     new_focal_mm = fx_mm
                # elif fy_mm is not None:
                #     new_focal_mm = fy_mm
                new_dist = (0.0,) * 5
                video_name = base_video_name + "_undistorted"
                original_entry = cam_matrices.get(base_video_name, {})
                updated_entry = {
                    "K": [list(row) for row in newK],
                    #"f": new_focal_mm,
                    "distortion": {
                        "rad_1": new_dist[0],
                        "rad_2": new_dist[1],
                        "tan_1": new_dist[2],
                        "tan_2": new_dist[3],
                        "rad_3": new_dist[4],
                    },
                }
                for aliases, key_name in [
                    (KEY_ALIASES['R'], 'R'),
                    (KEY_ALIASES['T'], 't'),
                    (KEY_ALIASES['P'], 'P'),
                ]:
                    val = _find_any(original_entry, aliases)
                    if val is not None:
                        updated_entry[key_name] = val
                if "Rt" in original_entry:
                    updated_entry["Rt"] = original_entry["Rt"]
                for fb_key in ["FROM_BLENDERWORLD", "from_blenderworld"]:
                    if fb_key in original_entry:
                        updated_entry["FROM_BLENDERWORLD"] = original_entry[fb_key]
                        break
                cam_matrices[video_name] = updated_entry
                needs_undistortion = True


        video_folder    = destination / video_name
        origin_folder   = video_folder / 'origin'
        origin_folder.mkdir(parents=True, exist_ok=True)

        files_csv_path  = video_folder / 'files.csv'
        with files_csv_path.open('w', newline='') as csv_out_file:
            csvwriter = csv.writer(csv_out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder'])

            capture = cv2.VideoCapture(str(video_path))

            image_count = 0
            frame_number = 0
            success = True

            try:
                if not capture.isOpened():
                    raise ValueError(f"couldn't open video: {video_path}")
                while capture.isOpened() and success:
                    success, frame = capture.read()
                    if not success:
                        break
                    if frame_indices is not None:
                        if frame_number > max_frame_number:
                            break
                        if frame_number not in frame_indices:
                            frame_number += 1
                            continue
                    if undistort and needs_undistortion:
                        frame = cv2.undistort(frame, np.array(K), undistort_coeffs, None, np.array(newK))
                    
                    filename = f"{video_name}_{frame_number}.png"
                    abs_file_path = origin_folder / filename
                    rel_file_path = Path(video_name) / 'origin' / filename

                    cv2.imwrite(str(abs_file_path), frame)
                    csvwriter.writerow([
                        frame_number,
                        str(rel_file_path),
                        'origin',
                        0,
                        video_name
                    ])
                    image_count += 1

                    frame_number += 1
                    dynamic_print(f'frame out: {frame_number}, total image: {image_count}')

                print(f'total image: {image_count}, done')

                json_index['frame_folders'].append(video_name)
                json_index['index_files'][video_name] = str(files_csv_path)
                json_index['image_sizes'][video_name] = [int(vwidth), int(vheight)]
            finally:
                capture.release()
    
        if also_create_frame2video_csv:
            # create a mapping 1:1 from original frame numbers to new frame numbers 
            # NOTE: this is a stub in case frame selection is implemented later
            frame2video_csv_path = video_folder / 'frame2video_1.csv'
            with frame2video_csv_path.open('w', newline='') as csv_out_file:
                csvwriter = csv.writer(csv_out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['origin_frame', 'new_frame'])

                for i in range(frame_indices[-1] + 1 if frame_indices is not None else image_count):
                    csvwriter.writerow([i, i])

    json_index['camera_matrices'] = cam_matrices

    json_index['status'] = 'origin'
    json_index['image_count'] = image_count

    with json_out_path.open("w") as f:
        json.dump(json_index, f, indent=2)


def get_image_np_from_path(image_path: str) -> np.ndarray:
    """
    Load a PIL image and convert it to a numpy array.
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)

def get_image_np_from_Image(img: Image.Image) -> np.ndarray:
    """
    Load a PIL image and convert it to a numpy array.
    """
    #img = img.convert("RGB")
    return np.array(img.getdata(0)).reshape(img.size[1], img.size[0])


def polygon_to_binary_mask(polygon, image_size, mode='1', fill=1, background=0) -> np.ndarray:
    """
    Create a binary (mask) PIL Image from a polygon.

    Parameters:
    -------------
    polygon : list of [x, y]
        List of 2D coordinates defining the polygon. Example:
        [[896.875, 578.125], [896.875, 617.1875], ..., [915.625, 578.125]]

    image_size : tuple of ints (width, height)
        Size of the output image in pixels.

    mode : str, default '1'
        Image mode, '1' for 1-bit pixels, black and white.
        Use 'L' for 8-bit grayscale if preferred.

    fill : int, default 1
        Pixel value to fill the polygon with (e.g., 1 or 255).

    background : int, default 0
        Pixel value for the background.

    Returns:
    --------
    PIL.Image.Image
        Binary mask image with the polygon filled.
    """
    # Convert float coordinates to integers
    coords = [(int(round(x)), int(round(y))) for x, y in polygon]

    # Create blank image
    mask_img = Image.new(mode, image_size, background)
    draw = ImageDraw.Draw(mask_img)

    # Draw filled polygon
    draw.polygon(coords, outline=fill, fill=fill)
    return np.array(mask_img, dtype=np.uint8)


# TODO: Instance tracking across views
def predict_masks_yolo(dataset_path: Path, model_path: Path, conf_threshold=0.8, frame_indices=None):
    """
    Use a pre-trained YOLO11n-seg model (model_path) to infer the segmentation masks of the dataset.
    This step is responsible for instance identification! 
    The subsequent steps rely on the correctness of the detected number of instances from this step.

    Augments the dataset as follows:

    <dataset_path>/
    ├── index.json          # Summary file already created in extract_frames()
    └── <frame_folder>/
        ├── origin/         # unchanged: all raw .png frames
        │   ├── ... N files of size HxWx3 (e.g., 2048x1040 RGB)
        ├── cropped/        # N_fish x N frames
        │   ├── image_0_0.png        # crop of fish #0 in frame 0, padded to square
        │   ├── image_0_1.png        # fish #1 in frame 0
        │   ├── image_1_0.png        # etc.
        │   └── ... up to 2 crops per frame (bounding boxes squared and padded)
        ├── mask/           # same count as `cropped/`
        │   ├── image_0_0_mask.png   # tight binary mask crop aligned with image_0_0.png
        │   ├── image_0_1_mask.png
        │   └── ...
        ├── mask_full/      # raw masks full-frame (optional: you could save them here)
        │   └── ... one full-frame mask per detection, size HxW (2048x1040)
        ├── bbox-masked_image/      # images at original size as but black everywhere
        │   │                         except inside the bounding boxes, where the original content is kept.
        |   ├── image_0_1_bbox-masked.png
        │   └── ... one full-frame bbox-masked image per detection, size HxW
        ├── files.csv       ← unchanged; already created in extract_frames()
        └── files_crop.csv  # one CSV per folder

    files_crop.csv:
        One row per saved crop or mask.
        Columns:
            - frame — original frame number
            - file_loc — relative path to the saved file (cropped or mask)
            - category — 'cropped', 'mask' or 'bbox-masked'
            - sub_index — detection index (number of the instance detected)
            - folder — the <frame_folder> name
            - bbox — [xmin, ymin, xmax, ymax] from the mask

        Example files_crop.csv:
            frame,file_loc,category,sub_index,folder,bbox
            0,video1/cropped/image_0_0.png,cropped,0,video1,"[120, 200, 360, 450]"
            0,video1/mask/image_0_0_mask.png,mask,0,video1,"[120, 200, 360, 450]"
            0,video1/cropped/image_0_1.png,cropped,1,video1,"[500, 100, 760, 420]"
            0,video1/mask/image_0_1_mask.png,mask,1,video1,"[500, 100, 760, 420]"
            1,video1/cropped/image_1_0.png,cropped,0,video1,"[130, 210, 370, 460]"
            ...
    """
    def crop_and_pad(
            image: np.ndarray,
            mask: np.ndarray,
            bbox: list[int]
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            1) Crops `image` and `mask` to the rectangle defined by `bbox`.
            2) Pads the shorter side so that the result is square.

            Args:
            image: HxWx3 RGB array.
            mask:  HxW binary mask array (0/1 or 0/255).
            bbox:  [xmin, ymin, xmax, ymax]

            Returns:
            (cropped_image, cropped_mask), both as square numpy arrays.
            """
            xmin, ymin, xmax, ymax = bbox
            # 1) crop
            crop_img  = image[      ymin:ymax,      xmin:xmax     ]
            crop_mask = mask[      ymin:ymax,      xmin:xmax     ]

            h, w = crop_img.shape[:2]
            diff = abs(h - w)

            # 2) pad to square
            if h < w:
                pad_top    = diff // 2
                pad_bottom = diff - pad_top
                crop_img  = np.pad(crop_img,
                                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                                mode='constant', constant_values=0)
                crop_mask = np.pad(crop_mask,
                                ((pad_top, pad_bottom), (0, 0)),
                                mode='constant', constant_values=0)
            elif w < h:
                pad_left  = diff // 2
                pad_right = diff - pad_left
                crop_img  = np.pad(crop_img,
                                ((0, 0), (pad_left, pad_right), (0, 0)),
                                mode='constant', constant_values=0)
                crop_mask = np.pad(crop_mask,
                                ((0, 0), (pad_left, pad_right)),
                                mode='constant', constant_values=0)

            return crop_img, crop_mask

    def save_crops(
            dataset_path: Path,
            folder: str,
            frame_counter: int,
            det_idx: int,
            cropped_img: np.ndarray,
            cropped_mask: np.ndarray,
            full_mask_np: np.ndarray
        ) -> None:
            """
            Saves:
            - the squared crop image,
            - the tight binary mask crop,
            - the full-frame mask.
            """
            base = dataset_path / folder

            # 1) crop image
            crop_path = os.path.join(
                base, 'cropped', f'image_{frame_counter}_{det_idx}.png'
            )
            Image.fromarray(cropped_img.astype(np.uint8)).save(crop_path)

            # 2) tight mask crop
            mask_path = os.path.join(
                base, 'mask', f'image_{frame_counter}_{det_idx}_mask.png'
            )
            Image.fromarray((cropped_mask * 255).astype(np.uint8)).save(mask_path)

            # 3) full-frame mask
            full_mask = (full_mask_np * 255).astype(np.uint8)
            full_mask_path = os.path.join(
                base, 'mask_full', f'image_{frame_counter}_{det_idx}_mask_full.png'
            )
            Image.fromarray(full_mask).save(full_mask_path)

    def run_infer_mask(
            model,
            input_path: Path,
            frame_indices: Optional[List[int]] = None,
            frame_number_by_name: Optional[dict[str, int]] = None
        ) -> dict[str, dict[str, list]]:
        frame_indices_set = set(frame_indices) if frame_indices is not None else None
        img_path2result = {}
        for img in Path(input_path).iterdir():
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                if frame_indices_set is not None:
                    frame_number = None
                    if frame_number_by_name is not None:
                        frame_number = frame_number_by_name.get(img.name)
                    if frame_number is None or frame_number not in frame_indices_set:
                        continue
                img_path2result[str(img)] = {
                    "bboxes": [],
                    "masks_xy": [],
                    "confs": []
                }
                bboxes, masks_xy, confs = infer_mask.inference(model, img)
                img_path2result[str(img)]["bboxes"]   = bboxes
                img_path2result[str(img)]["masks_xy"] = masks_xy
                img_path2result[str(img)]["confs"]    = confs
        return img_path2result

    
    # Read index.json
    with open(dataset_path / 'index.json') as jf:
        idx_json = json.load(jf)
    image_folders   = idx_json['frame_folders']
    max_n_instances = 0

    model = infer_mask.load_model(Path(model_path))

    for folder in image_folders:
        print(f"   processing frames for video {folder}...")
        image_size      = idx_json['image_sizes'][folder]
        for new_dir in ['cropped', 'mask', 'mask_full', 'bbox-masked_image']:
            if not os.path.exists(dataset_path / folder / new_dir):
                os.mkdir(dataset_path / folder / new_dir)

        csv_rows = []  # <-- Collect rows here instead of writing immediately

        files_csv_rows = list(csv.DictReader(open(dataset_path / folder / "files.csv")))
        frame_number_by_name = {
            Path(row["file_loc"]).name: int(row["frame"])
            for row in files_csv_rows
        }
        available_frames = sorted(frame_number_by_name.values())
        frame_indices_set = set(frame_indices) if frame_indices is not None else None
        target_frames = (
            [frame for frame in available_frames if frame in frame_indices_set]
            if frame_indices_set is not None
            else available_frames
        )
        frame_range_label = f"{target_frames[0]}-{target_frames[-1]}" if target_frames else "no-frames"
        img_path2prediction: dict[str, dict[str, list]] = run_infer_mask(
            model=model,
            input_path=dataset_path / folder / "origin",
            frame_indices=frame_indices,
            frame_number_by_name=frame_number_by_name
        )

        pbar = tqdm(
            total=len(target_frames),
            desc=f"      processing frames in {folder} [{frame_range_label}]"
        )
        for img_path, prediction in sorted(
            img_path2prediction.items(),
            key=lambda item: frame_number_by_name.get(Path(item[0]).name, -1)
        ):
            frame_number = frame_number_by_name.get(Path(img_path).name, -1)
            img_name = Path(img_path).name

            bboxes:   list[list[int]]         = prediction["bboxes"]
            masks_xy: list[list[list[float]]] = prediction["masks_xy"]
            confs:    list[float]             = prediction["confs"]
            #           |   └──> entry per instance
            #           └──> list of instances 
            for instance_number, (bbox, mask_xy, conf) in enumerate(zip(bboxes, masks_xy, confs)):
                dynamic_print(f"      processing instance {instance_number} in frame {frame_number} ({img_name}) with confidence {conf}")
                if conf > conf_threshold:
                    bbox_masked_image_fname = f"image_{frame_number}_{instance_number}_bbox-masked.png"
                    infer_mask.img2bbx(
                        img_path     = dataset_path / folder / "origin" / img_name, 
                        bbox         = bbox,
                        padding      = 20, 
                        out_dir      = dataset_path / folder / "bbox-masked_image",
                        out_filename = bbox_masked_image_fname
                    )
                    mask_img_np = polygon_to_binary_mask(mask_xy, image_size=image_size)
                    crop_img, crop_mask = crop_and_pad(get_image_np_from_path(str(img_path)), mask_img_np, bbox)
                    # Save outputs
                    save_crops(dataset_path, folder, frame_number, instance_number,
                            crop_img, crop_mask, mask_img_np)
                    # Collect CSV entries
                    rel_crop       = f"{folder}/cropped/image_{frame_number}_{instance_number}.png"
                    rel_mask       = f"{folder}/mask/image_{frame_number}_{instance_number}_mask.png"
                    rel_mask_full  = f"{folder}/mask_full/image_{frame_number}_{instance_number}_mask_full.png"
                    rel_bbx_masked = f"{folder}/bbox-masked_image/{bbox_masked_image_fname}"
                    csv_rows.append([frame_number, rel_crop, 'cropped', instance_number, folder, bbox])
                    csv_rows.append([frame_number, rel_mask, 'mask', instance_number, folder, bbox])
                    csv_rows.append([frame_number, rel_mask_full, 'mask_full', instance_number, folder, bbox])
                    csv_rows.append([frame_number, rel_bbx_masked, 'bbox-masked', instance_number, folder, bbox])

                    max_n_instances = max(max_n_instances, instance_number + 1)
            pbar.update(1)
        pbar.close()

        csv_rows.sort(key=lambda row: row[0])

        # Write to CSV
        with open(dataset_path / folder / 'files_crop.csv', 'w', newline='') as csv_out_file:
            csvwriter = csv.writer(csv_out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder', 'bbox'])
            csvwriter.writerows(csv_rows)
        
        idx_json['status'] = 'masks_detected'
        idx_json['max_n_instances'] = max_n_instances
        with open(dataset_path / 'index.json', 'w') as jf:
            json.dump(idx_json, jf, indent=2)


def get_frame_number(files_csv_rows:list, img_path:Path) -> int:
    img_name = str(Path(img_path).name)
    for row in files_csv_rows:
        if str(Path(row["file_loc"]).name) == img_name:
            return int(row["frame"])
    return -1


def draw_kpts_on_img(kpt2xyc: Dict[str, list], img_path: Path, out_path: Path, tenth_of_annot_radius: int = 1):
    img = cv2.imread(str(img_path))
    annot_radius = tenth_of_annot_radius * 10

    for name, (x,y,c) in kpt2xyc.items():
        if c <= 0:
            continue
        conf_scaled_annot_radius = int(
            int(c*10)/10   # will cut off second decimal (e.g 0.72 -> 0.7)
            *annot_radius  # if annot_radius is k*10 with k in N, this will yield an int
        )
        cv2.circle(img=img, center=(int(x),int(y)), radius=1, color=(255,0,0), lineType=-1) # center
        cv2.circle(img=img, center=(int(x),int(y)), radius=conf_scaled_annot_radius, color=(255,0,0), lineType=-1)
        cv2.putText(img, f"{int(c*100)/100}: {name}", (int(x),int(y+conf_scaled_annot_radius+10)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,0,0), thickness=1)

    cv2.imwrite(filename=str(out_path), img=img)


def detect_keypoints_yolo(dataset_path: Path, model_path: Path, kpt_names_dict: dict[int, str], frame_indices=None):
    """
    └── keypoint_results/
        └── keypoints_confs.pickle # expected to contain a dict with keypoints and confs indexed by frame number
    """

    model = infer_mask.load_model(model_path)

    with open(dataset_path / 'index.json', 'r') as jf:
        index_json = json.load(jf)

    def make_zero_dict() -> dict[str, list[float]]:
        return {
            kpt_name: [0.0, 0.0, 0.0]
            for kpt_name in kpt_names_dict.values()
        }

    def make_no_instance_detected_dict() -> dict[str, list[float]]:
        return {
            kpt_name: [-1.0, -1.0, -1.0]
            for kpt_name in kpt_names_dict.values()
        }

    views = index_json["frame_folders"]

    for view in views:
        print(f"processing frames for video {view}...")
        files_crop_csv_rows = list(csv.DictReader(open(dataset_path / view / "files_crop.csv")))
        os.makedirs(dataset_path / view / 'keypoints_results', exist_ok=True)
        frame2prediction: dict[str, InstancesKeypointsDict] = defaultdict(InstancesKeypointsDict)
        input_path = dataset_path / view / 'bbox-masked_image'

        frame_number_by_name = {
            Path(row["file_loc"]).name: int(row["frame"])
            for row in files_crop_csv_rows
        }
        frame_indices_set = set(frame_indices) if frame_indices is not None else None
        frame_to_images: dict[int, list[Path]] = defaultdict(list)
        for img in sorted(input_path.iterdir()):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            frame_number = frame_number_by_name.get(img.name)
            if frame_number is None:
                continue
            if frame_indices_set is not None and frame_number not in frame_indices_set:
                continue
            frame_to_images[frame_number].append(img)

        target_frames = sorted(frame_to_images.keys())
        frame_range_label = f"{target_frames[0]}-{target_frames[-1]}" if target_frames else "no-frames"
        pbar = tqdm(
            total=len(target_frames),
            desc=f"      processing frames in {view} [{frame_range_label}]"
        )
        for frame_number in target_frames:
            frame_number_str = str(frame_number)
            for img in sorted(frame_to_images[frame_number], key=lambda p: p.stem):
                instance_number: str = img.stem.split('_')[-2] # image_{frame}_{instance}_bbox-masked.png
                if frame_number_str not in frame2prediction:
                    frame2prediction[frame_number_str] = InstancesKeypointsDict()
                _, instances = infer_keypoints.inference(model, img)

                if len(instances) != 1:
                    """
                    this means either:
                        no instance was detected - however, 
                        we know that there is an instance in the image because all images used here 
                        are pre-filtered for containing instances by the instance segmentation step.
                    or:
                        multiple instances were detected - however,
                        we get a separate image per instance from the instance segmentation step,
                        so this is a false positive.
                    In both cases, we indicate that with a -1 instance number and all -1 coordinate and conf values.
                    """
                    print(f"   keypoint-detection missed all instances in frame {frame_number_str} ({Path(img).name})")
                    frame2prediction[frame_number_str]['-1'] = make_no_instance_detected_dict()
                    continue       
                
                # TODO: If keypoint detection detected multiple instances, get the instance with the best criteria, e.g. most keypoints detected. The other instance is considered a wrong detection.
                kpts = instances[0]
                dynamic_print(f"   processing instance {instance_number} in frame {frame_number_str} ({Path(img).name})")
                if len(kpts) == 0:
                    # we know there is an instance in the image because of the instance segmentation step,
                    # but keypoint detection missed it.
                    dynamic_print("      keypoint-detection missed this instance")
                    frame2prediction[frame_number_str][str(instance_number)] = make_no_instance_detected_dict()
                else:
                    frame2prediction[frame_number_str][str(instance_number)] = make_zero_dict()
                    for index, (x, y, c) in enumerate(kpts):
                        dynamic_print(f"      keypoint {kpt_names_dict[index]} at ({x}, {y}) with confidence {c}")
                        frame2prediction[frame_number_str][str(instance_number)][kpt_names_dict[index]] = [
                            x, y, (c if (x > 0 or y > 0) else 0.0) # undetected keypoints indicated by x=y=0 -> also conf=0.0
                        ]

                    draw_kpts_on_img(
                        frame2prediction[frame_number_str][str(instance_number)],
                        img,
                        dataset_path / view / 'keypoints_results' / f'keypoints_{frame_number_str}_{instance_number}.png'
                    )
            pbar.update(1)
        pbar.close()

        # low-confidence fish detections are already filtered out by mask detection!            
        with open(dataset_path / view / 'keypoints_results' / 'keypoints_confs.pickle', 'wb') as handle:
            pickle.dump(frame2prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        index_json['keypoint_list'] = list(kpt_names_dict.values())
        index_json['status'] = 'keypoints_detected'
        with open(dataset_path / 'index.json', 'w') as jf:
            json.dump(index_json, jf, indent=2)
