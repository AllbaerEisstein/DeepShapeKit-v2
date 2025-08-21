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
from src.types import *


def extract_from_video(videos: list[Path], out_dir: Path, dataset_folder_name: str = 'dataset', also_create_frame2video_csv: bool = False):
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

    dist = out_dir / dataset_folder_name
    dist.mkdir(exist_ok=True)

    json_out_path = dist / 'index.json'
    json_index = {
        'frame_folders': [],
        'index_files': {},
    }

    for video_path in videos:
        if not video_path.exists():
            raise Exception(f'Input video does not exist: {video_path}')

        print(f"processing video: {video_path}")

        video_name      = video_path.stem
        video_folder    = dist / video_name
        origin_folder   = video_folder / 'origin'
        origin_folder.mkdir(parents=True, exist_ok=True)

        files_csv_path       = video_folder / 'files.csv'
        with files_csv_path.open('w', newline='') as csv_out_file:
            csvwriter = csv.writer(csv_out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder'])

            capture = cv2.VideoCapture(str(video_path))

            image_count = 0
            frame_number = 0
            success = True

            while capture.isOpened() and success:
                success, frame = capture.read()
                if success:
                    vheight, vwidth = frame.shape[:2]
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
                print(f'frame out: {frame_number}, total image: {image_count}', end='\r')

            print(f'total image: {image_count}, done                  ')

            json_index['frame_folders'].append(video_name)
            json_index['index_files'][video_name] = str(files_csv_path)

            capture.release()
    
        if also_create_frame2video_csv:
            frame2video_csv_path = video_folder / 'frame2video_1.csv'
            with frame2video_csv_path.open('w', newline='') as csv_out_file:
                csvwriter = csv.writer(csv_out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['origin_frame', 'new_frame'])

                for i in range(image_count):
                    csvwriter.writerow([i, i])

    json_index['status'] = 'origin'
    json_index['image_count'] = image_count
    json_index['image_size'] = [int(vwidth), int(vheight)]

    with json_out_path.open("w") as f:
        json.dump(json_index, f, indent=2)


def process_input_folder(data_folder):
    """
    Takes the output from extract_from_video (specifically index.json) in data_folder.

    For the bottom and front camera views:
        - Writes a frame2video_1.csv file (1:1 mapping).
        - bottom: Flips all images horizontally and writes them back in-place.
        - front: Undistorts the images using predefined camera intrinsics.
        - Creates a full_size.mp4 video from the processed images in each view.

    📂 Expected Directory Structure Before Execution:

        data_folder/
        ├── index.json
        ├── bottom/
        │   ├── origin/
        │   │   ├── bottom_0.png
        │   │   ├── ...
        ├── front/
        │   ├── origin/
        │   │   ├── front_0.png
        │   │   ├── ...

    📄 Files Created:
        - bottom/frame2video_1.csv: Maps original to new frame (1:1).
            columns:
                - 'origin_frame' - one frame number (int) per row
                - 'new_frame' - each row the same entry as in 'origin_frame'
        - front/frame2video_1.csv: Same as above.
        - bottom/dlc_results/full_size.mp4: Reconstructed video from flipped frames.
        - front/dlc_results/full_size.mp4: Reconstructed video from undistorted frames.
        - Images in bottom/origin/ are flipped in-place.
        - Images in front/origin/ are undistorted in-place.

    🎥 Assumptions and Constraints:
        - index.json exists and has an image_count field.
        - bottom/origin/ contains images named like bottom_0.png, bottom_1.png, ...
        - front/origin/ contains images named like front_0.png, ...
        - Image resolution for bottom and front is assumed to be 2048x1040.
        - Hardcoded camera matrix and distortion coefficients (assumes a specific front camera calibration).
    """
    with open(os.path.join(data_folder, 'index.json')) as jf:
        video_meta = json.load(jf)

    n_frames = video_meta['image_count']

    # process bottom images
    with open(os.path.join(data_folder, 'bottom', 'frame2video_1.csv'), 'w') as csv_out_file:
        csvwriter = csv.writer(csv_out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['origin_frame', 'new_frame'])

        for i in range(n_frames):
            csvwriter.writerow([i, i])

    # create bottom video
    if not os.path.exists(os.path.join(data_folder, 'bottom', 'dlc_results')):
        os.mkdir(os.path.join(data_folder, 'bottom', 'dlc_results'))

    out = cv2.VideoWriter(os.path.join(data_folder, 'bottom', 'dlc_results', 'full_size.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (2048, 1040))
    frame_count = 0

    for i in range(n_frames):
        img1 = cv2.imread(os.path.join(data_folder, 'bottom', 'origin', 'bottom_{}.png'.format(i)))
        out.write(img1)
        print('bottom video frame out: {}'.format(frame_count), end='\r')
        frame_count += 1

    out.release()

    # flip bottom images
    for i in range(n_frames):
        img = cv2.imread(os.path.join(data_folder, 'bottom', 'origin', 'bottom_{}.png'.format(i)))
        img_flip_h = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(data_folder, 'bottom', 'origin', 'bottom_{}.png'.format(i)), img_flip_h)
        print('flipping frame: {}'.format(i), end='\r')

    # process front images
    with open(os.path.join(data_folder, 'front', 'frame2video_1.csv'), 'w') as csv_out_file:
        csvwriter = csv.writer(csv_out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['origin_frame', 'new_frame'])

        for i in range(n_frames):
            csvwriter.writerow([i, i])

    # undistort front images
    img_list = os.listdir(os.path.join(data_folder, 'front', 'origin'))
    mtx = np.array([[3946, 0, 1080], [0, 3934, 520], [0, 0, 1]])
    dist = np.array([-0.568857779226978, 0.151730496415158, 0, 0, 0])
    w = 2048
    h = 1040
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    iid = 1
    for img in img_list:
        if img[-4:] != '.png':
            continue

        dst = cv2.undistort(cv2.imread(os.path.join(data_folder, 'front', 'origin', img)), mtx, dist, None, newcameramtx)
        cv2.imwrite(os.path.join(data_folder, 'front', 'origin', img), dst)
        print('undistorting image: {}'.format(iid), end='\r')
        iid += 1

    # create front video
    if not os.path.exists(os.path.join(data_folder, 'front', 'dlc_results')):
        os.mkdir(os.path.join(data_folder, 'front', 'dlc_results'))

    out = cv2.VideoWriter(os.path.join(data_folder, 'front', 'dlc_results', 'full_size.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'), 20, (2048, 1040))
    frame_count = 0

    for i in range(n_frames):
        img1 = cv2.imread(os.path.join(data_folder, 'front', 'origin', 'front_{}.png'.format(i)))
        out.write(img1)
        print('front video frame out: {}'.format(frame_count), end='\r')
        frame_count += 1

    out.release()


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


def predict_masks_yolo(dataset_path: Path, model_path: Path, conf_threshold=0.8):
    """
    Use a pre-trained YOLO11n-seg model (model_path) to infer the segmentation masks of the dataset.
    Augment the dataset as follows:

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

    def run_infer_mask(model, input_path: Path) -> dict[str, dict[str, list]]:
        img_path2result = {}
        for img in Path(input_path).iterdir():
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
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
    image_size      = idx_json['image_size']

    model = infer_mask.load_model(Path(model_path))

    for folder in image_folders:
        print(f"   processing frames for video {folder}...")
        for new_dir in ['cropped', 'mask', 'mask_full', 'bbox-masked_image']:
            if not os.path.exists(dataset_path / folder / new_dir):
                os.mkdir(dataset_path / folder / new_dir)

        csv_rows = []  # <-- Collect rows here instead of writing immediately

        img_path2prediction: dict[str, dict[str, list]] = run_infer_mask(model=model, input_path=dataset_path / folder / "origin")
        files_csv_rows = list(csv.DictReader(open(dataset_path / folder / "files.csv")))

        for img_path, prediction in tqdm(sorted(img_path2prediction.items())):
            frame_number = get_frame_number(img_path=Path(img_path), files_csv_rows=files_csv_rows)
            img_name = Path(img_path).name

            bboxes:   list[list[int]]         = prediction["bboxes"]
            masks_xy: list[list[list[float]]] = prediction["masks_xy"]
            confs:    list[float]             = prediction["confs"]
            #           |   └──> entry per instance
            #           └──> list of instances 
            for instance_number, (bbox, mask_xy, conf) in enumerate(zip(bboxes, masks_xy, confs)):
                print(f"      processing instance {instance_number} in frame {frame_number} ({img_name}) with confidence {conf}")
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

        csv_rows.sort(key=lambda row: row[0])

        # Write to CSV
        with open(dataset_path / folder / 'files_crop.csv', 'w', newline='') as csv_out_file:
            csvwriter = csv.writer(csv_out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder', 'bbox'])
            csvwriter.writerows(csv_rows)


def get_frame_number(files_csv_rows:list, img_path:Path) -> int:
    img_name = str(Path(img_path).name)
    for row in files_csv_rows:
        if str(Path(row["file_loc"]).name) == img_name:
            return int(row["frame"])
    return -1


def detect_keypoints_yolo(dataset_path: Path, model_path: Path, kpt_names_dict: dict):
    """
    └── keypoint_results/
        └── keypoints_confs.pickle # expected to contain a dict with keypoints and confs indexed by frame number
    """

    model = infer_mask.load_model(model_path)

    with open(dataset_path / 'index.json', 'r') as jf:
        index_json = json.load(jf)

    zero_dict = { # for reusing
        kpt_name: [0.0, 0.0, 0.0]
        for kpt_name in kpt_names_dict.values()
    }
    no_instance_detected_dict = { # for reusing
        kpt_name: [-1.0, -1.0, -1.0]
        for kpt_name in kpt_names_dict.values()
    }

    views = index_json["frame_folders"]

    for view in views:
        print(f"processing frames for video {view}...")
        files_crop_csv_rows = list(csv.DictReader(open(dataset_path / view / "files_crop.csv")))
        os.makedirs(dataset_path / view / 'keypoints_results', exist_ok=True)
        frame2prediction: KeypointsDict = KeypointsDict()
        input_path = dataset_path / view / 'bbox-masked_image'

        for img in sorted(input_path.iterdir()):
            frame_number: str = str(get_frame_number(img_path=img, files_csv_rows=files_crop_csv_rows))
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                frame2prediction[frame_number] = {}
                _, instances = infer_keypoints.inference(model, img)

                if len(instances) == 0:
                    """
                    this means no instance was detected - however, 
                    we know that there is an instance in the image because all images used here 
                    are pre-filtered for containing instances by the instance segmentation step.
                    so we indicate that with -1-confidences. 
                    """
                    print(f"   keypoint-detection missed all instances in frame {frame_number} ({Path(img).name})")
                    frame2prediction[frame_number]['-1'] = no_instance_detected_dict

                for instance_number, kpts in enumerate(instances):
                    print(f"   processing instance {instance_number} in frame {frame_number} ({Path(img).name})")
                    if len(kpts) == 0:
                        print("      keypoint-detection missed this instance")
                        frame2prediction[frame_number][str(instance_number)] = no_instance_detected_dict
                    else:
                        frame2prediction[frame_number][str(instance_number)] = zero_dict
                    for index, (x, y, c) in enumerate(kpts):
                        print(f"      keypoint {kpt_names_dict[index]} at ({x}, {y}) with confidence {c}")
                        frame2prediction[frame_number][str(instance_number)][kpt_names_dict[index]] = [
                            x, y, (c if (x > 0 or y > 0) else 0) # undetected keypoints indicated by x=y=0
                        ] 

        # low-confidence fish detections are already filtered out by mask detection!            
        with open(dataset_path / view / 'keypoints_results' / 'keypoints_confs.pickle', 'wb') as handle:
            pickle.dump(frame2prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)