import cv2
import csv
import json
import os
import numpy as np
import torch
import deeplabcut

from src.dataloaders import UniLabDataset
from PIL import Image, ImageDraw
from tqdm import trange

import torchvision.transforms as T
import torch.nn.functional as F
import models.MaskRCNN as MRCNN
import infer_mask_cli_wrapper
import pathlib
from pathlib import Path

import pickle
import subprocess

def extract_from_video(videos, out_dir, out_folder='video_frames'):
    """
    📂 Expected/Assumed Directory Structure Before Execution:
        - out_dir exists. If not, the function raises an exception.
        - The videos exist at the paths listed in videos.
    
    Creates the following directory structure and contents:
        out_dir/
        └── video_frames/
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

    if not os.path.exists(out_dir):
        raise Exception('Output dir do not exist')

    dist = os.path.join(out_dir, out_folder)
    if not os.path.exists(dist):
        os.mkdir(dist)

    json_out_file = open(os.path.join(dist, 'index.json'), "w")
    json_index = {}
    json_index['frame_folders'] = []
    csv_list = {}

    for video_path in videos:
        if not os.path.exists(video_path):
            raise Exception('Input video do not exist')

        print("processing video: {}".format(video_path))

        video_name = video_path.split('/')[-1]
        ext_len = len(video_name.split('.')[-1])
        save_path = os.path.join(dist, video_name[:-ext_len-1], 'origin')
        if not os.path.exists(os.path.join(dist, video_name[:-ext_len-1])):
            os.mkdir(os.path.join(dist, video_name[:-ext_len - 1]))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        csv_out_file = open(os.path.join(dist, video_name[:-ext_len-1], 'files.csv'), 'w')
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder'])

        # cv2 extract frames
        capture = cv2.VideoCapture(video_path)
        vwidth  = capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float `width`
        vheight = capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)  # float `height`

        image_count = 0
        frame_number = 0
        success = True

        while capture.isOpened() and success:
            success, frame = capture.read()

            if success:
                file_loc = os.path.join(save_path, "{}_{}.png".format(video_name[:-ext_len-1], frame_number))
                cv2.imwrite(file_loc, frame)
                image_count += 1
                csvwriter.writerow([frame_number,
                                    os.path.join(video_name[:-ext_len-1], "origin/{}_{}.png".format(video_name[:-ext_len-1], frame_number)),
                                    'origin',
                                    0,
                                    video_name[:-ext_len-1]])

            frame_number += 1
            print('frame out: {}, total image: {}'.format(frame_number, image_count), end='\r')

            # if image_count > 5:
            #     break

        print('total image: {}, done                  '.format(image_count))

        json_index['frame_folders'].append(video_name[:-ext_len-1])
        csv_list[video_name[:-ext_len-1]] = os.path.join(dist, video_name[:-ext_len-1], 'files.csv')

        csv_out_file.close()
        capture.release()

    json_index['status']        = 'origin'
    json_index['index_files']   = csv_list
    json_index['image_count']   = image_count
    json_index['image_size']    = [int(vwidth), int(vheight)]

    json.dump(json_index, json_out_file)
    json_out_file.close()


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
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
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
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
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


def get_image_tensor(image_path: str, device: str) -> torch.Tensor:
    """
    Load a PIL image, convert to an RGB tensor in [0,1], and move to `device`.
    """
    img = Image.open(image_path).convert("RGB")
    return T.ToTensor()(img).to(device)


def polygon_to_binary_mask(polygon, image_size, mode='1', fill=1, background=0):
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

    return mask_img


def predict_masks_yolo(dataset_path, model_path, device, num_classes=2, yolo_env_name="yolo", conf_threshold=0.8):
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
            dataset_path: str,
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
            base = os.path.join(dataset_path, folder)

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
            full_mask = (full_mask_np.cpu().numpy() * 255).astype(np.uint8)
            full_mask_path = os.path.join(
                base, 'mask_full', f'image_{frame_counter}_{det_idx}_mask.png'
            )
            Image.fromarray(full_mask).save(full_mask_path)

    def run_infer_mask(input_path: Path, is_dir=False):
        cmd = [
            "conda", "run", "-n", yolo_env_name, "python", infer_mask_cli_wrapper,
            "inference",
            "--model", model_path,
            "--input", str(input_path),
            "--isdir", is_dir
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        res = json.loads(cp.stdout)
        if is_dir:
            return res
        else:
            bboxes = res["bboxes"]
            masks = res["masks_xy"]
            confs = res["confs"]
            return bboxes, masks, confs

    def run_img2bbx(input_path: Path, bboxes, out_dir: Path, padding: int, is_dir=False, out_filename=None):
        cmd = [
            "conda", "run", "-n", yolo_env_name, "python", infer_mask_cli_wrapper,
            "img2bbx",
            "--input", str(input_path),
            "--bboxes", json.dumps(bboxes),
            "--outdir", str(out_dir),
            "--pad", str(padding),
            "--isdir", is_dir
        ]
        if not is_dir:
            cmd += ["--outfilename", out_filename]
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(cp.stdout)

    # Read index.json
    with open(os.path.join(dataset_path, 'index.json')) as jf:
        idx_json = json.load(jf)
    image_folders   = idx_json['frame_folders']
    total_frames    = idx_json['image_count']
    image_size      = idx_json['image_size']

    csvwriters = {}
    for folder in image_folders:
        for new_dir in ['cropped', 'mask', 'mask_full', 'bbox-masked_image']:
            if not os.path.exists(os.path.join(dataset_path, folder, new_dir)):
                os.mkdir(os.path.join(dataset_path, folder, new_dir))

        csv_out_file = open(os.path.join(dataset_path, folder, 'files_crop.csv'), 'w')
        csvwriter = csv.writer(csv_out_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        # write the headers
        csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder', 'bbox'])
        csvwriters[folder] = csvwriter
        
        img_path2prediction = run_infer_mask(Path(dataset_path) / folder / "origin", is_dir=True)

        with open(dataset_path / "origin") as files_csv:
            files_csv_reader = csv.DictReader(files_csv)
            for img_path, prediction in img_path2prediction.items():
                # set frame number based on frame number of original file:
                frame_number = int([
                    row["frame"] for row in files_csv_reader 
                    if str(Path(row["file_loc"]).name) == img_path
                ][0])
                bboxes   = prediction["bboxes"]
                masks_xy = prediction["masks_xy"]
                confs    = prediction["confs"]
                for det_idx, (bbox, mask_xy, conf) in enumerate(zip(bboxes, masks_xy, confs)):
                    if conf > conf_threshold:
                        bbox_masked_image_fname = f"image_{frame_number}_{det_idx}_bbox-masked.png"
                        run_img2bbx(
                            input_path=Path(dataset_path) / folder / "origin" / img_path, 
                            bboxes=bbox,
                            padding=20, 
                            out_dir=Path(dataset_path) / folder / "bbox-masked_image",
                            out_filename=bbox_masked_image_fname,
                            is_dir=False
                        )
                        mask_img = polygon_to_binary_mask(mask_xy, image_size=image_size)
                        crop_img, crop_mask = crop_and_pad(get_image_tensor(str(img_path), device=device), mask_img, bbox)
                        # Save outputs
                        save_crops(dataset_path, folder, frame_number, det_idx,
                                crop_img, crop_mask, mask_img)
                        # Record CSV entries
                        rel_crop       = f"{folder}/cropped/image_{frame_number}_{det_idx}.png"
                        rel_mask       = f"{folder}/mask/image_{frame_number}_{det_idx}_mask.png"
                        rel_bbx_masked = f"{folder}/bbox-masked_image/{bbox_masked_image_fname}"
                        csvwriter.writerow([frame_number, rel_crop, 'cropped', det_idx, folder, bbox])
                        csvwriter.writerow([frame_number, rel_mask, 'mask', det_idx, folder, bbox])
                        csvwriter.writerow([frame_number, rel_bbx_masked, 'bbox-masked', det_idx, folder, bbox])


def detect_keypoints_yolo(dataset_path: str, model_path: str, yolo_env_name: str = "yolo", conf_threshold: float = 0.8):
    """
    ├── keypoint_results/
    └── keypoints_confs.pickle # expected to contain a dict with keypoints and confs indexed by frame number
    """
    def run_infer_kpts(input_path: Path, is_dir=False):
        cmd = [
            "conda", "run", "-n", yolo_env_name, "python", infer_mask_cli_wrapper,
            "keypoints",
            "--model", model_path,
            "--input", str(input_path),
            "--isdir", is_dir
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(cp.stdout)

    dataset_path = Path(dataset_path)
    views = [str(subdir) for subdir in dataset_path.listdir() if subdir.is_dir()]
    for view in views:
        os.makedirs(dataset_path / view / "keypoints_results", exist_ok=True)
        img_path2prediction = run_infer_kpts(Path(dataset_path) / view / "bbox-masked_images", is_dir=True)
        # low-confidence fish detections are already filtered out by mask detection!            
        with open('keypoints_confs.pickle', 'wb') as handle:
            # correspondence between keypoints and filename can later be established via files_crop.csv!
            pickle.dump(img_path2prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

def predict(dataset_path, model_path, device, num_classes=2):
    """
    <dataset_path>/
    └── <frame_folder>/
        ├── origin/         ← extracted frames
        ├── cropped/        ← to be created here
        ├── mask/           ← to be created here
        ├── mask_full/      ← to be created here
        └── files_crop.csv  ← newly created CSV

    """
    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def collate_fn(batch):
        return tuple(zip(*batch))

    # get the model using our helper function
    model = MRCNN.get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    dataset = UniLabDataset(dataset_path, get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    # create csv index files in each video folder
    jf = open(os.path.join(dataset_path, 'index.json'))
    index_json = json.load(jf)
    image_folders = index_json['frame_folders']

    csvwriters = {}
    for folder in image_folders:
        if not os.path.exists(os.path.join(dataset_path, folder, 'cropped')):
            os.mkdir(os.path.join(dataset_path, folder, 'cropped'))

        if not os.path.exists(os.path.join(dataset_path, folder, 'mask')):
            os.mkdir(os.path.join(dataset_path, folder, 'mask'))

        if not os.path.exists(os.path.join(dataset_path, folder, 'mask_full')):
            os.mkdir(os.path.join(dataset_path, folder, 'mask_full'))

        csv_out_file = open(os.path.join(dataset_path, folder, 'files_crop.csv'), 'w')
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                               quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder', 'bbox'])

        csvwriters[folder] = csvwriter

    target_frames = list(range(index_json['image_count']))

    model.eval()
    with torch.no_grad():
        k = 0
        pbar = trange(len(target_frames), desc="detect from frames")
        for image, label in data_loader_test:
            # only use non-overlap frames
            label = label[0]
            if int(label['frame']) not in target_frames:
                continue

            pbar.set_description('detecting frame {}'.format(label['frame']))
            images = torch.from_numpy(np.array(Image.open(image[0]).convert("RGB"))) #image[0]
            csvwriter = csvwriters[label['folder']]

            crop_image = images.permute(1,0,2)

            images = [images.to(device).permute(2, 0, 1) / 255.]
            predictions = model(images)

            for i in range(predictions[0]['boxes'].size()[0]):
                # only 2 fishes in the scene
                if i > 1:
                    break
                mask = predictions[0]['masks'][i, 0].cpu().numpy()

                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                bounding_box = [xmin, ymin, xmax, ymax]

                cropped = crop_image[int(bounding_box[0]):int(bounding_box[2]),
                          int(bounding_box[1]):int(bounding_box[3])]

                crop_mask = predictions[0]['masks'][i, 0].mul(255).permute(1, 0)
                cropped_mask = crop_mask[int(bounding_box[0]):int(bounding_box[2]),
                               int(bounding_box[1]):int(bounding_box[3])]

                diff = abs(bounding_box[2] - bounding_box[0] - (bounding_box[3] - bounding_box[1]))

                output = cropped.permute(2,0,1)
                out_mask = cropped_mask

                if bounding_box[2] - bounding_box[0] < bounding_box[3] - bounding_box[1]:
                    # padding height
                    output = F.pad(input=output,
                                   pad=(0, 0, int(diff / 2.0),
                                        int(diff / 2.0)),
                                   mode='constant', value=0)
                    out_mask = F.pad(input=cropped_mask,
                                     pad=(0, 0, int(diff / 2.0),
                                          int(diff / 2.0)),
                                     mode='constant', value=0)

                if bounding_box[2] - bounding_box[0] > bounding_box[3] - bounding_box[1]:
                    # padding height
                    output = F.pad(input=output,
                                   pad=(int(diff / 2.0),
                                        int(diff / 2.0), 0, 0),
                                   mode='constant', value=0)
                    out_mask = F.pad(input=cropped_mask,
                                     pad=(int(diff / 2.0),
                                          int(diff / 2.0), 0, 0),
                                     mode='constant', value=0)

                crop_out_dir = os.path.join(dataset_path, label['folder'], 'cropped')
                mask_out_dir = os.path.join(dataset_path, label['folder'], 'mask')
                mask_full_dir = os.path.join(dataset_path, label['folder'], 'mask_full')

                output = Image.fromarray(output.permute(2, 1, 0).cpu().byte().numpy())
                output.save(os.path.join(crop_out_dir, 'image_{}_{}.png'.format(k, i)))
                predicted_mask = Image.fromarray(out_mask.permute(1, 0).byte().cpu().numpy())
                predicted_mask.save(os.path.join(mask_out_dir, 'image_{}_{}_mask.png'.format(k, i)))
                full_mask = Image.fromarray(predictions[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
                full_mask.save(os.path.join(mask_full_dir, 'image_{}_{}_mask.png'.format(k, i)))

                crop_out_dir = os.path.join(crop_out_dir, 'image_{}_{}.png'.format(k, i))
                mask_out_dir = os.path.join(mask_out_dir, 'image_{}_{}_mask.png'.format(k, i))

                csvwriter.writerow([label['frame'],
                                    '/'.join(crop_out_dir.split('/')[-3:]),
                                    'cropped',
                                    str(i),
                                    label['folder'],
                                    str(bounding_box)])

                csvwriter.writerow([label['frame'],
                                    '/'.join(mask_out_dir.split('/')[-3:]),
                                    'mask',
                                    str(i),
                                    label['folder'],
                                    str(bounding_box)])

            pbar.update(1)
            k += 1
    print('\n finish')


def detect_dlc(data_folder,
               front_config_path='/home/lab/Documents/fish_mesh_eye_public/models/trained_models/master2021demo_front-Ruiheng Wu-2021-06-02/config.yaml',
               bottom_config_path='/home/lab/Documents/fish_mesh_eye_public/models/trained_models/master2021demo_bottom-Ruiheng Wu-2021-06-01/config.yaml'):

    deeplabcut.analyze_videos(front_config_path,
                              [os.path.join(data_folder, 'front', 'dlc_results', 'full_size.mp4')],
                              videotype='.mp4',
                              engine=deeplabcut.core.engine.Engine.TF)

    deeplabcut.analyze_videos(bottom_config_path,
                              [os.path.join(data_folder, 'bottom', 'dlc_results', 'full_size.mp4')],
                              videotype='.mp4',
                              engine=deeplabcut.core.engine.Engine.TF)