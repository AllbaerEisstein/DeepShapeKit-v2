import csv
import os
import cv2
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from PIL import Image
from src.types import *

class Multiview_Dataset(torch.utils.data.Dataset):
    """
    Following dataset structure is expected:
        <dataset_path>/
        ├── index.json          # Summary file with metadata
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
            ├── keypoint_results/
            │   └── keypoints_confs.pickle # expected to contain a dict with keypoints and confs indexed by frame number
            ├── files.csv       # Keeps track of the extracted frames
            ├── frame2video_1.csv # Maps original to new frame (1:1)
            └── files_crop.csv  # one CSV per folder

        files.csv:
            Keeps track of the extracted frames.
            One row per extracted frame.
            Columns: 
                - 'frame' - frame number
                - 'file_loc' - relative path to the saved file ("{video_name}_{frame_number}.png"
                - 'category' - field is always set to 'origin'
                - 'sub_index' - field is always set to 0
                - 'folder' - video name
        
        frame2video_1.csv: 
            Maps original to new frame (processed by pocess_frames) (1:1). 
            One row per video frame.
            Columns:
                - 'origin_frame' - frame number (int)
                - 'new_frame' - same entry as in 'origin_frame'

        files_crop.csv:
            One row per saved crop or mask.
            Columns:
                - frame — original frame number
                - file_loc — relative path to the saved file (cropped or mask)
                - category — 'cropped', 'mask' or 'bbox-masked'
                - sub_index — detection index (number of the instance detected)
                - folder — the <frame_folder> name
                - bbox — [xmin, ymin, xmax, ymax] from the mask
    """
    def __init__(self, root: str):
        self.root = Path(root)
        print("DATASET ROOT: "+str(self.root))
        with open(self.root / 'index.json', 'r') as jf:
            self.index_json = json.load(jf)
        self.views = self.index_json["frame_folders"]

        self.keypoints_confs: dict[str, KeypointsDict] = {
            v: pickle.load(
                open(self.root / v / 'keypoints_results' / f'keypoints_confs.pickle', 'rb')
            ) for v in self.views
        }

        # original frame file locations
        self.origin_frames_meta: dict[str, dict] = {
            #                         │    └──> the rows of the files.csv
            #                         └──> frame number string
            v: self._read_csv_dict(v, 'files.csv', key='frame')
            for v in self.views
        }

        # mask crop entries per view, grouped by origin frame
        self.masks_meta: dict[str, dict] = {
            #                  │    └──> the rows of the files_crop.csv with category mask
            #                  └──> frame number string
            v: self._group_csv(v, 'files_crop.csv', filter_cat='mask')
            for v in self.views
        }

        # mask crop entries per view, grouped by origin frame
        self.masks_full_meta: dict[str, dict] = {
            #                       │    └──> the rows of the files_crop.csv with category mask_full
            #                       └──> frame number string
            v: self._group_csv(v, 'files_crop.csv', filter_cat='mask_full')
            for v in self.views
        }

        self.cropped_meta: dict[str, dict] = {
            #                    │    └──> the rows of the files_crop.csv with category cropped
            #                    └──> frame number string
            v: self._group_csv(v, 'files_crop.csv', filter_cat='cropped')
            for v in self.views
        }

        self.bbox_masked_meta: dict[str, dict] = {
            #                       │    └──> the rows of the files_crop.csv with category bbox-masked
            #                       └──> frame number string
            v: self._group_csv(v, 'files_crop.csv', filter_cat='bbox-masked')
            for v in self.views
        }


        # mapping from reconstruction index -> origin frame per view
        self.frame_map = {
            v: self._read_csv_dict(v, 'frame2video_1.csv', key='new_frame', val='origin_frame')
            for v in self.views
        }

        self.prev_data = None



    def __len__(self) -> int:
        # assume same length across views
        return len(self.index_json["image_count"])
    


    def __getitem__(self, idx: int) -> dict:
        frame_key = str(idx)

        views_missing = 0
        for view in self.views:
            if frame_key not in self.masks_meta[view]:
                views_missing += 1
            if views_missing > 1:
                raise IndexError # to be handled by the caller
    
        sample = {'frames': [], 'imgpaths': [], 'instances': None}

        view_data = {}
        for view in self.views:
            origin_frame_number = self.frame_map[view][frame_key]
            # image path
            file_loc = self.origin_frames_meta[view][origin_frame_number]['file_loc']
            origin_img_path = self.root / file_loc

            # meta info for this frame
            masks_rows      = self.masks_meta[view].get(origin_frame_number, [])
            crops_rows      = self.cropped_meta[view].get(origin_frame_number, [])
            full_masks_rows = self.masks_full_meta[view].get(origin_frame_number, [])
            # sort by sub_index to keep instance order
            masks_rows      = sorted(masks_rows, key=lambda r: int(r['sub_index']))
            crops_rows      = sorted(crops_rows, key=lambda r: int(r['sub_index']))
            full_masks_rows = sorted(full_masks_rows, key=lambda r: int(r['sub_index']))

            # parse bboxes and load crops/full masks
            bboxes, crops, masks, full_masks = [], [], [], []
            for (crops_rows, masks_rows, full_masks_rows) in zip(crops_rows, masks_rows, full_masks_rows):
                bbox = self._parse_bbox(masks_rows['bbox'])
                bboxes.append(bbox)
                crops.append(self._load_grayscale_image(view, crops_rows['file_loc']))
                masks.append(self._load_grayscale_image(view, masks_rows['file_loc']))
                full_masks.append(self._load_grayscale_image(view, full_masks_rows['file_loc']))

            # extract DLC keypoints for each instance
            kpt_list = self._extract_keypoints(view, 
                                               self.keypoints_confs, 
                                               idx, 
                                               len(bboxes), 
                                               flip=(view=='bottom')
            )            

            view_data[view] = {
                'img_path':     str(origin_img_path),
                'bboxes':       torch.tensor(bboxes, dtype=torch.int64),              # (N,4)
                'crops':        torch.stack([torch.from_numpy(c) for c in crops]),   # (N,Hc,Wc)
                'masks':        torch.stack([torch.from_numpy(m) for m in masks]),
                'masks_full':   torch.stack([torch.from_numpy(m) for m in full_masks]),
                'keypoints':    torch.stack(kpt_list),                             # (N,K,3)
                'instances':    len(bboxes)
            }

        sample['imgpaths']    = [view_data[v]['img_path'] for v in self.views]
        sample['frames']      = list(range(len(self.views)))
        sample['instances']   = max(view_data[v]['instances'] for v in self.views)
        # stack per-view data for use by pipeline
        # view2n_instances = {v: data['bboxes'].shape[0] for v, data in view_data.items()}
        # sample['instances']   = torch.stack([view2n_instances[v]        for v in self.views])
        sample['bboxes']      = torch.stack([view_data[v]['bboxes']     for v in self.views])      # (V,N,4)
        sample['crops']       = torch.stack([view_data[v]['crops']      for v in self.views])
        sample['masks']       = torch.stack([view_data[v]['masks']      for v in self.views])      # (V,N,Hc,Wc)
        sample['masks_full']  = torch.stack([view_data[v]['masks_full'] for v in self.views])      # (V,N,Hf,Wf)
        sample['keypoints']   = torch.stack([view_data[v]['keypoints']  for v in self.views])      # (V,N,K,3)
        sample['full_kpts']   = True

        self.prev_data = sample
        return sample

    def _read_csv_dict(self, view: str, filename: str, key: str, val: str|None = None) -> dict:
        """
        Return a dict that has keys which are the values of a csv in a specific column (key).
        The values of the dict are either the complete row of the csv or an of the row in column 'value'.
        """
        path = self.root / view / filename
        with open(path) as f:
            reader = csv.DictReader(f)
            d = {}
            for row in reader:
                d[row[key]] = row if val is None else row[val]
            return d

    def _group_csv(self, view: str, filename: str, filter_cat: str) -> dict[int, dict]:
        """
        Returns a dict with one entry per frame if the frame has the specified category.
        {frame_number1: <files_crop.csv row for that file if the file has the specified category>}
        """
        path = self.root / view / filename
        d = {}
        with open(path) as f:
            reader = csv.DictReader(f, quotechar=' ')
            for row in reader:
                if row['category'] != filter_cat:
                    continue
                frame = row['frame']
                d.setdefault(frame, []).append(row)
        return d

    def _parse_bbox(self, bbox_str: str) -> list[int]:
        # string "[xmin, ymin, xmax, ymax]"
        nums = bbox_str.strip('[]').split(',')
        return [int(x) for x in nums]

    def _load_grayscale_image(self, view: str, rel_path: str) -> np.ndarray:
        # rel_path example: "video1/mask/image_0_0_mask.png"
        # try:
        #     img = Image.open(str(self.root / rel_path))
        #     img.verify()
        #     print("PIL load OK")
        # except Exception as e:
        #     print("PIL failed:", e)
        return np.array(cv2.imread(str(self.root / rel_path), cv2.IMREAD_GRAYSCALE))

    def _extract_keypoints(
        self,
        view: str,
        keypoints_dict: dict,
        idx: int,
        n_instances: int,
        flip: bool = False
    ) -> list[torch.Tensor]:
        """
        args:
            - keypoints_dict: dict containing keypoints and confidences per view(!), format:
                {
                    '0': {                          x      y    conf
                            'individual1': {        ↓      ↓     ↓
                                'mouth tip':    [123.5,  87.2, 0.998],
                                'gill':         [0.0,    0.0,  0.0],   <- keypoint was not detected (occluded)
                                … 
                            },
                            'individual2': {
                                'mouth tip':    [-1.0,  -1.0, -1.0,],  <- keypoint detection missed this instance
                                'gill':         [-1.0,  -1.0, -1.0,],
                                … 
                            },
                            …
                        },
                    '1': {…}
                }
            - idx: index of the frame of which the keypoints shall be extracted
            - n_instances:
            - flip: 
        Build a list of (K,3) tensors for each instance from keypoints outputs:
            - keypoints_dict['coordinates']: list of length K each shape (M,2)
            - keypoints_dict['confidence']:   list of length K each shape (M,1)
        """
        #filename = self.get_files_for_frame([view], ['bbox-masked'], [idx])[view]['bbox_masked'][idx][0]
        coords = keypoints_dict[view][str(idx)]
        kpt_list = []
        for inst in range(n_instances):
            inst = str(inst)
            pts = []
            for kpt_name in coords[inst]:
                x,y     = (coords[inst][kpt_name][0], coords[inst][kpt_name][1])
                conf    = coords[inst][kpt_name][2]
                if flip:
                    # assume width-known; you may parametrize
                    x = 2048 - x
                pts.append([x,y,conf])                              # one tuple for each kpt
            kpt_list.append(torch.tensor(pts, dtype=torch.float32)) # one tensor for reach instance
        return kpt_list

    def _get_files_for_frame(self, views: list[str], categories: list[str], idxs: list[int]) -> dict[str, dict[str, dict[int, list[Path]]]]:
        matching_files = {
            view: {
                category: {
                    idx: [] for idx in idxs
                } for category in categories
            } for view in views
        }
        for view in views:
            for category in categories:
                match category:
                    case 'mask':
                        meta_info = self.masks_meta[view]
                    case 'cropped':
                        meta_info = self.masks_meta[view]
                    case 'bbox-masked':
                        meta_info = self.masks_meta[view]
                    case 'origin':
                        meta_info = self.origin_frames_meta[view]
                    case _:
                        break
                for idx in idxs:
                    matching_files[view][category][idx] = [ 
                        self.root / row['folder'] / row['file_loc'] 
                        for row in meta_info
                        if  row['frame']     == idx
                        and row['category']  == category
                    ]
        return matching_files



class DeepFishDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images/valid"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks/valid"))))

    def __getitem__(self, idx):
        # load images and masks
        img_name = self.imgs[idx]
        mask_name = self.masks[idx]
        if img_name.split('.')[0] != mask_name.split('.')[0]:
            print(img_name)
            print(mask_name)
            raise Exception('image: ' + img_name + 'image and mask do not match' + mask_name)

        img_path = os.path.join(self.root, "images/valid", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks/valid", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class UniKNDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_name = self.imgs[idx]
        mask_name = self.masks[idx]
        if img_name.split('.')[0] != mask_name.split('.')[0]:
            raise Exception('image and mask do not match')

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path).convert("RGB")
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        mask_mono = ((mask[:,:,0] > 254) + 0.0) + 2 * ((mask[:,:,1] > 254) + 0.0)  # we use rerd and green to encode different instances
        # instances are encoded as different colors
        obj_ids = np.unique(mask_mono)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask_mono == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class UniLabDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        jf = open(os.path.join(root, 'index.json'))
        index_json = json.load(jf)
        image_folders = index_json['frame_folders']

        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(root)))

        self.imgs = []
        files = []
        for folder in image_folders:
            cf = open(os.path.join(root, folder, 'files.csv'))
            #cf2 = open(os.path.join(root, folder, 'frame2video.csv'))
            index_csv = csv.DictReader(cf)
            #frame_csv = csv.DictReader(cf2)
            for row in index_csv:
                self.imgs.append(row)

            # for row in frame_csv:
            #     self.imgs.append(files[int(row['origin_frame'])])

            cf.close()

        jf.close()



    def __getitem__(self, idx):
        # load images and masks
        image_dict = self.imgs[idx]

        img_path = os.path.join(self.root, image_dict['file_loc'])
        #img = Image.open(img_path).convert("RGB")

        label = image_dict

        # if self.transforms is not None:
        #     img = self.transforms(img, label)

        return img_path, label

    def __len__(self):
        return len(self.imgs)



