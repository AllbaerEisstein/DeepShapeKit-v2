from collections import defaultdict
import csv
import os
from typing import List, Optional
import cv2
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from PIL import Image
from src.types import *
from src.parse_cams_json import CameraSet

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
            ├── bbox-masked_image/      # images at original size but black everywhere
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
    def __init__(self, root: str, views: Optional[List[str]] = None):
        self.root = Path(root)
        print("DATASET ROOT: "+str(self.root))
        with open(self.root / 'index.json', 'r') as jf:
            index_json = json.load(jf)

        available_views = index_json.get("frame_folders", [])
        if not isinstance(available_views, list) or not available_views:
            raise ValueError("Dataset index does not list any frame folders.")

        if views is None:
            selected_views = available_views
        else:
            missing_views = [view for view in views if view not in available_views]
            if missing_views:
                raise ValueError(
                    f"Requested view(s) not present in dataset: {', '.join(missing_views)}"
                )
            selected_views = [view for view in views if view in available_views]
            if not selected_views:
                raise ValueError("No valid views selected for dataset.")

        self.views: list[str] = list(selected_views)
        self.index_json = dict(index_json)
        self.index_json["frame_folders"] = self.views
        for key in ("index_files", "image_sizes", "camera_matrices"):
            if key in self.index_json and isinstance(self.index_json[key], dict):
                self.index_json[key] = {
                    k: v for k, v in self.index_json[key].items() if k in self.views
                }

        self.view_2_frames_2_instances_2_kpts: dict[str, dict[str, InstancesKeypointsDict]] = {
            #                                        |         |     └──> dict mapping instance number str to the dict kptname->[x,y,c]
            #                                        |         └──> frame number string
            #                                        └──> view name
            v: pickle.load(
                open(self.root / v / 'keypoints_results' / 'keypoints_confs.pickle', 'rb')
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
        self.masks_meta: dict[str, dict[str, list[dict]]] = {
            #                  |         |     └──> the rows of the files_crop.csv for this view and this frame with category mask
            #                  |         └──> frame number string
            #                  └──> view name
            v: self._group_csv(v, 'files_crop.csv', filter_cat='mask')
            for v in self.views
        }

        # mask crop entries per view, grouped by origin frame
        self.masks_full_meta: dict[str, dict[str, list[dict]]] = {
            #                       |         |     └──> the rows of the files_crop.csv for this view and this frame with category mask
            #                       |         └──> frame number string
            #                       └──> view name
            v: self._group_csv(v, 'files_crop.csv', filter_cat='mask_full')
            for v in self.views
        }

        self.cropped_meta: dict[str, dict[str, list[dict]]] = {
            #                    |         |     └──> the rows of the files_crop.csv for this view and this frame with category mask
            #                    |         └──> frame number string
            #                    └──> view name
            v: self._group_csv(v, 'files_crop.csv', filter_cat='cropped')
            for v in self.views
        }

        self.bbox_masked_meta: dict[str, dict[str, list[dict]]] = {
            #                        |         |     └──> the rows of the files_crop.csv for this view and this frame with category mask
            #                        |         └──> frame number string
            #                        └──> view name
            v: self._group_csv(v, 'files_crop.csv', filter_cat='bbox-masked')
            for v in self.views
        }


        # mapping from reconstruction index -> origin frame per view
        self.frame_map = {
            v: self._read_csv_dict(v, 'frame2video_1.csv', key='new_frame', val='origin_frame')
            for v in self.views
        }

        self.uniform_img_size = (max([value[0] for value in self.index_json['image_sizes'].values()]), 
                                 max([value[1] for value in self.index_json['image_sizes'].values()]))

        self.cams = CameraSet(self.index_json, self.views, uniform_img_size=self.uniform_img_size)

        self.instance_indices = list(range(self.index_json['max_n_instances']))

        self.prev_data = None


    def __len__(self) -> int:
        # assume same length across views
        return len(self.index_json["image_count"])


    def __getitem__(self, frame_idx: int, instance_idx: Optional[int] = None) -> dict:
        """
        Note: This class is only responsible for constructing a sample - it has no semantic responsibility over the number of views needed, etc.
        Returns:
            sample (dict): a dict with the following fields:
            sample['imgpaths'] (list[str]): list of paths to the original image for each view
            sample['frames'] (list[int]): the list of the view indices
            sample['instances'] (list[int]): the indices of instances in this sample (= list(range(max_n_instances)))
            sample['bboxes'] (n_instances, n_views, 4): bounding boxes of the segmentation masks in the format (x0, y0, x1, y1)
            sample['masks_full'] (n_instances, n_views, uniform_img_width, uniform_img_height): tensor storing the binary segmentation mask
            sample['keypoints'] (n_instances, n_views, n_keypoints, 3): tensor storing (x,y,conf) for each keypoint (keypoint missing -> conf=-1)
            sample['seg_mask_present_mask'] (n_instances, n_views): boolean values indicating if an instance has mask detections for each view
            sample['kpt_present_mask'] (n_instances, n_views, n_keypoints): boolean values indicating if an instance has a keypoint detection for each view and each keypoint
        OR, if `instance_idx` is specified, the following fields are different:
            sample['instances'] (int): the index of the instance in this sample
            sample['bboxes'] (n_views, 4): bounding boxes of the segmentation masks in the format (x0, y0, x1, y1)
            sample['masks_full'] (n_views, uniform_img_width, uniform_img_height): tensor storing the binary segmentation mask
            sample['keypoints'] (n_views, n_keypoints, 3): tensor storing (x,y,conf) for each keypoint (keypoint missing -> conf=-1)
            sample['seg_mask_present_mask'] (n_views): boolean values indicating if this instance has mask detections for each view
            sample['kpt_present_mask'] (n_views, n_keypoints): boolean values indicating if this instance has a keypoint detection for each view and each keypoint
        """
        frame_key = str(frame_idx)

        ## Legacy code
        # # raise an index error if we don't have at least two views with annotations for this instance
        # if instance_idx is not None:
        #     views_with_annot = len(self.views)
        #     for view in self.views:
        #         frame2mask_rows = self.masks_meta[view]
        #         if frame_key not in [str(frame) for frame in frame2mask_rows]:
        #             views_with_annot -= 1
        #         elif str(instance_idx) not in [row['sub_index'] for rows in frame2mask_rows.values() for row in rows]:
        #             views_with_annot -= 1
        #         # two views are the minimum to perform 3d reconstruction
        #         if views_with_annot < 2:
        #             raise IndexError # to be handled by the caller
    
        sample = {'frames': [], 'imgpaths': [], 'instances': None}
        seg_mask_present_mask = [[False]*len(self.views)]*len(self.instance_indices)
        kpt_present_mask = [[[False]*len(self.index_json['keypoint_list'])]*len(self.views)]*len(self.instance_indices)

        view_data = {}
        for view_index, view in enumerate(self.views):
            origin_frame_number = self.frame_map[view][frame_key]
            # image path
            file_loc = self.origin_frames_meta[view][origin_frame_number]['file_loc']
            origin_img_path = self.root / file_loc
        

            # extract keypoints for each instance
            for instance_number in self.instance_indices:
                if str(instance_number) not in self.view_2_frames_2_instances_2_kpts[view][frame_key].keys():
                    kpt_present_mask[instance_number][view_index] = [False]*len(self.index_json["keypoint_list"])
                else:
                    for kpt_index, kpt_name in enumerate(self.index_json["keypoint_list"]):
                        if self.view_2_frames_2_instances_2_kpts[view][frame_key][str(instance_number)].get(kpt_name, None) is not None:
                            kpt_present_mask[instance_number][view_index][kpt_index] = (
                                self.view_2_frames_2_instances_2_kpts[view][frame_key][str(instance_number)][kpt_name][2] != -1
                            )
                        else:
                            kpt_present_mask[instance_number][view_index][kpt_index] = False
            kpt_list: list[torch.Tensor] = self._extract_keypoints(
                view, 
                self.view_2_frames_2_instances_2_kpts, 
                frame_idx, 
                flip=False
            )

            # pad the image to match the maximum image size and also adjust keypoint coordinates accordingly
            orig_img_size = (self.index_json['image_sizes'][view][0], self.index_json['image_sizes'][view][1])
            needs_padding = orig_img_size != self.uniform_img_size
            pad_x = 0.0
            pad_y = 0.0
            if needs_padding:
                w, h = orig_img_size
                max_w, max_h = self.uniform_img_size
                pad_x = int((max_w - w) / 2.0)
                pad_y = int((max_h - h) / 2.0)
                for inst_kpts in kpt_list:
                    valid = inst_kpts[:, 2] != -1
                    if valid.any():
                        inst_kpts[valid, 0] += pad_x
                        inst_kpts[valid, 1] += pad_y

            # meta info for this frame
            masks_row_per_instance      = self.masks_meta[view].get(origin_frame_number, [])
            crops_row_per_instance      = self.cropped_meta[view].get(origin_frame_number, [])
            full_masks_row_per_instance = self.masks_full_meta[view].get(origin_frame_number, [])

            # add placeholders for missing instances
            for inst in range(self.index_json['max_n_instances']):
                if not any(r['sub_index'] == str(inst) for r in masks_row_per_instance):
                    # add empty entries for missing instances
                    masks_row_per_instance.append({
                        'frame':        origin_frame_number,
                        'file_loc':     '', # no mask
                        'category':     'mask',
                        'sub_index':    str(inst),
                        'folder':       view,
                        'bbox':         '[0,0,0,0]',
                    })
                if not any(r['sub_index'] == str(inst) for r in crops_row_per_instance):
                    crops_row_per_instance.append({
                        'frame':        origin_frame_number,
                        'file_loc':     '', # no crop
                        'category':     'cropped',
                        'sub_index':    str(inst),
                        'folder':       view,
                    })
                if not any(r['sub_index'] == str(inst) for r in full_masks_row_per_instance):
                    full_masks_row_per_instance.append({
                        'frame':        origin_frame_number,
                        'file_loc':     '', # no full mask
                        'category':     'mask_full',
                        'sub_index':    str(inst),
                        'folder':       view,
                    })
            masks_row_per_instance      = sorted(masks_row_per_instance, key=lambda r: int(r['sub_index']))
            crops_row_per_instance      = sorted(crops_row_per_instance, key=lambda r: int(r['sub_index']))
            full_masks_row_per_instance = sorted(full_masks_row_per_instance, key=lambda r: int(r['sub_index']))

            # parse bboxes and load crops/full masks for each instance
            bboxes, crops, masks, full_masks = [], [], [], []
            for instance_number, (crops_row, masks_row, full_masks_row) in enumerate(zip(crops_row_per_instance, masks_row_per_instance, full_masks_row_per_instance)):
                if crops_row['file_loc'] == '' or masks_row['file_loc'] == '' or full_masks_row['file_loc'] == '':
                    # missing instance -> add empty entries
                    bboxes.append([0,0,0,0])
                    crops.append(np.zeros((1, 1), dtype=np.uint8))
                    masks.append(np.zeros((1, 1), dtype=np.uint8))
                    full_masks.append(np.zeros((self.uniform_img_size[1], self.uniform_img_size[0]), dtype=np.uint8))
                    seg_mask_present_mask[instance_number][view_index] = False
                    continue

                bbox = self._parse_bbox(masks_row['bbox'])
                crop = self._load_grayscale_image(view, crops_row['file_loc'])
                mask = self._load_grayscale_image(view, masks_row['file_loc'])
                full_mask = self._load_grayscale_image(view, full_masks_row['file_loc'])

                # pad the image to match the maximum image size and also adjust the bbox coordinates accordingly
                if needs_padding:
                    # bbox is specified in x1, y1, x2, y2 format
                    bbox = [bbox[0]+pad_x, bbox[1]+pad_y, bbox[2]+pad_x, bbox[3]+pad_y]
                    full_mask = np.pad(full_mask, ((pad_y, pad_y), (pad_x, pad_x)))
                    
                bboxes.append(bbox)
                crops.append(crop)
                masks.append(mask)
                full_masks.append(full_mask)
                seg_mask_present_mask[instance_number][view_index] = True
        

            view_data[view] = {
                'img_path':     str(origin_img_path),
                'bboxes':       torch.tensor(bboxes, dtype=torch.int64),              # (N,4)
                #'crops':        torch.stack([torch.from_numpy(c) for c in crops]),   # (N,Hc,Wc)
                #'masks':        torch.stack([torch.from_numpy(m) for m in masks]),
                'masks_full':   torch.stack([torch.from_numpy(m) for m in full_masks]),
                'keypoints':    torch.stack(kpt_list),                             # (N,K,3)
            }

        sample['imgpaths']    = [view_data[v]['img_path'] for v in self.views]
        sample['frames']      = list(range(len(self.views)))
        sample['instances']   = list(range(self.index_json['max_n_instances']))
        sample['bboxes']      = torch.stack([view_data[v]['bboxes']     for v in self.views])      # (N,V,4)
        #sample['crops']       = torch.stack([view_data[v]['crops']      for v in self.views])
        #sample['masks']       = torch.stack([view_data[v]['masks']      for v in self.views])      # (N,V,Wc,Hc)
        sample['masks_full']  = torch.stack([view_data[v]['masks_full'] for v in self.views])      # (N,V,Wf,Hf)
        sample['keypoints']   = torch.stack([view_data[v]['keypoints']  for v in self.views])      # (N,V,K,3)
        sample['seg_mask_present_mask'] = seg_mask_present_mask
        sample['kpt_present_mask']      = kpt_present_mask
        # -> sample["masks_full"][<view_index>][<instance_index>] gives the mask image loaded as matrix of that instance in that view
        # -> sample["keypoints"][<view_index>][<instance_index>][<keypoint_index>] gives the (x,y,conf) of that keypoint of that instance in that view

        # change size: output should be indexable by instance number first, then by view number
        for attr in ['bboxes', 'masks_full', 'keypoints']:
            sample[attr] = sample[attr].transpose(0, 1)
        # -> sample["masks_full"][<instance_index>][<view_index>] gives the mask image loaded as matrix of that instance in that view
        # -> sample["keypoints"][<instance_index>][<view_index>][<keypoint_index>] gives the (x,y,conf) of that keypoint of that instance in that view

        self.prev_data = sample

        # return sample pre-filtered for the desired instance, if specified
        if instance_idx is not None:
            sample['instances']   = instance_idx
            sample['bboxes']      = sample['bboxes'][instance_idx]      # (V,4)
            sample['masks_full']  = sample['masks_full'][instance_idx]  # (V,Wf,Hf)
            sample['keypoints']   = sample['keypoints'][instance_idx]   # (V,K,3)
            sample['seg_mask_present_mask'] = seg_mask_present_mask[instance_idx]
            sample['kpt_present_mask']      = kpt_present_mask[instance_idx]

        return sample

    def _read_csv_dict(self, view: str, filename: str, key: str, val: str|None = None) -> dict:
        """
        Return a dict that has keys which are the values of a csv in a specific column (key).
        The values of the dict are either the complete row of the csv or the value of the row in column 'value'.
        """
        path = self.root / view / filename
        with open(path) as f:
            reader = csv.DictReader(f, quotechar='"')
            d = {}
            for row in reader:
                d[row[key]] = row if val is None else row[val]
            return d

    def _group_csv(self, view: str, filename: str, filter_cat: str) -> dict[str, list[dict]]:
        """
        Returns a dict with one entry per frame if the frame has the specified category.
        {frame_number1: <files_crop.csv row for that file if the file has the specified category>}
        """
        path = self.root / view / filename
        d = defaultdict(list)
        with open(path) as f:
            reader = csv.DictReader(f, quotechar='"')
            for row in reader:
                if row['category'] != filter_cat:
                    continue
                frame = str(row['frame'])
                d[frame].append(row)
        return d

    def _parse_bbox(self, bbox_str: str) -> list[int]:
        # string "[xmin, ymin, xmax, ymax]"
        nums = bbox_str.strip('[]').split(',')
        return [int(x) for x in nums]

    def _load_grayscale_image(self, view: str, rel_path: str) -> np.ndarray:
        """
        Use cv2.imread() to read an image to np.ndarray and transpose because cv2 read to row-major order but we require column-major order.
        """
        # rel_path example: "video1/mask/image_0_0_mask.png"
        # try:
        #     img = Image.open(str(self.root / rel_path))
        #     img.verify()
        #     print("PIL load OK")
        # except Exception as e:
        #     print("PIL failed:", e)
        return np.array(cv2.imread(str(self.root / rel_path), cv2.IMREAD_GRAYSCALE))
    
    def _get_present_instances(self, f_idx: int) -> list[int]:
        """
        Get the instances that shall be reconstructed in frame f_idx.
        This is determined by the maximum number of detected instances across all views.
        """
        present_instances = set()
        for view in self.views:
            n = set(int(row["sub_index"]) for row in self.masks_meta[view].get(str(f_idx), []) if row['category']=='mask')
            present_instances = n.union(present_instances)
        return list(present_instances)

    def _extract_keypoints(
        self,
        view: str,
        view_2_frames_2_instances_2_kpts: dict[str, dict[str, InstancesKeypointsDict]],
        f_idx: int,
        flip: bool = False
    ) -> list[torch.Tensor]:
        """
        args:
            - view_2_frames_2_instances_2_kpts: dict containing keypoints and confidences per view(!), format:
                {
                    'view1': {                      x      y    conf
                            'individual1': {        ↓      ↓     ↓
                                'mouth tip':    [123.5,  87.2, 0.998],
                                'gill':         [0.0,    0.0,  0.0],   <- keypoint was not detected (occluded)
                                … 
                            },

                            OR:
                            '-1': {                                    <- keypoint detection detected the wrong number of instances
                                'mouth tip':    [-1.0,  -1.0, -1.0,],  
                                'gill':         [-1.0,  -1.0, -1.0,],
                                … 
                            },

                            OR: 
                            'individual1': {
                                'mouth tip':    [-1.0,  -1.0, -1.0,],  <- keypoint detection detected no keypoints for this (definitly present) instance
                                'gill':         [-1.0,  -1.0, -1.0,],
                                … 
                            },
                            …
                        },
                    'view2': {…}
                }
            - f_idx: index of the frame of which the keypoints shall be extracted
            - flip: 
        Build a list of (K,3) tensors for each instance from keypoints outputs.
        """
        #filename = self.get_files_for_frame([view], ['bbox-masked'], [idx])[view]['bbox_masked'][idx][0]
        inst_2_kpts: InstancesKeypointsDict = view_2_frames_2_instances_2_kpts[view][str(f_idx)]
        kpt_list = []
        for inst in range(self.index_json['max_n_instances']):
            inst = str(inst)
            pts = []
            if inst_2_kpts.get(inst, None) is None:
                # instance was not detected by keypoint detection
                pts = [[-1.0, -1.0, -1.0] for _ in range(len(self.index_json['keypoint_list']))]
                kpt_list.append(torch.tensor(pts, dtype=torch.float32))
                continue
            for x,y,conf in inst_2_kpts[inst].values():
                if flip:
                    x = self.index_json["image_size"][0] - x
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


