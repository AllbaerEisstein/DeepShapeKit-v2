import csv
import os
import cv2
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from PIL import Image

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
        # self.views = ['front', 'bottom']
        self.views = [str(subdir.name) for subdir in self.root.iterdir() if subdir.is_dir()]

        # DLC outputs per view
        """
        pickle.load is expected to yield
            {
                "frame<frame_number>": {
                    'coordinates':  [(x1, y1), ...]
                    'confidence':   [c1, ...]
                }
                ...
            }
        """
        self.keypoints_confs = {
            v: pickle.load(
                open(self.root / v / 'keypoints_results' / f'keypoints_confs.pickle', 'rb')
            ) for v in self.views
        }

        # original frame file locations
        self.files = {
            v: self._read_csv_dict(v, 'files.csv', key='frame')
            for v in self.views
        }

        # mask crop entries per view, grouped by origin frame
        self.mask_rows = {
            v: self._group_csv(v, 'files_crop.csv', filter_cat='mask')
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
        return len(self.frame_map[self.views[0]])
    


    def __getitem__(self, idx: int) -> dict:
        frame_key = str(idx)
        sample = {'frames': [], 'imgpaths': [], 'instances': None}

        view_data = {}
        for view in self.views:
            origin_frame_number = self.frame_map[view][frame_key]
            # image path
            file_loc = self.files[view][origin_frame_number]['file_loc']
            origin_img_path = self.root / file_loc

            # mask rows for this frame
            rows = self.mask_rows[view].get(origin_frame_number, [])
            # sort by sub_index to keep instance order
            rows = sorted(rows, key=lambda r: int(r['sub_index']))

            # parse bboxes and load crops/full masks
            bboxes, crops, masks, full_masks = [], [], [], []
            for row in rows:
                bbox = self._parse_bbox(row['bbox'])
                bboxes.append(bbox)
                crops.append(self._load_mask_crop(view, row['file_loc']))
                masks.append(self._load_full_mask(view, row['file_loc']))

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
        sample['masks']       = torch.stack([view_data[v]['crops']      for v in self.views])      # (V,N,Hc,Wc)
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

    def _group_csv(self, view: str, filename: str, filter_cat: str) -> dict:
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

    def _load_mask_crop(self, view: str, rel_path: str) -> np.ndarray:
        # rel_path example: "video1/mask/image_0_0_mask.png"
        return cv2.imread(str(self.root / rel_path), cv2.IMREAD_GRAYSCALE)

    def _load_full_mask(self, view: str, rel_path: str) -> np.ndarray:
        # replace '/mask/' with '/mask_full/' in path
        parts = Path(rel_path).parts
        parts = [view if i==view else p for i,p in enumerate(parts)]
        parts = list(parts)
        parts[-2] = 'mask_full'
        return cv2.imread(str(self.root.joinpath(*parts)), cv2.IMREAD_GRAYSCALE)

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
            - keypoints_dict: dict containing keypoints and confidences, format:
                {
                    <filename1>: {                x      y    conf
                            'individual1': {        ↓      ↓     ↓
                                'mouth tip':    [123.5,  87.2, 0.998],
                                'gill':         [ 98.4,  65.1, 0.987],
                                … 
                            },
                            'individual2': {
                                'mouth tip':    [223.1,  87.9, 0.992],
                                'gill':         [198.3,  60.7, 0.981],
                                … 
                            },
                            …
                        },
                    <filename1>: {…}
                }
            - idx: index of the frame of which the keypoints shall be extracted
            - n_instances:
            - flip: 
        Build a list of (K,3) tensors for each instance from keypoints outputs:
            - keypoints_dict['coordinates']: list of length K each shape (M,2)
            - keypoints_dict['confidence']:   list of length K each shape (M,1)
        """
        filename = self.get_files_for_frame([view], ['bbox-masked'], [idx])[view]['bbox_masked'][idx][0]
        coords = keypoints_dict[filename]
        kpt_list = []
        for inst in range(n_instances):
            pts = []
            for kpt_name in coords[inst]:
                x,y     = (coords[inst][kpt_name,0], coords[inst][kpt_name,1])
                conf    = coords[inst][kpt_name,2]
                if flip:
                    # assume width-known; you may parametrize
                    x = 2048 - x
                pts.append([x,y,conf])                              # one tuple for each kpt
            kpt_list.append(torch.tensor(pts, dtype=torch.float32)) # one tensor for reach instance
        return kpt_list

    def get_files_for_frame(self, views: list[str], categories: list[str], idxs: list[int]) -> dict[str, dict[str, dict[int, list[Path]]]]:
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
                        meta_info = self.mask_rows[view]
                    case 'cropped':
                        meta_info = self.mask_rows[view]
                    case 'bbox-masked':
                        meta_info = self.mask_rows[view]
                    case 'origin':
                        meta_info = self.files[view]
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


    # def __init__(self, root='data/detection_1/frames'):
    #     self.root = root

    #     self.detections_front = pickle.load(open(os.path.join(root, "front", "dlc_results",
    #                                                            "full_sizeDLC_resnet50_master2021demo_frontJun2shuffle1_30000_full.pickle"), "rb"))
    #     self.detections_bottom = pickle.load(open(os.path.join(root, "bottom", "dlc_results",
    #                                                            "full_sizeDLC_resnet50_master2021demo_bottomJun1shuffle1_15000_full.pickle"), "rb"))

    #     # saving the rows of the video's csv containing information about each file in front_index, indexed by frame_number.
    #     # i.e.
    #     # front_index = {frame_number1: files.csv-row1, ...}
    #     self.front_index = {}
    #     with open(os.path.join(root, "front", "files.csv")) as fif:
    #         reader = csv.DictReader(fif)
    #         for row in reader:
    #             self.front_index[row['frame']] = row

    #     self.bottom_index = {}
    #     with open(os.path.join(root, "bottom", "files.csv")) as bif:
    #         reader = csv.DictReader(bif)
    #         for row in reader:
    #             self.bottom_index[row['frame']] = row

    #     self.mask_front_index = {}
    #     with open(os.path.join(root, "front", "files_crop.csv")) as fif:
    #         reader = csv.DictReader(fif, quotechar=' ')
    #         for row in reader:
    #             if row['category'] == 'mask' and row['sub_index'] == '0':
    #                 self.mask_front_index[row['frame']] = [row]
    #             elif row['category'] == 'mask' and row['sub_index'] == '1':
    #                 self.mask_front_index[row['frame']].append(row)

    #     self.mask_bottom_index = {}
    #     with open(os.path.join(root, "bottom", "files_crop.csv")) as bif:
    #         reader = csv.DictReader(bif, quotechar=' ')
    #         for row in reader:
    #             if row['category'] == 'mask' and row['sub_index'] == '0':
    #                 if row['frame'] not in self.mask_bottom_index.keys():
    #                     self.mask_bottom_index[row['frame']] = [row]
    #                 else:
    #                     self.mask_bottom_index[row['frame']].append(row)
    #             elif row['category'] == 'mask' and row['sub_index'] == '1':
    #                 if row['frame'] not in self.mask_bottom_index.keys():
    #                     self.mask_bottom_index[row['frame']] = [row]
    #                 else:
    #                     self.mask_bottom_index[row['frame']].append(row)

    #     self.front_match = {}
    #     with open(os.path.join(root, "front", "frame2video_1.csv")) as fif:
    #         reader = csv.DictReader(fif)
    #         for row in reader:
    #             self.front_match[row['new_frame']] = row['origin_frame']

    #     self.bottom_match = {}
    #     with open(os.path.join(root, "bottom", "frame2video_1.csv")) as bif:
    #         reader = csv.DictReader(bif)
    #         for row in reader:
    #             self.bottom_match[row['new_frame']] = row['origin_frame']

    #     self.prev_data = None

    # def __getitem__(self, index):
    #     path_front = self.front_index[self.front_match[str(index)]]['file_loc']
    #     path_bottom = self.bottom_index[self.bottom_match[str(index)]]['file_loc']

    #     front_bboxs = [row['bbox'] for row in self.mask_front_index[self.front_match[str(index)]]]
    #     bbox_front = []
    #     for candidate_box in front_bboxs:
    #         bbox_list = []
    #         for x in candidate_box[1:-1].split(','):
    #             try:
    #                 bbox_list.append(int(x))
    #             except:
    #                 pass
    #         if len(bbox_list) != 0:
    #             bbox_front.append(bbox_list)

    #     front_individual_index = 0
    #     if len(bbox_front) == 2:
    #         if bbox_front[0][0] < bbox_front[1][0]:
    #             bbox_front2 = bbox_front[1]
    #             bbox_front = bbox_front[0]
    #             row_front = self.mask_front_index[self.front_match[str(index)]][0]
    #             row_front2 = self.mask_front_index[self.front_match[str(index)]][1]
    #         else:
    #             front_individual_index = 1
    #             bbox_front2 = bbox_front[0]
    #             bbox_front = bbox_front[1]
    #             row_front = self.mask_front_index[self.front_match[str(index)]][1]
    #             row_front2 = self.mask_front_index[self.front_match[str(index)]][0]
    #     else:  # len == 1
    #         #return {'full_kpts': False}
    #         bbox_front = bbox_front[0]
    #         bbox_front2 = bbox_front
    #         row_front = self.mask_front_index[self.front_match[str(index)]][0]
    #         row_front2 = row_front

    #     bottom_bboxs = [row['bbox'] for row in self.mask_bottom_index[self.bottom_match[str(index)]]]
    #     bbox_bottom = []

    #     for candidate_box in bottom_bboxs:
    #         bbox_list = []
    #         for x in candidate_box[1:-1].split(','):
    #             try:
    #                 bbox_list.append(int(x))
    #             except:
    #                 pass
    #         if len(bbox_list) != 0:
    #             bbox_bottom.append(bbox_list)

    #     bottom_individual_index = 0
    #     if len(bbox_bottom) == 2:
    #         if bbox_bottom[0][0] < bbox_bottom[1][0]:
    #             bbox_bottom2 = bbox_bottom[1]
    #             bbox_bottom = bbox_bottom[0]
    #             row_bottom = self.mask_bottom_index[self.bottom_match[str(index)]][0]
    #             row_bottom2 = self.mask_bottom_index[self.bottom_match[str(index)]][1]
    #         else:
    #             bottom_individual_index = 1
    #             bbox_bottom2 = bbox_bottom[0]
    #             bbox_bottom = bbox_bottom[1]
    #             row_bottom = self.mask_bottom_index[self.bottom_match[str(index)]][1]
    #             row_bottom2 = self.mask_bottom_index[self.bottom_match[str(index)]][0]
    #     else:  # len == 1
    #         return {'full_kpts': False}
    #         bbox_bottom2 = None
    #         bbox_bottom = bbox_bottom[0]
    #         row_bottom = self.mask_bottom_index[self.bottom_match[str(index)]][0]
    #         row_bottom2 = None

    #     # bbox_front = self.mask_front_index[self.front_match[str(index)]]['bbox']
    #     # bbox_front_list = []
    #     # for x in bbox_front[1:-1].split(' '):
    #     #     try:
    #     #         bbox_front_list.append(int(x))
    #     #     except:
    #     #         pass
    #     # if len(bbox_front_list) != 0:
    #     #     bbox_front = bbox_front_list  # [w_low, h_low, w_up, h_up]
    #     # else:
    #     #     raise Exception('front bbox empty')
    #     #
    #     # bbox_bottom = self.mask_bottom_index[self.bottom_match[str(index)]]['bbox']
    #     # # bbox_bottom = [int(x) for x in bbox_bottom[1:-1].split(' ')]
    #     # bbox_bottom_list = []
    #     # for x in bbox_bottom[1:-1].split(' '):
    #     #     try:
    #     #         bbox_bottom_list.append(int(x))
    #     #     except:
    #     #         pass
    #     # if len(bbox_bottom_list) != 0:
    #     #     bbox_bottom = bbox_bottom_list
    #     # else:
    #     #     raise Exception('bottom bbox empty')

    #     # get keypoints from pickle files
    #     item_key = 'frame' + '0' * (len(list(self.detections_front.keys())[-1]) - 5 - len(str(index))) + str(index)


    #     """
    #     #### ----------------------- detections_front is being read (keypoints pickle) ---------------------- ####
    #     """

    #     front_confdences = self.detections_front[item_key]['confidence']
    #     front_kpts = self.detections_front[item_key]['coordinates'][0]
    #     kpt_front = []
    #     kpt_front2 = []
    #     for i in range(len(front_confdences)):
    #         if i == 5:  # skip eye keypoint
    #             continue
    #         #kpt_id = np.argmax(front_confdences[i])
    #         kpt_ids = front_confdences[i][:, 0].argsort()[-2:]  # two most confident kpt
    #         if len(kpt_ids) < 2:
    #             kpt_id = kpt_ids[0]
    #             kpt_id_2 = kpt_ids[0]

    #             if not self.prev_data:
    #                 return {'full_kpts': False}

    #             if i < 5:
    #                 current_id = i
    #             else:
    #                 current_id = i - 1

    #             prev_front_1 = self.prev_data['keypoints'][0,current_id,:2]
    #             prev_front_2 = self.prev_data['keypoints2'][0,current_id,:2]

    #             w = front_kpts[i][kpt_id, 0]
    #             h = front_kpts[i][kpt_id, 1]

    #             d1 = (w - prev_front_1[0]) ** 2 + (h - prev_front_1[1]) ** 2
    #             d2 = (w - prev_front_2[0]) ** 2 + (h - prev_front_2[1]) ** 2

    #             if d1 < d2:
    #                 kpt_front.append(np.array([w, h, front_confdences[i][kpt_id, 0]]))
    #                 kpt_front2.append(np.array([prev_front_2[0], prev_front_2[1], 0]))
    #             else:
    #                 kpt_front2.append(np.array([w, h, front_confdences[i][kpt_id, 0]]))
    #                 kpt_front.append(np.array([prev_front_1[0], prev_front_1[1], 0]))
    #         else:
    #             #kpt_id = kpt_ids[1]  # np.argmin(front_kpts[i][:2, 0])
    #         # kpt_front.append(np.append(front_kpts[i][kpt_id],front_confdences[i][kpt_id]))
    #         # kpt_id = kpt_ids[0]  # np.argmax(front_kpts[i][:2, 0])
    #         # kpt_front2.append(np.append(front_kpts[i][kpt_id], front_confdences[i][kpt_id]))
    #             # check if kpt in bounding box
    #             kpt_id = kpt_ids[0]  # np.argmax(bottom_kpts[i][:2, 0])
    #             kpt_id_2 = kpt_ids[1]
    #             w1 = front_kpts[i][kpt_id, 0]
    #             w2 = front_kpts[i][kpt_id_2, 0]
    #             h1 = front_kpts[i][kpt_id, 1]
    #             h2 = front_kpts[i][kpt_id_2, 1]

    #             # if bbox_front[0] - 5 < w1 < bbox_front[2] + 5 and bbox_front[1] - 5 < h1 < bbox_front[3] + 5:
    #             #     kpt_front.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
    #             #     kpt_front2.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))
    #             # else:
    #             #     kpt_front2.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
    #             #     kpt_front.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))

    #             if i in [0, 1]:
    #                 bound_w = [bbox_front[0] - 5, (bbox_front[0] + bbox_front[2]) / 2]
    #                 bound_h = [bbox_front[1] - 5, bbox_front[3] + 5]
    #             elif i in [2, 3, 6]:
    #                 bound_w = [(bbox_front[0] + bbox_front[2]) / 2, bbox_front[2] + 5]
    #                 bound_h = [bbox_front[1] - 5, bbox_front[3] + 5]
    #             else:
    #                 bound_w = [bbox_front[0] - 5, bbox_front[2] + 5]
    #                 bound_h = [bbox_front[1] - 5, bbox_front[3] + 5]

    #             if bound_w[0] < w1 < bound_w[1] and bound_h[0] < h1 < bound_h[1]:
    #                 kpt_front.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
    #                 kpt_front2.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))
    #             elif bound_w[0] < w2 < bound_w[1] and bound_h[0] < h2 < bound_h[1]:
    #                 kpt_front2.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
    #                 kpt_front.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))
    #             else:
    #                 dist1 = (w1 - ((bbox_front[0] + bbox_front[2]) / 2)) ** 2 + (
    #                         h1 - ((bbox_front[1] + bbox_front[3]) / 2)) ** 2
    #                 dist2 = (w2 - ((bbox_front[0] + bbox_front[2]) / 2)) ** 2 + (
    #                         h2 - ((bbox_front[1] + bbox_front[3]) / 2)) ** 2
    #                 if dist1 < dist2:
    #                     kpt_front.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
    #                     kpt_front2.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))
    #                 else:
    #                     kpt_front2.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
    #                     kpt_front.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))

    #         # if (len(front_kpts[i]) > 1):
    #         #     kpt_front.append(np.append(front_kpts[i][front_individual_index], front_confdences[i][front_individual_index]))
    #         # else:
    #         #     kpt_front.append(
    #         #         np.append(front_kpts[i][0], front_confdences[i][0]))

    #     # front_eyes = []
    #     # kpt_front_new = []
    #     # kpt_front2_new = []
    #     # for i in range(len(kpt_front)):
    #     #     if i == 5:
    #     #         front_eyes.append(kpt_front[i])
    #     #         front_eyes.append(kpt_front2[i])
    #     #     else:
    #     #         kpt_front_new.append(kpt_front[i])
    #     #         kpt_front2_new.append(kpt_front2[i])
    #     #
    #     # kpt_front = kpt_front_new
    #     # kpt_front2 = kpt_front2_new

    #     # if index == 1:
    #     #     print('debug')
    #     bottom_confdences = self.detections_bottom[item_key]['confidence']
    #     if len(self.detections_bottom[item_key]['coordinates']) > 1:
    #         print('debug')
    #     bottom_kpts = self.detections_bottom[item_key]['coordinates'][0]
    #     kpt_bottom = []
    #     kpt_bottom2 = []
    #     for i in range(len(bottom_confdences)):
    #         if bottom_confdences[i].shape[0] < 2:
    #             kpt_id = kpt_ids[0]
    #             kpt_id_2 = kpt_ids[0]

    #             if not self.prev_data:
    #                 return {'full_kpts': False}

    #             current_id = i

    #             prev_bottom_1 = self.prev_data['keypoints'][1, current_id, :2]
    #             prev_bottom_2 = self.prev_data['keypoints2'][1, current_id, :2]

    #             w = bottom_kpts[i][kpt_id, 0]
    #             h = bottom_kpts[i][kpt_id, 1]

    #             d1 = (w - prev_bottom_1[0]) ** 2 + (h - prev_bottom_1[1]) ** 2
    #             d2 = (w - prev_bottom_2[0]) ** 2 + (h - prev_bottom_2[1]) ** 2

    #             if d1 < d2:
    #                 kpt_bottom.append(np.array([w, h, bottom_confdences[i][kpt_id, 0]]))
    #                 kpt_bottom2.append(np.array([prev_bottom_2[0], prev_bottom_2[1], 0]))
    #             else:
    #                 kpt_bottom2.append(np.array([w, h, bottom_confdences[i][kpt_id, 0]]))
    #                 kpt_bottom.append(np.array([prev_bottom_1[0], prev_bottom_1[1], 0]))
    #             continue
    #         elif bottom_confdences[i].shape[0] < 3:
    #             kpt_ids = [0,1]
    #         else:
    #             if bottom_confdences[i][:, 0].max() == bottom_confdences[i][:, 0].min():
    #                 kpt_ids = [0,1]
    #             else:
    #                 kpt_ids = bottom_confdences[i][:, 0].argsort()[-2:]  # two most confident kpt
    #         #kpt_id = np.argmax(bottom_confdences[i])

    #         # check if kpt in bounding box
    #         kpt_id = kpt_ids[0]  # np.argmax(bottom_kpts[i][:2, 0])
    #         kpt_id_2 = kpt_ids[1]
    #         w1 = 2048 - bottom_kpts[i][kpt_id, 0]
    #         w2 = 2048 - bottom_kpts[i][kpt_id_2,0]
    #         h1 = bottom_kpts[i][kpt_id, 1]
    #         h2 = bottom_kpts[i][kpt_id_2, 1]

    #         margin = 5

    #         k1_in = False
    #         k2_in = False
    #         if bbox_bottom[0] - margin < w1 < bbox_bottom[2] + margin and bbox_bottom[1] - margin < h1 < bbox_bottom[3] + margin:
    #             # kpt_bottom.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id,0]]))
    #             # kpt_bottom2.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2,0]]))
    #             k1_in = True
    #         if bbox_bottom[0] - margin < w2 < bbox_bottom[2] + margin and bbox_bottom[1] - margin < h2 < bbox_bottom[3] + margin:
    #             # kpt_bottom2.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
    #             # kpt_bottom.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))
    #             k2_in = True

    #         if bbox_bottom2[0] - margin < w1 < bbox_bottom2[2] + margin and bbox_bottom2[1] - margin < h1 < bbox_bottom2[3] + margin:
    #             # kpt_bottom.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id,0]]))
    #             # kpt_bottom2.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2,0]]))
    #             k2_in = True
    #         if bbox_bottom2[0] - margin < w2 < bbox_bottom2[2] + margin and bbox_bottom2[1] - margin < h2 < bbox_bottom2[3] + margin:
    #             # kpt_bottom2.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
    #             # kpt_bottom.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))
    #             k1_in = True

    #         if k1_in and not k2_in:
    #             kpt_bottom.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
    #             kpt_bottom2.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))
    #         elif k2_in and not k1_in:
    #             kpt_bottom2.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
    #             kpt_bottom.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))
    #         else:
    #             dist1 = (w1 - ((bbox_bottom[0] + bbox_bottom[2]) / 2)) ** 2 + (
    #                         h1 - ((bbox_bottom[1] + bbox_bottom[3]) / 2)) ** 2
    #             dist2 = (w2 - ((bbox_bottom[0] + bbox_bottom[2]) / 2)) ** 2 + (
    #                         h2 - ((bbox_bottom[1] + bbox_bottom[3]) / 2)) ** 2
    #             if dist1 < dist2:
    #                 kpt_bottom.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
    #                 kpt_bottom2.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))
    #             else:
    #                 kpt_bottom2.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
    #                 kpt_bottom.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))

    #         # if (len(bottom_kpts[i]) > 1):
    #         #     kpt_bottom.append(np.append(bottom_kpts[i][bottom_individual_index], bottom_confdences[i][bottom_individual_index]))
    #         # else:
    #         #     kpt_bottom.append(
    #         #         np.append(bottom_kpts[i][0], bottom_confdences[i][0]))

    #     kpt_front = torch.tensor(kpt_front)
    #     kpt_bottom = torch.tensor(kpt_bottom)
    #     kpt_front2 = torch.tensor(kpt_front2)
    #     kpt_bottom2 = torch.tensor(kpt_bottom2)

    #     # kpt_front = torch.cat([torch.tensor(self.detections_front[item_key]['coordinates']).squeeze(),
    #     #                        torch.tensor(self.detections_front[item_key]['confidence']).squeeze(1)], -1)
    #     # kpt_bottom = torch.cat([torch.tensor(self.detections_bottom[item_key]['coordinates']).squeeze(),
    #     #                        torch.tensor(self.detections_bottom[item_key]['confidence']).squeeze(1)], -1)

    #     width = bbox_front[2] - bbox_front[0]
    #     height = bbox_front[3] - bbox_front[1]
    #     margin_front = int((width - height) / 2)
    #     #kpt_front[:,0] = (width / 368.0) * kpt_front[:,0] + bbox_front[0]
    #     #kpt_front[:,1] = (width / 368.0) * kpt_front[:,1] + bbox_front[1] - 0.5 * (width - height)

    #     # undistort keypoints
    #     # kpt_image = np.zeros((constants.in_size[0], constants.in_size[1], kpt_front.shape[0]))
    #     # for i in range(kpt_front.shape[0]):
    #     #     kpt_image[kpt_front[i,0].int(), kpt_front[i,1].int(), i] = 5
    #     #     kpt_image[kpt_front[i, 0].int()-1, kpt_front[i, 1].int(), i] = 5
    #     #     kpt_image[kpt_front[i, 0].int()+1, kpt_front[i, 1].int(), i] = 5
    #     #     kpt_image[kpt_front[i, 0].int(), kpt_front[i, 1].int()-1, i] = 5
    #     #     kpt_image[kpt_front[i, 0].int(), kpt_front[i, 1].int()+1, i] = 5
    #     #
    #     # kpt_image = kpt_image.transpose()
    #     # intrinsic_mat = np.array(constants.intrinsic_mat)
    #     # distortion = np.array(constants.distortion[0])
    #     # h, w = kpt_image.shape[:2]
    #     # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_mat, distortion, (w, h), 1, (w, h))
    #     #
    #     # mapx, mapy = cv2.initUndistortRectifyMap(intrinsic_mat, distortion, None, newcameramtx, (w, h), 5)
    #     # for i in range(kpt_front.shape[0]):
    #     #     undistort_kpt = cv2.undistort(kpt_image[i], intrinsic_mat, distortion, None, newcameramtx)

    #     width = bbox_bottom[2] - bbox_bottom[0]
    #     height = bbox_bottom[3] - bbox_bottom[1]
    #     margin_bottom = int((width - height) / 2)
    #     kpt_bottom[:, 0] = 2048. - kpt_bottom[:,0]  # for flipped detection
    #     kpt_bottom2[:, 0] = 2048. - kpt_bottom2[:, 0]
    #     # kpt_bottom[:,0] = (width / 368.0) * kpt_bottom[:,0] + bbox_bottom[0]
    #     # kpt_bottom[:,1] = (width / 368.0) * kpt_bottom[:,1] + bbox_bottom[1] - 0.5 * (width - height)

    #     kpt_bottom = torch.cat([kpt_bottom, torch.zeros([1,3])])
    #     kpt_bottom2 = torch.cat([kpt_bottom2, torch.zeros([1,3])])

    #     # img_size = 368
    #     # front_mask = cv2.resize(cv2.imread(os.path.join(self.root, 'front_mask', self.front_masks[index])), (img_size, img_size))
    #     # bottom_mask = cv2.resize(cv2.imread(os.path.join(self.root, 'bottom_mask', self.bottom_masks[index])), (img_size, img_size))

    #     # boxes = []
    #     # masks = [front_mask, bottom_mask]
    #     # for i in range(2):
    #     #     pos = np.where(masks[i])
    #     #     xmin = np.min(pos[1])
    #     #     xmax = np.max(pos[1])
    #     #     ymin = np.min(pos[0])
    #     #     ymax = np.max(pos[0])
    #     #     boxes.append([xmin, ymin, xmax, ymax])

    #     front_mask_crop = cv2.imread(os.path.join(self.root, row_front['file_loc']), cv2.IMREAD_GRAYSCALE)
    #     bottom_mask_crop = cv2.imread(os.path.join(self.root, row_bottom['file_loc']), cv2.IMREAD_GRAYSCALE)

    #     front_mask_crop = cv2.resize(front_mask_crop, (380, 380))
    #     bottom_mask_crop = cv2.resize(bottom_mask_crop, (380, 380))

    #     front_mask_crop2 = cv2.imread(os.path.join(self.root, row_front2['file_loc']), cv2.IMREAD_GRAYSCALE)
    #     bottom_mask_crop2 = cv2.imread(os.path.join(self.root, row_bottom2['file_loc']), cv2.IMREAD_GRAYSCALE)

    #     front_mask_crop2 = cv2.resize(front_mask_crop2, (380, 380))
    #     bottom_mask_crop2 = cv2.resize(bottom_mask_crop2, (380, 380))

    #     full_front_mask_loc = row_front['file_loc'].split('/')
    #     full_front_mask_loc[-2] = 'mask_full'
    #     front_mask_full = cv2.imread(os.path.join(self.root, '/'.join(full_front_mask_loc)), cv2.IMREAD_GRAYSCALE)

    #     full_bottom_mask_loc = row_bottom['file_loc'].split('/')
    #     full_bottom_mask_loc[-2] = 'mask_full'
    #     bottom_mask_full = cv2.imread(os.path.join(self.root, '/'.join(full_bottom_mask_loc)), cv2.IMREAD_GRAYSCALE)

    #     full_front_mask_loc2 = row_front2['file_loc'].split('/')
    #     full_front_mask_loc2[-2] = 'mask_full'
    #     front_mask_full2 = cv2.imread(os.path.join(self.root, '/'.join(full_front_mask_loc2)), cv2.IMREAD_GRAYSCALE)

    #     full_bottom_mask_loc2 = row_bottom2['file_loc'].split('/')
    #     full_bottom_mask_loc2[-2] = 'mask_full'
    #     bottom_mask_full2 = cv2.imread(os.path.join(self.root, '/'.join(full_bottom_mask_loc2)), cv2.IMREAD_GRAYSCALE)

    #     # front_mask = np.zeros((1040, 2048))
    #     # bottom_mask = np.zeros((1040, 2048))
    #     #
    #     # front_mask[bbox_front[1]:bbox_front[3], bbox_front[0]:bbox_front[2]] = front_mask_crop[margin_front:-margin_front,:,-1]
    #     # bottom_mask[bbox_bottom[1]:bbox_bottom[3], bbox_bottom[0]:bbox_bottom[2]] = bottom_mask_crop[margin_bottom:-margin_bottom,:,-1]

    #     data = {'imgpaths': [os.path.join(self.root, path_front), os.path.join(self.root, path_bottom)],
    #             'frames': [0, 1],
    #             'keypoints': torch.stack([kpt_front, kpt_bottom]),
    #             'keypoints2': torch.stack([kpt_front2, kpt_bottom2]),
    #             'keypoints_norm': torch.stack([self.normalize_kpt(kpt_front), self.normalize_kpt(kpt_bottom)]),
    #             'keypoints_norm2': torch.stack([self.normalize_kpt(kpt_front2), self.normalize_kpt(kpt_bottom2)]),
    #             'masks': torch.stack([torch.from_numpy(front_mask_crop), torch.from_numpy(bottom_mask_crop)]),
    #             'masks2': torch.stack([torch.from_numpy(front_mask_crop2), torch.from_numpy(bottom_mask_crop2)]),
    #             'masks_full': torch.stack([torch.from_numpy(front_mask_full), torch.from_numpy(bottom_mask_full)]),
    #             'masks_full2': torch.stack([torch.from_numpy(front_mask_full2), torch.from_numpy(bottom_mask_full2)]),
    #             'bboxes': torch.tensor([bbox_front, bbox_bottom]),
    #             'bboxes2': torch.tensor([bbox_front2, bbox_bottom2]),
    #             'full_kpts': True,
    #             #'detected_eyes': torch.tensor(front_eyes)
    #             }

    #     self.prev_data = data
    #     return data

    # def normalize_kpt(self, kpt):
    #     kpt_norm = torch.zeros_like(kpt)
    #     for i in range(kpt_norm.size(0)):
    #         if i == 0:
    #             kpt_norm[i, 2] = kpt[i, 2]
    #         else:
    #             kpt_norm[i, :2] = kpt[i, :2] - kpt[0, :2]
    #             kpt_norm[i, 2] = kpt[i, 2]

    #     return kpt_norm

    # def __len__(self):
    #     return len(self.front_match.keys())

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



