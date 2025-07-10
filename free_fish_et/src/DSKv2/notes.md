## Dataset:

### Following dataset structure is expected:

```
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
    ├── files.csv       # Keeps track of the extracted frames
    ├── frame2video_1.csv # Maps original to new frame (1:1)
    └── files_crop.csv  # one CSV per folder
```

### files.csv:

    Keeps track of the extracted frames.
    One row per extracted frame.
    Columns:
        - 'frame' - frame number
        - 'file_loc' - relative path to the saved file ("{video_name}_{frame_number}.png"
        - 'category' - field is always set to 'origin'
        - 'sub_index' - field is always set to 0
        - 'folder' - video name

### frame2video_1.csv:

    Maps original to new frame (processed by pocess_frames) (1:1).
    One row per video frame.
    Columns:
        - 'origin_frame' - frame number (int)
        - 'new_frame' - same entry as in 'origin_frame'

### files_crop.csv:

    One row per saved crop or mask.
    Columns:
        - frame — original frame number
        - file_loc — relative path to the saved file (cropped or mask)
        - category — 'cropped', 'mask' or 'bbox-masked'
        - sub_index — detection index (number of the instance detected)
        - folder — the <frame_folder> name
        - bbox — [xmin, ymin, xmax, ymax] from the mask

## Process:

Segmentation masks and keypoints are detected and stored in specific dataset format. multiview_reconstruction.reconstruct is called. This attempts to create the multiview dataset specified in dataloaders and to initalize an optimizer as well as the fish mesh and a silhouette renderer. Then, for each frame, it loads the frame from the dataset and calls multiview.fit_mesh on it.

## Todo:

- Understand process_files() in extract_frames.py
  - Specifically, frames2video.csv
- Modify dataloader to use bbox-masked images (incorporate info from files_crop.csv)
- Modify dataloaders to fit with YOLO keypoints.
- Modify dataloaders so that it can work with arbitrary amount of views
- Get calibration data
- Understand multivew
- Adjust multiview error function weights
