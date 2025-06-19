import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def inference(model: YOLO, img_path: Path):
    """
    Run inference on an image and return masks and bounding boxes.

    Args:
        model (YOLO): Loaded Ultralytics YOLO model.
        img_path (Path): Path to the input image.

    Returns:
        bboxes (List[Tuple[int,int,int,int]]): List of bounding boxes (x1, y1, x2, y2).
        masks (List[np.ndarray]): List of mask matrices (H x W) per object.
    """
    # Run prediction
    results = model(str(img_path))

    bboxes = []
    masks = []

    for result in results:
        # Extract bounding boxes (xyxy format) and convert to ints
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            bboxes.append((x1, y1, x2, y2))

        # Extract mask matrices (num_objects x H x W)
        if result.masks is not None:
            for mask in result.masks.data.cpu().numpy():
                masks.append(mask)

    return bboxes, masks


def process_and_save(img_path: Path, bboxes, out_dir: Path):
    """
    Create an output image the same size as input but black everywhere
    except inside the bounding boxes, where the original content is kept.
    """
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Black canvas
    canvas = np.zeros((h, w, 3), dtype=img.dtype)

    # Copy ROIs
    for (x1, y1, x2, y2) in bboxes:
        # Ensure coords within image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        canvas[y1:y2, x1:x2] = img[y1:y2, x1:x2]

    # Prepare output path
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), canvas)
    print(f"Saved processed image to: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Blacken images except for detected object bounding boxes using a trained YOLO model."
    )
    p.add_argument('--indir', required=True, help="Directory with input images (.jpg, .png)")
    p.add_argument('--outdir', required=True, help="Directory to save output images")
    p.add_argument('--model', default="segment_bluegill.pt", help="Path to your YOLO .pt model")
    args = p.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    model_path = args.model

    if not indir.is_dir():
        print(f"Error: {indir} is not a directory.")
        sys.exit(1)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load your custom trained YOLO model
    model = YOLO(model_path)

    # Iterate over images
    for img_path in indir.iterdir():
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            # Inference
            bboxes, masks = inference(model, img_path)

            # Process and save
            process_and_save(img_path, bboxes, outdir)


if __name__ == '__main__':
    main()
