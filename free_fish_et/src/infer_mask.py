import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def load_model(model_path: Path):
    return YOLO(model_path)


def inference(model: YOLO, image_path: Path) -> tuple[list, list, list]:
    """
    Run inference on an image and return masks and bounding boxes.

    Args:
        model (YOLO): Loaded Ultralytics YOLO model.
        img_path (Path): Path to the input image.

    Returns:
        bboxes (list[list[int,int,int,int]]): list of one bounding box (x1, y1, x2, y2) per instance.

        masks (list[list[list[float,float]]]): list of mask polygons (list of points) per instance.

        confs (list[float]): Confidence for each detection.
    """
    # Run prediction
    results = model(str(image_path), verbose=False)

    bboxes = []
    masks_xy = []
    confs = []

    for result in results:
        if result.masks is not None:
            for instance in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, instance)
                bboxes.append(list((x1, y1, x2, y2)))

            for instance in result.masks.xy:
                masks_xy.append(instance.tolist())
            
            for instance in result.boxes.conf:
                confs.append(float(instance))
        else:
            return [[]], [[[]]], [0.0] # None detection to be filtered out by confidence

    return bboxes, masks_xy, confs
        


def img2bbx(img_path: Path, bbox, out_dir: Path, padding, out_filename: None | str = None):
    """
    Create an output image the same size as input but black everywhere
    except inside the bounding box(es), where the original content is kept.
    """
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Black canvas
    canvas = np.zeros((h, w, 3), dtype=img.dtype)

    (x1, y1, x2, y2) = bbox
    # Ensure coords within image
    x1, y1 = max(0, x1-padding), max(0, y1-padding)
    x2, y2 = min(w, x2+padding), min(h, y2+padding)
    canvas[y1:y2, x1:x2] = img[y1:y2, x1:x2]

    # Prepare output path
    if out_filename is None:
        out_path = out_dir / img_path.name
    else:
        out_path = out_dir / out_filename
    cv2.imwrite(str(out_path), canvas)


def main():
    p = argparse.ArgumentParser(
        description="Blacken images except for detected object bounding boxes using a trained YOLO model."
    )
    p.add_argument('--indir', required=True, help="Directory with input images (.jpg, .png)")
    p.add_argument('--outdir', required=True, help="Directory to save output images")
    p.add_argument('--model', default="segment_bluegill.pt", help="Path to your YOLO .pt model")
    p.add_argument('--padding', type=int, default=0, help="Area around bounding box in pixel that will be visible in the output additionally.")
    args = p.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    model_path = args.model
    padding = args.padding

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
            bboxes, masks, confs = inference(model, img_path)

            # Process and save
            img2bbx(img_path, bboxes, outdir, padding)


if __name__ == '__main__':
    main()
