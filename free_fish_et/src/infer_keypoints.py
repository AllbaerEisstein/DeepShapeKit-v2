import argparse
import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2


def load_model(model_path: Path):
    return YOLO(model_path)


def inference(model: YOLO, img_path: Path) -> tuple[list[tuple[int,int,int,int]], list[list[tuple[float, float, float]]]]:
    """
    Run inference on an image and return bounding boxes of the keypoints and keypoints.

    Args:
        model (YOLO): Loaded Ultralytics YOLO model.
        img_path (Path): Path to the input image.

    Returns:
        bboxes (list[tuple[int,int,int,int]]): List of bounding boxes (x1, y1, x2, y2).

        kpts (list[list[tuple[float,float,float]]]): 
        List of the lists of keypoint tuples (x, y, visibility) per detected instance.

        Important! 
        An empty results.keypoints means "no objects were detected," not "all keypoints are missing for a detected object." 
        If you have detections, you should always get keypoint results (possibly including zeros for undetected keypoints); 
        if you have no detections, results.keypoints is empty.
    """
    results = model(str(img_path), verbose=False)

    bboxes: list[tuple[int,int,int,int]]           = []
    kpts:   list[list[tuple[float, float, float]]] = []

    for result in results:
        if result.keypoints is not None: # else: no instance was detected
            for instance in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, instance)
                bboxes.append((x1, y1, x2, y2))

            for instance_kpts in result.keypoints.data.cpu().numpy(): # shape: (num_instances, num_keypoints, 3)
                kpts.append([(float(x), float(y), float(v)) for x, y, v in instance_kpts])
    
    return bboxes, kpts


def create_discrete_color_map(kpt_names, cmap=cv2.COLORMAP_RAINBOW, RGB=False):
    """
    From a pre-defined cv2-colormap and a list of keypoints, assign equally distributed colors from cmap to each keypoint.
    
    Returns:
        kpt_name_2_color (Dict[str, Tuple[int, int, int]]) containing the keypoint names as keys and a 3-tuple (0-255) RGB color as values.
    """
    n = len(kpt_names)
    if n < 2:
        raise ValueError("At least two keypoint names are required for color mapping.")

    step_values = np.linspace(0, 255, n, dtype=np.uint8)
    kpt_name_2_color = {}

    for kpt, val in zip(kpt_names, step_values):
        gray = np.uint8([[val]])  # 1x1 grayscale image
        color_bgr = cv2.applyColorMap(src=gray, colormap=cmap)[0, 0]
        color_rgb = tuple(int(c) for c in color_bgr[::-1])  # Convert BGR to RGB
        kpt_name_2_color[kpt] = [int(v) for v in color_rgb] if RGB else [int(v) for v in color_bgr]

    return kpt_name_2_color


def draw_kpts(img_path: Path, kpts, out_dir: Path, kpt_name_2_color, annot_radius = 5):
    """
    Args:
        kpts (List[List[np.ndarray]]): List of the lists of keypoint tuples (x, y, visibility) per detected instance.
    Create an output image with keypoint annotation.
    """
    img = cv2.imread(str(img_path))
    kpt_names = [name for name, _ in kpt_name_2_color.items()]

    for instance in kpts:
        for i, (x, y, v) in enumerate(instance):
            if v and (x!=0 and y!=0):
                kpt_name = kpt_names[i]
                cv2.circle(img, center=(int(x),int(y)), radius=annot_radius, color=kpt_name_2_color[kpt_name], lineType=-1)

    # Prepare output path
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"Saved processed image to: {out_path}")
    return kpt_name_2_color


def main():
    p = argparse.ArgumentParser(
        description="""
        Annotate images with keypoints using a trained YOLO model. 
        Example usage: 
        python3 keypoints/infer_keypoints.py --indir=../data/bluegill_videos/05142025/4-cam-2/bbx_images/ --outdir=../data/bluegill_videos/05142025/4-cam-2/annotations/ --model=keypoints/bluegill_pose.pt --number=100 --kpts "mouth tip" "gill" "root of pelvic fin" "caudal peduncle" "middle of caudal fin" "lower tip of caudal fin"
        """
    )
    p.add_argument('--indir', required=True, help="Directory with input images (.jpg, .png)")
    p.add_argument('--outdir', required=True, help="Directory to save output images")
    p.add_argument('--model', default="bluegill_pose.pt", help="Path to your YOLO .pt model")
    p.add_argument('--kpts', nargs='+', required=True, help="Ordered list of keypoint names to include")
    p.add_argument('--number', default=None, type=int, help="Limit the number of files to be processed")
    args = p.parse_args()

    indir       = Path(args.indir)
    outdir      = Path(args.outdir)
    model_path  = args.model
    kpt_names   = args.kpts
    number_of_images = args.number

    kpt_name_2_color = create_discrete_color_map(kpt_names)

    if not indir.is_dir():
        print(f"Error: {indir} is not a directory.")
        sys.exit(1)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load your custom trained YOLO model
    model = YOLO(model_path)

    # Iterate over images
    img_counter = 0
    for img_path in indir.iterdir():
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img_counter += 1
            if number_of_images is not None:
                if img_counter > number_of_images:
                    break
            # Inference
            bboxes, kpts = inference(model, img_path)

            print(kpts)

            # Process and save
            if kpts is not None:
                kpt_name_2_color = draw_kpts(img_path, kpts, outdir, kpt_name_2_color)


if __name__ == '__main__':
    main()
