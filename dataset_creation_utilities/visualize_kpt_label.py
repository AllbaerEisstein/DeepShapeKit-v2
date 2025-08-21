"""
Show or save keypoints + bounding boxes from a YOLO-pose label file.

Example label line:
0 0.044195 0.554729 0.072572 0.124529 0.080481 0.584129 2 0.041421 0.558276 2 0.059753 0.616993 2 ...

Usage:
    python show_kps_and_bbox.py --label labels/0001.txt --width 1280 --height 720
    python show_kps_and_bbox.py --label labels/0001.txt --width 640 --height 480 --save debug.png
    python show_kps_and_bbox.py --label labels/0001.txt --width 640 --height 480 --save debug.png --show
"""
import argparse
import os
from typing import List, Tuple
import cv2
import numpy as np

# Default colors (B, G, R) for vis 0,1,2
DEFAULT_COLORS = {
    0: (0, 0, 255),    # red
    1: (0, 255, 255),  # yellow
    2: (0, 255, 0),    # green
}

EPS = 1e-9

def parse_tokens_from_line(line: str) -> List[str]:
    s = line.strip()
    if not s or s.startswith("#"):
        return []
    return s.split()

def is_kp_only_format(tokens: List[str]) -> bool:
    # 1 + 3*n
    if len(tokens) < 4:
        return False
    try:
        float(tokens[0])
    except ValueError:
        return False
    return (len(tokens) - 1) % 3 == 0

def is_yolopose_format(tokens: List[str]) -> bool:
    # 5 + 3*n
    if len(tokens) < 5:
        return False
    try:
        for i in range(5):
            float(tokens[i])
    except ValueError:
        return False
    return (len(tokens) - 5) % 3 == 0

def safe_float_to_int_vis(v_str: str) -> int:
    """Return integer visibility 0/1/2 from a string (may be '2.000000')."""
    try:
        fv = float(v_str)
    except Exception:
        return 0
    # round to nearest int and clamp to 0/1/2
    ri = int(round(fv))
    if ri < 0: ri = 0
    if ri > 2: ri = 2
    return ri

def clamp_coord(x: float, max_val: int) -> int:
    """Clamp normalized or pixel coordinate into [0, max_val-1] integer pixel coord."""
    xi = int(round(x))
    if xi < 0: return 0
    if xi >= max_val: return max_val - 1
    return xi

def denorm_x(px_norm: float, img_w: int) -> int:
    return clamp_coord(px_norm * img_w, img_w)

def denorm_y(py_norm: float, img_h: int) -> int:
    return clamp_coord(py_norm * img_h, img_h)

def bbox_from_yolo_center(xc_norm: float, yc_norm: float, w_norm: float, h_norm: float, img_w: int, img_h: int) -> Tuple[int,int,int,int]:
    """Return (x1,y1,x2,y2) integer pixel coordinates clamped into image."""
    x_center = xc_norm * img_w
    y_center = yc_norm * img_h
    bw = w_norm * img_w
    bh = h_norm * img_h
    x1 = int(round(x_center - bw/2.0))
    y1 = int(round(y_center - bh/2.0))
    x2 = int(round(x_center + bw/2.0))
    y2 = int(round(y_center + bh/2.0))
    # clamp
    x1 = max(0, min(img_w-1, x1))
    y1 = max(0, min(img_h-1, y1))
    x2 = max(0, min(img_w-1, x2))
    y2 = max(0, min(img_h-1, y2))
    return x1, y1, x2, y2

def bbox_from_kps_normalized(kps: List[Tuple[float,float]], img_w: int, img_h: int) -> Tuple[int,int,int,int]:
    """Compute bbox that encloses present KPs (in normalized coords), fallback to full-frame."""
    present = [(x,y) for x,y in kps if x is not None and y is not None]
    if not present:
        # full frame
        return 0, 0, img_w-1, img_h-1
    xs = [x for x,y in present]
    ys = [y for x,y in present]
    xmin = min(xs) * img_w
    xmax = max(xs) * img_w
    ymin = min(ys) * img_h
    ymax = max(ys) * img_h
    # ensure non-zero
    if xmax - xmin < EPS:
        xmax = xmin + 1.0
    if ymax - ymin < EPS:
        ymax = ymin + 1.0
    x1 = int(round(xmin))
    y1 = int(round(ymin))
    x2 = int(round(xmax))
    y2 = int(round(ymax))
    x1 = max(0, min(img_w-1, x1))
    y1 = max(0, min(img_h-1, y1))
    x2 = max(0, min(img_w-1, x2))
    y2 = max(0, min(img_h-1, y2))
    return x1,y1,x2,y2

def draw_bbox_center_dimensions(img: np.ndarray,
                                bbox: Tuple[int,int,int,int],
                                img_w: int,
                                img_h: int,
                                color: Tuple[int,int,int],
                                thickness: int = 1,
                                font_scale: float = 0.5):
    """
    Draw the bbox center and annotate width/height.
    - draws a small cross at the center
    - draws horizontal and vertical thin lines through center spanning the bbox
    - writes width and height (pixels and normalized) near the bbox
    """
    x1, y1, x2, y2 = bbox
    # center in pixels
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    w_px = max(1, x2 - x1)
    h_px = max(1, y2 - y1)
    w_norm = w_px / float(img_w)
    h_norm = h_px / float(img_h)

    # cross at center
    cross_len = max(6, min(w_px, h_px) // 10)  # adapt size
    cv2.line(img, (cx - cross_len, cy), (cx + cross_len, cy), color, thickness=thickness)
    cv2.line(img, (cx, cy - cross_len), (cx, cy + cross_len), color, thickness=thickness)
    cv2.circle(img, (cx, cy), radius=max(1, cross_len//3), color=color, thickness=-1)

    # thin lines across bbox center to indicate width / height visually
    # horizontal line across bbox center
    cv2.line(img, (x1, cy), (x2, cy), color, thickness=max(1, thickness//1))
    # vertical line across bbox center
    cv2.line(img, (cx, y1), (cx, y2), color, thickness=max(1, thickness//1))

    # prepare annotation text: pixel and normalized
    txt_w = f"W: {w_px}px ({w_norm:.3f})"
    txt_h = f"H: {h_px}px ({h_norm:.3f})"
    # place the text above the bbox (if room), otherwise below
    text_pad = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((tw_w, th_w), _) = cv2.getTextSize(txt_w, font, font_scale, thickness)
    ((tw_h, th_h), _) = cv2.getTextSize(txt_h, font, font_scale, thickness)
    txt_w_total = max(tw_w, tw_h)
    txt_height_total = th_w + th_h + text_pad

    # compute background rect coords
    bx1 = x1
    by1 = max(0, y1 - txt_height_total - 6)
    bx2 = min(img_w - 1, bx1 + txt_w_total + 8)
    by2 = by1 + txt_height_total + 6
    # draw semi-opaque background by filled rectangle (solid color)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, thickness=-1)
    # draw texts in contrasting color (black or white depending on bbox color brightness)
    # compute brightness to choose text color
    brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
    text_color = (0,0,0) if brightness > 128 else (255,255,255)

    ty = by1 + th_w + 2
    cv2.putText(img, txt_w, (bx1 + 4, ty), font, font_scale, text_color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.putText(img, txt_h, (bx1 + 4, ty + th_w + 2), font, font_scale, text_color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_instance_on_image(img: np.ndarray,
                           bbox: Tuple[int,int,int,int],
                           keypoints_px: List[Tuple[int,int]],
                           vis_list: List[int],
                           class_idx: int,
                           colors: dict,
                           img_w: int,
                           img_h: int,
                           point_radius: int = 4,
                           thickness: int = 2):
    """Draw bbox and keypoints on the image."""
    x1,y1,x2,y2 = bbox
    # choose bbox color by max visibility of keypoints
    if vis_list:
        max_vis = max(vis_list)
    else:
        max_vis = 0
    bbox_color = colors.get(max_vis, colors[0])
    # draw bbox
    cv2.rectangle(img, (x1,y1), (x2,y2), bbox_color, thickness=thickness)
    # draw class text at top-left
    text = str(class_idx)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_th = 2
    # compute text background
    ((tw, th), _) = cv2.getTextSize(text, font, font_scale, text_th)
    # background rectangle
    pad = 4
    bx1 = x1
    by1 = max(0, y1 - th - pad)
    bx2 = bx1 + tw + pad
    by2 = y1
    cv2.rectangle(img, (bx1, by1), (bx2, by2), bbox_color, thickness=-1)
    cv2.putText(img, text, (bx1 + 2, by2 - 4), font, font_scale, (0,0,0), thickness=text_th, lineType=cv2.LINE_AA)

    draw_bbox_center_dimensions(img, bbox, img_w, img_h, bbox_color, thickness=1, font_scale=0.45)

    # draw keypoints
    for (px,py), vis in zip(keypoints_px, vis_list):
        col = colors.get(vis, colors[0])
        if px is None or py is None:
            continue
        cv2.circle(img, (px, py), point_radius, col, thickness=-1)
        # also draw small outline
        cv2.circle(img, (px, py), max(1, point_radius//2), (0,0,0), thickness=1)

def process_label_file(path: str, img_w: int, img_h: int, colors: dict):
    """
    Read a label file (may contain multiple lines/instances).
    Return an image (BGR ndarray) with drawings and number of instances processed.
    """
    # black image
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    instances_processed = 0
    for line in lines:
        toks = parse_tokens_from_line(line)
        if not toks:
            continue

        if is_yolopose_format(toks):
            # class + bbox + triplets
            cls = int(float(toks[0]))
            xc = float(toks[1])
            yc = float(toks[2])
            bw = float(toks[3])
            bh = float(toks[4])
            # bbox
            bbox_px = bbox_from_yolo_center(xc, yc, bw, bh, img_w, img_h)

            # keypoints
            triplet_vals = toks[5:]
            n_trip = len(triplet_vals) // 3
            keypoints_px = []
            vis_list = []
            for i in range(n_trip):
                px_n = float(triplet_vals[3*i + 0])
                py_n = float(triplet_vals[3*i + 1])
                vis_tok = triplet_vals[3*i + 2]
                vis = safe_float_to_int_vis(vis_tok)
                # if px_n and py_n are exactly 0.0 and vis==0, treat as missing (skip drawing)
                if abs(px_n) < EPS and abs(py_n) < EPS and vis == 0:
                    keypoints_px.append((None, None))
                    vis_list.append(0)
                else:
                    keypoints_px.append((denorm_x(px_n, img_w), denorm_y(py_n, img_h)))
                    vis_list.append(vis)

            draw_instance_on_image(img, bbox_px, keypoints_px, vis_list, cls, colors, img_w, img_h)
            instances_processed += 1
            continue

        if is_kp_only_format(toks):
            # no bbox in file; compute bbox from present keypoints
            cls = int(float(toks[0]))
            triplet_vals = toks[1:]
            n_trip = len(triplet_vals) // 3
            kps_norm = []
            keypoints_px = []
            vis_list = []
            for i in range(n_trip):
                px_n = float(triplet_vals[3*i + 0])
                py_n = float(triplet_vals[3*i + 1])
                vis_tok = triplet_vals[3*i + 2]
                vis = safe_float_to_int_vis(vis_tok)
                if abs(px_n) < EPS and abs(py_n) < EPS and vis == 0:
                    kps_norm.append((None, None))
                    keypoints_px.append((None, None))
                    vis_list.append(0)
                else:
                    kps_norm.append((px_n, py_n))
                    keypoints_px.append((denorm_x(px_n, img_w), denorm_y(py_n, img_h)))
                    vis_list.append(vis)
            bbox_px = bbox_from_kps_normalized([p for p in kps_norm if p[0] is not None], img_w, img_h) if any(p[0] is not None for p in kps_norm) else (0,0,img_w-1,img_h-1)
            draw_instance_on_image(img, bbox_px, keypoints_px, vis_list, cls, colors, img_w, img_h)
            instances_processed += 1
            continue

        # Unknown format: skip
        print(f"Warning: skipping unknown-format line in {path!r}: {line.strip()[:120]}")

    return img, instances_processed

def main():
    p = argparse.ArgumentParser(description="Visualize YOLO-pose keypoints + bounding boxes on a black canvas.")
    p.add_argument("--label", required=True, help="Path to the label .txt file")
    p.add_argument("--width", type=int, default=1024, help="Image width (pixels)")
    p.add_argument("--height", type=int, default=576, help="Image height (pixels)")
    p.add_argument("--save", default=None, help="If given, save the visualization to this image path instead of (or in addition to) showing")
    p.add_argument("--no-show", dest="show", action="store_false", help="Do not open a GUI window (useful in headless or when using --save)")
    p.add_argument("--radius", type=int, default=5, help="Keypoint circle radius in pixels")
    p.add_argument("--thickness", type=int, default=2, help="BBox thickness in pixels")
    args = p.parse_args()

    lab_path = args.label
    if not os.path.isfile(lab_path):
        print(f"Error: label file '{lab_path}' not found.")
        return

    img_w = args.width
    img_h = args.height

    # use defaults and override radius/thickness by monkeypatching draw function via closures:
    colors = DEFAULT_COLORS.copy()

    # process file
    # set the drawing style by temporarily binding the passed radii/thickness into call by using lambda wrapper
    # but simpler: set global defaults via parameters passed to draw_instance call (we'll pass them)
    img, n = process_label_file(lab_path, img_w, img_h, colors)

    # apply radius/thickness by re-drawing: we already used defaults. To respect args.radius/thickness,
    # we'll re-create img with those values by calling process_label_file with updated draw parameters.
    # Simpler: just call a wrapper that uses requested radii/thickness. To keep code simple and explicit:
    # Re-run drawing function with desired params by re-implementing process_label_file call that passes them.
    # (To avoid duplicating code, we will modify draw_instance_on_image's default values by calling a small wrapper.)
    # Instead, to keep the script simple, we'll just re-create using the provided radius/thickness.
    # (So call an internal variant that uses them.)

    # rebuild image properly with user radii/thickness
    def process_label_file_with_style(path, img_w, img_h, colors, radius, thickness):
        img_local = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        instances_processed = 0
        for line in lines:
            toks = parse_tokens_from_line(line)
            if not toks:
                continue
            if is_yolopose_format(toks):
                cls = int(float(toks[0]))
                xc = float(toks[1]); yc = float(toks[2]); bw = float(toks[3]); bh = float(toks[4])
                bbox_px = bbox_from_yolo_center(xc, yc, bw, bh, img_w, img_h)
                triplet_vals = toks[5:]
                n_trip = len(triplet_vals) // 3
                keypoints_px = []
                vis_list = []
                for i in range(n_trip):
                    px_n = float(triplet_vals[3*i + 0])
                    py_n = float(triplet_vals[3*i + 1])
                    vis_tok = triplet_vals[3*i + 2]
                    vis = safe_float_to_int_vis(vis_tok)
                    if abs(px_n) < EPS and abs(py_n) < EPS and vis == 0:
                        keypoints_px.append((None, None)); vis_list.append(0)
                    else:
                        keypoints_px.append((denorm_x(px_n, img_w), denorm_y(py_n, img_h))); vis_list.append(vis)
                draw_instance_on_image(img_local, bbox_px, keypoints_px, vis_list, cls, colors, img_w, img_h, point_radius=radius, thickness=thickness)
                instances_processed += 1
                continue
            if is_kp_only_format(toks):
                cls = int(float(toks[0])); triplet_vals = toks[1:]; n_trip = len(triplet_vals)//3
                kps_norm = []; keypoints_px = []; vis_list = []
                for i in range(n_trip):
                    px_n = float(triplet_vals[3*i + 0]); py_n = float(triplet_vals[3*i + 1]); vis_tok = triplet_vals[3*i + 2]
                    vis = safe_float_to_int_vis(vis_tok)
                    if abs(px_n) < EPS and abs(py_n) < EPS and vis == 0:
                        kps_norm.append((None,None)); keypoints_px.append((None,None)); vis_list.append(0)
                    else:
                        kps_norm.append((px_n,py_n)); keypoints_px.append((denorm_x(px_n, img_w), denorm_y(py_n, img_h))); vis_list.append(vis)
                bbox_px = bbox_from_kps_normalized([p for p in kps_norm if p[0] is not None], img_w, img_h) if any(p[0] is not None for p in kps_norm) else (0,0,img_w-1,img_h-1)
                draw_instance_on_image(img_local, bbox_px, keypoints_px, vis_list, cls, colors, img_w, img_h, point_radius=radius, thickness=thickness)
                instances_processed += 1
                continue
            print(f"Warning: skipping unknown-format line: {line.strip()[:120]}")
        return img_local, instances_processed

    img, n = process_label_file_with_style(lab_path, img_w, img_h, colors, radius=args.radius, thickness=args.thickness)

    if args.save:
        saved = cv2.imwrite(args.save, img)
        if saved:
            print(f"Saved visualization to {args.save}")
        else:
            print(f"Failed to save {args.save}")

    if args.show:
        winname = f"KP visualization: {os.path.basename(lab_path)}"
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.imshow(winname, img)
        print("Press any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not args.show and not args.save:
        # nothing done? but script will have created image but not shown/saved
        print("No --save provided and --no-show set; nothing was displayed or written.")

    print(f"Processed {n} instance(s) from {lab_path}")

if __name__ == "__main__":
    main()
