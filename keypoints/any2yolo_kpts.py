import os
import sys
import json
import glob
import argparse

def find_jsons(indir):
    return sorted(glob.glob(os.path.join(indir, '*.json')))

def load_points(json_path):
    """Return (w,h, dict[label]->(x,y)) or None on failure."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not all(k in data for k in ("shapes","imageWidth","imageHeight")):
        return None
    w = data["imageWidth"]
    h = data["imageHeight"]
    pts = {}
    for s in data["shapes"]:
        if s.get("shape_type")!="point":
            continue
        lab = s.get("label")
        coords = s.get("points",[])
        if lab and coords:
            x,y = coords[0]
            pts[lab] = (x,y)
    return w,h,pts

def compute_bbox(pts):
    """pts: list of (x,y). Return xmin,ymin,xmax,ymax."""
    xs = [x for x,y in pts]
    ys = [y for x,y in pts]
    return min(xs), min(ys), max(xs), max(ys)

def make_yolo_line(class_idx, w, h, pts_dict, kp_list):
    """
    Returns the full line string for one image.
    Missing kps get (0,0,0). vis=1 if present.
    """
    # collect only present points for bbox
    present = list(pts_dict.values())
    if present:
        xmin,ymin,xmax,ymax = compute_bbox(present)
        x_ctr = (xmin + xmax) / 2.0 / w
        y_ctr = (ymin + ymax) / 2.0 / h
        bw    = (xmax - xmin) / w
        bh    = (ymax - ymin) / h
    else:
        # fallback to entire frame
        x_ctr,y_ctr,bw,bh = 0.5,0.5,1.0,1.0

    parts = [
        str(class_idx),
        f"{x_ctr:.6f}",
        f"{y_ctr:.6f}",
        f"{bw:.6f}",
        f"{bh:.6f}",
    ]
    # now keypoints
    for kp in kp_list:
        if kp in pts_dict:
            x,y = pts_dict[kp]
            parts += [
                f"{x / w:.6f}",
                f"{y / h:.6f}",
                "1"
            ]
        else:
            parts += ["0.000000","0.000000","0"]
    return " ".join(parts)

def main():
    p = argparse.ArgumentParser(
        description="Convert AnyLabeling keypoint JSONs → YOLOv8-pose txts"
    )
    p.add_argument('--indir',     required=True,
                   help="directory of .json files")
    p.add_argument('--outdir',    default=None,
                   help="where to write .txt files (default: same as --indir)")
    p.add_argument('--class-idx', type=int, default=0,
                   help="integer class index (default 0)")
    p.add_argument('--kps',       nargs='+', required=True,
                   help="ordered list of keypoint names to include")
    args = p.parse_args()

    indir  = args.indir
    outdir = args.outdir or indir
    kps    = args.kps
    cls    = args.class_idx

    if not os.path.isdir(indir):
        print(f"❌ '{indir}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(outdir, exist_ok=True)

    jsons = find_jsons(indir)
    if not jsons:
        print("⚠️  No .json files found.", file=sys.stderr)
        sys.exit(0)

    for js in jsons:
        res = load_points(js)
        if res is None:
            print(f"Skipping {os.path.basename(js)} (missing fields)", file=sys.stderr)
            continue
        w,h,pts = res
        line = make_yolo_line(cls, w, h, pts, kps)
        txt_name = os.path.splitext(os.path.basename(js))[0] + ".txt"
        with open(os.path.join(outdir, txt_name), 'w') as fo:
            fo.write(line + "\n")
        print(f"✅ {os.path.basename(js)} → {txt_name}")

if __name__=="__main__":
    main()
