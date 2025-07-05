#!/miniforge3/envs/yolo/bin/python
import sys
import json
import argparse
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import PIL.Image

import infer_mask  


def encode_mask(mask: np.ndarray) -> str:
    """Turn a binary mask into base64-encoded PNG."""
    img = PIL.Image.fromarray((np.array(mask) * 255).astype('uint8'))
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def run_inference(model_path: str, input_path: str, is_dir=False):
    model = infer_mask.load_model(Path(model_path))

    out = {}
    if is_dir:
        img_path2result = {}
        for img in Path(input_path).iterdir():
            img_path2result[str(img)] = {
                "bboxes": [],
                "masks_xy": [],
                "confs": []
            }
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                bboxes, masks_xy, confs = confs = infer_mask.inference(model, img)
                img_path2result[str(img)]["bboxes"]   = bboxes
                img_path2result[str(img)]["masks_xy"] = masks_xy
                img_path2result[str(img)]["confs"]    = confs
        out = img_path2result
    else:
        bboxes, masks_xy, confs = infer_mask.inference(model, Path(input_path))
        out = {
            "bboxes": bboxes,
            "masks_xy": masks_xy,
            "confs": confs,
        }

    print(json.dumps(out))


def run_img2bbx(input_path: str, bboxes, out_dir: str, padding: int, is_dir=False):
    csv_return = []
    if is_dir:
        frame_counter = 0
        folder_name = Path(input_path).name
        for img in Path(input_path).iterdir():
            if str(img) in bboxes and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                for det_idx, bbox in enumerate(bboxes[str(img)]):
                    out_fname = f"image_{frame_counter}_{det_idx}_bbox-masked.png"
                    infer_mask.img2bbx(
                        img_path=img,
                        bbox=bbox,
                        out_dir=Path(out_dir),
                        padding=padding,
                        out_filename=out_fname
                    )
                    rel_path = f"{folder_name}/bbox-masked_image/{out_fname}"
                    csv_return.append([
                        frame_counter,
                        rel_path,
                        'bbox-masked',
                        det_idx,
                        folder_name,
                        bbox
                    ])
            frame_counter += 1
    else:
        infer_mask.img2bbx(
            img_path=Path(input_path),
            bbox=bboxes,
            out_dir=Path(out_dir),
            padding=padding
        )
    print(json.dumps(csv_return))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    inf = sub.add_parser("inference")
    inf.add_argument("--model", required=True)
    inf.add_argument("--input", required=True)
    inf.add_argument("--isdir", default=False, type=bool)

    bbx = sub.add_parser("img2bbx")
    bbx.add_argument("--input", required=True)
    bbx.add_argument("--bboxes", required=True,
                     help="JSON list of [x1,y1,x2,y2] boxes")
    bbx.add_argument("--outdir", required=True)
    bbx.add_argument("--pad", type=int, default=0)
    bbx.add_argument("--isdir", default=False, type=bool)

    args = p.parse_args()

    if args.mode == "inference":
        run_inference(args.model, args.input, args.isdir)
    else:
        # parse bboxes JSON
        bboxes = json.loads(args.bboxes)
        run_img2bbx(args.input, bboxes, args.outdir, args.pad)


if __name__ == "__main__":
    main()
