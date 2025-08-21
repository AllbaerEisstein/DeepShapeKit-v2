"""
Convert keypoint-only label lines or "almost-YOLO" lines into strict YOLO-pose format
(in-place). Rules:

Input keypoint-only form:
  <class> <px1> <py1> <v1> <px2> <py2> <v2> ...
(converted ->)
YOLO-pose form:
  <class> <x_center> <y_center> <width> <height> <px1> <py1> <v1> <px2> <py2> <v2> ...

Notes about visibility (v):
- Output visibility tokens MUST be integers 0, 1 or 2 (no decimal points).
- If input v is float but equal to 0,1 or 2 (within 1e-6), it will be trimmed to the integer.
- Otherwise, on conversion from arbitrary confidences we map:
    v <= 0      -> 0
    0 < v <= 0.5 -> 1
    v  > 0.5    -> 2
  (This is a deterministic mapping; change thresholds if you'd prefer different behavior.)

The script also will "fix" lines already looking like YOLO-pose but having float visibilities like 2.00000 -> 2.
"""

import os
import argparse
import glob

EPS = 1e-6

def parse_line_tokens(line):
    s = line.strip()
    if not s:
        return []
    if s.startswith("#"):
        return []
    return s.split()

def is_input_kp_only(tokens):
    """Return True if tokens look like: 1 + 3*n (class + triplets px,py,vis)."""
    if len(tokens) < 4:
        return False
    try:
        float(tokens[0])
    except ValueError:
        return False
    return (len(tokens) - 1) % 3 == 0

def is_yolopose_like(tokens):
    """Return True if tokens look like: 5 + 3*n (class + bbox(4) + triplets)."""
    if len(tokens) < 5:
        return False
    try:
        # check first five numeric
        for i in range(5):
            float(tokens[i])
    except ValueError:
        return False
    return (len(tokens) - 5) % 3 == 0

def float_is_integer_like(v):
    """Return integer value if v is within EPS of integer and that integer in {0,1,2}, else None."""
    try:
        fv = float(v)
    except Exception:
        return None
    ri = int(round(fv))
    if abs(fv - ri) <= EPS and ri in (0,1,2):
        return ri
    return None

def map_conf_to_vis(v):
    """Map an arbitrary float confidence to discrete vis {0,1,2}."""
    try:
        fv = float(v)
    except Exception:
        return 0
    # If it's already integer-like (0/1/2) return that
    int_like = float_is_integer_like(fv)
    if int_like is not None:
        return int_like
    # Otherwise map by threshold:
    if fv <= 0.0:
        return 0
    else:
        return 2

def normalize_number_str(x):
    """Format numeric x to 6 decimals (for px/py/bbox), but keep ints for visibility handled separately."""
    return f"{float(x):.6f}"

def convert_input_kp_only_to_yolopose(tokens):
    """
    tokens: [cls, px1,py1,v1, px2,py2,v2, ...] where px/py in normalized coords and v is float/conf.
    Returns new token list in strict YOLO-pose: [cls, x_ctr, y_ctr, w, h, px1,py1,v1_int, ...]
    """
    cls_tok = tokens[0]
    kp_vals = tokens[1:]
    n_triplets = len(kp_vals) // 3

    triples = []
    for i in range(n_triplets):
        px_s = kp_vals[3*i + 0]
        py_s = kp_vals[3*i + 1]
        v_s  = kp_vals[3*i + 2]
        try:
            px = float(px_s)
            py = float(py_s)
            v  = map_conf_to_vis(v_s)
        except Exception:
            px, py, v = 0.0, 0.0, 0
        triples.append((px, py, v))

    present_pts = [(px,py) for (px,py,v) in triples if v > 0]

    if present_pts:
        xs = [p[0] for p in present_pts]
        ys = [p[1] for p in present_pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        x_ctr = (xmin + xmax) / 2.0
        y_ctr = (ymin + ymax) / 2.0
        w_box = (xmax - xmin)
        h_box = (ymax - ymin)
        # avoid degenerate zero-size boxes
        if w_box <= 0:
            w_box = 1e-6
        if h_box <= 0:
            h_box = 1e-6
    else:
        # fallback full frame
        x_ctr, y_ctr, w_box, h_box = 0.5, 0.5, 1.0, 1.0

    out = [cls_tok,
           f"{x_ctr:.6f}", f"{y_ctr:.6f}", f"{w_box:.6f}", f"{h_box:.6f}"]

    for px, py, v in triples:
        out.append(f"{px:.6f}")
        out.append(f"{py:.6f}")
        vis_int = map_conf_to_vis(v)
        out.append(str(vis_int))

    return out

def fix_yolopose_visibilities(tokens):
    """
    Given tokens that are already yolopose-like (5 + 3n), ensure visibility tokens are integers 0/1/2.
    Return modified token list (may be identical if no changes).
    """
    out = tokens[:5]  # class + bbox
    n_triplets = (len(tokens) - 5) // 3
    changed = False
    for i in range(n_triplets):
        px_s = tokens[5 + 3*i + 0]
        py_s = tokens[5 + 3*i + 1]
        vis_s = tokens[5 + 3*i + 2]

        # Format px/py to six decimals
        try:
            px_f = float(px_s)
            py_f = float(py_s)
        except Exception:
            px_f, py_f = 0.0, 0.0

        # normalize visibility:
        vis_int_like = float_is_integer_like(vis_s)
        if vis_int_like is not None:
            vis_out = str(vis_int_like)
            if vis_s != vis_out:
                changed = True
        else:
            # not exactly integer-like: map using thresholds
            vis_out = str(map_conf_to_vis(vis_s))
            if vis_s != vis_out:
                changed = True

        out.append(f"{px_f:.6f}")
        out.append(f"{py_f:.6f}")
        out.append(vis_out)
    return out, changed

def process_file(path, recalc=False):
    """
    Read file, convert applicable lines (input kp-only -> YOLO-pose) and fix visibilities
    for already-yolopose-like lines. Overwrite file if any changes occurred.
    Returns number of lines changed.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    out_lines = []
    changed_count = 0
    recalculated_count = 0

    for raw in raw_lines:
        toks = parse_line_tokens(raw)
        if not toks:
            # preserve blank/comments exactly
            out_lines.append(raw.rstrip("\n"))
            continue

        if is_input_kp_only(toks):
            try:
                new_toks = convert_input_kp_only_to_yolopose(toks)
                out_lines.append(" ".join(new_toks))
                changed_count += 1
            except Exception as e:
                print(f"⚠️  Failed to convert line in {path}: {e}")
                out_lines.append(raw.rstrip("\n"))
            continue

        if recalc and is_yolopose_like(toks):
            try:
                kpt_toks = [toks[0]]+toks[5:]  # skip bbox
                toks = convert_input_kp_only_to_yolopose(kpt_toks)
                out_lines.append(" ".join(toks))
                recalculated_count += 1
            except Exception as e:
                print(f"⚠️  Failed to fix YOLO-pose line in {path}: {e}")
                out_lines.append(raw.rstrip("\n"))
            continue

        if is_yolopose_like(toks):
            new_toks, changed = fix_yolopose_visibilities(toks)
            out_lines.append(" ".join(new_toks))
            if changed:
                changed_count += 1
            continue

        # Unknown format: keep as-is
        out_lines.append(raw.rstrip("\n"))

    if changed_count > 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
    return changed_count, recalculated_count

def find_txt_files(indir, recursive=False):
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(glob.glob(os.path.join(indir, pattern), recursive=recursive))

def main():
    p = argparse.ArgumentParser(description="Convert keypoint-only label lines -> strict YOLO-pose (in-place).")
    p.add_argument("--indir", required=True, help="directory containing .txt label files (will scan recursively)")
    p.add_argument("--recursive", dest="recursive", default=False, help="also scan subdirectories")
    p.add_argument("--recalculate", dest="recalc", default=False, help="recalculate keypoint bounding boxes and centers")
    args = p.parse_args()

    indir = args.indir
    recursive = args.recursive
    recalc = args.recalc

    if not os.path.isdir(indir):
        print(f"❌ '{indir}' is not a directory.")
        return

    files = find_txt_files(indir, recursive=recursive)
    if not files:
        print("⚠️  No .txt files found.")
        return

    total_files_modified = 0
    total_format_corrected = 0
    total_bbox_recalculated = 0
    for fn in files:
        changed, recalculated = process_file(fn, recalc=recalc)
        if recalculated > 0 or changed > 0:
            total_files_modified += 1
        if changed > 0:
            print(f"✅ Modified {changed} line(s) in {fn}")
            total_format_corrected += changed
        if recalculated > 0:
            print(f"✅ Recalculated bbox for {recalculated} line(s) in {fn}")
            total_bbox_recalculated += recalculated

        else:
            print(f"✅ All labels in {fn} are already in strict YOLO-pose format.")

    print(f"\nDone. Files modified: {total_files_modified}.\nLines with wrong format corrected: {total_format_corrected}." + f"\nBounding Boxes recalculated: {total_bbox_recalculated}." if recalc else "")

if __name__ == "__main__":
    import argparse
    main()