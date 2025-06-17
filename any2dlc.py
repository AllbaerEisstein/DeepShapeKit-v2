import os
import json
import glob
import argparse
import csv

def find_json_files(indir):
    return sorted(glob.glob(os.path.join(indir, '*.json')))

def collect_labels(json_files):
    labels = []
    for jf in json_files:
        data = json.load(open(jf, 'r'))
        for shape in data.get('shapes', []):
            if shape.get('shape_type') != 'point':
                continue
            label = shape.get('label')
            pts = shape.get('points', [])
            if not label or len(pts)==0:
                continue
            if label not in labels:
                labels.append(label)
    return labels

def extract_keypoints(json_file):
    data = json.load(open(json_file, 'r'))
    imgpath = data.get('imagePath')
    kp = {}
    for shape in data.get('shapes', []):
        if shape.get('shape_type') != 'point':
            continue
        label = shape.get('label')
        pts = shape.get('points', [])
        if not label or len(pts)==0:
            continue
        x,y = pts[0]
        kp[label] = (x, y)
    return imgpath, kp

def write_dlc_csv(outpath, records, labels, scorer, session):
    """
    records: list of tuples (imgpath, keypoint_dict)
    labels: ordered list of labels
    """
    n_kp = len(labels)
    # Build header rows
    # 1) scorer
    header1 = ['scorer', '', ''] + [scorer for _ in range(n_kp*2)]
    # 2) bodyparts
    header2 = ['bodyparts', '', ''] + [l for l in labels for _ in (0,1)]
    # 3) coords
    header3 = ['coords', '', ''] + ['x' if i%2==0 else 'y' for i in range(n_kp*2)]
    with open(outpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header1)
        w.writerow(header2)
        w.writerow(header3)
        # rows
        for imgpath, kpdict in records:
            row = ['labeled-data', session, imgpath]
            for lab in labels:
                if lab in kpdict:
                    x,y = kpdict[lab]
                    row += [f"{x:.6f}", f"{y:.6f}"]
                else:
                    row += ['', '']
            w.writerow(row)

def main():
    p = argparse.ArgumentParser(description="Convert AnyLabeling JSONs → DeepLabCut CSV")
    p.add_argument('--indir',   required=True,  help="Directory containing .json files")
    p.add_argument('--outdlc',  required=True,  help="Output DLC‐formatted CSV path")
    p.add_argument('--scorer',  default='user', help="Scorer name to put in CSV header")
    p.add_argument('--session', default=None,   help="Session name (default: basename of indir)")
    args = p.parse_args()

    indir = args.indir
    outcsv = args.outdlc
    scorer = args.scorer
    session = args.session or os.path.basename(os.path.abspath(indir))

    # 1) find all jsons
    json_files = find_json_files(indir)
    if not json_files:
        print(f"No .json files found in {indir}")
        return

    # 2) collect all labels across files
    labels = collect_labels(json_files)
    if not labels:
        print("No point‐type labels found in any JSON.")
        return

    # 3) extract per‐file keypoints
    records = []
    for jf in json_files:
        imgpath, kpdict = extract_keypoints(jf)
        # prefix the image path with the directory if needed
        records.append((imgpath, kpdict))

    # 4) write dlc csv
    write_dlc_csv(outcsv, records, labels, scorer, session)
    print(f"Written {len(records)} frames × {len(labels)} keypoints to {outcsv}")

if __name__ == '__main__':
    main()