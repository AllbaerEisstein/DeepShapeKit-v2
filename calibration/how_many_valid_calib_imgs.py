import numpy as np
import cv2 as cv
import glob
import os
import argparse
import json


def main():    
    imgpoints_data = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("indir1")
    parser.add_argument("indir2")
    parser.add_argument("indir3")
    args = parser.parse_args()
    indirs = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir1),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir2),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir3)
            ]
            

    for indir in indirs:
    # Load all imgpoints JSON files
        for file in glob.glob(os.path.join(indir, "*imgpoints_processed.json")):
            with open(file, "r") as f:
                imgpoints_data.append(json.load(f))
    
    valid_imgnames = set([
        filename for filename in (list(imgpoints_data[0]["imgpoints"].keys()) + list(imgpoints_data[1]["imgpoints"].keys()) + list(imgpoints_data[2]["imgpoints"].keys()))
        if ((filename in imgpoints_data[0]["imgpoints"].keys()) and (filename in imgpoints_data[1]["imgpoints"].keys())) and (filename in imgpoints_data[2]["imgpoints"].keys())
    ])

    imgpoints1  = dict(sorted({
        filename: imgpoint for filename, imgpoint in imgpoints_data[0]["imgpoints"].items()
        if filename in valid_imgnames
    }.items()))
    imgpoints1 = np.array([imgpoint for filename, imgpoint in imgpoints1.items()], dtype=np.float32)

    print(f"\nnumber of images where all points have been detected in all three views: {len(imgpoints1)}\n")

if __name__ == "__main__":
    main()