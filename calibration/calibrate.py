import numpy as np
import cv2 as cv
import os
import argparse
import json

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    args = parser.parse_args()
    infile_path = args.infile
    infile = os.path.join(os.path.dirname(os.path.realpath(__file__)), infile_path)
    
    imgpoints_processed = None
    with open(infile, "r") as f:
        imgpoints_processed = json.load(f)
    
    perspective_name = imgpoints_processed["perspective_name"]
    filename2index = imgpoints_processed["filename2index"]
    objpoints   = np.array(imgpoints_processed["objpoints_list"], dtype=np.float32)
    imgpoints   = np.array(imgpoints_processed["imgpoints_list"], dtype=np.float32)
    

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (2048,2048), None, None, criteria=criteria)

    # Convert NumPy arrays to lists for JSON serialization
    calibration_data = {
        "perspective_name": perspective_name,
        "ret": ret,
        "mtx": mtx.tolist(),
        "dist": dist.tolist(),
        "rvecs": [r.tolist() for r in rvecs],
        "tvecs": [t.tolist() for t in tvecs],
        "filename2index": filename2index
    }

    # Save to a JSON file
    outdir = os.path.dirname(infile)
    with open(os.path.join(outdir, perspective_name+"calibration_parameters.json"), "w") as f:
        json.dump(calibration_data, f, indent=4)

    print(f"Calibration parameters saved to {outdir+perspective_name}calibration_parameters.json")

if __name__ == "__main__":
    main()
