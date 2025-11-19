import numpy as np
import cv2 as cv
import glob
import os
import argparse
import json


#change this if stereo calibration not good.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
stereocalibration_flags = cv.CALIB_FIX_INTRINSIC

def main():    
    calibration_data = []
    imgpoints_data = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("indir1")
    parser.add_argument("indir2")
    args = parser.parse_args()
    indirs = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir1),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir2)
            ]
            
    for indir in indirs:
    # Load all camera calibration JSON files
        for file in glob.glob(os.path.join(indir, "*calibration_parameters.json")):
            with open(file, "r") as f:
                calibration_data.append(json.load(f))

    for indir in indirs:
    # Load all imgpoints JSON files
        for file in glob.glob(os.path.join(indir, "*imgpoints_processed.json")):
            with open(file, "r") as f:
                imgpoints_data.append(json.load(f))
    
    valid_imgnames = set([
        filename for filename in (list(imgpoints_data[0]["imgpoints"].keys()) + list(imgpoints_data[1]["imgpoints"].keys()))
        if (filename in imgpoints_data[0]["imgpoints"].keys()) and (filename in imgpoints_data[1]["imgpoints"].keys())
    ])

    perspective_name1 = calibration_data[0]["perspective_name"]
    perspective_name2 = calibration_data[1]["perspective_name"]

    
    objpoints   = dict(sorted({
        filename: objpoint for filename, objpoint in imgpoints_data[0]["objpoints"].items()
        if filename in valid_imgnames
    }.items()))
    objpoints = np.array([objpoint for filename, objpoint in objpoints.items()], dtype=np.float32)

    imgpoints1  = dict(sorted({
        filename: imgpoint for filename, imgpoint in imgpoints_data[0]["imgpoints"].items()
        if filename in valid_imgnames
    }.items()))
    imgpoints1 = np.array([imgpoint for filename, imgpoint in imgpoints1.items()], dtype=np.float32)

    imgpoints2  = dict(sorted({
        filename: imgpoint for filename, imgpoint in imgpoints_data[1]["imgpoints"].items()
        if filename in valid_imgnames
    }.items()))
    imgpoints2 = np.array([imgpoint for filename, imgpoint in imgpoints2.items()], dtype=np.float32)

    print(f"\nnumber of images where all points have been detected in both views: {len(imgpoints1)}\n")

    mtx1        = np.array(calibration_data[0]["mtx"], dtype=np.float32)
    dist1       = np.array(calibration_data[0]["dist"], dtype=np.float32)
    mtx2        = np.array(calibration_data[1]["mtx"], dtype=np.float32)
    dist2       = np.array(calibration_data[1]["dist"], dtype=np.float32)
    
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx1, dist1,
    mtx2, dist2, (2048, 2048), criteria = criteria, flags = stereocalibration_flags)


    # Convert NumPy arrays to lists for JSON serialization
    stereo_calibration_data = {
        "perspective_name": perspective_name1+perspective_name2,
        "ret":              ret,
        "CM1":              CM1.tolist(),       # camera matrix 1
        "dist1":            dist1.tolist(),     # distortion coefficients 1
        "CM2":              CM2.tolist(),       # camera matrix 2
        "dist2":            dist2.tolist(),     # distortion coefficients 2
        "R":                R.tolist(),         # rotation matrix of perspective 2 relative to perspective 1
        "T":                T.tolist(),         # rotation matrix of perspective 1 relative to perspective 2
        "E":                E.tolist(),         # essential matrix
        "F":                F.tolist()          # fundamental matrix
    }

    # Save to a JSON file
    with open(perspective_name1+perspective_name2+"stereo_camera_calibration.json", "w") as f:
        json.dump(stereo_calibration_data, f, indent=4)

    print(f"Calibration parameters saved to {perspective_name1+perspective_name2}stereo_camera_calibration.json")


if __name__ == "__main__":
    main()
