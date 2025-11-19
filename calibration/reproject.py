import numpy as np
import cv2 as cv
import glob
import os
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    args = parser.parse_args()
    indir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir)

    # Load calibration data
    calib_file = glob.glob(os.path.join(indir,"annotated/*camera_calibration.json"))[0]
    with open(calib_file, "r") as f:
        calibration_data = json.load(f)

    mtx = np.array(calibration_data["mtx"])
    dist = np.array(calibration_data["dist"])
    rvecs = [np.array(r) for r in calibration_data["rvecs"]]
    tvecs = [np.array(t) for t in calibration_data["tvecs"]]
    objpoints = {filename: np.array(objp) for filename, objp in calibration_data["objpoints"].items()}
    filename2index = calibration_data["filename2index"]

    # Visualize the reprojected points
    images = glob.glob(os.path.join(indir,"*.jpg")) + glob.glob(os.path.join(indir,"*.png")) # Load all images

    for i, fname in enumerate(images):
        img = cv.imread(fname)
        
        # Resize image to 50% of its original dimensions
        img_resized = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        
        fname = os.path.splitext(os.path.basename(fname))[0]
        imgpoints2, _ = cv.projectPoints(objpoints[fname], rvecs[filename2index[fname]], tvecs[filename2index[fname]], mtx, dist)
        
        for j, p in enumerate(imgpoints2):
            scaled_p = (int(p[0][0] / 2), int(p[0][1] / 2))  # Scale points accordingly
            cv.circle(img_resized, scaled_p, 5, (0, 0, 255), -1)  # Red points
            cv.putText(img_resized, str(j + 1), (scaled_p[0] + 5, scaled_p[1] - 5),
                cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA)

        cv.imshow(f"Reprojection Check {i+1}", img_resized)
        cv.waitKey(2000)

    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
