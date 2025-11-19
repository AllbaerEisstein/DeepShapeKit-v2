import numpy as np
import cv2 as cv
import glob
import os
import argparse
import json
import process_imgpoints

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")
    
    # Add arguments
    parser.add_argument("indir",
                        help="input directory with calibration images")
    parser.add_argument("--incr_contr", type=float, default=-1.0,
                        help="contrast value between 1.0 and 3.0")
    parser.add_argument("--cbwidth", type=int, default=12,
                        help="checkerboard width")
    parser.add_argument("--cbheight", type=int, default=12,
                        help="checkerboard height")

    # Parse the arguments
    args = parser.parse_args()
    
    indir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir)
    cbheight = args.cbheight
    cbwidth = args.cbwidth

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbwidth*cbheight,3), np.float32)
    objp[:,:2] = np.mgrid[0:cbheight,0:cbwidth].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = {} # 3d point in real world space
    imgpoints = {} # 2d points in image plane.

    images = glob.glob(indir+'*.png') + glob.glob(indir+'*.jpg')
    print(images)

    i = 0
    for fname in images:
        filename = os.path.splitext(os.path.basename(fname))[0]
        print(os.path.basename(fname))
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        if args.incr_contr != -1.0:
            # Increase contrast
            alpha = 3.0
            gray = cv.convertScaleAbs(gray, alpha=alpha)
            #for y in range(gray.shape[0]):
             #   for x in range(gray.shape[1]):
              #      gray[y,x] = np.clip(alpha*gray[y,x], 0, 255)

        # Find the chess board corners
        ret, corners = cv.findChessboardCornersSB(gray, (cbheight,cbwidth), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints[filename] = objp.tolist()

            corners2 = cv.cornerSubPix(gray,corners, (cbheight-1,cbwidth-1), (-1,-1), criteria)
            print("success\n")
            imgpoints[filename] = corners2.tolist()
            imgpoints[filename] = [point[0] for point in imgpoints[filename]] # remove one unnecessary dimension in the array (points were unnecessarily stored as [[x,y]] instead of [x,y])
            
            # Extract four key corners
            first = imgpoints[filename][0]
            second = imgpoints[filename][cbheight - 1]
            third = imgpoints[filename][cbheight * cbwidth - cbheight]
            forth = imgpoints[filename][cbheight * cbwidth - 1]
            key_corners = np.array([first, second, third, forth])
            
            facing_up = process_imgpoints.identify_upwards_facing_edge(key_corners)

            # Extract the two points of the upwards-facing edge
            p1, p2 = map(tuple, map(np.int32, facing_up))  # Ensure points are integer tuples

            # Draw the red line for the facing-up edge
            cv.line(img, p1, p2, (0, 255, 0), 10)  # Red color (BGR: 0,0,255), thickness: 2

            # Draw and display the chessboard corners
            first_corner = (int(corners2[0][0][0]), int(corners2[0][0][1]))
            last_corner = (int(corners2[-1][0][0]), int(corners2[-1][0][1]))
            cv.putText(img, '0', first_corner, cv.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 8, cv.LINE_AA)  # First corner
            cv.putText(img, str(len(corners2) - 1), last_corner, cv.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 8, cv.LINE_AA)  # Last corner
            cv.drawChessboardCorners(img, (cbheight, cbwidth), corners2, ret)
            cv.imwrite(indir + "annotated/" + os.path.basename(fname), img)


    with open(indir+"annotated/objpoints.json", 'w') as outfile:
            json.dump(objpoints, outfile)
    with open(indir+"annotated/imgpoints.json", 'w') as outfile2:
            json.dump(imgpoints, outfile2)

if __name__ == "__main__":
    main()
