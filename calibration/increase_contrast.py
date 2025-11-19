import numpy as np
import cv2 as cv
import glob
import os
import argparse

def main():
    # Create the parser
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument("indir",
                        help="input directory with calibration images")

    # Parse the arguments
    args = parser.parse_args()
    
    indir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir)
    
    if os.path.exists(indir+"increased_contrast/"):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        images = glob.glob(indir+'*.png') + glob.glob(indir+'*.jpg')

        i = 0;
        for fname in images:
            print(os.path.basename(fname))
            img = cv.imread(fname)
            
            # Increase contrast
            alpha = 3.0
            img = cv.convertScaleAbs(img, alpha=alpha)
            cv.imwrite(indir+"increased_contrast/"+os.path.basename(fname), img)
            i = i + 1
            
    else:
	    print("output path doesn't exist")

if __name__ == "__main__":
    main()
