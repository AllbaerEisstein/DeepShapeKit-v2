import numpy as np
import cv2 as cv
import glob
import os
import argparse
import json
import re


complete_corner_mapping = {
    "AB": ["A","B","D","C"],
    "BC": ["B","C","A","D"],
    "CD": ["C","D","B","A"],
    "DA": ["D","A","C","B"],
    "AD": ["A","D","B","C"],
    "DC": ["D","C","A","B"],
    "CB": ["C","B","D","A"],
    "BA": ["B","A","C","D"],
}

clockwise_rotation = ["AD","DC","CB","BA"]

default_corner_order = "BA"


"""
A matrix rotation of 90 degree is a transpose with reordering.
Elements that were in a row have to be into a column after rotation, hence transpose.
If we then reverse the order within the rows (i.e. we push around columns) we get each former row (now column) to it's desired rotated position.
"""
def rotate_90_clockwise(m, iterations=1):
    for i in range(iterations):
        m = np.flip(m.transpose([1,0,2]), 1)
    return np.flip(m.transpose([1,0,2]), 1)
    #return m.transpose([1,0,2])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    args = parser.parse_args()
    indir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir)
    
    perspective_name = ""
    video_perspective = None
    if re.match(r".*/a2/.*", indir):
        video_perspective = 0
        perspective_name = "a2_"
    if re.match(r".*/b2/.*", indir):
        video_perspective = 1
        perspective_name = "b2_"
    if re.match(r".*/c2/.*", indir):
        video_perspective = 2
        perspective_name = "c2_"
        
    print(f"\nvideo perspective: {video_perspective}\n")
    
    objpoints = {}
    objpoints_list = []
    imgpoints_list = []
    filename2index = {}
    
    imgpoints_manual = []
    for file in glob.glob(os.path.join(indir, "*labeled_points.json")):
        with open(file, "r") as f:
            imgpoints_manual.append(json.load(f))

    with open(glob.glob(os.path.join(indir, "imgpoints.json"))[0], "r") as f:
        imgpoints = json.load(f)
    imgpoints = {imgname: [point[0] for point in points] for imgname, points in imgpoints.items()}

    with open(glob.glob("three-perspective_videos/calib-corner-mapping.json")[0], "r") as f:
        corner_mapping = json.load(f)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((12*12,3), np.float32)
    objp[:,:2] = np.mgrid[0:12,0:12].T.reshape(-1,2)
    
    for el in imgpoints_manual:
        imgname = list(el)[0]
        imgpoints[imgname] = [point for point in el[imgname]]
    imgpoints = {imgname:np.array(imgpoints[imgname]) for imgname in sorted(imgpoints.keys())}
    #print(imgpoints)
    for imgname, points in imgpoints.items(): print(f"{imgname}: [{points[0]}, ...]")
    
    """
    Problem: We detected each point in each image. 
    However, the points are ordered differently in some images.
    We want to be able to establish a correspondence between points via order.
    So, in the following code, point ordering is unified for each image.
    """
    for i, (imgname, points) in enumerate(imgpoints.items()):
        imgprefix = imgname[:2]
        print(f"\n\n{imgname}\n")
        if video_perspective is not None:
            #objp2 = np.copy(objp).reshape(12,12,3)
            imgpt2 = np.copy(points).reshape(12,12,2)
            this_corner_order = corner_mapping[imgprefix][video_perspective]
            print(f"corner order before: {this_corner_order}\n")
            """ 
            if clockwise corner-assignment: reverse rows 
            """
            if this_corner_order not in clockwise_rotation: 
                #print(f"{objp2[0]}\n{objp2[1]}\n ... ... ")
                #objp2 = np.flip(objp2, 1)
                imgpt2 = np.flip(imgpt2, 1)
                #print(f"{objp2[0]}\n{objp2[1]}\n ... ... ")
                this_corner_order = this_corner_order[::-1]
                print(f"corner order clockwise: {this_corner_order}\n")
            """ 
            rotate so that every detected point corresponds to the correct point in each other image 
            """
            while this_corner_order != default_corner_order:
                index = clockwise_rotation.index(this_corner_order)
                #objp2 = rotate_90_clockwise(objp2)
                imgpt2 = rotate_90_clockwise(imgpt2, iterations=3)
                #print(f"{objp2[0]}\n{objp2[1]}\n ... ... ")
                index = (index+1) % len(clockwise_rotation)
                this_corner_order = clockwise_rotation[index]
                print(f"corner order rotated: {this_corner_order}\n")
            objpoints[imgname] = np.copy(objp).reshape(144,3)
            imgpoints[imgname] = imgpt2.reshape(144,2)
            #print(objp2.reshape(144,3))
            
            #objpoints_list.append(objpoints[imgname])
            objpoints_list.append(np.copy(objp).reshape(144,3))
            objpoints_list = [np.array(pts, dtype=np.float32) for pts in objpoints_list]
            imgpoints_list.append(imgpoints[imgname])
            imgpoints_list = [np.array(pts, dtype=np.float32) for pts in imgpoints_list]
            filename2index[imgname] = i

    objpoints_list = np.array(objpoints_list)
    imgpoints_list = np.array(imgpoints_list)
    #print(objpoints_list)
    #print(imgpoints_list)
    
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints_list, imgpoints_list, (2048,2048), None, None)

    # Convert NumPy arrays to lists for JSON serialization
    calibration_data = {
        "perspective_name": perspective_name,
        "ret": ret,
        "mtx": mtx.tolist(),
        "dist": dist.tolist(),
        "rvecs": [r.tolist() for r in rvecs],
        "tvecs": [t.tolist() for t in tvecs],
        "objpoints": {filename: points.tolist() for filename, points in objpoints.items()},
        "imgpoints": {filename: points.tolist() for filename, points in imgpoints.items()},
        "objpoints_list": [objp.tolist() for objp in objpoints_list],
        "imgpoints_list": [imgpt.tolist() for imgpt in imgpoints_list],
        "filename2index": filename2index
    }

    # Save to a JSON file
    with open(indir+perspective_name+"camera_calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=4)

    print(f"Calibration parameters saved to {indir+perspective_name}camera_calibration.json")


if __name__ == "__main__":
    main()
