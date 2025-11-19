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
    "BA": ["B","A","C","D"]
}

clockwise_rotation = ["AB","BC","CD","DA"]

default_corner_order = "BC"


"""
Identify the upwards facing side based on two criteria:
    1) It has to include the top-most corner
    2) It has to be close to horizontal
"""
def identify_upwards_facing_edge(key_corners):
    neighbours_by_index = {
        0: [key_corners[1],key_corners[2]],
        1: [key_corners[0],key_corners[3]],
        2: [key_corners[0],key_corners[3]],
        3: [key_corners[1],key_corners[2]]
    }

    # Sort by Y coordinate
    sorted_by_y = sorted(key_corners, key=lambda p: p[1])[::-1] # opencv y-coordinates are inverted
    top_most = sorted_by_y[2:4]  # Two highest corners
    top_most_corner = top_most[1]
    top_most_index = next(i for i, c in enumerate(key_corners) if np.array_equal(c, top_most_corner))

    # each edge that includes the top-most corner is a candidate for the upwards-facing side.
    facing_up_candidates = []
    if top_most[1][1] != top_most[0][1]: # if there is exactly one top most corner, i.e. the upper side is not horizontal:
        for neighbour in neighbours_by_index[top_most_index]:
            facing_up_candidates.append((top_most_corner, neighbour))
    else: # in the unlikely case that the top two corners share their y-coordinate
        facing_up_candidates.append((top_most[1], top_most[0]))
    
    def compute_angle(p1, p2):
        return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

    minDeviation = 90  # Start with the worst possible case
    facing_up = None
    for candidate in facing_up_candidates:
        angle = compute_angle(*candidate)
        deviation = min(abs(angle), abs(180 - abs(angle)))  # Ensure we measure closeness to horizontal
        if deviation < minDeviation:
            facing_up = candidate
            minDeviation = deviation

    return facing_up


"""
The (arbitrarily chosen) corner-naming for the checkerboard looks like this:
A         B
  |-|-|-|
  | | | |
  | | | |
D |-|-|-| C
  loopbio

Based on coordinates of the imagepoints and based on a user-given ground-truth about positions of corners in the image, a correspondence between the above real-world corners and imgpoints is recovered.
"""
def detect_corner_order(imgpoints, cbwidth, cbheight, upwards_facing_side):
    # Extract four key corners
    first = imgpoints[0]
    second = imgpoints[cbheight - 1]
    third = imgpoints[cbheight * cbwidth - cbheight]
    forth = imgpoints[cbheight * cbwidth - 1]
    key_corners = np.array([first, second, third, forth])
    
    facing_up = identify_upwards_facing_edge(key_corners)
    facing_up_left  = facing_up[0] if facing_up[0][0] < facing_up[1][0] else facing_up[1]
    facing_up_right = facing_up[0] if facing_up[0][0] > facing_up[1][0] else facing_up[1]
    
    # Now, we have a correspondence between the key-corners and real-world chessboard points
    # Identify the corresponding index
    index_of_facing_up_left_end = next(
        i for i, c in enumerate(key_corners) if np.all(np.isclose(c, facing_up_left))
    )
    index_of_facing_up_right_end = next(
        i for i, c in enumerate(key_corners) if np.all(np.isclose(c, facing_up_right))
    )

    print(f"facing up left corner index: {index_of_facing_up_left_end} ({facing_up_left})\nfacing up right corner index: {index_of_facing_up_right_end} ({facing_up_right})\n")

    # Find out which of the 8 corner orders is the order of the currently detected imagepoints based on where in the order the upwards pointing corners are:
    corner_order = None
    for short, order in complete_corner_mapping.items():
        if order[index_of_facing_up_left_end] == upwards_facing_side[0] and order[index_of_facing_up_right_end] == upwards_facing_side[1]:
            corner_order = short
            
    return corner_order
    

"""
A matrix rotation of 90 degree is a transpose with reordering.
Elements that were in a row have to be into a column after rotation, hence transpose.
If we then reverse the order within the rows (i.e. we push around columns) we get each former row (now column) to it's desired rotated position.
"""
def rotate_90_clockwise(m, iterations=1):
    for i in range(iterations):
        m = np.flip(m.transpose([1,0,2]), 0)
    return m
    #return m.transpose([1,0,2])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("--cbheight", type=int, default=8)
    parser.add_argument("--cbwidth", type=int, default=6)
    parser.add_argument("--upwards_facing_side", type=str, default=None)
    parser.add_argument("--source_images", type=str, default=None)
    parser.add_argument("--upwards_facing_side_file", default=None)
    args = parser.parse_args()
    height = args.cbheight
    width = args.cbwidth
    upwards_facing_side = args.upwards_facing_side
    source_images_dir = args.source_images
    indir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir)
    upwards_facing_side_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.upwards_facing_side_file) if args.upwards_facing_side_file is not None else None
    
    perspective_name = ""
    video_perspective = None
    if re.match(r".*/a\d*/.*", indir):
        video_perspective = 0
        perspective_name = "a_"+str(width)+"x"+str(height)+"_"
    if re.match(r".*/b\d*/.*", indir):
        video_perspective = 1
        perspective_name = "b_"+str(width)+"x"+str(height)+"_"
    if re.match(r".*/c\d*/.*", indir):
        video_perspective = 2
        perspective_name = "c_"+str(width)+"x"+str(height)+"_"
        
    print(f"\nvideo perspective: {video_perspective}\n")
    
    objpoints = {}
    objpoints_list = []
    imgpoints_list = []
    filename2index = {}
    
    filename_to_upside = None
    if upwards_facing_side_file is not None:
        with open(upwards_facing_side_file, "r") as f:
            filename_to_upside = json.load(f)

    def find_ground_truth_upside(num):
        for entry in filename_to_upside:
            if entry["start"] <= num <= entry["end"]:
                return entry["upside"]
        return None

    with open(glob.glob(os.path.join(indir, "imgpoints.json"))[0], "r") as f:
        imgpoints = json.load(f)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((height*width,3), np.float32)
    objp[:,:2] = np.mgrid[0:height,0:width].T.reshape(-1,2)
    
    imgpoints = {imgname:np.array(imgpoints[imgname]) for imgname in sorted(imgpoints.keys())}

    for imgname, points in imgpoints.items(): 
        print(f"{imgname}: [{points[0]}, ...]")
    
    if video_perspective is not None:
        for i, (imgname, points) in enumerate(imgpoints.items()):
            print(f"\n\n{imgname}\n")
            imgnum = int(imgname)
            ground_truth_upside = find_ground_truth_upside(imgnum) if filename_to_upside is not None else upwards_facing_side
            print(f"ground truth upside: {ground_truth_upside}\n")
            this_corner_order = detect_corner_order(points, cbheight=height, cbwidth=width, upwards_facing_side=ground_truth_upside)
            imgpt2 = np.copy(points).reshape(height,width,2)
            print(f"corner order before: {this_corner_order}\n")
            if this_corner_order is not None: 
                """ 
                if counter-clockwise corner-assignment: reverse rows 
                """
                if this_corner_order not in clockwise_rotation: 
                    imgpt2 = np.flip(imgpt2, 1)
                    this_corner_order = this_corner_order[::-1]
                    print(f"corner order clockwise: {this_corner_order}\n")
                """ 
                rotate imgpoints matrix so that every detected point corresponds to the correct point in each other image 
                """
                while this_corner_order != default_corner_order:
                    index = clockwise_rotation.index(this_corner_order)
                    imgpt2 = rotate_90_clockwise(imgpt2, iterations=1)
                    #print(f"{objp2[0]}\n{objp2[1]}\n ... ... ")
                    index = (index-1) % len(clockwise_rotation)
                    this_corner_order = clockwise_rotation[index]
                    print(f"corner order rotated: {this_corner_order}\n")
                    
                objpoints[imgname] = np.copy(objp).reshape(height*width,3)
                imgpoints[imgname] = imgpt2.reshape(height*width,2)

                objpoints_list.append(np.copy(objp).reshape(height*width,3))
                objpoints_list = [np.array(pts, dtype=np.float32) for pts in objpoints_list]
                imgpoints_list.append(imgpoints[imgname])
                imgpoints_list = [np.array(pts, dtype=np.float32) for pts in imgpoints_list]
                filename2index[imgname] = i
                
                if source_images_dir is not None and os.path.exists(os.path.join(indir,"refined/")):
                    fname = glob.glob(os.path.join(source_images_dir, imgname)+".jpg")[0]
                    img = cv.imread(fname)
                    first_corner = (int(imgpoints[imgname][0][0]), int(imgpoints[imgname][0][1]))
                    last_corner = (int(imgpoints[imgname][height*width-1][0]), int(imgpoints[imgname][height*width-1][1]))
                    cv.putText(img, '0', first_corner, cv.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 8, cv.LINE_AA)  # First corner
                    cv.putText(img, str(len(imgpoints[imgname])-1), last_corner, cv.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 8, cv.LINE_AA)  # Last corner
                    cv.drawChessboardCorners(img, (height,width), np.array(imgpoints[imgname], dtype=np.float32), True)
                    cv.imwrite(indir+"refined/"+os.path.basename(fname), img)

    print("\n+++++ imgpoints processing done +++++\n\n")
    
    objpoints_list = np.array(objpoints_list)
    imgpoints_list = np.array(imgpoints_list)
    #print(objpoints_list)
    #print(imgpoints_list)

    # Convert NumPy arrays to lists for JSON serialization
    calibration_data = {
        "perspective_name": perspective_name,
        "objpoints": {filename: points.tolist() for filename, points in objpoints.items()},
        "imgpoints": {filename: points.tolist() for filename, points in imgpoints.items()},
        "objpoints_list": [objp.tolist() for objp in objpoints_list],
        "imgpoints_list": [imgpt.tolist() for imgpt in imgpoints_list],
        "filename2index": filename2index
    }

    # Save to a JSON file
    outdir = source_images_dir if source_images_dir is not None else indir
    with open(outdir+perspective_name+"imgpoints_processed.json", "w") as f:
        json.dump(calibration_data, f, indent=4)

    print(f"Imagepoints and objectpoints saved to {outdir+perspective_name}imgpoints_processed.json")


if __name__ == "__main__":
    main()
