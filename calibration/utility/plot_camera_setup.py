import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import argparse
import json
import glob

def transform_to_first_camera(rvecs, tvecs, rvecs_ref, tvecs_ref):
    """Transform a camera's pose into the first camera's coordinate system"""
    R, _ = cv.Rodrigues(rvecs)
    T = np.array(tvecs).reshape((3, 1))

    R_ref, _ = cv.Rodrigues(rvecs_ref)
    T_ref = np.array(tvecs_ref).reshape((3, 1))

    R_rel = R_ref.T @ R
    T_rel = R_ref.T @ (T - T_ref)

    return R_rel, T_rel

def plot_all_checkerboards(cameras):
    """Plots all cameras and checkerboards in the first camera's coordinate system"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use the first camera as the reference frame
    rvecs_ref = np.array(cameras[0]["rvecs"][0])
    tvecs_ref = np.array(cameras[0]["tvecs"][0])

    # Colors for different cameras
    colors = ['red', 'blue', 'green']
    checkerboard_colors = ['magenta', 'turquoise', 'yellow']
    
    for idx, camera in enumerate(cameras):
        rvecs = [np.array(r) for r in camera["rvecs"]]
        tvecs = [np.array(t) for t in camera["tvecs"]]
        objpoints_list = np.zeros((len(tvecs),3), np.float32)

        camera_color = colors[idx % len(colors)]  # Assign color per camera
        checkerboard_color = checkerboard_colors[idx % len(checkerboard_colors)]  # Assign color per camera

        if idx > 0:  # Transform cameras 2 and 3 to Camera 1's coordinate system
            R_cam, T_cam = transform_to_first_camera(rvecs[0], tvecs[0], rvecs_ref, tvecs_ref)  # Camera's own transformation
            cam_origin = T_cam.T[0]  # Camera’s new position
            R_cam_inv = R_cam.T  # Inverse rotation (to align with Camera 1)
        else:
            cam_origin = tvecs[0].T[0]  # Camera 1 stays in place
            R_cam_inv = np.eye(3)  # Identity matrix (no rotation needed)

        # Plot camera position
        ax.scatter(cam_origin[0], cam_origin[1], cam_origin[2], 
                   color=camera_color, marker='s', s=100, label=f'Camera {idx+1}')

        # Plot each checkerboard position for this camera
        for i in range(len(rvecs)):
            R_rel, T_rel = cv.Rodrigues(rvecs[i])[0], tvecs[i]  # Get rotation matrix and translation

            # **Transform checkerboard rotation into Camera 1's coordinate frame**
            R_new = R_cam_inv @ R_rel  # Rotate checkerboard with transformed camera rotation
            T_new = (R_cam_inv @ T_rel).T + cam_origin  # Transform translation as before

            # Transform checkerboard points
            obj_cam = (R_new @ objpoints_list[i].T).T + T_new  # Apply rotation and shift

            # Set transparency based on checkerboard index
            alpha = 0.2 + 0.8 * (i / len(rvecs))

            ax.scatter(obj_cam[:, 0], obj_cam[:, 1], obj_cam[:, 2], 
                       color=checkerboard_color, alpha=alpha, 
                       label=f'Checkerboard Cam {idx+1}' if i == 0 else None)



    # Labels and legend
    ax.set_xlabel("X (camera 1)")
    ax.set_ylabel("Y (camera 1)")
    ax.set_zlabel("Z (camera 1)")
    ax.set_title("All Cameras & Checkerboard Positions in Camera 1 Frame")
    ax.legend()

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    args = parser.parse_args()
    indir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.indir)

    calibration_data = []

    # Load all camera calibration JSON files
    for file in glob.glob(os.path.join(indir, "*calibration_parameters.json")):
        with open(file, "r") as f:
            calibration_data.append(json.load(f))

    plot_all_checkerboards(calibration_data)

if __name__ == "__main__":
    main()

