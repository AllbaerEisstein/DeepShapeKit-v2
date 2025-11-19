import argparse
import glob
import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def rvec_tvec_to_board_pose(rvec, tvec):
    """
    Given a rotation vector and translation vector from solvePnP (which maps board points
    to camera coordinates as: X_cam = R * X_board + t), compute the inverse transformation
    so that we obtain the board pose relative to the camera.
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(np.array(rvec))
    t = np.array(tvec).reshape(3, 1)
    # Invert the transformation: 
    # if X_cam = R * X_board + t, then X_board = R^T * (X_cam - t)
    R_inv = R.T
    t_inv = -R_inv @ t
    # Build homogeneous transformation matrix T_board (in camera coordinates)
    T = np.eye(4)
    T[:3, :3] = R_inv
    T[:3, 3] = t_inv.flatten()
    return T

def create_checkerboard_corners(num_cols, num_rows, square_size):
    """
    Create a rectangle (4 corner points) spanning the area from the first to the last corner.
    In many calibration setups, the board is defined on the z=0 plane with the first corner at (0,0,0)
    and the last at ((num_cols-1)*square_size, (num_rows-1)*square_size, 0).
    """
    w = (num_cols - 1) * square_size
    h = (num_rows - 1) * square_size
    # Define the four corners of the rectangle in board coordinate system
    pts = np.array([
        [0, 0, 0],
        [w, 0, 0],
        [w, h, 0],
        [0, h, 0]
    ], dtype=np.float32)
    return pts

def transform_points(T, points):
    """
    Apply a 4x4 homogeneous transformation T to an array of 3D points.
    """
    n = points.shape[0]
    hom_pts = np.hstack((points, np.ones((n, 1))))
    pts_transformed = (T @ hom_pts.T).T
    return pts_transformed[:, :3]

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct and plot multi-camera checkerboard poses in 3D."
    )
    parser.add_argument("directory", help="Directory containing *camera_parameters.json files")
    parser.add_argument("--num_cols", type=int, default=9,
                        help="Number of checkerboard corners in horizontal direction (default: 9)")
    parser.add_argument("--num_rows", type=int, default=6,
                        help="Number of checkerboard corners in vertical direction (default: 6)")
    parser.add_argument("--square_size", type=float, default=1.0,
                        help="Size of one square (units, default: 1.0)")
    args = parser.parse_args()

    # Find all JSON files that match the pattern.
    pattern = os.path.join(args.directory, "*calibration_parameters.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No camera parameter files found in", args.directory)
        return

    # Dictionary to store each camera's data.
    # Each entry will contain a list of board poses (4x4 matrices) computed from the (rvec, tvec) samples.
    camera_data = {}
    for cam_idx, file in enumerate(files):
        with open(file, 'r') as f:
            data = json.load(f)
        rvecs = data["rvecs"]
        tvecs = data["tvecs"]
        # Sample every 20th pose
        sampled_indices = list(range(0, len(rvecs), 20))
        poses = []
        for i in sampled_indices:
            rvec = rvecs[i]
            tvec = tvecs[i]
            # Invert the transformation to get board pose relative to the camera.
            T_board = rvec_tvec_to_board_pose(rvec, tvec)
            poses.append(T_board)
        camera_data[cam_idx] = {"file": file, "poses": poses}

    # Use the first camera (camera 0) as the reference coordinate frame.
    # Its transformation to itself is identity.
    ref_camera = camera_data[0]
    ref_T = np.eye(4)
    camera_data[0]["T_to_ref"] = ref_T

    # For every other camera, compute the transformation to the reference frame.
    # Here we use the first sample's board pose: 
    # T_cam_to_ref = T_board_ref_first * inv(T_board_cam_first)
    T_board_ref_first = ref_camera["poses"][0]
    for cam_idx in camera_data:
        if cam_idx == 0:
            continue
        T_board_cam_first = camera_data[cam_idx]["poses"][0]
        T_to_ref = T_board_ref_first @ np.linalg.inv(T_board_cam_first)
        camera_data[cam_idx]["T_to_ref"] = T_to_ref

    # Create the checkerboard rectangle in board coordinates.
    board_corners = create_checkerboard_corners(args.num_cols, args.num_rows, args.square_size)

    # Set up a 3D plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'm', 'c', 'y']  # cycle colors for multiple cameras

    # For each camera, plot its center and the board poses (rectangles).
    for cam_idx in camera_data:
        color = colors[cam_idx % len(colors)]
        T_to_ref = camera_data[cam_idx]["T_to_ref"]
        # The camera center in its own coordinate frame is at (0,0,0,1).
        cam_center = T_to_ref @ np.array([0, 0, 0, 1])
        ax.scatter(cam_center[0], cam_center[1], cam_center[2],
                   c=color, marker='o', s=100, label=f'Camera {cam_idx}')
        # Plot each board (checkerboard) pose.
        for T_board in camera_data[cam_idx]["poses"]:
            # Transform the board pose to the reference (camera 1) coordinate frame.
            T_board_ref = T_to_ref @ T_board
            # Transform the rectangle corners from board coordinates into world coordinates.
            corners_world = transform_points(T_board_ref, board_corners)
            # Close the rectangle for plotting
            corners_plot = np.vstack([corners_world, corners_world[0]])
            ax.plot(corners_plot[:, 0], corners_plot[:, 1], corners_plot[:, 2],
                    c=color, alpha=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Camera positions and checkerboard poses in Camera 1's frame")
    plt.show()

if __name__ == "__main__":
    main()
