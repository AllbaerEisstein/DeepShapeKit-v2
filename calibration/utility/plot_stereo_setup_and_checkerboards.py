import argparse
import json
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

def rvec_tvec_to_board_pose(rvec, tvec):
    """
    Given a rotation vector and translation vector from solvePnP (which maps board points
    to camera coordinates as: X_cam = R * X_board + t), invert the transformation to get
    the board's pose in the camera coordinate system.
    """
    R, _ = cv2.Rodrigues(np.array(rvec))
    t = np.array(tvec).reshape(3, 1)
    # Invert the transformation: X_board = R^T * (X_cam - t)
    R_inv = R.T
    t_inv = -R_inv @ t
    T = np.eye(4)
    T[:3, :3] = R_inv
    T[:3, 3] = t_inv.flatten()
    return T

def create_checkerboard_corners(num_cols, num_rows, square_size):
    """
    Create a rectangle defined by the first and last checkerboard corners.
    Assumes the board lies in the z=0 plane and the first corner is at (0,0,0)
    and the last at ((num_cols-1)*square_size, (num_rows-1)*square_size, 0).
    """
    w = (num_cols - 1) * square_size
    h = (num_rows - 1) * square_size
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

def plot_camera(ax, center, R, label, color, arrow_length=10):
    """
    Plot a camera center and its viewing direction.
    The viewing direction is assumed to be the camera's optical axis (the z-axis).
    """
    ax.scatter(center[0], center[1], center[2], c=color, marker='o', s=100, label=label)
    # The viewing direction is given by R applied to [0,0,1].
    view_dir = R @ np.array([0, 0, 1])
    view_dir = view_dir / np.linalg.norm(view_dir)
    ax.quiver(center[0], center[1], center[2],
              view_dir[0], view_dir[1], view_dir[2],
              length=arrow_length, color=color)

def compute_epipole(F, which=1):
    """
    Compute an epipole from the fundamental matrix F.
    which==1: epipole in image 1 (right null space of F)
    which==2: epipole in image 2 (right null space of F^T)
    Returns the epipole in homogeneous image coordinates.
    """
    if which == 1:
        U, S, Vt = np.linalg.svd(F)
        e = Vt[-1]
    else:
        U, S, Vt = np.linalg.svd(F.T)
        e = Vt[-1]
    return e / e[-1]

def unproject_epipole(e, CM):
    """
    Convert an epipole (in homogeneous image coordinates) into a 3D ray direction
    in the camera coordinate system by applying the inverse of the intrinsic matrix.
    """
    e = np.array(e)
    ray = np.linalg.inv(CM) @ np.array([e[0], e[1], 1])
    return ray / np.linalg.norm(ray)

def main():
    parser = argparse.ArgumentParser(
        description="Plot a two-camera stereo calibration setup with checkerboard poses (from camera1), viewing directions, and mirror the board poses along the x-y plane."
    )
    parser.add_argument("--stereo_calib_file", required=True,
                        help="Path to the stereo calibration JSON file (with CM1, CM2, R, T, E, F, etc.)")
    parser.add_argument("--calib_params_file", required=True,
                        help="Path to the *calibration_parameters.json file for camera1 (contains rvecs, tvecs)")
    parser.add_argument("--num_cols", type=int, default=9,
                        help="Number of checkerboard corners in horizontal direction (default: 9)")
    parser.add_argument("--num_rows", type=int, default=6,
                        help="Number of checkerboard corners in vertical direction (default: 6)")
    parser.add_argument("--square_size", type=float, default=1.0,
                        help="Size of one square (units, default: 1.0)")
    args = parser.parse_args()

    # ----------------------
    # Load stereo calibration data
    # ----------------------
    with open(args.stereo_calib_file, 'r') as f:
        stereo_data = json.load(f)
    
    # Convert camera matrices and other arrays to numpy arrays.
    CM1 = np.array(stereo_data["CM1"])
    CM2 = np.array(stereo_data["CM2"])
    R_stereo = np.array(stereo_data["R"])
    T_stereo = np.array(stereo_data["T"]).flatten()  # translation vector from cam1 to cam2
    F = np.array(stereo_data["F"])
    E = np.array(stereo_data["E"])
    
    # Define camera1 pose in world coordinates (reference frame):
    cam1_center = np.zeros(3)
    cam1_R = np.eye(3)
    
    # For camera2, assume that the stereo calibration provides the transformation:
    # X_cam1 = R_stereo * X_cam2 + T_stereo, so camera2 center (X_cam2=0) in cam1 coordinates is T_stereo.
    cam2_center = T_stereo
    cam2_R = R_stereo  # Rotation of camera2 in camera1's coordinate frame

    # ----------------------
    # Load calibration parameters for camera1 (checkerboard poses)
    # ----------------------
    with open(args.calib_params_file, 'r') as f:
        calib_data = json.load(f)
    
    rvecs = calib_data["rvecs"]
    tvecs = calib_data["tvecs"]
    # Sample every 20th pose
    sampled_indices = list(range(0, len(rvecs), 20))
    board_poses = []
    for i in sampled_indices:
        rvec = rvecs[i]
        tvec = tvecs[i]
        T_board = rvec_tvec_to_board_pose(rvec, tvec)
        board_poses.append(T_board)
    
    # Create checkerboard rectangle (in board coordinates)
    board_corners = create_checkerboard_corners(args.num_cols, args.num_rows, args.square_size)
    
    # Mirror transformation matrix for reflecting along the x-y plane (z=0)
    mirror_M = np.diag([1, 1, -1, 1])

    # Increase the z-coordinate of the checkerboards by a constant offset (e.g., 10 units)
    z_offset = 10.0
    x_offset = -5.0
    translation_M = np.array([
        [1, 0, 0, x_offset],
        [0, 1, 0, 0],
        [0, 0, 1, z_offset],
        [0, 0, 0, 1]
    ])

    # ----------------------
    # Compute epipoles from the fundamental matrix
    # (Note: Epipoles are defined in image coordinates.)
    # Here we unproject them into 3D ray directions.
    # For camera1:
    e1 = compute_epipole(F, which=1)
    ray1 = unproject_epipole(e1, CM1)
    # For camera2:
    e2 = compute_epipole(F, which=2)
    ray2 = unproject_epipole(e2, CM2)

    # ----------------------
    # Start 3D Plot
    # ----------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for the two cameras
    color_cam1 = 'blue'
    color_cam2 = 'red'
    
    # Plot camera centers and their viewing directions
    plot_camera(ax, cam1_center, cam1_R, label="Camera 1", color=color_cam1)
    plot_camera(ax, cam2_center, cam2_R, label="Camera 2", color=color_cam2)
    
    # Plot epipolar ray directions (derived from F) as dashed arrows.
    arrow_len = 10
    # ax.quiver(cam1_center[0], cam1_center[1], cam1_center[2],
    #           ray1[0], ray1[1], ray1[2], length=arrow_len, color='cyan', linestyle='dashed',
    #           label="Cam1 epipolar ray")
    # ax.quiver(cam2_center[0], cam2_center[1], cam2_center[2],
    #           ray2[0], ray2[1], ray2[2], length=arrow_len, color='magenta', linestyle='dashed',
    #           label="Cam2 epipolar ray")
    
    # Plot the checkerboard poses (only from camera1's calibration data)
    # Here we apply a mirror transformation along z (the x-y plane) to each board pose.
    for T_board in board_poses:
        T_board = mirror_M @ T_board
        # Apply the translation to increase the z-coordinate
        T_board = translation_M @ T_board
        corners_world = transform_points(T_board, board_corners)
        # Close the rectangle for plotting
        corners_plot = np.vstack([corners_world, corners_world[0]])
        ax.plot(corners_plot[:, 0], corners_plot[:, 1], corners_plot[:, 2],
                c=color_cam1, alpha=0.5)


    # Set labels and title.
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Stereo Calibration Setup with Mirrored Checkerboard Poses (Camera1)")
    
    plt.show()

    # ----------------------
    # Commentary on Fundamental and Essential Matrix Visualization
    # ----------------------
    print("\nNote:")
    print("Direct visualization of the fundamental and essential matrices in a 3D plot is non-trivial,")
    print("because these matrices describe mappings between image points (i.e. epipolar geometry).")
    print("In this script, we compute and display the epipolar ray directions (derived from the fundamental matrix)")
    print("as a hint toward the inter-camera geometry. To fully visualize the epipolar geometry, one would typically")
    print("overlay epipolar lines on the image planes rather than in the 3D world coordinate system.")

if __name__ == "__main__":
    main()
