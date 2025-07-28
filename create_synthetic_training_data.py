import os
import sys

import bpy
import bmesh
import bpy_extras
from mathutils import Matrix
from mathutils import Vector

from pathlib import Path
import numpy as np
import json
import cv2
    

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------


def get_calibration_matrix_K_from_blender(camd):
    """
    Build intrinsic camera parameters from Blender camera data

    See notes on this in 
    blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
    """
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

def get_3x4_RT_matrix_from_blender(cam):
    """
    Returns camera rotation and translation matrices from Blender.

    There are 3 coordinate systems involved:
    1. The World coordinates: "world"
        - right-handed
    2. The Blender camera coordinates: "bcam"
        - x is horizontal
        - y is up
        - right-handed: negative z look-at direction
    3. The desired computer vision camera coordinates: "cv"
        - x is horizontal
        - y is down (to align to the actual pixel coordinates 
            used in digital images)
        - right-handed: positive z look-at direction
        bcam stands for blender camera
    """
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

# ----------------------------------------------------------
def project_by_object_utils(cam, point):
    """        
    Alternate 3D coordinates to 2D pixel coordinate projection code
    adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
    to have the y axes pointing up and origin at the top-left corner
    """
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


#---------------------------------------------------------------
# render animation and retrieve ground-truth 3D-data
#---------------------------------------------------------------

def prepare_blender_data():
    obj = bpy.context.active_object
    bm = bmesh.from_edit_mesh(obj.data)
    faces = [[v.index for v in f.verts] for f in bm.faces]
    vertices = [list(v.co) for v in obj.data.vertices]
    n_verts = len(obj.data.vertices)

    keypoint_list = [
        'mouth tip', 
        'gill', 
        'root of pelvic fin', 
        'caudal peduncle', 
        'middle of caudal fin', 
        'lower tip of caudal fin'
    ]
    keypoint2index = {
        kpt_name: index 
        for index, kpt_name in enumerate(keypoint_list)
    }
    keypoint_groups = [
        group for group in obj.vertex_groups
        if group.name in keypoint_list
    ]
    
    return obj, bm, faces, vertices, n_verts, keypoint_list, keypoint2index, keypoint_groups

def get_median_kpt_coords(obj, keypoint_list, keypoint_groups):
    kpt2coords = {
        kpt_name: []
        for kpt_name in keypoint_list
    }
    
    kpt2verts = {
        group.name: [
            v for v in obj.data.vertices 
            if group.name in [
                obj.vertex_groups[vg.group].name for vg in v.groups
                ]
            ]
        for group in keypoint_groups
    }
    
    for kpt, verts in kpt2verts.items():
        xs, ys, zs = [], [], []
        for vert in verts:
            x, y, z = vert.co
            xs.append(x)
            ys.append(y)
            zs.append(z)
        kpt2coords[kpt] = (np.average(xs),np.averga(ys),np.average(zs))
    
    return kpt2coords

def create_discrete_color_map(kpt_names, cmap=cv2.COLORMAP_RAINBOW, RGB=False):
    """
    From a pre-defined cv2-colormap and a list of keypoints, assign equally distributed colors from cmap to each keypoint.
    
    Returns:
        kpt_name_2_color (Dict[str, Tuple[int, int, int]]) containing the keypoint names as keys and a 3-tuple (0-255) RGB color as values.
    """
    n = len(kpt_names)
    if n < 2:
        raise ValueError("At least two keypoint names are required for color mapping.")

    step_values = np.linspace(0, 255, n, dtype=np.uint8)
    kpt_name_2_color = {}

    for kpt, val in zip(kpt_names, step_values):
        gray = np.uint8([[val]])  # 1x1 grayscale image
        color_bgr = cv2.applyColorMap(src=gray, colormap=cmap)[0, 0]
        color_rgb = tuple(int(c) for c in color_bgr[::-1])  # Convert BGR to RGB
        kpt_name_2_color[kpt] = [int(v) for v in color_rgb] if RGB else [int(v) for v in color_bgr]

    return kpt_name_2_color

def render_image_from_cam(out_path):
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    return out_path

def draw_kpts_on_img(kpt2coords, img_path, out_path, annot_radius = 5):
    """
    Args:
        kpts (List[List[np.ndarray]]): List of the lists of keypoint tuples (x, y, visibility) per detected instance.
    Create an output image with keypoint annotation.
    """
    img = cv2.imread(str(img_path))
    cmap = create_discrete_color_map(list(kpt2coords))

    for name,x,y in kpt2coords:
        cv2.circle(img, center=(int(x),int(y)), radius=annot_radius, color=kpt_name_2_color[kpt_name], lineType=-1)

    cv2.imwrite(str(out_path), img)
    
def get_seg_mask_polygon(cam_matrix, faces):
    """
    get seg mask polygon
    frontfaces = []
    for face in faces:
        if face.normal dot cam.forward < 0: # face points to opposite direction of cam (towards cam)
            frontfaces.append(face)
    for index, face in enumerate(frontfaces):
        f = frontfaces.pop(index)
        frontfaces.insert(index, [cam_matrix@vertex for vertex in f.vertices])
        
    """

obj, bm, faces, vertices, n_verts, keypoint_list, keypoint2index, keypoint_groups = prepare_blender_data()

kpt2coords = get_median_kpt_coords(obj, keypoint_list, keypoint_groups)
obj_camera = bpy.context.scene.camera
cam_matrix = get_3x4_P_matrix_from_blender(obj_camera)
kpt2coords = {
    kpt: cam_matrix@coords
    for kpt, coords in kpt2coords.items()
}

img_path = 'C:\\Users\\User\\Documents\\Studium\\Bachelor\\deepshapekit-v2\\bluegill_data\\image_1.jpg'
out_path = 'C:\\Users\\User\\Documents\\Studium\\Bachelor\\deepshapekit-v2\\bluegill_data\\image_1_annot.jpg'
render_image_from_cam(img_path)
draw_kpts_on_img(kpt2coords, img_path, out_path)