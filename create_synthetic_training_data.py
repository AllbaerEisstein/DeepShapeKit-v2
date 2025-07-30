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
# Keypoint and Mask Data Retrieval
#---------------------------------------------------------------

def get_mesh_data(collection_name, object_name):
    obj = bpy.data.collections[collection_name].objects[object_name]
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces = [[v.index for v in f.verts] for f in bm.faces]
    vertices = [list(v.co) for v in obj.data.vertices]
    n_verts = len(obj.data.vertices)
    bm.free()

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
        kpt2coords[kpt] = (np.average(xs),np.average(ys),np.average(zs))
    
    return kpt2coords


'''
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
        gray = np.array(np.uint8([[val]]))  # 1x1 grayscale image
        color_bgr = cv2.applyColorMap(gray, cmap)[0, 0]
        color_rgb = tuple(int(c) for c in color_bgr[::-1])  # Convert BGR to RGB
        kpt_name_2_color[kpt] = [int(v) for v in color_rgb] if RGB else [int(v) for v in color_bgr]

    return kpt_name_2_color
'''


def draw_kpts_on_img(kpt2coords, img_path, out_path, annot_radius = 5):
    """
    Args:
        kpts (List[List[np.ndarray]]): List of the lists of keypoint tuples (x, y, visibility) per detected instance.
    Create an output image with keypoint annotation.
    """
    img = cv2.imread(str(img_path))
    #cmap = create_discrete_color_map(list(kpt2coords))

    for name,(x,y) in kpt2coords.items():
        cv2.circle(img=img, center=(int(x),int(y)), radius=annot_radius, color=(255, 0, 0), lineType=-1)
        cv2.putText(img, name, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=1)

    cv2.imwrite(str(out_path), img)
    
    
def blender2d_to_opencv2d(point2d):
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return (point2d[0], render_size[1] - point2d[1])

    
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


def calculate_metadata(collection_name, object_name):
    obj, bm, faces, vertices, n_verts, keypoint_list, keypoint2index, keypoint_groups = get_mesh_data(collection_name, object_name)

    kpt2coords_world = get_median_kpt_coords(obj, keypoint_list, keypoint_groups)

    obj_camera = bpy.context.scene.camera
    P, K, RT = get_3x4_P_matrix_from_blender(obj_camera)

    kpt2coords_camera = {
        kpt: P@Vector(coords)
        for kpt, coords in kpt2coords_world.items()
    }
    kpt2coords_image = {
        kpt: blender2d_to_opencv2d((coords_homog.x / coords_homog.z, coords_homog.y / coords_homog.z))
        for kpt, coords_homog in kpt2coords_camera.items()
    }

    return kpt2coords_image



#---------------------------------------------------------------
# Rendering
#---------------------------------------------------------------

def render_image_from_active_cam(out_path):
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    return out_path


class TimedRender(bpy.types.Operator):
    """
    A modal operator in Blender is an operator that stays active after it's invoked, handling input events (mouse, keyboard, timer) continuously until it explicitly finishes.
    Render multiple images with periodic checks if blender is ready to render so that it doesn't crash"
    """
    bl_idname = "render.timed_render"
    bl_label = "Render multiple images with periodic checks if blender is ready to render so that it doesn't crash."

    cancel_render:  bool | None = None
    rendering:      bool | None = None
    render_queue:   list | None = None
    timer_event = None
    total: int | None = None

    OUT_DIR   = "//synthetic_data"
    ANNOT_DIR = "//synthetic_data" + os.sep + "annot"

    last_metadata: dict | None = None


    def render_init(self, scene, depsgraph):
        self.rendering = True
        print("RENDER INIT")
    
    def render_complete(self, scene, depsgraph):
        if not self.render_queue:
            self.report(
                {'ERROR_INVALID_INPUT'}, message=f"Render queue is empty!"
            )
            return {'CANCELLED'}
        else:
            self.render_queue.pop(0)
            self.rendering = False
    
    def render_cancel(self, scene, depsgraph):
        self.cancel_render = True
        print("RENDER CANCEL")


    def make_filename(self, qitem):
        return f'{qitem["view"]}_{qitem["frame"]}' 

    def exists(self, blender_formatted_filepath):
        filepath = bpy.path.abspath(blender_formatted_filepath)
        return os.path.exists(filepath)
    

    def execute(self, context):
        self.cancel_render = False
        self.rendering = False
        self.render_queue = []

        cam_objects = bpy.data.collections['Cameras'].objects

# --- Build render queue ------------------------------------------------------

        for cam_name, cam_object in cam_objects.items():
            for frame_index in range(bpy.context.scene.frame_start,
                                     bpy.context.scene.frame_end):
                self.render_queue.append({
                    "view":  cam_name,
                    "frame": frame_index
                })
        
        self.total = len(self.render_queue)
        print("number of images to render: "+str(self.total))

# -----------------------------------------------------------------------------

        bpy.app.handlers.render_init.clear()
        bpy.app.handlers.render_init.append(self.render_init)

        bpy.app.handlers.render_complete.clear()
        bpy.app.handlers.render_complete.append(self.render_complete)

        bpy.app.handlers.render_cancel.clear()
        bpy.app.handlers.render_cancel.append(self.render_cancel)

        bpy.types.RenderSettings.use_lock_interface = True

        # Add a timer to the given window, to generate periodic ‘TIMER’ events
        self.timer_event = bpy.context.window_manager.event_timer_add(0.3, window=bpy.context.window)
        # Add a modal handler to the window manager, for the given modal operator (this is expected to be a class that extends bpy.types.Operator!) (called by invoke() with self, just before returning {‘RUNNING_MODAL’})
        # -> A modal handler handles events!
        context.window_manager.modal_handler_add(self)  # <- This is key

        return {'RUNNING_MODAL'}


    def cleanup(self, context):
        # remove all render callbacks
        bpy.app.handlers.render_init.clear()
        bpy.app.handlers.render_complete.clear()
        bpy.app.handlers.render_cancel.clear()
        # remove timer
        context.window_manager.event_timer_remove(self.timer_event)
        bpy.types.RenderSettings.use_lock_interface = False

    def annotate(self, metadata):
        if metadata is not None:
            try:
                annot_filepath = bpy.path.abspath(self.ANNOT_DIR + os.sep + metadata["annot_filename"])
                #print(f'Annotating image, reading from {metadata["path"]}, saving to {annot_filepath}')
                if metadata["frame_index"] <= self.total + 1 - len(self.render_queue):
                    draw_kpts_on_img(
                        kpt2coords = metadata["kpt2coords"], 
                        img_path   = metadata["rendered_image_path"], 
                        out_path   = annot_filepath
                    )
            except:
                self.report(
                    {'ERROR_INVALID_INPUT'}, message=f"Error while annotating previous image: {metadata['rendered_image_path']}"
                )
                return {'CANCELLED'}

    def modal(self, context, event):
        # cancelling manually via pressing escape
        if event.type == 'ESC':
            bpy.types.RenderSettings.use_lock_interface = False
            print("CANCELLED")
            return {'CANCELLED'}
        # react to timer event
        elif event.type == 'TIMER':
            if not self.render_queue:
                self.report(
                    {'ERROR_INVALID_INPUT'}, message=f"Render queue is empty!"
                )
                return {'CANCELLED'}
            else:
                # if cancelled or there are no items left to render, first cleanup and finish, then annotate the last rendered image
                if len(self.render_queue) == 0 or self.cancel_render == True:
                    self.cleanup(context)
                # nothing is rendering and there are items in queue
                elif self.rendering == False:
                    sc = context.scene
                    # render_filename = self.make_filename(qitem)
                    # img_out_file = self.OUT_DIR + os.sep + render_filename + '.png'
                    
                    # # skip if the file exists
                    # if self.exists(img_out_file):
                    #     self.render_queue.pop(0)
                    #     print("Skipping " + render_filename)
                    
                    #else:
                    qitem = self.render_queue.pop(0)
                    frame_index = qitem["frame"]
                    cam_name = qitem["view"]

                    # switch camera
                    sc.camera = bpy.data.objects.get(cam_name)
                    if not sc.camera:
                        self.report({'ERROR'}, f"Camera {cam_name} not found!")
                        return {'CANCELLED'}

                    # set frame
                    sc.frame_set(frame_index)

                    # build an exact filename
                    render_basename = f"{cam_name}_{str(frame_index).zfill(4)}"
                    img_file = os.path.join(self.OUT_DIR, render_basename + ".png")

                    print(f'Rendering {str(self.total + 1 - len(self.render_queue))} / {str(self.total)}: {img_file}')

                    # render to that exact path
                    sc.render.filepath = img_file
                    bpy.ops.render.render(write_still=True)

                    # annotate immediately
                    try:
                        draw_kpts_on_img(
                            kpt2coords = calculate_metadata("Bluegill","Body"),
                            img_path   = bpy.path.abspath(img_file),
                            out_path   = os.path.join(bpy.path.abspath(self.ANNOT_DIR), render_basename + "_annot.png"),
                        )
                    except Exception as e:
                        self.report({'ERROR'}, f"Annotation failed on {bpy.path.abspath(img_file)}: {e}")


        return {'PASS_THROUGH'}

    

def unregister_timed_render():
    bpy.utils.unregister_class(TimedRender)

if __name__ == "__main__":
    bpy.utils.register_class(TimedRender)
    bpy.ops.render.timed_render()