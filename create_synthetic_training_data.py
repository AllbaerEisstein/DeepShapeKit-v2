import os

import bpy
import bmesh
import bpy_extras
from mathutils import Matrix
from mathutils import Vector

import numpy as np
import json
import cv2


#---------------------------------------------------------------
# Global variables
#---------------------------------------------------------------


keypoint_list:   list[str] = []
collection_name: str       = ""
object_name:     str       = ""
use_cam_matrix:  bool      = True


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# !! With the unaltered code, x- and z-axis are swapped and the direction of the z-axis is inverted - also, the axes have a scale factor > 1 !!
#---------------------------------------------------------------


def get_calibration_matrix_K_Blendercam2Blenderimage(camd):
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
        # the sensor HEIGHT is FIXED (sensor fit is vertical), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor WIDTH is FISXED (sensor fit is horizontal), 
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
        (  (alpha_u,  skew  , u_0),
           (   0   , alpha_v, v_0),
           (   0   ,    0   ,  1 )  ))
    return K


def get_3x4_RT_matrix_Blender2Blendercam(cam):
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
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
    ))

    return RT, R_world2bcam, T_world2bcam


def get_3x4_P_matrix_Blender2Blenderimage(cam):
    K = get_calibration_matrix_K_Blendercam2Blenderimage(cam.data)
    RT, R, T = get_3x4_RT_matrix_Blender2Blendercam(cam)
    return K@RT, K, R, T, RT

# -----------------------------

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


def blender2d_to_opencv2d(point2d):
    """
.    openCV: 
.            
.          .´ [Z camera: positive Z look-at]                                                                       .´ Z
.        .´                                                                                                      .´    
.        0/0---X--->                                                                                             0/0---X--->                  
.        |                                                                                                       |                                   
.        |    image:                                                                                             |     world:             
.        Y    Point(x,y) -> column-major                                                                         Y     right-handed coordinate system, y pointing down
.        |    Mat(y,x)   -> row-major                                                                            |     axes around positive x are clockwise Y, Z
.        |                                                                                                       |     
.        v                     ^    __                                                                           v        ^ 
.                              |   |`.                                                                                    |
.               (1,  0,  0)    |      `.                                                                                  |    (1,  0,  0)
.               (0, -1,  0)----+        `.                                                                                +----(0,  0, -1)     = 90° ccw about x
.               (0,  0, -1)    |          `.                   K @ Blcam2cvcam @ RT                                       |    (0,  1,  0)
.               Blcam2cvcam    |            `.                                                                            |  Blworld2cvworld
.                              |              `.___________________________________________________________________       |                                              
.    Blender:                                                                                                      `.                    
.                                                                                                                    `.                   
.        ^                                                                                                       ^     world:
.        |                                    <---------------------------.----------------.------------         |     right-handed coordinate system, z pointing up                     
.        |    image:                                                    .´                  `.                   |     axes around positive x are clockwise Y, Z             
.        Y    Vector(x,y) -> column-major                      (  a_w,  skew,  u_0)   ( Rxx,  Rxy,  Rxz,  tx)    Z
.        |    Matrix(y,x) -> row-major                         (   0 ,   a_u,  v_0) @ ( Ryx,  Ryy,  Ryz,  ty)    |   .´ Y
.        |                                                     (   0 ,    0 ,   1 )   ( Rzx,  Rzy,  Rzz,  tz)    | .´
.        0/0---X--->                                                     K                      RT               0/0---X--->
.      .´
.    .´ [Z camera: negative Z look-at]   
.          
.            bpy_extras.object_utils.world_to_camera_view(scene, obj, coord)
.            Returns the camera space coords for a 3d point. (also known as: normalized device coordinates - NDC).
.            Where (0, 0) is the bottom left and (1, 1) is the top right of the camera frame. values outside 0-1 are also supported. 
.            A negative 'z' value means the point is behind the camera.
    """
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return (point2d[0], render_size[1] - point2d[1])


#---------------------------------------------------------------
# Keypoint and Mask Data Retrieval
#---------------------------------------------------------------


def get_deformed_mesh_data(collection_name, object_name):
    obj = bpy.data.collections[collection_name].objects[object_name]
    deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)

    obj2world = obj_eval.matrix_world

    # create a mesh with modifiers, armature, shapekeys applied
    # -> docs: create a Mesh data-block from the current state of the object. The object owns the data-block. 
    # The result is temporary and cannot be used by objects from the main database.
    mesh_eval = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=deps)
    bm = bmesh.new()
    bm.from_mesh(mesh_eval)

    faces    = [[v.index for v in f.verts] for f in bm.faces]
    vertices = [tuple(v.co)                for v in bm.verts]
    normals  = [tuple(v.normal)            for v in bm.verts]
    bm.free()

    # build a mapping group-name -> list of vertex indices in that group
    kpt2idx = {}
    for vg in obj_eval.vertex_groups:
        if vg.name in keypoint_list:
            # find all vertices in mesh_eval whose group indices include vg.index
            verts_in_group = [
                vi for vi, v in enumerate(mesh_eval.vertices)
                if any(g.group == vg.index for g in v.groups)
            ]
            kpt2idx[vg.name] = verts_in_group

    # now get their coords in world space:
    kpt2verts_worldco = {
        kpt: [tuple(obj2world @ mesh_eval.vertices[i].co) for i in idx_list]
        for kpt, idx_list in kpt2idx.items()
    }

    bm.free()
    # docs: The object owns the mesh data-block. To force free it use to_mesh_clear(). 
    obj_eval.to_mesh_clear()
    
    return faces, vertices, normals, kpt2verts_worldco


def calculate_metadata(collection_name, object_name, cam_name):
    faces, vertices, normals, kpt2verts_co = get_deformed_mesh_data(collection_name, object_name)

    kpt2coords_world = get_avg_kpt_coords(kpt2verts_co)

    obj_camera = bpy.data.objects.get(cam_name)
    Blender_world_2_Blender_image, Blender_cam_2_Blender_image, R, T, Blender_world_2_Blender_cam = get_3x4_P_matrix_Blender2Blenderimage(obj_camera)

    # invert y and z:
    Blender_cam_2_cv_cam = Matrix(
        (( 1,  0,  0),  # -> x becomes x
         ( 0, -1,  0),  # -> y becomes negative y
         ( 0,  0, -1))) # -> z becomes negative z
    
    Blender_world_2_cv_image = Blender_cam_2_Blender_image @ Blender_cam_2_cv_cam @ Blender_world_2_Blender_cam

    kpt2coords_camera = {
        kpt: Blender_world_2_cv_image@Vector(coords + (1,)) if use_cam_matrix else project_by_object_utils(obj_camera, Vector(coords))
        for kpt, coords in kpt2coords_world.items()
    }

    kpt2coords_image = {
        kpt: (coords_homog.x / coords_homog.z, coords_homog.y / coords_homog.z)
            if use_cam_matrix
            else (coords_homog.x, coords_homog.y)
        for kpt, coords_homog in kpt2coords_camera.items()
    }

    return kpt2coords_image

# -----------------------------

def get_avg_kpt_coords(kpt2verts_co:dict):
    kpt2coords = {}
    
    for kpt, coord in kpt2verts_co.items():
        xs, ys, zs = [], [], []
        for vert in coord:
            x, y, z = vert
            xs.append(x)
            ys.append(y)
            zs.append(z)
        kpt2coords[kpt] = (np.average(xs),np.average(ys),np.average(zs))
    
    return kpt2coords


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
        cv2.putText(img, name, (int(x),int(y+annot_radius+10)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=1)

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
        return f"{qitem['view']}_{str(qitem['frame']).zfill(4)}" 

    def exists(self, blender_formatted_filepath):
        filepath = bpy.path.abspath(blender_formatted_filepath)
        return os.path.exists(filepath)
    

    def execute(self, context):
        self.cancel_render = False
        self.rendering = False
        self.render_queue = []

        cam_objects = bpy.data.collections['Cameras'].objects

# --- Build render queue ------------------------------------------------------

        for cam_name in cam_objects.keys():
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
        self.timer_event = bpy.context.window_manager.event_timer_add(0.5, window=bpy.context.window)
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
                                        
                    # # skip if the file exists
                    # if self.exists(img_out_file):
                    #     self.render_queue.pop(0)
                    #     print("Skipping " + render_filename)
                    
                    #else:
                    qitem = self.render_queue.pop(0)
                    frame_index = qitem["frame"]
                    cam_name = qitem["view"]
                    
                    render_basename = self.make_filename(qitem)

                    # switch camera
                    sc.camera = bpy.data.objects.get(cam_name)
                    if not sc.camera:
                        self.report({'ERROR'}, f"Camera {cam_name} not found!")
                        return {'CANCELLED'}

                    # set frame
                    sc.frame_set(frame_index)

                    img_file = os.path.join(self.OUT_DIR, render_basename + ".png")

                    print(f'Rendering {str(self.total + 1 - len(self.render_queue))} / {str(self.total)}: {img_file}')

                    # render to that exact path
                    sc.render.filepath = img_file
                    bpy.ops.render.render(write_still=True)

                    # annotate immediately
                    # try:
                    draw_kpts_on_img(
                        kpt2coords = calculate_metadata(collection_name, object_name, cam_name),
                        img_path   = bpy.path.abspath(img_file),
                        out_path   = os.path.join(bpy.path.abspath(self.ANNOT_DIR), render_basename + "_annot.png"),
                    )
                    # except Exception as e:
                    #     self.report({'ERROR'}, f"Annotation failed on {bpy.path.abspath(img_file)}: {e}")


        return {'PASS_THROUGH'}


#---------------------------------------------------------------
# Execution context
#---------------------------------------------------------------


def unregister_timed_render():
    bpy.utils.unregister_class(TimedRender)

if __name__ == "__main__":

    test           = False
    use_cam_matrix = True
    ortho          = False


    if test:
        keypoint_list = [
            'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'origin'
        ]
        collection_name = "Test"
        object_name = "Cube"

    else:
        keypoint_list = [
            'mouth tip', 'gill',
            'root of pelvic fin','caudal peduncle',
            'middle of caudal fin','lower tip of caudal fin'
        ]
        collection_name = "Bluegill"
        object_name = "Body"


    cam_objects = bpy.data.collections['Cameras'].objects
    for cam_name in cam_objects.keys():
        cam = bpy.data.objects.get(cam_name)
        cam.data.type = 'ORTHO' if ortho else 'PERSP'
        if ortho:
            cam.data.ortho_scale = 30


    bpy.utils.register_class(TimedRender)
    bpy.ops.render.timed_render()