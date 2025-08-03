import os
import math

import bpy
import bmesh
import bpy_extras
from mathutils import Matrix
from mathutils import Vector

import numpy as np
import json
import cv2
import alphashape
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection


#---------------------------------------------------------------
# Global variables
#---------------------------------------------------------------

global RENDER_SCALE
global IMAGE_WIDTH
global IMAGE_HEIGHT
global SCENE
global evaluated_depsgraph
global CAM_OBJECTS


global KEYPOINT_LIST
global COLLECTION_NAME
global OBJECT_NAME

global OUT_DIR_BL
global ANNOT_DIR_BL

use_cam_matrix:        bool      = True
render_binary:         bool      = False
keypoint_visible_threshold_percentage: float = 0.1
contour_threshold:     float     = 0.01
alphashape_alpha:      float     = 1.0
draw_lattice_for_kpt_annot: bool = False

# invert y and z:
Blender_cam_2_cv_cam = Matrix((
    ( 1,  0,  0),  # -> x becomes x
    ( 0, -1,  0),  # -> y becomes negative y
    ( 0,  0, -1))) # -> z becomes negative z

mirror_y = Matrix((
    ( 1,  0,  0),
    ( 0, -1,  0),
    ( 0,  0,  1)))


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
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


def blender2d_origin_to_opencv2d_origin(point2d):
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


def get_deformed_mesh_data(deps, collection_name, object_name):
    obj = bpy.data.collections[collection_name].objects[object_name]
    if deps is None:
        deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)

    obj2world = obj_eval.matrix_world

    # create a mesh with modifiers, armature, shapekeys applied
    # -> docs: create a Mesh data-block from the current state of the object. The object owns the data-block. 
    # The result is temporary and cannot be used by objects from the main database.
    mesh_eval = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=deps)
    bm = bmesh.new()
    bm.from_mesh(mesh_eval)

    faces    = [[tuple(v.co) for v in f.verts]   for f in bm.faces]
    vertices = [tuple(v.co)                      for v in bm.verts]
    normals  = [(tuple(v.co), tuple(v.normal))   for v in bm.verts]
    bm.free()

    # build a mapping group-name -> list of vertex indices in that group
    kpt2idx = {}
    for vg in obj_eval.vertex_groups:
        if vg.name in KEYPOINT_LIST:
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


def get_projected_keypoint_coords_cv(collection_name, object_name, cam_name):
    evaluated_depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_obj = bpy.data.objects.get(cam_name)

    faces, vertices, normals, kpt_2_coord_list_world = get_deformed_mesh_data(evaluated_depsgraph, collection_name, object_name)

    occluded_kpts = []
    kpt_2_coord_list_world_visible = {}
    for kpt, coords_list in kpt_2_coord_list_world.items():
        visible_verts = [v for v in coords_list if not is_vertex_occluded(evaluated_depsgraph, cam_obj, Vector(v))]
        if len(visible_verts)/len(kpt_2_coord_list_world[kpt]) >= keypoint_visible_threshold_percentage:
            kpt_2_coord_list_world_visible[kpt] = visible_verts
        else:
            occluded_kpts.append(kpt)
    for kpt in occluded_kpts:
        kpt_2_coord_list_world.pop(kpt)

    kpt_2_coords_world = get_avg_kpt_coords_3d(kpt_2_coord_list_world)


    KRT_Blender_world_2_Blender_image, K_Blender_cam_2_Blender_image, R, T, RT_Blender_world_2_Blender_cam = get_3x4_P_matrix_Blender2Blenderimage(cam_obj)
    def get_height_dependent_matrices(h):
        Blender2d_origin_to_opencv2d_origin = Matrix((
            ( 1,  0,  0),
            ( 0, -1,  h),  # -> y, after mirroring, is still at y, but the origin got shifted by +h and the direction go changed, so it needs to be translated to h - y
            ( 0,  0,  1)))
        
        translate_by_hdiv2 = Matrix((
            ( 1,  0,  0),
            ( 0,  1, h/2),
            ( 0,  0,  1)))

        translate_by_neghdiv2 = Matrix((
            ( 1,  0,  0),
            ( 0,  1, -h/2),
            ( 0,  0,  1)))
        
        return Blender2d_origin_to_opencv2d_origin, translate_by_hdiv2, translate_by_neghdiv2
    Blender2d_origin_to_opencv2d_origin, translate_by_hdiv2, translate_by_neghdiv2 = get_height_dependent_matrices(IMAGE_HEIGHT)
 
    Blender_world_2_cv_image = K_Blender_cam_2_Blender_image @ Blender_cam_2_cv_cam @ RT_Blender_world_2_Blender_cam

    kpt_2_coords_camera = {
        kpt: Blender_world_2_cv_image@Vector(coords + (1,)) 
            if use_cam_matrix 
            else project_by_object_utils(cam_obj, Vector(coords))
        for kpt, coords in kpt_2_coords_world.items()
    }

    kpt_2_coords_image = {
        kpt: (
                (coords_homog.x / coords_homog.z, coords_homog.y / coords_homog.z), # cv image coords
                len(kpt_2_coord_list_world_visible[kpt]) / len(kpt_2_coord_list_world[kpt]) # "confidence" (percentage of vertices of this keypoint visible)
            ) # ((x,y),c) in the end
            if use_cam_matrix
            else (
                (coords_homog.x, coords_homog.y),
                len(kpt_2_coord_list_world_visible[kpt]) / len(kpt_2_coord_list_world[kpt])
            )
        for kpt, coords_homog in kpt_2_coords_camera.items()
        # check if keypoint is in rendered frame
        if  0 <= (coords_homog.x / coords_homog.z) < IMAGE_WIDTH
        and 0 <= (coords_homog.y / coords_homog.z) < IMAGE_HEIGHT
    }

    return kpt_2_coords_image

# -----------------------------

def get_avg_kpt_coords_3d(kpt2verts_co:dict):
    """
    Given a dict mapping keypoint names to lists of 3D vertex coords,
    return a dict mapping each keypoint to its (x,y,z) mean.
    """
    kpt2coords = {}
    for kpt, verts in kpt2verts_co.items():
        if not verts:
            # no vertices → skip or assign NaNs
            continue  
        arr = np.array(verts, dtype=float)       # shape (N,3)
        mean_xyz = arr.mean(axis=0)              # shape (3,)
        kpt2coords[kpt] = tuple(mean_xyz.tolist())
    return kpt2coords


def is_vertex_occluded(deps, cam_obj, vertex_co_world, eps=1e-5):
    # 1) Get depsgraph
    if deps is None:
        deps = bpy.context.evaluated_depsgraph_get()

    # 2) Ray origin / direction  
    cam_co = cam_obj.matrix_world.translation
    dir_vec = (vertex_co_world - cam_co)
    dist_to_pt = dir_vec.length
    if dist_to_pt < eps:
        return False    # at camera, always “visible”
    dir_vec.normalize()

    # 3) Nudge the origin off the camera to avoid immediate self-hit
    origin = cam_co + dir_vec * eps

    # 4) Cast the ray
    hit, hit_loc, _, _, hit_obj, _ = SCENE.ray_cast(deps, origin, dir_vec)

    # 5) If nothing was hit → visible
    if not hit:
        return False

    # 6) Otherwise compare distances **from the same origin**:
    dist_hit = (hit_loc - origin).length

    # occluded if something is strictly *before* the target point
    return dist_hit < (dist_to_pt - eps)


def draw_kpts_on_img(kpt2coords, img_path, out_path, tenth_of_annot_radius: int = 1):
    """
    Args:
        kpts (List[List[np.ndarray]]): List of the lists of keypoint tuples (x, y, visibility) per detected instance.
    Create an output image with keypoint annotation.
    """
    img = cv2.imread(str(img_path))
    #cmap = create_discrete_color_map(list(kpt2coords))
    annot_radius = tenth_of_annot_radius * 10

    for name,((x,y),c) in kpt2coords.items():
        conf_scaled_annot_radius = int(
            int(c*10)/10   # will cut off second decimal (e.g 0.72 -> 0.7)
            *annot_radius  # if annot_radius is k*10 with k in N, this will yield an int
        )
        cv2.circle(img=img, center=(int(x),int(y)), radius=conf_scaled_annot_radius, color=(255, 0, 0), lineType=-1)
        cv2.putText(img, f"{int(c*100)/100}: {name}", (int(x),int(y+conf_scaled_annot_radius+10)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=1)

    cv2.imwrite(str(out_path), img)


def draw_lattice_lines(image_path, output_path=None,
                       line_color=(0, 255, 0),  # BGR tuple: green lines
                       middle_thickness=3,
                       other_thickness=1,
                       draw_horizontal=True,
                       draw_vertical=True):
    """
    Reads an image, draws horizontal and/or vertical lines at intervals of 0.1 * dimension
    with the central line thicker and centered.

    :param image_path: Path to the input image file
    :param output_path: Optional path to save the modified image
    :param line_color: Color for the lines in BGR format (default green)
    :param middle_thickness: Thickness of the central lines in pixels
    :param other_thickness: Thickness of the other lines in pixels
    :param draw_horizontal: Whether to draw horizontal lines
    :param draw_vertical: Whether to draw vertical lines
    :return: The image with lines drawn (as a NumPy array)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at '{image_path}'")

    height, width = image.shape[:2]

    # Horizontal lines
    if draw_horizontal:
        h_step = int(0.1 * height)
        mid_y = height // 2
        # central horizontal line
        cv2.line(image, (0, mid_y), (width, mid_y), line_color, thickness=middle_thickness)
        # lines below
        offset = h_step
        while mid_y + offset < height:
            y = mid_y + offset
            cv2.line(image, (0, y), (width, y), line_color, thickness=other_thickness)
            offset += h_step
        # lines above
        offset = h_step
        while mid_y - offset > 0:
            y = mid_y - offset
            cv2.line(image, (0, y), (width, y), line_color, thickness=other_thickness)
            offset += h_step

    # Vertical lines
    if draw_vertical:
        v_step = int(0.1 * width)
        mid_x = width // 2
        # central vertical line
        cv2.line(image, (mid_x, 0), (mid_x, height), line_color, thickness=middle_thickness)
        # lines right
        offset = v_step
        while mid_x + offset < width:
            x = mid_x + offset
            cv2.line(image, (x, 0), (x, height), line_color, thickness=other_thickness)
            offset += v_step
        # lines left
        offset = v_step
        while mid_x - offset > 0:
            x = mid_x - offset
            cv2.line(image, (x, 0), (x, height), line_color, thickness=other_thickness)
            offset += v_step

    # Save or return
    if output_path:
        cv2.imwrite(output_path, image)
    return image

# -----------------------------

def get_contour_polygon_points_3d(RT, normals, contour_threshold=0.1):
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
    cam_forward = (RT @ Vector((0, 0, 1))).normalized()
    outline_points_3d = []
    for co, normal in normals:
        if abs(Vector(normal).normalized().dot(cam_forward)) < contour_threshold:
            outline_points_3d.append(tuple(co))

    return outline_points_3d


def nearest_neighbour(points, use_two_opt=False):

    def euclid(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def tour_length(tour):
        """Compute total length of a closed tour."""
        L = 0.0
        N = len(tour)
        for i in range(N):
            L += euclid(tour[i], tour[(i+1) % N])
        return L

    def two_opt(tour):
        """
        Perform 2-opt until no improving swap is found.
        tour: list of (x,y) points in initial visit order.
        Returns a new list (possibly the same) with shorter length.
        """
        best = tour
        improved = True
        while improved:
            improved = False
            best_dist = tour_length(best)
            N = len(best)

            for i in range(1, N - 2):
                for j in range(i+1, N):
                    # skip adjacent edges, and the case i=0, j=N-1 which is the same closing edge
                    if j - i == 1 or (i == 0 and j == N-1):
                        continue

                    # create a new tour by reversing the segment between i and j
                    new_tour = best[:i] + best[i:j][::-1] + best[j:]
                    new_dist = tour_length(new_tour)
                    if new_dist + 1e-8 < best_dist:  # found an improvement
                        best = new_tour
                        best_dist = new_dist
                        improved = True
                        # break out to restart searching from the top
                        break
                if improved:
                    break

        return best

    ordered = []
    next_vertex_x, next_vertex_y = points.pop(0)
    neighbour_index = 0
    
    # Nearest-Neighbour-Heuristic for ordering the polygon
    while len(points) > 0:
        min_dist = 10e6
        for candidate_index, (candidate_x, candidate_y) in enumerate(points):
            dist_squared = (next_vertex_x-candidate_x)**2 + (next_vertex_y-candidate_y)**2
            if dist_squared < min_dist:
                min_dist = dist_squared
                neighbour_index = candidate_index

        next_vertex_x, next_vertex_y = points.pop(neighbour_index)
        ordered.append((next_vertex_x, next_vertex_y)) 
    
    return two_opt(ordered) if use_two_opt else ordered


def collect_polys(g):
    polys = []
    if isinstance(g, Polygon):
        polys.append(g)
    elif isinstance(g, (MultiPolygon, GeometryCollection)):
        for sub in g.geoms:
            collect_polys(sub)
    
    return polys


def get_mask_polygons_from_binary_image(img_path):
    # 1. Read as binary mask
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 2. Find external contours only
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL,     # only outer boundaries
        cv2.CHAIN_APPROX_NONE  # no approximation → get every pixel
    )

    # 3. contours is a list of NumPy arrays of shape (N,1,2)
    #    We’ll flatten each to a list of (x,y) tuples
    polygons = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)  # shape (N,2)
        polygons.append([tuple(pt) for pt in pts])

    # If you know there’s only one object, return polygons[0]
    return polygons


def draw_polygons(img_path, out_path, polygons):
    img = cv2.imread(img_path)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32, ndmin=2).reshape(-1, 1, 2)
        cv2.polylines(img=img, pts=[pts], isClosed=True, color=(255,0,0), thickness=1)
    cv2.imwrite(out_path, img)

# -----------------------------

def write_pose_labels_yolo(instances, kpt_order, image_width, image_height, class_idx, out_path):
    """
    Write YOLOv8-pose label file from a list of detected-instance keypoint dicts.

    Parameters
    ----------
    instances : List[Dict[str, Tuple[ Tuple[float,float], float ]]]
        Each dict maps keypoint name → ((x, y), confidence).
    kpt_order : List[str]
        The exact ordering of keypoint names to write.
    image_width : int
        Width of the image.
    image_height : int
        Height of the image.
    class_idx : int
        The class index for all instances.
    out_path : str
        Path to the output .txt file.
    """
    lines = []
    for inst in instances:
        # 1) compute bbox over only the keypoints present
        pts = []
        for ((x, y), _) in inst.values():
            pts.append((x, y))
        if pts:
            xs, ys = zip(*pts)
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            x_ctr = ((xmin + xmax) / 2) / image_width
            y_ctr = ((ymin + ymax) / 2) / image_height
            w_box = (xmax - xmin) / image_width
            h_box = (ymax - ymin) / image_height
        else:
            # fallback to full image
            x_ctr, y_ctr, w_box, h_box = 0.5, 0.5, 1.0, 1.0

        parts = [
            str(class_idx),
            f"{x_ctr:.6f}",
            f"{y_ctr:.6f}",
            f"{w_box:.6f}",
            f"{h_box:.6f}"
        ]

        # 2) append each keypoint in the fixed order
        for kpt in kpt_order:
            if kpt in inst:
                (x, y), _ = inst[kpt]
                parts.append(f"{x / image_width:.6f}")
                parts.append(f"{y / image_height:.6f}")
            else:
                # missing → zero
                parts.append("0.000000")
                parts.append("0.000000")

        lines.append(" ".join(parts))

    # write out
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_polygons_to_yolo(polygons, image_width, image_height, out_path):
    """
    Write a list of polygons to a YOLOv8 segmentation txt file.

    Parameters
    ----------
    polygons : List[List[Tuple[float, float]]]
        A list of polygons. Each polygon is a list of (x, y) tuples in pixel coords.
    image_width : int
        Width of the image the polygons came from.
    image_height : int
        Height of the image the polygons came from.
    out_path : str
        Path to the output .txt file. Will be overwritten if it exists.

    Output file format (one polygon per line):
      <instance_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    where coordinates are normalized to [0,1] by dividing x by image_width
    and y by image_height.
    """
    lines = []
    for idx, polygon in enumerate(polygons):
        # normalize each point and format to three decimal places
        norm_pts = []
        for x, y in polygon:
            x_n = x / image_width
            y_n = y / image_height
            norm_pts.append(f"{x_n:.3f}")
            norm_pts.append(f"{y_n:.3f}")
        # assemble the line: instance-id followed by all normalized coords
        line = f"{idx} " + " ".join(norm_pts)
        lines.append(line)

    # write all lines to the output file
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


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
    timer_event                 = None
    total:           int | None = None

    OUT_DIR_BL   = "//synthetic_data"
    ANNOT_DIR_BL = "//synthetic_data" + os.sep + "annot"
    KPT_LABEL_DIR = bpy.path.abspath(os.path.join(ANNOT_DIR_BL, "labels_keypoints"))
    MASK_LABEL_DIR = bpy.path.abspath(os.path.join(ANNOT_DIR_BL, "labels_masks"))
    for dir in [bpy.path.abspath(ANNOT_DIR_BL), KPT_LABEL_DIR, MASK_LABEL_DIR]:
        os.makedirs(dir, exist_ok=True)



    def render_init(self, scene, depsgraph):
        self.rendering = True
        print("RENDER INIT")
    
    def render_complete(self, scene, depsgraph):
        if self.render_queue == None:
            self.report(
                {'ERROR_INVALID_INPUT'}, message=f"Render queue is empty!"
            )
            return {'CANCELLED'}
        else:
            self.rendering = False
    
    def render_cancel(self, scene, depsgraph):
        self.cancel_render = True
        print("RENDER CANCEL")


    def make_prefix_cam_frame(self, qitem):
        prefix_for_cam = qitem['view'].split('.')[1]+'_'+qitem['view'].split('.')[0]
        return f"{str(qitem['frame']).zfill(4)}_{prefix_for_cam}" 
    

    def execute(self, context):
        self.cancel_render = False
        self.rendering = False
        self.render_queue = []

        cam_objects = sorted(bpy.data.collections['Cameras'].objects, key=lambda cam_obj: cam_obj.name.split('.')[1])

# --- Build render queue ------------------------------------------------------

        modes = ["regular", "binary"] if render_binary else ["regular"]
        for cam in cam_objects:
            for frame_index in range(bpy.context.scene.frame_start,
                                    bpy.context.scene.frame_end+1):
                for mode in modes: # an additional entry for the binary render (used for mask annotation) appended to the end of the queue
                    self.render_queue.append({
                        "view":  cam.name,
                        "frame": frame_index,
                        "mode":  mode
                    })

        # for entry in self.render_queue:
        #     for key, value in entry.items():
        #         print(f"{key}: {value}")
        #     print("\n")
        
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
            self.cleanup(context)
            print("CANCELLED")
            self.report(
                {'INFO'}, message=f"Cancelled via pressing ESC."
            )
            return {'CANCELLED'}
        # react to timer event
        elif event.type == 'TIMER':
            if self.render_queue is None:
                self.report(
                    {'ERROR_INVALID_INPUT'}, message=f"Render queue is empty!"
                )
                return {'CANCELLED'}
            else:
                # if cancelled or there are no items left to render, first cleanup and finish, then annotate the last rendered image
                if len(self.render_queue) == 0 or self.cancel_render == True:
                    self.cleanup(context)
                    self.report(
                        {'INFO'}, message=f"Done! Render queue is empty."
                    )
                    return {'FINISHED'}
                # nothing is rendering and there are items in queue
                elif self.rendering == False:
                    sc = context.scene
                                        
                    # # skip if the file exists
                    # if self.exists(img_out_file):
                    #     self.render_queue.pop(0)
                    #     print("Skipping " + render_filename)
                    
                    #else:
                    qitem       = self.render_queue.pop(0)
                    frame_index = qitem["frame"]
                    cam_name    = qitem["view"]
                    mode        = qitem["mode"]

                    # switch camera
                    sc.camera = bpy.data.objects.get(cam_name)
                    if not sc.camera:
                        self.report({'ERROR'}, f"Camera {cam_name} not found!")
                        return {'CANCELLED'}
                    # set frame
                    sc.frame_set(frame_index)
                    # render to that exact path
                    render_prefix_cam_frame = self.make_prefix_cam_frame(qitem)
                    render_out_file_path_bl = os.path.join(self.OUT_DIR_BL, render_prefix_cam_frame + ".png")
                    sc.render.filepath = render_out_file_path_bl

                    def reset_render_settings():
                        sc.node_tree.nodes["Alpha Over"].inputs[1].default_value = (0, 0, 0, 0)
                        sc.node_tree.nodes["Brightness/Contrast"].inputs[1].default_value = 0
                        sc.node_tree.nodes["Brightness/Contrast"].inputs[2].default_value = 0


                    if mode == "regular":
                        print(f'Rendering {str(self.total + 1 - len(self.render_queue))} / {str(self.total)}: {render_out_file_path_bl} in mode \'regular\'')

                        reset_render_settings()
                        bpy.ops.render.render(write_still=True)

                        kpt_2_coords_cvimg = get_projected_keypoint_coords_cv(COLLECTION_NAME, OBJECT_NAME, cam_name)

                        # annotate keypoints
                        keypoint_annot_out_file_path = os.path.join(
                            self.KPT_LABEL_DIR, render_prefix_cam_frame + "_annot_kpt.png"
                        )
                        draw_kpts_on_img(
                            kpt2coords = kpt_2_coords_cvimg,
                            img_path   = bpy.path.abspath(render_out_file_path_bl),
                            out_path   = keypoint_annot_out_file_path
                        )
                        if draw_lattice_for_kpt_annot:
                            draw_lattice_lines(
                                image_path  = keypoint_annot_out_file_path,
                                output_path = keypoint_annot_out_file_path,
                                middle_thickness = 2
                            )

                        kpt_label_out_path =  os.path.join(self.KPT_LABEL_DIR, render_prefix_cam_frame + ".txt")
                        write_pose_labels_yolo([kpt_2_coords_cvimg], KEYPOINT_LIST, IMAGE_WIDTH, IMAGE_HEIGHT, 0, kpt_label_out_path)

                    elif mode == "binary":
                        print(f'Rendering {str(self.total + 1 - len(self.render_queue))} / {str(self.total)}: {render_out_file_path_bl} in mode \'binary\'')

                        sc.node_tree.nodes["Alpha Over"].inputs[1].default_value = (0, 0, 0, 1)
                        sc.node_tree.nodes["Brightness/Contrast"].inputs[1].default_value = 50
                        sc.node_tree.nodes["Brightness/Contrast"].inputs[2].default_value = 100

                        render_binary_out_path_bl = os.path.join(
                            self.MASK_LABEL_DIR, render_prefix_cam_frame + "_annot_poly.png"
                        )
                        sc.render.filepath = render_binary_out_path_bl

                        bpy.ops.render.render(write_still=True)

                        polygons = get_mask_polygons_from_binary_image(bpy.path.abspath(render_binary_out_path_bl))
                        draw_polygons(
                            polygons = polygons,
                            img_path = bpy.path.abspath(render_out_file_path_bl),
                            out_path = bpy.path.abspath(render_binary_out_path_bl)
                        )
                        
                        mask_label_out_path =  os.path.join(self.MASK_LABEL_DIR, render_prefix_cam_frame + ".txt")
                        write_polygons_to_yolo(polygons, IMAGE_WIDTH, IMAGE_HEIGHT, mask_label_out_path)

                        reset_render_settings()


        return {'PASS_THROUGH'}


#---------------------------------------------------------------
# Execution context
#---------------------------------------------------------------


def unregister_timed_render():
    bpy.utils.unregister_class(TimedRender)

if __name__ == "__main__":

    test           = False
    use_cam_matrix = True
    render_binary  = True
    ortho          = False

    keypoint_visible_threshold_percentage = 0.1
    contour_threshold = 0.1
    alphashape_alpha  = 2.0
    draw_lattice_for_kpt_annot = False

    SCENE = bpy.context.scene
    RENDER_SCALE = SCENE.render.resolution_percentage / 100
    IMAGE_HEIGHT, IMAGE_WIDTH = (
        SCENE.render.resolution_x * RENDER_SCALE,
        SCENE.render.resolution_y * RENDER_SCALE,
    )

    OUT_DIR_BL   = "//synthetic_data"
    ANNOT_DIR_BL = "//synthetic_data" + os.sep + "annot"

    if test:
        KEYPOINT_LIST = [
            'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'origin'
        ]
        COLLECTION_NAME = "Test"
        OBJECT_NAME = "Cube"

    else:
        KEYPOINT_LIST = [
            'mouth tip', 'gill',
            'root of pelvic fin','caudal peduncle',
            'middle of caudal fin','lower tip of caudal fin'
        ]
        COLLECTION_NAME = "Bluegill"
        OBJECT_NAME = "Body"


    CAM_OBJECTS = bpy.data.collections['Cameras'].objects
    for cam_name in CAM_OBJECTS.keys():
        cam = bpy.data.objects.get(cam_name)
        cam.data.type = 'ORTHO' if ortho else 'PERSP'
        if ortho:
            cam.data.ortho_scale = 30


    bpy.utils.register_class(TimedRender)
    bpy.ops.render.timed_render()