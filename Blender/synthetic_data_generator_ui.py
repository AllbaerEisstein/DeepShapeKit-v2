bl_info = {
    "name": "Synthetic Dataset UI + TimedRender for YOLO Pose & Seg",
    "author": "ChatGPT",
    "version": (0, 4),
    # target Blender 4.5 and backwards-compatible with 2.8+
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Synthetic Data",
    "description": "UI for setting globals and an embedded TimedRender operator that renders frames, writes binary masks and keypoint labels for YOLO datasets.",
    "warning": "Experimental",
    "category": "Import-Export",
}

import csv
import bpy
import bmesh
import os
import json
import shutil
import re
import glob
from collections import defaultdict
from mathutils import Vector, Matrix
from bpy.props import (
    StringProperty, BoolProperty, FloatProperty, IntProperty,
    PointerProperty
)
from bpy.types import Panel, Operator, PropertyGroup
import cv2
import numpy as np


# --------------------------- Globals --------------------------------------

# cache for camera matrices (per-camera)
cam_name_2_matrix = {}

# conversion matrix from Blender camera coordinates to usual CV camera coordinates
BLENDER_CAM_2_CV_CAM = Matrix((
    (1, 0, 0),
    (0, -1, 0),
    (0, 0, -1)
))


# --------------------------- Property Group ---------------------------------

class SYNTH_PropertyGroup(PropertyGroup):
    # Paths
    render_out_dir: StringProperty(
        name="Render Out Dir",
        description="Relative or absolute output directory for rendered images (blender // path supported)",
        subtype='DIR_PATH',
        default="//synthetic_data"
    )

    annot_out_dir: StringProperty(
        name="Annotations Dir",
        description="Base annotation directory",
        subtype='DIR_PATH',
        default="//synthetic_data/annot"
    )

    kpt_label_dir: StringProperty(
        name="Keypoint Label Dir",
        description="Directory for keypoint label files",
        subtype='DIR_PATH',
        default="//synthetic_data/annot/labels_keypoints"
    )

    mask_label_dir: StringProperty(
        name="Mask Label Dir",
        description="Directory for mask label files",
        subtype='DIR_PATH',
        default="//synthetic_data/annot/labels_masks"
    )

    # Scene / render
    render_scale: FloatProperty(
        name="Render Scale",
        description="Render scale (percentage / 100)",
        default=1.0,
        min=0.01,
        max=2.0
    )

    image_width_px: IntProperty(
        name="Image Width (px)",
        description="Rendered image width in pixels",
        default=1024,
        min=1
    )

    image_height_px: IntProperty(
        name="Image Height (px)",
        description="Rendered image height in pixels",
        default=576,
        min=1
    )

    # Objects / keypoints
    collection_name: StringProperty(
        name="Collection",
        description="Name of the collection containing the animated object",
        default="Bluegill"
    )

    object_name: StringProperty(
        name="Object",
        description="Name of the mesh object to sample keypoints from",
        default="Body"
    )

    keypoint_list_csv: StringProperty(
        name="Keypoint List",
        description="Comma separated list of keypoint (vertex group) names",
        default='mouth tip,gill,root of pelvic fin,caudal peduncle,middle of caudal fin,lower tip of caudal fin'
    )

    # Timers and rendering behaviour
    event_timer_interval: FloatProperty(
        name="Timer Interval",
        description="Seconds between render queue checks",
        default=0.35,
        min=0.01
    )

    render_binary: BoolProperty(
        name="Render Binary Masks",
        description="Necessary for mask annotation. Renders each frame a second time but as a binary image. This is either done by using a user-defined compositor or by temporarily overriding materials. Choose by setting 'Use Compositor For Binary Render' option. This render is immediately used to determine the silhoutte.",
        default=True
    )

    use_compositor: BoolProperty(
        name="Use Compositor For Binary Render",
        description="If checked: use a user-defined compositor (requirements: black background, the only non-black pixels should be where the object is).",
        default=True
    )

    create_annotated_images: BoolProperty(
        name="Create Annotated Images",
        description="For every rendered file, create and save keypoint and mask annotations to Keypoint Label Dir and Mask Label Dir.",
        default=True
    )

    check_keypoint_visibility: BoolProperty(
        name="Check Keypoint Visibility",
        default=True
    )

    keypoint_visible_threshold: FloatProperty(
        name="Keypoint Visible Threshold",
        description="Minimum visible area fraction to export a keypoint",
        default=0.1,
        min=0.0,
        max=1.0
    )

    keep_occluded_keypoints: BoolProperty(
        name="Keep Occluded Keypoints",
        description="""
        If checked, keep coordinates of occluded keypoints and set their 'visibility' to 1. This reflects the convention 0=missing, 1=occluded, 2=visible.
        In context of YOLO pose model training/inference:
        Ultralytics models generally treat visibility/confidence as a continuous value, not strictly as discrete 0/1/2 flags. However, in practice, datasets like COCO and hand-keypoints may use such flags for annotation. For training, Ultralytics typically treats both 1 (occluded) and 2 (visible) as present and contributing to loss calculation, while 0 means the keypoint is ignored.
        Visibility information for keypoints in Ultralytics is indicated by the has_visible attribute of the Keypoints class, which tells you if the keypoint data includes a visibility/confidence value. This information is typically stored as the third value in the keypoints tensor (shape [N, K, 3]), where the three elements are (x, y, conf) or (x, y, visibility).
        """,
        default=False
    )

    draw_every_keypoint_vertex: BoolProperty(
        name="Draw Every Visible Keypoint Vertex",
        default=False
    )

    draw_every_keypoint_face: BoolProperty(
        name="Draw Every Visible Keypoint Face",
        default=True
    )

    # Misc
    draw_lattice_for_kpt_annot: BoolProperty(
        name="Draw Lattice On KPT Annot",
        default=False
    )

    create_yolo_datasets: BoolProperty(
        name="Create YOLO Datasets On Finish",
        default=True
    )


# --------------------------- Utilities -------------------------------------

def resolve(path):
    return bpy.path.abspath(path)


# ---- camera intrinsic helpers (from your original script) -----------------

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def get_calibration_matrix_K_Blendercam2Blenderimage(camd, scene):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    f_in_mm = camd.lens
    resolution_x_in_px = scene.render.resolution_x * (scene.render.resolution_percentage / 100.0)
    resolution_y_in_px = scene.render.resolution_y * (scene.render.resolution_percentage / 100.0)
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(camd.sensor_fit, scene.render.pixel_aspect_x * resolution_x_in_px, scene.render.pixel_aspect_y * resolution_y_in_px)
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1.0 / pixel_size_mm_per_px
    s_v = 1.0 / pixel_size_mm_per_px / pixel_aspect_ratio
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0.0
    K = Matrix(((s_u, skew, u_0), (0.0, s_v, v_0), (0.0, 0.0, 1.0)))
    return f_in_mm, K


def get_3x4_RT_matrix_Blender2Blendercam(cam):
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world_2_blcam = rotation.to_matrix().transposed()
    loc_world_2_blcam = -1 * R_world_2_blcam @ location
    Rt = Matrix((
        R_world_2_blcam[0][:] + (loc_world_2_blcam[0],),
        R_world_2_blcam[1][:] + (loc_world_2_blcam[1],),
        R_world_2_blcam[2][:] + (loc_world_2_blcam[2],)
    ))
    return Rt, R_world_2_blcam.to_4x4(), Matrix.Translation(tuple(loc_world_2_blcam))


def get_3x4_P_matrix_Blendercam2Blenderimage(cam, scene):
    f, K = get_calibration_matrix_K_Blendercam2Blenderimage(cam.data, scene)
    Rt, R, T = get_3x4_RT_matrix_Blender2Blendercam(cam)
    return f, K @ Rt, K, R, T, Rt


def export_cam_matrices(context):
    scene = context.scene
    p = scene.synth_props
    out = {}
    cam_collection = bpy.data.collections.get('Cameras')
    cam_objs = cam_collection.objects if cam_collection else [o for o in bpy.data.objects if o.type == 'CAMERA']
    cam_name_2_matrix.clear()
    for cam in cam_objs:
        if cam.type != 'CAMERA':
            continue
        try:
            mats = get_cam_matrix_for_cam(cam, scene)
        except Exception as e:
            raise ValueError(f"Failed to compute matrix for {cam.name}: {e}")
            continue
        def mat_to_list(m):
            return [[float(v) for v in row] for row in m]
        T = mats.get('T')
        if T is None:
            T_list = None
        else:
            try:
                T_list = mat_to_list(T)
            except Exception:
                T_list = None
        P = mats.get('P')
        f = mats.get('f')
        Rt = mats.get('Rt')
        entry = {
            'f': float(f) if f is not None else None,
            'K': mat_to_list(mats['K']),
            'R': mat_to_list(mats['R']),
            'T': T_list,
            'Rt': mat_to_list(Rt),
            'P': mat_to_list(P),
            'camera_name': cam.name
        }
        video_name = cam.name.split('.', 1)[1] + '_' + cam.name.split('.', 1)[0] if '.' in cam.name else cam.name
        out[video_name] = entry
    try:
        out_path = os.path.join(resolve(p.annot_out_dir), 'cam_matrices.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        return out_path
    except Exception as e:
        raise ValueError(f"Failed to write cam_matrices.json: {e}")


# --------------------------- Mesh / Keypoint extraction --------------------

def get_deformed_mesh_data(deps, collection_name, object_name, kpt_list):
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

    faces = []
    for poly in mesh_eval.polygons:
        face_data = {
            "id":   poly.index,
            "area": poly.area,
            "verts": [vid for vid in poly.vertices]
        }
        faces.append(face_data)

    vertices = [
        {
            "id": v.index,
            "co": tuple(v.co)
        }
        for v in mesh_eval.vertices
    ]
    normals  = [(tuple(v.co), tuple(v.normal))   for v in bm.verts]
    bm.free()

    # build a mapping group-name -> list of vertex indices in that group
    kpt_2_verts_objco = {}
    for vg in obj_eval.vertex_groups:
        if vg.name in kpt_list:
            # find all vertices in mesh_eval whose group indices include vg.index
            verts_in_group = [
                vi for vi, v in enumerate(mesh_eval.vertices)
                if any(g.group == vg.index for g in v.groups)
            ]
            kpt_2_verts_objco[vg.name] = verts_in_group

    # now get their coords in world space:
    kpt_2_verts_worldco = {
        kpt: [{
                "id": i, "co": tuple(obj2world @ mesh_eval.vertices[i].co)
            } for i in idx_list
        ]
        for kpt, idx_list in kpt_2_verts_objco.items()
    }

    # Keypoint → world-space faces (strictly associated)
    kpt_2_faces_worldco = {}
    for kpt, idx_list in kpt_2_verts_objco.items():
        verts_set = set(idx_list)
        face_list = []
        for face in faces:
            # strict: all face verts must be in the keypoint's vertices
            if set(face["verts"]).issubset(verts_set):
                # build list of world‐space coords for this face
                coords = [
                    tuple(obj2world @ mesh_eval.vertices[i].co)
                    for i in face["verts"]
                ]
                face_list.append({"coords": coords, "area": face["area"]})
        kpt_2_faces_worldco[kpt] = face_list

    bm.free()
    # docs: The object owns the mesh data-block. To force free it use to_mesh_clear(). 
    obj_eval.to_mesh_clear()
    
    return faces, vertices, normals, kpt_2_verts_worldco, kpt_2_faces_worldco


def get_mesh_json(context):
    collection_name = context.scene.synth_props.collection_name
    object_name = context.scene.synth_props.object_name
    kpt_list = [kp.strip() for kp in context.scene.synth_props.keypoint_list_csv.split(',') if kp.strip()]

    col = bpy.data.collections.get(collection_name)
    if col is None:
        raise ValueError(f"Collection '{collection_name}' not found")
    obj = col.objects.get(object_name)
    if obj is None:
        raise ValueError(f"Object '{object_name}' not found in collection '{collection_name}'")

    # -- find armature modifier & armature object
    arm = None
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE':
            arm = modifier.object
            break
    if arm is None:
        raise ValueError("Saving mesh to json failed. Specified object has no armature modifier.")

    # --- JOINTS & KINTREE (ensure joint indices align with joints list)
    bone_list = list(arm.data.bones)
    roots = [b for b in bone_list if b.parent is None]
    if not roots:
        raise ValueError("No root bones found in armature")

    # BFS traversal to preserve topology order
    bone_names_tree = {b.name: {'p': str, 'c': []} for b in bone_list} # very inefficient way to store a tree
    ordered_bones = []
    queue = roots[:]
    while queue:
        b = queue.pop(0)
        ordered_bones.append(b)
        bone_names_tree[b.name]['p'] = b.parent.name if b.parent is not None else ''
        for ch in b.children:
            queue.append(ch)
            bone_names_tree[b.name]['c'].append(ch.name)


    # head/tail helpers (armature-space local coordinates)
    def head_pos(b): return b.head_local.copy()
    def tail_pos(b): return b.tail_local.copy()

    def key_from_vec(v, prec=6):
        return (round(v.x, prec), round(v.y, prec), round(v.z, prec))

    pos_to_joint_idx = {}
    joint_positions = []
    joint_names = []
    parent_indices = []

    def ensure_joint_at_position(pos, name=None):
        k = key_from_vec(pos)
        if k in pos_to_joint_idx:
            return pos_to_joint_idx[k]
        idx = len(joint_positions)
        pos_to_joint_idx[k] = idx
        joint_positions.append(pos)
        joint_names.append(name or f"joint_{idx}")
        parent_indices.append(-1)
        return idx

    # Create joints (heads & tails), set parent relationships for joints
    for b in ordered_bones:
        hi = ensure_joint_at_position(head_pos(b), name=f"{b.name}_head")
        ti = ensure_joint_at_position(tail_pos(b), name=f"{b.name}_tail")

        # the tail joint's parent is the head joint of the same bone
        parent_indices[ti] = hi

        # the head joint's parent is the tail joint of the parent bone (if parent exists)
        if b.parent is not None:
            p_tail_idx = ensure_joint_at_position(tail_pos(b.parent), name=f"{b.parent.name}_tail")
            if p_tail_idx != hi and parent_indices[hi] == -1:
                parent_indices[hi] = p_tail_idx
        else:
            parent_indices[hi] = -1

    # Now detect missing physical bone connections and add virtual bones
    virtual_bone_2_joint_idx = {}   # vname -> (parent_joint_idx, child_joint_idx)
    virtual_bone_names = []
    for b in ordered_bones:
        if b.parent is None:
            continue
        p_tail_key = key_from_vec(tail_pos(b.parent))
        child_head_key = key_from_vec(head_pos(b))
        if p_tail_key != child_head_key:
            vname = f"virtual_{b.parent.name}_to_{b.name}"
            # in bone-tree, replace entry for original child bone with entry for virtual bone (insert node and edge into tree)
            bone_names_tree[vname] = {'p': b.parent.name, 'c': [b.name]}
            bone_names_tree[b.name]['p'] = vname
            bone_names_tree[b.parent.name]['c'] = [vname if c == b.name else c for c in bone_names_tree[b.parent.name]['c']]

            virtual_bone_names.append(vname)
            p_idx = pos_to_joint_idx.get(p_tail_key)
            c_idx = pos_to_joint_idx.get(child_head_key)
            if p_idx is None:
                p_idx = ensure_joint_at_position(tail_pos(b.parent), name=f"{b.parent.name}_tail")
            if c_idx is None:
                c_idx = ensure_joint_at_position(head_pos(b), name=f"{b.name}_head")
            virtual_bone_2_joint_idx[vname] = (p_idx, c_idx)
    
    # topo-sort the tree
    bone_names_ordered = []
    queue = [node for node, p_c_dict in bone_names_tree.items() if not p_c_dict['p']]
    while queue:
        n = queue.pop(0)
        bone_names_ordered.append(n)
        for ch in bone_names_tree[n]['c']:
            queue.append(ch)

    # Build bone_to_joint mapping: for each exported bone name (real or virtual),
    # assign the joint index that the bone 'controls' (the tail/end joint).
    bone_to_joint = {}
    # real bones => tail joint index
    for b in ordered_bones:
        tail_k = key_from_vec(tail_pos(b))
        bone_to_joint[b.name] = pos_to_joint_idx[tail_k]
    # virtual bones => child-head joint index (stored as c_idx in virtual_bone_map)
    for vname, (p_idx, c_idx) in virtual_bone_2_joint_idx.items():
        bone_to_joint[vname] = c_idx

    # Convert joint positions to lists (object-space coordinates)
    joints = [[float(c) for c in v] for v in joint_positions]
    joint_indices = list(range(len(joints)))
    kintree_unique_joints = [parent_indices, joint_indices]

    # geometry
    verts = [[float(c) for c in v.co] for v in obj.data.vertices]
    faces = [list(p.vertices) for p in obj.data.polygons]

    # weights: include columns for virtual bones (zeros)
    n_verts = len(obj.data.vertices)
    n_bone_groups = len(bone_names_ordered)
    weights = [[0.0]*n_bone_groups for _ in range(n_verts)]
    bone_name_2_index = {name:i for i,name in enumerate(bone_names_ordered)}
    for v in obj.data.vertices:
        for g in v.groups:
            group_name = obj.vertex_groups[g.group].name
            if group_name in bone_name_2_index:
                weights[v.index][bone_name_2_index[group_name]] = float(g.weight)
    # note: virtual bones don't have a vertex group so their weights will remain 0, which is intended

    # -- v2k
    v2k = np.zeros((len(kpt_list), n_verts))
    # for every keypoint...
    for kpt_index, kpt_name in enumerate(kpt_list):
        vertices_for_this_kpt = []
        # loop over every vertex...
        for v_index, v in enumerate(obj.data.vertices):
            # check if the vertex belongs to a keypoint...
            if kpt_name in [obj.vertex_groups[g.group].name for g in v.groups]:
                # if it does, record the index of this vertex
                vertices_for_this_kpt.append(v_index)
        normalized_weight = 1/len(vertices_for_this_kpt) if vertices_for_this_kpt else 0.0
        for vertex_index in vertices_for_this_kpt:
            v2k[kpt_index][vertex_index] = normalized_weight
    
    v2k = [list(keypoint) for keypoint in v2k]

    out = {
        'V': verts,
        'F': faces,
        'J': joints,
        'vert2kpt': v2k,
        'kintree_table': kintree_unique_joints,
        'weights': weights,
        'n_bones': n_bone_groups,
        'kpt_list': kpt_list,
        'bone_order': bone_names_ordered,            # bone order used for export
        'bone_names_tree': bone_names_tree, # a tree-dict of parent-child-relationships of bones
        'bone_to_joint': bone_to_joint,       # maps bone_name -> joint_index it controls (tail joint)
        'virtual_bone_names': virtual_bone_names # for identifying virtual bones
    }

    return out



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


# --------------------------- Visibility / Occlusion ------------------------

def is_vertex_occluded(deps, cam_obj, vertex_co_world, eps=1e-4):
    if deps is None:
        deps = bpy.context.evaluated_depsgraph_get()
    cam_co = cam_obj.matrix_world.translation
    dir_vec = (Vector(vertex_co_world) - cam_co)
    dist_to_pt = dir_vec.length
    if dist_to_pt < eps:
        return False
    dir_vec.normalize()
    origin = cam_co + dir_vec * eps
    hit, hit_loc, _, _, hit_obj, _ = bpy.context.scene.ray_cast(deps, origin, dir_vec)
    if not hit:
        return False
    dist_hit = (hit_loc - origin).length
    return dist_hit < (dist_to_pt - eps)


def get_keypoint_visibility_from_faces(deps, kpt_2_faces_worldco, cam_obj):
    """
    For each keypoint, kpt_2_faces_worldco[kpt] is a list of faces,
    each face is a list of world-space (x,y,z) tuples.

    Returns:
      - kpt_2_visibility_pct: { kpt: visible_area/total_area }
      - kpt_2_visible_faces: { kpt: [ face_coords, … ] } for faces fully visible
    """

    kpt_2_visibility_pct = {}
    kpt_2_visible_faces  = defaultdict(list)

    for kpt, face_list in kpt_2_faces_worldco.items():
        total_area   = 0.0
        visible_area = 0.0

        for face in face_list:
            face_coords = face["coords"]
            # 1) compute this face's area
            #area = _polygon_area_3d(face_coords)
            area = face["area"]
            total_area += area

            # 2) test all vertices for visibility
            all_visible = False
            for coord in face_coords:
                if not is_vertex_occluded(deps, cam_obj, Vector(coord)):
                    all_visible = True
                    break

            if all_visible:
                visible_area += area
                kpt_2_visible_faces[kpt].append(face_coords)

        if total_area > 0:
            kpt_2_visibility_pct[kpt] = visible_area / total_area
        else:
            # keypoint has no associated faces
            kpt_2_visibility_pct[kpt] = 0.0

    return kpt_2_visibility_pct, kpt_2_visible_faces


# --------------------------- Projection helpers ----------------------------

def get_cam_matrix_for_cam(cam_obj, scene):
    """Compute and cache camera matrices for a camera object.
    Returns dict with keys 'f','K','R','T','P' in CV convention (y is down, z is positive camera look-at) where P maps Blender world -> CV image homogeneous coords.
    """
    cam_name = cam_obj.name
    if cam_name in cam_name_2_matrix and cam_name_2_matrix[cam_name].get('P') is not None:
        return cam_name_2_matrix[cam_name]

    f, KRT, K, R, T, Rt = get_3x4_P_matrix_Blendercam2Blenderimage(cam_obj, scene)
    # P: we want mapping from Blender world -> cv image (with y down and positive z forward)
    P = K @ BLENDER_CAM_2_CV_CAM @ Rt
    cam_name_2_matrix[cam_name] = {'f': f, 'K': K, 'R': BLENDER_CAM_2_CV_CAM.to_4x4() @ R, 'T': BLENDER_CAM_2_CV_CAM.to_4x4() @ T, 'P': P, 'Rt': BLENDER_CAM_2_CV_CAM @ Rt}
    # P = K @ Rt
    # cam_name_2_matrix[cam_name] = {'f': f, 'K': K, 'R': R, 'T': T, 'P': P, 'Rt': Rt}
    return cam_name_2_matrix[cam_name]


def project_world_point_with_cam_matrix(P, world_coord):
    ph = P @ Vector((*world_coord, 1.0))
    if abs(ph.z) < 1e-8:
        return None
    return (ph.x / ph.z, ph.y / ph.z)


# --------------------------- Mask & YOLO writers ---------------------------

def render_binary_mask_keep_occluders_black(scene, target_obj, out_path):
    """
    Render a binary mask (white target, black occluders) WITHOUT requiring a prepared compositor.
    - Replaces materials on all mesh objects: target -> white emission, others -> black emission.
    - Renders to out_path, then restores all original materials and scene state.
    - target_obj may be an object or an object name (str).
    """
    # resolve target_obj if user passed a name
    if isinstance(target_obj, str):
        target_obj = bpy.data.objects.get(target_obj)
    if target_obj is None:
        raise ValueError("target_obj not found")

    out_abspath = bpy.path.abspath(out_path)

    # --- create temp materials ------------------------------------------------
    white_em = bpy.data.materials.new(name="SYNTH_tmp_white_emission")
    white_em.use_nodes = True
    ntw = white_em.node_tree
    # clear nodes (be defensive)
    for n in list(ntw.nodes):
        ntw.nodes.remove(n)
    emis = ntw.nodes.new('ShaderNodeEmission')
    emis.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    outm = ntw.nodes.new('ShaderNodeOutputMaterial')
    ntw.links.new(emis.outputs['Emission'], outm.inputs['Surface'])

    black_em = bpy.data.materials.new(name="SYNTH_tmp_black_emission")
    black_em.use_nodes = True
    ntb = black_em.node_tree
    for n in list(ntb.nodes):
        ntb.nodes.remove(n)
    bemis = ntb.nodes.new('ShaderNodeEmission')
    bemis.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    outbm = ntb.nodes.new('ShaderNodeOutputMaterial')
    ntb.links.new(bemis.outputs['Emission'], outbm.inputs['Surface'])

    # --- save state -----------------------------------------------------------
    orig_filepath = scene.render.filepath
    try:
        orig_film_transparent = scene.render.film_transparent
    except Exception:
        orig_film_transparent = False

    # Save original materials per object (list of materials, may be empty)
    orig_materials = {}
    all_mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
    for o in all_mesh_objects:
        orig_materials[o.name] = [slot.material for slot in o.material_slots]

    # Save world and set black background (optional but ensures no stray background)
    orig_world = scene.world
    black_world = None
    try:
        black_world = bpy.data.worlds.new(name="SYNTH_tmp_black_world")
        black_world.use_nodes = True
        # clear nodes
        for n in list(black_world.node_tree.nodes):
            black_world.node_tree.nodes.remove(n)
        bg = black_world.node_tree.nodes.new('ShaderNodeBackground')
        bg.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        outw = black_world.node_tree.nodes.new('ShaderNodeOutputWorld')
        black_world.node_tree.links.new(bg.outputs['Background'], outw.inputs['Surface'])
        scene.world = black_world
    except Exception:
        # if creating a new world fails, keep original world
        black_world = None

    # --- assign materials: target -> white, others -> black --------------------
    try:
        for o in all_mesh_objects:
            # ensure we have material slots to assign into
            if len(o.data.materials) == 0:
                o.data.materials.append(None)  # create 1 slot

            # prepare a clean material slot array
            o.data.materials.clear()

            if o.name == target_obj.name:
                # assign white emission to all slots (1 slot is enough)
                o.data.materials.append(white_em)
            else:
                # assign black emission
                o.data.materials.append(black_em)

        # ensure output dir exists
        os.makedirs(os.path.dirname(out_abspath), exist_ok=True)

        # set render settings and render
        scene.render.filepath = out_abspath
        try:
            scene.render.film_transparent = False
        except Exception:
            pass

        bpy.ops.render.render(write_still=True)

    finally:
        # --- restore materials ------------------------------------------------
        for o in all_mesh_objects:
            orig = orig_materials.get(o.name, [])
            # clear then restore original materials
            try:
                o.data.materials.clear()
            except Exception:
                # fallback: attempt to set each slot if clear is unsupported
                pass
            for m in orig:
                if m is None:
                    # append an empty slot
                    o.data.materials.append(None)
                else:
                    o.data.materials.append(m)

        # restore world, filepath, film transparency
        try:
            scene.world = orig_world
        except Exception:
            pass
        try:
            scene.render.filepath = orig_filepath
        except Exception:
            pass
        try:
            scene.render.film_transparent = orig_film_transparent
        except Exception:
            pass

        # cleanup temporary materials/world if unused
        try:
            if white_em.users == 0:
                bpy.data.materials.remove(white_em, do_unlink=True)
        except Exception:
            pass
        try:
            if black_em.users == 0:
                bpy.data.materials.remove(black_em, do_unlink=True)
        except Exception:
            pass
        try:
            if black_world is not None and black_world.users == 0:
                bpy.data.worlds.remove(black_world, do_unlink=True)
        except Exception:
            pass


def get_mask_polygons_from_binary_image(img_path):
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        polygons.append([tuple(map(int, p)) for p in pts])
    return polygons


def draw_polygons(img_path, out_path, polygons, color=(255,255,255)):
    img = cv2.imread(img_path)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32, ndmin=2).reshape(-1, 1, 2)
        cv2.polylines(img=img, pts=[pts], isClosed=True, color=color, thickness=1)
    cv2.imwrite(out_path, img)


def write_polygons_to_yolo(polygons, image_width, image_height, out_path):
    lines = []
    for idx, polygon in enumerate(polygons):
        norm_pts = []
        for x, y in polygon:
            norm_pts.append(f"{(x / image_width):.3f}")
            norm_pts.append(f"{(y / image_height):.3f}")
        lines.append(f"{idx} " + " ".join(norm_pts))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")



def draw_points_on_img(points, img_path, out_path, annot_radius = 1):
    img = cv2.imread(str(img_path))
    for point in points:
        cv2.circle(img=img, center=(int(point[0]),int(point[1])), radius=annot_radius, color=(255, 255, 255), lineType=-1)
    cv2.imwrite(str(out_path), img)


def draw_kpts_on_img(kpt2coords, kpt_2_vis_status, kpt_2_vis_ptg, img_path, out_path, tenth_of_annot_radius: int = 1):
    """
    Args:
        kpts (List[List[np.ndarray]]): List of the lists of keypoint tuples (x, y, visibility) per detected instance.
    Create an output image with keypoint annotation.
    """
    img = cv2.imread(str(img_path))
    #cmap = create_discrete_color_map(list(kpt2coords))
    annot_radius = tenth_of_annot_radius * 10

    for name, (x,y) in kpt2coords.items():
        if kpt_2_vis_status[name] == 0:
            continue
        conf = kpt_2_vis_ptg[name]
        color = (255, 0, 0) if kpt_2_vis_status[name] == 2 else (255, 0, 216) if kpt_2_vis_status[name] == 1 else (69, 0, 255)
        conf_scaled_annot_radius = int(
            int(conf*10)/10   # will cut off second decimal (e.g 0.72 -> 0.7)
            *annot_radius  # if annot_radius is k*10 with k in N, this will yield an int
        )
        cv2.circle(img=img, center=(int(x),int(y)), radius=1, color=color, lineType=-1) # center
        cv2.circle(img=img, center=(int(x),int(y)), radius=conf_scaled_annot_radius, color=color, lineType=-1)
        cv2.putText(img, f"{int(conf*100)/100}: {name}", (int(x),int(y+conf_scaled_annot_radius+10)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=color, thickness=1)

    cv2.imwrite(str(out_path), img)


def write_pose_labels_yolo(instances, instances_vis_status, kpt_order, image_width, image_height, instances_class_idx, out_path):
    """
    Write YOLO-style pose labels.

    Each line (one instance) is:
      <class-index> <bbox_center_x> <bbox_center_y> <bbox_width> <bbox_height> <px1> <py1> <p1-vis> ... <pxN> <pyN> <pN-vis>

    - x,y and width,height are normalized to [0,1] (image dimensions).
    - Each keypoint triple is: normalized_x normalized_y visibility (0, 1, 2).
    - visibility can have one of three values:
        0: The keypoint is not labeled or is out-of-view (not visible and not labeled).
        1: The keypoint is labeled but not visible (occluded).
        2: The keypoint is labeled and visible (fully visible).
        During training, both 1 (occluded) and 2 (visible) are treated as present and contribute to the loss calculation, 
        while 0 means the keypoint is ignored in training. The model learns to predict keypoint locations and a visibility score.
    - Missing keypoints are written as 0 0 0.
    """
    # basic validation to avoid division by zero
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive numbers")

    lines = []
    for kpt_2_coords, kpt_2_vis_status, class_idx in zip(instances, instances_vis_status, instances_class_idx):
        # collect all present keypoint coordinates to compute bbox
        pts = []
        for kpt in kpt_order:
            if kpt not in kpt_2_coords or kpt not in kpt_2_vis_status:
                continue
            x, y = kpt_2_coords[kpt]
            vis = kpt_2_vis_status[kpt]
            if vis > 0:  # only consider visible or occluded keypoints
                pts.append((x, y))

        if pts:
            xs, ys = zip(*pts)
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            # normalized center and size
            x_ctr = ((xmin + xmax) / 2.0) / float(image_width)
            y_ctr = ((ymin + ymax) / 2.0) / float(image_height)
            w_box = (xmax - xmin) / float(image_width)
            h_box = (ymax - ymin) / float(image_height)
        else:
            # fallback: full image
            x_ctr, y_ctr, w_box, h_box = 0.5, 0.5, 1.0, 1.0

        parts = [
            str(int(class_idx)),
            f"{x_ctr:.6f}",
            f"{y_ctr:.6f}",
            f"{w_box:.6f}",
            f"{h_box:.6f}"
        ]

        # append each keypoint in the fixed order as: x y visibility (all normalized / scaled)
        for kpt in kpt_order:
            if kpt in kpt_2_coords and kpt in kpt_2_vis_status:
                x, y = kpt_2_coords[kpt]
                vis_status = kpt_2_vis_status[kpt]
                parts.append(f"{(x / float(image_width)):.6f}" if vis_status > 0 else "0.000000")
                parts.append(f"{(y / float(image_height)):.6f}" if vis_status > 0 else "0.000000")
                parts.append("2" if vis_status == 2 else "1" if vis_status == 1 else "0")
            else:
                # missing → three zeros
                parts.append("0.000000")
                parts.append("0.000000")
                parts.append("0")

        lines.append(" ".join(parts))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))


# --------------------------- TimedRender Operator ---------------------------

class TimedRender(Operator):
    bl_idname = "render.timed_render"
    bl_label = "Timed Render for Synthetic Dataset"
    bl_description = """INFO: Cancel periodic rendering by pressing ESC.

Automatically creates synthetic training data for YOLO pose and segmentation models. 
A TimedRender Operator is used to periodically render images from a render_queue, 
which includes two entries each for every frame in an animation for every camera in the collection 'cameras'. 
In addition to rendering, the renders get annotated with keypoint and silhouette labels. 
Keypoints are determined by projecting average coordinates of visible keypoints (vertex groups) 
onto the image plane using each camera's camera calibration matrix. 
Silhouette annotation is achieved by rendering a binary image with white just where the animated object is so that 
using opencv's contour detection can create a silhouette annotation from the binary image."""

    render_queue: list | None = None
    timer_event = None
    rendering = False
    cancel_render = False
    total = 0

    def make_prefix_cam_frame(self, cam_name, frame_number):
        prefix_for_cam = cam_name.split('.', 1)[1] + '_' + cam_name.split('.', 1)[0] if '.' in cam_name else cam_name
        return f"{prefix_for_cam}_{str(frame_number).zfill(4)}"

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props
        self.cancel_render = False
        self.rendering = False
        self.render_queue = []

        cam_collection = bpy.data.collections.get('Cameras')
        # TODO: add filtering based on checkboxes or dropdown in ui
        cam_objects = cam_collection.objects if cam_collection else bpy.data.objects
        cam_objects = sorted(cam_objects, key=lambda c: c.name)

        modes = ["regular"] + (["binary"] if p.render_binary else [])

        skipped_count = 0
        for cam in cam_objects:
            if cam.type != 'CAMERA':
                continue
            for frame_index in range(scene.frame_start, scene.frame_end + 1):
                for mode in modes:
                    render_prefix = self.make_prefix_cam_frame(cam.name, frame_index)
                    render_path_os = os.path.join(resolve(p.render_out_dir), render_prefix + ".png")
                    mask_label_path_os = os.path.join(resolve(p.mask_label_dir), render_prefix + ".txt")
                    kpt_label_path_os = os.path.join(resolve(p.kpt_label_dir), render_prefix + ".txt")
                    if (
                            (not os.path.exists(render_path_os))
                            or ((not os.path.exists(mask_label_path_os)) if p.render_binary else True)
                            or (not os.path.exists(kpt_label_path_os))
                        ):
                        self.render_queue.append({
                            'view':                     cam.name,
                            'frame':                    frame_index,
                            'mode':                     mode,
                            'render_prefix_cam_frame':  render_prefix,
                            'render_path_bl':           os.path.join(p.render_out_dir, render_prefix + ".png"),
                            'render_path_os':           render_path_os,
                            'mask_annot_path':          os.path.join(resolve(p.mask_label_dir), render_prefix + ".png"),
                            'kpt_annot_path':           os.path.join(resolve(p.kpt_label_dir),  render_prefix + ".png"),
                            'mask_label_path':          mask_label_path_os,
                            'kpt_label_path':           kpt_label_path_os,
                        })
                    else:
                        skipped_count += 1

        self.total = len(self.render_queue)
        self.report({'INFO'}, f"Queued {self.total} renders (skipped {skipped_count})")

        # add timer
        self.timer_event = context.window_manager.event_timer_add(p.event_timer_interval, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        if self.timer_event:
            try:
                context.window_manager.event_timer_remove(self.timer_event)
            except Exception:
                pass
            self.timer_event = None


    def handle_render_item(self, context, qitem):
        scene = context.scene
        def reset_render_settings():
            scene.node_tree.nodes["Alpha Over"].inputs[1].default_value = (0, 0, 0, 0)
            scene.node_tree.nodes["Brightness/Contrast"].inputs[1].default_value = 0
            scene.node_tree.nodes["Brightness/Contrast"].inputs[2].default_value = 0

        p = scene.synth_props
        kpt_list = [kp.strip() for kp in p.keypoint_list_csv.split(',') if kp.strip()]

        cam_name                 = qitem['view']
        frame_index              = qitem['frame']
        mode                     = qitem['mode']
        render_prefix_cam_frame  = qitem['render_prefix_cam_frame']
        render_out_file_path_bl  = qitem['render_path_bl']
        render_out_file_path_os  = qitem['render_path_os']
        mask_annot_out_file_path = qitem['mask_annot_path']
        kpt_annot_out_file_path  = qitem['kpt_annot_path']
        kpt_label_out_path       = qitem['kpt_label_path']
        mask_label_out_path      = qitem['mask_label_path']


        cam_obj = bpy.data.objects.get(cam_name)
        if not cam_obj:
            self.report({'ERROR'}, f"Camera {cam_name} not found")
            return
        scene.camera = cam_obj
        scene.frame_set(frame_index)

        scene.render.filepath = render_out_file_path_bl
        if p.use_compositor:
            reset_render_settings()
        bpy.ops.render.render(write_still=True)


        try:
            deps = bpy.context.evaluated_depsgraph_get()

            faces, vertices, normals, kpt_2_verts_list_world, kpt_2_faces_list_world = get_deformed_mesh_data(deps, p.collection_name, p.object_name, kpt_list)
            

            kpt_2_visibility_pct, kpt_2_visible_faces = (
                get_keypoint_visibility_from_faces(deps, kpt_2_faces_list_world, cam_obj) 
                if p.check_keypoint_visibility 
                else ({k: 1.0 for k in kpt_2_verts_list_world.keys()}, kpt_2_faces_list_world)
            )

            # no filtering yet
            kpt_2_visible_vert_coords_list_world = {
                kpt: list(set([
                    point 
                    for poly in kpt_2_visible_faces[kpt] 
                    for point in poly
                ]))
                for kpt, vis in kpt_2_visibility_pct.items() 
            }
            kpt_2_coords_list_world = {
                kpt: list(set([
                    point 
                    for poly in kpt_2_faces_list_world[kpt] 
                    for point in poly["coords"]
                ]))
                for kpt, vis in kpt_2_visibility_pct.items() 
            }

            # avg over the visible vertices coords
            kpt_2_avg_coords_world_visible = get_avg_kpt_coords_3d(kpt_2_visible_vert_coords_list_world)
            # avg over all the verts
            kpt_2_avg_coords_world = get_avg_kpt_coords_3d(kpt_2_coords_list_world)

            # compute camera matrix P and project using matrix (keeps parity with original script)
            cam_mats = get_cam_matrix_for_cam(cam_obj, scene)
            P = cam_mats['P']

            EPS = 1e-8
            img_w = int(scene.render.resolution_x * (scene.render.resolution_percentage / 100.0))
            img_h = int(scene.render.resolution_y * (scene.render.resolution_percentage / 100.0))
            def is_outside_image_bounds(x, y):
                return x < EPS or x > (img_w-EPS) or y < EPS or y > (img_h-EPS)
            
            def project_world_keypoints(kpt_2_coords):
                kpt_2_projected = {}
                for kpt, coords in kpt_2_coords.items():
                    ph = P @ Vector((*coords, 1.0))
                    # require a meaningful positive depth (z)
                    if not (ph.z > EPS):
                        # either skip or explicitly mark as missing; write_pose_labels_yolo expects missing keys possible
                        # skipping is fine
                        continue
                    x_img = ph.x / ph.z
                    y_img = ph.y / ph.z
                    kpt_2_projected[kpt] = (x_img, y_img)
                return kpt_2_projected
                
            kpt_2_coords_image_filtered_by_vis = project_world_keypoints(kpt_2_avg_coords_world_visible)
            kpt_2_coords_image_all_faces_count = project_world_keypoints(kpt_2_avg_coords_world)

            # 0: The keypoint is not labeled or is out-of-view (not visible and not labeled).
            # 1: The keypoint is labeled but not visible (occluded).
            # 2: The keypoint is labeled and visible (fully visible).
            occluded_status = 1 if p.keep_occluded_keypoints else 0 # status 1 may only occur if the flag is set
            kpt_2_vis_status = {
                kpt: 
                    0 if is_outside_image_bounds(*coords)
                    else 
                    occluded_status if kpt_2_visibility_pct[kpt] < p.keypoint_visible_threshold
                    else 
                    2
                for kpt, coords in kpt_2_coords_image_filtered_by_vis.items() 
            }
            for kpt in [kp.strip() for kp in context.scene.synth_props.keypoint_list_csv.split(',') if kp.strip()]:
                if kpt not in kpt_2_vis_status:
                    kpt_2_vis_status[kpt] = occluded_status
                    
            # if the keypoint is occluded, its coordinates are considered to be the center of all its vertices, not only the visible ones (because there are possible none visible)
            for kpt, status in kpt_2_vis_status.items():
                if status == 1:
                    kpt_2_coords_image_filtered_by_vis[kpt] = kpt_2_coords_image_all_faces_count[kpt]
                elif status == 0:
                    kpt_2_coords_image_filtered_by_vis[kpt] = (0,0)
            

            img_annot_source_file_path = render_out_file_path_os

            if p.create_annotated_images and p.draw_every_keypoint_vertex:
                visible_verts = []
                for vertex_list in kpt_2_verts_list_world.values():
                    for vertex in vertex_list:
                        if (not is_vertex_occluded(deps, cam_obj, Vector(vertex["co"])) if p.check_keypoint_visibility else True):
                            vertex_bl_cam = P @ Vector(tuple(vertex["co"]) + (1,))
                            visible_verts.append((vertex_bl_cam.x / vertex_bl_cam.z, vertex_bl_cam.y / vertex_bl_cam.z))
                draw_points_on_img(visible_verts, img_annot_source_file_path, kpt_annot_out_file_path)
                img_annot_source_file_path = kpt_annot_out_file_path

            
            if p.create_annotated_images and p.draw_every_keypoint_face:
                visible_faces = []
                for face_list in kpt_2_visible_faces.values():
                    for face in face_list:
                        projected_face = []
                        for vertex_world in face:
                            ph = P @ Vector((*vertex_world, 1))
                            projected_face.append((ph.x / ph.z, ph.y / ph.z))
                        visible_faces.append(projected_face)
                draw_polygons(img_annot_source_file_path, kpt_annot_out_file_path, visible_faces)
                img_annot_source_file_path = kpt_annot_out_file_path

            if p.create_annotated_images:
                draw_kpts_on_img(
                    kpt_2_coords_image_filtered_by_vis, 
                    kpt_2_vis_status, 
                    kpt_2_visibility_pct, 
                    img_annot_source_file_path, 
                    kpt_annot_out_file_path
                )
                
            write_pose_labels_yolo(
                [kpt_2_coords_image_filtered_by_vis],
                [kpt_2_vis_status],
                kpt_list, 
                int(p.image_width_px), 
                int(p.image_height_px), 
                [0], 
                kpt_label_out_path
            )

        except Exception as e:
            self.report({'WARNING'}, f"Keypoint generation failed: {e}")

        if p.render_binary and mode == 'binary':
            if p.use_compositor:
                scene.node_tree.nodes["Alpha Over"].inputs[1].default_value = (0, 0, 0, 1)
                scene.node_tree.nodes["Brightness/Contrast"].inputs[1].default_value = 50
                scene.node_tree.nodes["Brightness/Contrast"].inputs[2].default_value = 100
                scene.render.filepath = mask_annot_out_file_path
                bpy.ops.render.render(write_still=True)
            else:
                render_binary_mask_keep_occluders_black(scene,
                                            bpy.data.objects.get(p.object_name),
                                            mask_annot_out_file_path)
            if os.path.exists(mask_annot_out_file_path):
                polygons = get_mask_polygons_from_binary_image(mask_annot_out_file_path)
                if p.create_annotated_images:
                    draw_polygons(render_out_file_path_os, mask_annot_out_file_path, polygons)
                write_polygons_to_yolo(polygons, int(p.image_width_px), int(p.image_height_px), mask_label_out_path)
            else:
                try:
                    img = cv2.imread(render_out_file_path_os)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                        tmp_mask = os.path.join(resolve(p.mask_label_dir), render_prefix_cam_frame + "_tmp_mask.png")
                        os.makedirs(os.path.dirname(tmp_mask), exist_ok=True)
                        cv2.imwrite(tmp_mask, thresh)
                        polygons = get_mask_polygons_from_binary_image(tmp_mask)
                        write_polygons_to_yolo(polygons, int(p.image_width_px), int(p.image_height_px), mask_label_out_path)
                        os.remove(tmp_mask)
                except Exception as e:
                    self.report({'WARNING'}, f"Mask extraction failed: {e}")


    def modal(self, context, event):
        if event.type == 'ESC':
            self.cancel_render = True
            self.cleanup(context)
            self.report({'INFO'}, 'Render cancelled by user')
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if not self.render_queue or len(self.render_queue) == 0 or self.cancel_render:
                self.cleanup(context)
                if context.scene.synth_props.create_yolo_datasets:
                    try:
                        create_yolo_dataset(
                            imgs_dir=resolve(context.scene.synth_props.render_out_dir),
                            label_dir=context.scene.synth_props.kpt_label_dir,
                            dataset_name="keypoint_dataset_yolo",
                            train_pct=0.8, test_pct=0.15, val_pct=0.05,
                            class_list=["fish"],
                            kpt_list=[kp.strip() for kp in context.scene.synth_props.keypoint_list_csv.split(',') if kp.strip()]
                        )
                        self.report({'INFO'}, f"Created YOLO pose estimation training dataset at {resolve(context.scene.synth_props.render_out_dir)}")
                    except Exception as e:
                       self.report({'WARNING'}, f"YOLO keypoint dataset creation failed: {e}")
                    try:
                        create_yolo_dataset(
                            imgs_dir=resolve(context.scene.synth_props.render_out_dir),
                            label_dir=context.scene.synth_props.mask_label_dir,
                            dataset_name="mask_dataset_yolo",
                            train_pct=0.8, test_pct=0.15, val_pct=0.05,
                            class_list=["fish"],
                        )
                        self.report({'INFO'}, f"Created YOLO mask segmentation training dataset at {resolve(context.scene.synth_props.render_out_dir)}")
                    except Exception as e:
                       self.report({'WARNING'}, f"YOLO mask dataset creation failed: {e}")
                self.report({'INFO'}, 'TimedRender finished')
                return {'FINISHED'}

            if not self.rendering:
                qitem = self.render_queue.pop(0)
                try:
                    self.handle_render_item(context, qitem)
                except Exception as e:
                    self.report({'WARNING'}, f"Render failed for item: {e}")

        return {'PASS_THROUGH'}


# --------------------------- YOLO helpers (create dataset) ------------------

def get_available_dir_name(imgs_dir, base_name):
    candidate = base_name
    counter = 1
    while os.path.exists(os.path.join(imgs_dir, candidate)):
        candidate = f"{base_name}_{counter:02d}"
        counter += 1
    return candidate


def create_directory_structure(imgs_dir, dataset_name):
    os.makedirs(os.path.join(imgs_dir, dataset_name), exist_ok=True)
    for folder in ["images", "labels"]:
        for subset in ["train", "test", "val"]:
            os.makedirs(os.path.join(imgs_dir, dataset_name, folder, subset), exist_ok=True)


def split_labels(label_dir, train_pct, test_pct, val_pct):
    all_files = os.listdir(label_dir)
    label_files = [f for f in all_files if os.path.isfile(os.path.join(label_dir, f)) and f.lower().endswith('.txt')]
    np.random.shuffle(label_files)
    total = len(label_files)
    num_train = int(total * train_pct)
    num_test = int(total * test_pct)
    train_labels = label_files[:num_train]
    test_labels = label_files[num_train:num_train+num_test]
    val_labels = label_files[num_train+num_test:]
    subsets = {'train': set(os.path.splitext(f)[0] for f in train_labels), 'test': set(os.path.splitext(f)[0] for f in test_labels), 'val': set(os.path.splitext(f)[0] for f in val_labels)}
    return train_labels, test_labels, val_labels, subsets


def move_label_files(label_dir, dataset_dir, train_labels, test_labels, val_labels):
    for subset, file_list in zip(["train", "test", "val"], [train_labels, test_labels, val_labels]):
        for filename in file_list:
            src = os.path.join(label_dir, filename)
            dst = os.path.join(dataset_dir, "labels", subset, filename)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def process_images(imgs_dir, subsets, dataset_name):
    all_files = os.listdir(imgs_dir)
    image_files = [f for f in all_files if os.path.isfile(os.path.join(imgs_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img in image_files:
        basename, _ = os.path.splitext(img)
        dest_subset = None
        if basename in subsets['train']:
            dest_subset = 'train'
        elif basename in subsets['test']:
            dest_subset = 'test'
        elif basename in subsets['val']:
            dest_subset = 'val'
        src = os.path.join(imgs_dir, img)
        if dest_subset:
            dst_dir = os.path.join(imgs_dir, dataset_name, "images", dest_subset)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, img))
        else:
            os.remove(src)


def create_dataset_yaml(class_list, dataset_path, kpt_list=None):
    lines = ["train: images/train", "val:   images/val", "test:  images/test"]
    if kpt_list is not None:
        lines.append(f"kpt_shape: [{len(kpt_list)}, 3]")
        lines.append(f"flip_idx: {list(range(len(kpt_list)))}")
    lines.append("names:")
    for idx, name in enumerate(class_list):
        lines.append(f"  {idx}: {name}")
    content = "\n".join(lines) + "\n"
    os.makedirs(dataset_path, exist_ok=True)
    yaml_path = os.path.join(dataset_path, os.path.basename(dataset_path) + ".yaml")
    with open(yaml_path, 'w') as f:
        f.write(content)


def create_yolo_dataset(imgs_dir, label_dir, dataset_name, train_pct, test_pct, val_pct, class_list, kpt_list=None):
    if abs((train_pct+test_pct+val_pct)-1.0) > 1e-6:
        raise ValueError('Train/Test/Val percentages must sum to 1')
    if not os.path.isdir(imgs_dir) or not os.path.isdir(label_dir):
        raise FileNotFoundError('Invalid images or label directory')
    dataset_name = get_available_dir_name(imgs_dir, dataset_name)
    create_directory_structure(imgs_dir, dataset_name)
    train_labels, test_labels, val_labels, subsets = split_labels(label_dir, train_pct, test_pct, val_pct)
    move_label_files(label_dir, os.path.join(imgs_dir, dataset_name), train_labels, test_labels, val_labels)
    process_images(imgs_dir, subsets, dataset_name)
    create_dataset_yaml(class_list, os.path.join(imgs_dir, dataset_name), kpt_list)


# --------------------------- UI Operators & Panel --------------------------
class SYNTH_OT_apply_settings(Operator):
    bl_idname = "synth.apply_settings"
    bl_label = "Apply And Save Settings"
    bl_description = "Apply settings, save them to a synth_config.json file in the annotation directory and, if not there already, create the expected (empty) directory structure at the render out dir"

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props
        try:
            scene.render.resolution_x = int(p.image_width_px)
            scene.render.resolution_y = int(p.image_height_px)
            scene.render.resolution_percentage = int(p.render_scale * 100)
        except Exception as e:
            self.report({'WARNING'}, f"Failed to set render resolution/scale: {e}")
        for path_prop in [p.render_out_dir, p.annot_out_dir, p.kpt_label_dir, p.mask_label_dir]:
            try:
                os.makedirs(resolve(path_prop), exist_ok=True)
            except Exception as e:
                self.report({'WARNING'}, f"Could not create path {path_prop}: {e}")
        # invalidate camera matrix cache so K/R/T/P will be recomputed with new settings
        cam_name_2_matrix.clear()
        cfg = {
            'RENDER_OUT_DIR_BL': p.render_out_dir,
            'ANNOT_OUT_DIR_BL': p.annot_out_dir,
            'KPT_LABEL_DIR': p.kpt_label_dir,
            'MASK_LABEL_DIR': p.mask_label_dir,
            'KEYPOINT_LIST': [kp.strip() for kp in p.keypoint_list_csv.split(',') if kp.strip()],
            'COLLECTION_NAME': p.collection_name,
            'OBJECT_NAME': p.object_name,
            'EVENT_TIMER_INTERVAL': p.event_timer_interval,
            'render_binary': p.render_binary,
            'use_compositor': p.use_compositor,
            'create_annotated_images': p.create_annotated_images,
            'check_keypoint_visibility': p.check_keypoint_visibility,
            'KEYPOINT_VISIBLE_THRESHOLD': p.keypoint_visible_threshold,
            'draw_every_keypoint_vertex': p.draw_every_keypoint_vertex,
            'keep_occluded_keypoints': p.keep_occluded_keypoints,
            'draw_every_keypoint_face': p.draw_every_keypoint_face,
            'draw_lattice_for_kpt_annot': p.draw_lattice_for_kpt_annot,
            'create_yolo_datasets': p.create_yolo_datasets,
        }
        try:
            cfg_path = os.path.join(resolve(p.annot_out_dir), 'synth_config.json')
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=4)
            self.report({'INFO'}, f"Wrote config to {cfg_path}")
        except Exception as e:
            self.report({'WARNING'}, f"Could not write config json: {e}")
        return {'FINISHED'}


class SYNTH_OT_load_config(Operator):
    bl_idname = "synth.load_config"
    bl_label = "Load Config"
    bl_description = "Select a synth_config.json file and apply its settings to the UI and scene"

    filepath: StringProperty(subtype='FILE_PATH', default="")

    def invoke(self, context, event):
        # show the file selector
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props

        path = bpy.path.abspath(self.filepath)
        if not os.path.isfile(path):
            self.report({'ERROR'}, f"Config file not found: {path}")
            return {'CANCELLED'}

        try:
            with open(path, 'r') as f:
                cfg = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read config: {e}")
            return {'CANCELLED'}

        # helper to set prop if present
        def safe_set_prop(prop_name, value):
            if hasattr(p, prop_name):
                try:
                    setattr(p, prop_name, value)
                except Exception as e:
                    self.report({'WARNING'}, f"Could not set {prop_name}: {e}")
            else:
                # silently ignore unknown entries (backwards compatibility)
                pass

        # mapping of config keys -> synth_props names
        key_map = {
            'RENDER_OUT_DIR_BL': 'render_out_dir',
            'ANNOT_OUT_DIR_BL': 'annot_out_dir',
            'KPT_LABEL_DIR': 'kpt_label_dir',
            'MASK_LABEL_DIR': 'mask_label_dir',
            'COLLECTION_NAME': 'collection_name',
            'OBJECT_NAME': 'object_name',
            'EVENT_TIMER_INTERVAL': 'event_timer_interval',
            'render_binary': 'render_binary',
            'use_compositor': 'use_compositor',
            'create_annotated_images': 'create_annotated_images',
            'check_keypoint_visibility': 'check_keypoint_visibility',
            'KEYPOINT_VISIBLE_THRESHOLD': 'keypoint_visible_threshold',
            'draw_every_keypoint_vertex': 'draw_every_keypoint_vertex',
            'keep_occluded_keypoints':'keep_occluded_keypoints',
            'draw_every_keypoint_face': 'draw_every_keypoint_face',
            'draw_lattice_for_kpt_annot': 'draw_lattice_for_kpt_annot',
            'create_yolo_datasets': 'create_yolo_datasets',
        }

        # apply mapped simple scalar/bool/string values
        for ck, prop_name in key_map.items():
            if ck in cfg:
                safe_set_prop(prop_name, cfg[ck])

        # KEYPOINT_LIST (list -> csv string)
        if 'KEYPOINT_LIST' in cfg:
            try:
                kp_list = cfg['KEYPOINT_LIST']
                if isinstance(kp_list, (list, tuple)):
                    safe_set_prop('keypoint_list_csv', ','.join([str(k) for k in kp_list]))
                else:
                    safe_set_prop('keypoint_list_csv', str(kp_list))
            except Exception as e:
                self.report({'WARNING'}, f"Could not set keypoint list: {e}")

        # If the config included image size/scale keys, apply them to the scene.
        # Some older configs may not contain these keys; only set if present.
        if 'IMAGE_WIDTH' in cfg:
            try:
                scene.render.resolution_x = int(cfg['IMAGE_WIDTH'])
                safe_set_prop('image_width_px', int(cfg['IMAGE_WIDTH']))
            except Exception as e:
                self.report({'WARNING'}, f"Could not set IMAGE_WIDTH: {e}")
        if 'IMAGE_HEIGHT' in cfg:
            try:
                scene.render.resolution_y = int(cfg['IMAGE_HEIGHT'])
                safe_set_prop('image_height_px', int(cfg['IMAGE_HEIGHT']))
            except Exception as e:
                self.report({'WARNING'}, f"Could not set IMAGE_HEIGHT: {e}")
        if 'RENDER_SCALE' in cfg:
            try:
                scene.render.resolution_percentage = int(float(cfg['RENDER_SCALE']) * 100)
                safe_set_prop('render_scale', float(cfg['RENDER_SCALE']))
            except Exception as e:
                self.report({'WARNING'}, f"Could not set RENDER_SCALE: {e}")

        # create directories referenced by config (be permissive)
        for dirkey in ('render_out_dir', 'annot_out_dir', 'kpt_label_dir', 'mask_label_dir'):
            val = getattr(p, dirkey, None)
            if val:
                try:
                    os.makedirs(resolve(val), exist_ok=True)
                except Exception as e:
                    self.report({'WARNING'}, f"Could not create {dirkey} dir {val}: {e}")

        # invalidate camera matrix cache so matrices are recomputed with new settings
        try:
            cam_name_2_matrix.clear()
        except Exception:
            pass

        self.report({'INFO'}, f"Loaded config from {path}")
        return {'FINISHED'}


class SYNTH_OT_export_keypoint_list(Operator):
    bl_idname = "synth.export_keypoint_list"
    bl_label = "Export Keypoint List"

    def execute(self, context):
        p = context.scene.synth_props
        kp_list = [kp.strip() for kp in p.keypoint_list_csv.split(',') if kp.strip()]
        try:
            out_file = os.path.join(resolve(p.annot_out_dir), 'keypoint_list.csv')
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w') as f:
                f.write(",".join(kp_list))
            self.report({'INFO'}, f"Wrote {len(kp_list)} keypoints to {out_file}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to write keypoint list: {e}")
        return {'FINISHED'}


class SYNTH_PT_main_panel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Synthetic Data'
    bl_label = 'Synthetic Data Generator'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        p = scene.synth_props
        box = layout.box()
        box.label(text="Output Paths")
        box.prop(p, 'render_out_dir')
        box.prop(p, 'annot_out_dir')
        box.prop(p, 'kpt_label_dir')
        box.prop(p, 'mask_label_dir')
        box = layout.box()
        box.label(text="Render / Image")
        box.prop(p, 'render_scale')
        row = box.row(align=True)
        row.prop(p, 'image_width_px')
        row.prop(p, 'image_height_px')
        box = layout.box()
        box.label(text="Target Object & Keypoints")
        box.prop(p, 'collection_name')
        box.prop(p, 'object_name')
        box.prop(p, 'keypoint_list_csv')
        box.operator('synth.export_keypoint_list', icon='EXPORT')
        box = layout.box()
        box.label(text="Behaviour")
        box.prop(p, 'render_binary')
        box.prop(p, 'use_compositor')
        box = layout.box()
        box.label(text="Keypoint Options")
        box.prop(p, 'check_keypoint_visibility')
        box.prop(p, 'keypoint_visible_threshold')
        box.prop(p, 'keep_occluded_keypoints')
        box.prop(p, 'draw_every_keypoint_vertex')
        box.prop(p, 'draw_every_keypoint_face')
        box.prop(p, 'draw_lattice_for_kpt_annot')
        box = layout.box()
        box.label(text="Misc")
        box.prop(p, 'create_annotated_images')
        box.prop(p, 'create_yolo_datasets')
        box.prop(p, 'event_timer_interval')
        row = layout.row()
        row.operator('synth.apply_settings', icon='CHECKMARK')
        row.operator('synth.load_config', icon='IMPORT')
        row = layout.row()
        row.operator('render.timed_render', icon='RENDER_STILL')
        row.operator('synth.unregister_timed_render', icon='CANCEL')
        row.operator('synth.create_videos', icon='SEQUENCE')
        row.operator('synth.export_camera_matrices', icon='FILE_FOLDER')
        row.operator('synth.export_mesh', icon='FILE_FOLDER')
        row.operator('synth.export_pose_time_series_json', icon='SEQUENCE')



class SYNTH_OT_unregister_timed_render(Operator):
    bl_idname = "synth.unregister_timed_render"
    bl_label = "Unregister TimedRender"

    def execute(self, context):
        try:
            bpy.utils.unregister_class(TimedRender)
            self.report({'INFO'}, 'Unregistered TimedRender')
            return {'FINISHED'}
        except Exception as e:
            self.report({'WARNING'}, f'Failed to unregister TimedRender: {e}')
            return {'CANCELLED'}



# --------------------------- Camera matrices export operator ----------------

class SYNTH_OT_export_camera_matrices(Operator):
    bl_idname = "synth.export_camera_matrices"
    bl_label = "Export Camera Matrices"
    bl_description = "Export computed camera parameters & matrices (f, K, R, T, P, Rt) for [Blender world -> CV image]-conversion for all scene cameras to cam_matrices.json in the annotation folder. -- NOTE -- K expects coordinates in CV-convention; y is down, z is positive camera look-at (forward), x is right -- P, R, T, Rt expect coordinates in Blender world convention: z is up, y is forward, x is right"

    def execute(self, context):
        try:
            path_matrices_were_saved_to = export_cam_matrices(context)
            self.report({'INFO'}, f"Wrote camera parameters & matrices to {path_matrices_were_saved_to}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'WARNING'}, f"Failed to export camera matrices: {e}")
            return {'CANCELLED'}


# --------------------------- Mesh export operator ----------------

class SYNTH_OT_export_mesh(Operator):
    bl_idname = "synth.export_mesh"
    bl_label = "Export Mesh"
    bl_description = "Export mesh + armature weights & joints, and keypoints (all in local/model coordinates) to JSON"

    def execute(self, context):
        synth_props = context.scene.synth_props
        out_dir = synth_props.annot_out_dir
        collection_name = synth_props.collection_name
        object_name = synth_props.object_name
        try:
            out = get_mesh_json(context)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{collection_name}_{object_name}_mesh.json")
            with open(out_path, 'wt') as f:
                json.dump(out, f, indent=2)
            self.report({'INFO'}, f"Saved mesh JSON to {out_path}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'WARNING'}, f"Failed to export mesh: {e}")
            return {'CANCELLED'}
        

# --------------------------- Create Videos Operator --------------------------

class SYNTH_OT_create_videos(Operator):
    bl_idname = "synth.create_videos"
    bl_label = "Create Videos from Renders"
    bl_description = "Create one MP4 video per camera from rendered frames in render_out_dir"

    def _prefix_for_cam(self, cam_name: str) -> str:
        return cam_name.split('.', 1)[1] + '_' + cam_name.split('.', 1)[0] if '.' in cam_name else cam_name

    def _find_frames_for_prefix(self, folder: str, prefix: str):
        """
        Return sorted list of tuples (frame_int, filepath) for files in folder that match prefix_{frame}.{ext}.
        Accepts .png/.jpg/.jpeg (case-insensitive). Returns empty list if none found.
        """
        candidates = []
        # search for common image extensions
        for ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
            pattern = os.path.join(folder, f"{prefix}_*.{ext}")
            for fp in glob.glob(pattern):
                base = os.path.basename(fp)
                # look for trailing _<digits>.<ext>
                m = re.search(r'_(\d+)\.[^.]+$', base)
                if not m:
                    continue
                frame_str = m.group(1)
                try:
                    frame_i = int(frame_str)
                except Exception:
                    continue
                candidates.append((frame_i, fp))
        # sort by frame number
        candidates.sort(key=lambda x: x[0])
        return candidates

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props

        # Resolve output folder
        render_dir = resolve(p.render_out_dir)  # use your resolve helper to expand // paths
        if not os.path.isdir(render_dir):
            self.report({'ERROR'}, f"Render out dir not found: {render_dir}")
            return {'CANCELLED'}

        # Determine camera list: prefer Cameras collection if present
        cam_collection = bpy.data.collections.get('Cameras')
        cam_objs = cam_collection.objects if cam_collection else [o for o in bpy.data.objects if o.type == 'CAMERA']

        if not cam_objs:
            self.report({'WARNING'}, "No cameras found in scene.")
            return {'CANCELLED'}

        # Determine desired fps from Blender scene
        try:
            fps = scene.render.fps / scene.render.fps_base
        except Exception:
            fps = float(scene.render.fps)  # fallback

        # target folder for videos (use render_dir itself)
        out_folder = render_dir
        os.makedirs(out_folder, exist_ok=True)

        videos_created = 0
        cameras_skipped_no_frames = 0
        failed = []

        for cam in sorted(cam_objs, key=lambda c: c.name):
            if getattr(cam, "type", None) != 'CAMERA':
                continue

            prefix = self._prefix_for_cam(cam.name)
            frames = self._find_frames_for_prefix(render_dir, prefix)

            if not frames:
                cameras_skipped_no_frames += 1
                continue

            # frames is list of (frame_int, filepath), sorted
            frame_nums = [f for f, _ in frames]
            min_frame, max_frame = frame_nums[0], frame_nums[-1]

            # choose output name: prefix.mp4 if full range present, else prefix_min-max.mp4
            expected_frames_count = scene.frame_end - scene.frame_start + 1
            has_full_range = (min_frame == scene.frame_start and max_frame == scene.frame_end and len(frame_nums) == expected_frames_count)

            if has_full_range:
                out_name = f"{prefix}.mp4"
            else:
                out_name = f"{prefix}_{min_frame}-{max_frame}.mp4"

            out_path = os.path.join(out_folder, out_name)

            # read first image to get frame size (width,height)
            first_img_path = frames[0][1]
            img0 = cv2.imread(first_img_path)
            if img0 is None:
                failed.append((prefix, "Could not read first image"))
                continue
            h, w = img0.shape[:2]
            # ensure integer fps for VideoWriter; VideoWriter accepts float fps but some backends prefer ints
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            try:
                writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
            except Exception as e:
                failed.append((prefix, f"Failed to create VideoWriter: {e}"))
                continue

            # write frames in order; if any frame differs in size, resize to first image size
            try:
                for frame_i, fp in frames:
                    img = cv2.imread(fp)
                    if img is None:
                        # skip missing/unreadable frames but report
                        self.report({'WARNING'}, f"Skipping unreadable frame {fp} for {prefix}")
                        continue
                    if img.shape[0] != h or img.shape[1] != w:
                        # resize to first image size
                        img = cv2.resize(img, (w, h))
                    writer.write(img)
                writer.release()
                videos_created += 1
                self.report({'INFO'}, f"Wrote video: {out_path}")
            except Exception as e:
                try:
                    writer.release()
                except Exception:
                    pass
                failed.append((prefix, str(e)))
                continue

        summary = f"Created {videos_created} videos"
        if cameras_skipped_no_frames:
            summary += f", skipped {cameras_skipped_no_frames} cameras with no frames"
        if failed:
            summary += f", {len(failed)} failures"
            for (cam_pref, msg) in failed:
                self.report({'WARNING'}, f"{cam_pref}: {msg}")

        self.report({'INFO'}, summary)
        return {'FINISHED'}


# --------------------------- Export Pose Time Series Operator --------------------------


class SYNTH_OT_export_pose_time_series_json(Operator):
    bl_idname = "synth.export_pose_time_series_json"
    bl_label = "Export Pose Time Series (JSON)"
    bl_description = "Export per-frame root position, bone relative rotations in exponential map and bone length scalings to a JSON file"

    def _find_armature_for_scene(self, context):
        p = context.scene.synth_props
        try:
            col = bpy.data.collections.get(p.collection_name)
            if col:
                obj = col.objects.get(p.object_name)
                if obj:
                    for mod in obj.modifiers:
                        if mod.type == 'ARMATURE' and mod.object:
                            return mod.object
        except Exception:
            return None

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props

        arm_obj = self._find_armature_for_scene(context)
        if arm_obj is None:
            self.report({'ERROR'}, "No armature found (checked modifier on target object).")
            return {'CANCELLED'}

        # Use get_mesh_json to derive bone order, joint list and maps
        mesh_info = get_mesh_json(context)
        bone_names_ordered = mesh_info['bone_order']            # includes virtual bones appended
        bone_names_tree = mesh_info['bone_names_tree']
        virtual_bone_names = mesh_info['virtual_bone_names']
        bone_to_joint = mesh_info['bone_to_joint']     # bone_name -> joint_index
        joints = mesh_info['J']                        # list of joint positions (object-space)
        # compute rest lengths from joint positions: for bone -> (head_joint, tail_joint)
        # For real bones, tail_joint is bone_to_joint[b], head_joint = parent of that tail (kintree parent)
        parent_indices = mesh_info['kintree_table'][0]

        # Build a quick map joint_index -> parent_joint_index (from kintree)
        joint_to_parent = {i: parent_indices[i] for i in range(len(parent_indices))}

        # rest lengths per bone (real + virtual)
        bone_name_2_rest_length = {}
        # we need head joint index for each bone:
        for bname in bone_names_ordered:
            tail_j = bone_to_joint.get(bname)
            head_j = joint_to_parent.get(tail_j, -1)
            head_pos = Vector(joints[head_j])
            tail_pos = Vector(joints[tail_j])
            dist = float((tail_pos - head_pos).length)
            bone_name_2_rest_length[bname] = dist if dist > 0.0 else 1.0

        # Prepare path and timing
        out_dir = resolve(p.annot_out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"pose_time_series_{p.collection_name}_{p.object_name}.json")

        deps = bpy.context.evaluated_depsgraph_get()
        frame_start = scene.frame_start
        frame_end = scene.frame_end
        try:
            fps = float(scene.render.fps) / float(scene.render.fps_base)
        except Exception:
            fps = float(scene.render.fps)

        data = {
            "meta": {
                "armature": arm_obj.name,
                "bone_order": bone_names_ordered,
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "fps": float(fps),
                "axis_order": "XYZ (Blender world coordinate convention)",
                "units": "meters (world space)"
            },
            "frames": []
        }

        # iterate frames
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            deps.update()
            arm_eval = arm_obj.evaluated_get(deps)
            pose_bones = arm_eval.pose.bones

            # root: first bone in bone_names must be a real bone (it was originally from ordered bones)
            first_bone = bone_names_ordered[0]
            if first_bone not in pose_bones:
                self.report({'ERROR'}, f"Root bone '{first_bone}' missing in pose at frame {frame}")
                return {'CANCELLED'}

            pb_root = pose_bones[first_bone]
            # root global translation: use root bone head in world space
            root_bone_translation_world = arm_eval.matrix_world @ Vector(pb_root.head)
            root_bone_rot_matrix_world = arm_eval.matrix_world @ pb_root.matrix
            root_ori_world = root_bone_rot_matrix_world.to_3x3().to_quaternion().to_exponential_map()

            global_t = [float(root_bone_translation_world.x), float(root_bone_translation_world.y), float(root_bone_translation_world.z)]
            global_ori = [float(root_ori_world.x), float(root_ori_world.y), float(root_ori_world.z)]

            body_pose = []
            body_bone_length = []

            # iterate bone_names in order, skipping the first (root)
            for bname in bone_names_ordered[1:]:
                # Real bone: present in pose_bones
                if bname in pose_bones:
                    pb = pose_bones[bname]
                    parent_pb = pb.parent
                    if parent_pb is None:
                        self.report({'ERROR'}, f"Pose bone {bname} unexpectedly has no parent at frame {frame}")
                        return {'CANCELLED'}
                    rel_mat = parent_pb.matrix.inverted() @ pb.matrix
                    q = rel_mat.to_3x3().to_quaternion()
                    exp_map = q.to_exponential_map()
                    body_pose.append([float(exp_map.x), float(exp_map.y), float(exp_map.z)])

                    # length scaling using world head/tail
                    world_head = arm_eval.matrix_world @ Vector(pb.head)
                    world_tail = arm_eval.matrix_world @ Vector(pb.tail)
                    cur_len = float((world_tail - world_head).length)
                    rest_len = bone_name_2_rest_length.get(bname, 1.0)
                    length_factor = cur_len / rest_len if rest_len != 0 else 1.0
                    body_bone_length.append(length_factor)
                    continue

                # Virtual bone
                if bname in virtual_bone_names:
                    p_name = bone_names_tree[bname]['p']
                    c_name = bone_names_tree[bname]['c'][0]
                    # ensure parent & child pose bones exist
                    if (p_name not in [b.name for b in pose_bones]) or (c_name not in [b.name for b in pose_bones]):
                        self.report({'ERROR'}, f"Virtual bone {bname} refers to missing pose bones at frame {frame}")
                        return {'CANCELLED'}

                    parent_pb = pose_bones[p_name]
                    child_pb = pose_bones[c_name]

                    # build frames at parent tail and child head (armature space -> world)
                    parent_tail = Vector(parent_pb.tail)
                    child_head = Vector(child_pb.head)
                    parent_rot_world = (arm_eval.matrix_world @ parent_pb.matrix).to_3x3().to_4x4()
                    child_rot_world = (arm_eval.matrix_world @ child_pb.matrix).to_3x3().to_4x4()
                    world_parent_tail = arm_eval.matrix_world @ parent_tail
                    world_child_head = arm_eval.matrix_world @ child_head

                    parent_tail_frame_world = Matrix.Translation(world_parent_tail) @ parent_rot_world
                    child_head_frame_world = Matrix.Translation(world_child_head) @ child_rot_world

                    rel_mat = parent_tail_frame_world.inverted() @ child_head_frame_world
                    q = rel_mat.to_3x3().to_quaternion()
                    exp_map = q.to_exponential_map()
                    body_pose.append([float(exp_map.x), float(exp_map.y), float(exp_map.z)])

                    cur_dist = float((world_child_head - world_parent_tail).length)
                    rest_len = bone_name_2_rest_length.get(bname, 1.0)
                    length_factor = cur_dist / rest_len if rest_len != 0 else 1.0
                    body_bone_length.append(length_factor)
                    continue

                # neither real nor virtual (shouldn't happen)
                body_pose.append([0.0, 0.0, 0.0])
                body_bone_length.append(1.0)

            frame_entry = {
                "frame": int(frame),
                "time": float((frame - frame_start) / fps),
                "global_t": global_t,
                "global_ori": global_ori,
                "body_pose": body_pose,
                "body_bone_length": body_bone_length
            }
            data["frames"].append(frame_entry)

        # write json
        with open(out_path, 'w') as jf:
            json.dump(data, jf, indent=2)

        self.report({'INFO'}, f"Wrote pose time series JSON to {out_path}")
        return {'FINISHED'}




# --------------------------- Registration ----------------------------------
classes = (
    SYNTH_PropertyGroup,
    SYNTH_OT_apply_settings,
    SYNTH_OT_load_config,
    SYNTH_OT_export_keypoint_list,
    SYNTH_PT_main_panel,
    SYNTH_OT_unregister_timed_render,
    TimedRender,
    SYNTH_OT_export_camera_matrices,
    SYNTH_OT_export_mesh,
    SYNTH_OT_create_videos,
    SYNTH_OT_export_pose_time_series_json,
)



def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.synth_props = PointerProperty(type=SYNTH_PropertyGroup)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, 'synth_props'):
        del bpy.types.Scene.synth_props


if __name__ == '__main__':
    register()