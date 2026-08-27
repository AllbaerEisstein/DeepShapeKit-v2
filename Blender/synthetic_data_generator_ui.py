bl_info = {
    "name": "Synthetic Dataset UI + TimedRender for YOLO Pose & Seg",
    "author": "Jonathan Häßler",
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
import math
from collections import defaultdict
from mathutils import Vector, Matrix
from bpy.props import (
    StringProperty, BoolProperty, FloatProperty, IntProperty,
    PointerProperty, CollectionProperty,
)
from bpy_extras.io_utils import ImportHelper
from bpy.types import Panel, Operator, PropertyGroup, UIList 
import cv2
import numpy as np


# --------------------------- Globals --------------------------------------

# cache for camera matrices (per-camera)
cam_name_2_matrix = {}

# cache for priors UI convenience toggle
armature_pose_toggle_cache = {
    "is_rest_mode": False,
    "armature_name": None,
    "bone_mats": {},
}

# conversion matrices between conventions
BLENDER_CAM_2_CV_CAM = Matrix((
    (1, 0, 0),
    (0, -1, 0),
    (0, 0, -1)
))

BLENDERWORLD_2_CVWORLD = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=float)


# --------------------------- Property Group ---------------------------------
class SYNTH_BoneGroupItem(PropertyGroup):
    names_csv: StringProperty(
        name="Bones/Keypoints",
        description="Comma-separated bone and/or keypoint names",
        default=""
    )
    include_children: BoolProperty(
        name="Include Children",
        description="For bones listed here: also add all descendants recursively",
        default=False
    )

class SYNTH_CameraSelectionItem(PropertyGroup):
    camera_name: StringProperty(
        name="Camera Name",
        default=""
    )
    enabled: BoolProperty(
        name="Enabled",
        default=True
    )

class SYNTH_BonePriorItem(PropertyGroup):
    bone_name: StringProperty(
        name="Bone Name",
        default=""
    )
    swing_x: FloatProperty(name="Swing X", default=180.0)
    twist_y: FloatProperty(name="Twist Y", default=360.0)
    swing_z: FloatProperty(name="Swing Z", default=180.0)

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

    bone_groups: CollectionProperty(type=SYNTH_BoneGroupItem)
    bone_groups_index: IntProperty(default=-1)
    camera_selections: CollectionProperty(type=SYNTH_CameraSelectionItem)
    bone_priors_ui_item_collection: CollectionProperty(type=SYNTH_BonePriorItem)
    show_priors_explanation: BoolProperty(
        name="Show/Hide Explanation",
        default=False
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


def get_scene_cameras_sorted():
    cam_collection = bpy.data.collections.get('Cameras')
    cam_objects = cam_collection.objects if cam_collection else bpy.data.objects
    return sorted([cam for cam in cam_objects if getattr(cam, "type", None) == 'CAMERA'], key=lambda c: c.name)


def sync_camera_selections(scene):
    p = scene.synth_props
    cam_objects = get_scene_cameras_sorted()
    cam_names = {cam.name for cam in cam_objects}

    for idx in reversed(range(len(p.camera_selections))):
        if p.camera_selections[idx].camera_name not in cam_names:
            p.camera_selections.remove(idx)

    existing = {item.camera_name for item in p.camera_selections}
    for cam in cam_objects:
        if cam.name not in existing:
            item = p.camera_selections.add()
            item.camera_name = cam.name
            item.enabled = True

    return cam_objects


def get_target_object(scene):
    p = scene.synth_props
    col = bpy.data.collections.get(p.collection_name)
    if col is None:
        return None
    return col.objects.get(p.object_name)


def find_target_armature(scene):
    obj = get_target_object(scene)
    if obj is None:
        return None
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE' and modifier.object:
            return modifier.object
    return None


def get_target_armature_bone_names_sorted(scene):
    arm_obj = find_target_armature(scene)
    if arm_obj is None:
        return None, []
    return arm_obj, sorted([b.name for b in arm_obj.data.bones])


def sync_bone_priors_ui_item_collection(scene):
    """
    Add an UI item for all existing bones in the armature.
    Remove items for bones that no longer exist.
    """
    p = scene.synth_props
    arm_obj, bone_names = get_target_armature_bone_names_sorted(scene)
    bone_names_set = set(bone_names)

    for idx in reversed(range(len(p.bone_priors_ui_item_collection))):
        if p.bone_priors_ui_item_collection[idx].bone_name not in bone_names_set:
            p.bone_priors_ui_item_collection.remove(idx)

    existing = {item.bone_name for item in p.bone_priors_ui_item_collection}
    for bone_name in bone_names:
        if bone_name not in existing:
            item = p.bone_priors_ui_item_collection.add()
            item.bone_name = bone_name

    return arm_obj, bone_names


# ---- camera intrinsic helpers -----------------

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

    # CLAUDE FIX (B1): `bm.free()` was called a second time here on a BMesh that had already been
    # freed a few lines above. The duplicate call raises ReferenceError, which propagated out of
    # this function and was swallowed by the broad `except` in TimedRender.handle_render_item --
    # silently dropping every keypoint label for every frame.
    # docs: The object owns the mesh data-block. To force free it use to_mesh_clear(). 
    obj_eval.to_mesh_clear()
    
    return faces, vertices, normals, kpt_2_verts_worldco, kpt_2_faces_worldco



# --------------------------- Mesh extraction --------------------

def get_mesh_json(context):
    def _parse_csv_names(csv_text: str):
        return [n.strip() for n in csv_text.split(",") if n.strip()]

    def _dedupe_preserve_order(seq):
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    p = context.scene.synth_props
    collection_name = p.collection_name
    object_name = p.object_name
    kpt_list = _parse_csv_names(p.keypoint_list_csv)

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
    # CLAUDE FIX (A9): everything downstream -- the kintree, `body_pose` indexing in the
    # pose_time_series exporter/importer, and LBS's chain, which now rejects a second joint with
    # parent -1 -- assumes a single kinematic root. With two roots the pose arrays are one entry
    # short and every bone index silently shifts. Refuse here instead.
    if len(roots) > 1:
        raise ValueError(
            f"The armature '{arm.name}' has {len(roots)} root bones "
            f"({', '.join(sorted(b.name for b in roots))}). The template format supports exactly "
            f"one root; parent the extra roots under a single root bone and re-export."
        )

    # head/tail helpers (armature-space local coordinates)
    def head_pos(b): return b.head_local.copy()
    def tail_pos(b): return b.tail_local.copy()

    # CLAUDE FIX (A15): J (head_local/tail_local) is ARMATURE space, `rest_rot_world` was WORLD
    # space and V came from obj.data.vertices, i.e. MESH-OBJECT space -- three different spaces in
    # one template file, which only ever agreed because the demo scene has identity object
    # transforms. `rest_rot` is now emitted in armature space (the same space as J), the mesh is
    # baked into armature space, and the armature's own world matrix is recorded separately.
    # `rest_rot_world` is still written so that older consumers keep working.
    arm_world = arm.matrix_world.copy()
    arm_world_3 = arm_world.to_3x3()
    obj_to_armature = arm_world.inverted() @ obj.matrix_world

    # BFS traversal to preserve topology order
    # CLAUDE FIX (B13): the placeholder for 'p' used to be the *type* `str`, which is truthy, so a
    # bone that the BFS never reaches was treated as a non-root, never entered `bone_order`, and
    # still consumed a weights column index derived from `bone_order`. Use None and verify that
    # every bone was visited.
    bone_names_tree = {b.name: {'p': None, 'c': [], 'joints': [], 'joints_idx': [-1,-1], 'rest_rot': [], 'rest_rot_world': []} for b in bone_list}
    ordered_bones = []
    queue = roots[:]
    while queue:
        b = queue.pop(0)
        ordered_bones.append(b)
        bone_names_tree[b.name]['p'] = b.parent.name if b.parent is not None else ''
        bone_names_tree[b.name]['joints'] = [head_pos(b), tail_pos(b)]
        rest_rot_armature = b.matrix_local.to_3x3().normalized()      # bone rest -> armature space
        bone_names_tree[b.name]['rest_rot'] = rest_rot_armature
        bone_names_tree[b.name]['rest_rot_world'] = (arm_world_3 @ rest_rot_armature).normalized()
        for ch in b.children:
            queue.append(ch)
            bone_names_tree[b.name]['c'].append(ch.name)

    unvisited = [b.name for b in bone_list if bone_names_tree[b.name]['p'] is None]
    if unvisited:
        raise ValueError(f"Bones not reachable from the root bone: {sorted(unvisited)}")

    pos_to_joint_idx = {}
    joint_positions = []
    joint_names = []
    parent_indices = []

    # NOTE: Logic needs to be checked again
    def get_virtual_bone_rest_matrix_from_bones(parentb, childb):
        # CLAUDE FIX (A15): built in ARMATURE space so that it matches J and `rest_rot`.
        # CLAUDE FIX (B15): a zero-length gap used to reach `.normalized()` on a zero vector and
        # only failed later, in a confusing place. Reject it here, matching the exporter.
        parent_tail = tail_pos(parentb)
        child_head = head_pos(childb)
        dir_arm = child_head - parent_tail
        if dir_arm.length < 1e-8:
            raise Exception(
                f"virtual bone between '{parentb.name}' and '{childb.name}' has zero length; "
                f"the child's head coincides with the parent's tail, so no virtual bone is needed."
            )
        y = dir_arm.normalized()   # virtual bone +Y in armature space
        # choose a stable roll reference using parent's REST axes:
        parent_rest_R = bone_names_tree[parentb.name]['rest_rot']  # 3x3, armature space
        # try parent's rest Z (local +Z) first:
        parent_z = parent_rest_R @ Vector((0.0, 0.0, 1.0))
        # if nearly parallel to y, try parent local X
        if abs(parent_z.normalized().dot(y)) > 0.999:
            parent_x = parent_rest_R @ Vector((1.0, 0.0, 0.0))
            # compute Z = parent_x × Y (ensure orthogonality). error out if Z is degenerate
            z = parent_x.cross(y)
            if z.length < 1e-8:
                raise Exception(f"z-axis is degenerate while constructing rest_rot for bone {childb.name}")
            z.normalize()
            x = y.cross(z)
            virtual_rest_R = Matrix((x, y, z)).transposed()
        else:
            # compute X = parent_z × Y (ensure orthogonality). error out if X is degenerate
            x = parent_z.cross(y)
            if x.length < 1e-8:
                raise Exception(f"x-axis is degenerate while constructing rest_rot for bone {childb.name}")
            x.normalize()
            z = y.cross(x)
            virtual_rest_R = Matrix((x, y, z)).transposed()
        return virtual_rest_R


    def key_from_vec(v, prec=6):
        return (round(v.x, prec), round(v.y, prec), round(v.z, prec))

    def ensure_joint_at_position(pos, name=None):
        """
        If a joint was already ensured at this position, return the index of that joint.
        Else, create a new index for this position. Indices auto-increment.
        This function also appends to the list `joint_positions` every time it is called. 
        So, if it is called in hierarchical order, `joint_positions` is ordered according to this hierarchy.
        """
        k = key_from_vec(pos)
        if k in pos_to_joint_idx:
            return pos_to_joint_idx[k]
        idx = len(joint_positions)
        pos_to_joint_idx[k] = idx
        joint_positions.append(pos)
        joint_names.append(name or f"joint_{idx}")
        # fill up with default -1 parent for each new pos
        parent_indices.append(-1)
        return idx
    
    # Detect missing physical bone connections and add virtual bones
    virtual_bone_names = []
    for b in ordered_bones:
        if b.parent is None:
            continue
        p_tail_key = key_from_vec(tail_pos(b.parent))
        child_head_key = key_from_vec(head_pos(b))
        if p_tail_key != child_head_key:
            vname = f"virtual_{b.parent.name}_to_{b.name}"
            virtual_bone_names.append(vname)
            # in bone-tree, replace entry for original child bone with entry for virtual bone (insert node and edge into tree)
            bone_names_tree[vname] = {
                    'p': b.parent.name, 'c': [b.name],
                    'joints': [tail_pos(b.parent), head_pos(b)],
                    'joints_idx': [-1,-1],
                    'rest_rot': [], 'rest_rot_world': [],
                }
            bone_names_tree[b.name]['p'] = vname
            bone_names_tree[b.parent.name]['c'] = [vname if c == b.name else c for c in bone_names_tree[b.parent.name]['c']]

            virtual_rest_R = get_virtual_bone_rest_matrix_from_bones(b.parent, b)
            bone_names_tree[vname]['rest_rot'] = virtual_rest_R
            bone_names_tree[vname]['rest_rot_world'] = (arm_world_3 @ virtual_rest_R).normalized()
    
    # topo-sort the tree and add joint information, in the same go create joint indexing and joint parent information
    # this ensures indexing of joints that corresponds to the bone hierarchy
    bone_names_ordered = []
    queue = [node for node, p_c_dict in bone_names_tree.items() if p_c_dict['p'] == '']
    while queue:
        n = queue.pop(0)
        bone_names_ordered.append(n)

        hi = ensure_joint_at_position(bone_names_tree[n]['joints'][0], name=f"{n}_head")
        ti = ensure_joint_at_position(bone_names_tree[n]['joints'][1], name=f"{n}_tail")
        bone_names_tree[n]['joints_idx'][0] = hi
        bone_names_tree[n]['joints_idx'][1] = ti

        # the parent of the tail joint is the head joint of the same bone
        parent_indices[ti] = hi

        # the parent of the head joint has been set already since every head joint (except for root joint) is a also a tail joint.
        # just the parent index of the head joint has to be set manually.
        if bone_names_tree[n]['p'] == '':
            parent_indices[hi] = -1

        for ch in bone_names_tree[n]['c']:
            queue.append(ch)

    # -- Bone groups from UI (CSV per group)
    def collect_children_recursive(bone_name, tree):
        """Return [bone_name] + all descendants' names (DFS)."""
        stack = [bone_name]
        result = []
        while stack:
            b = stack.pop()
            if b not in result:
                result.append(b)
                stack.extend(tree[b]['c'])
        return result

    bone_groups_out = []
    missing_names = []  # collect all invalid names for a consolidated error, while skipping them

    for group_item in p.bone_groups:
        raw_names = _parse_csv_names(group_item.names_csv)
        group_bone_names = []
        group_kpt_names = []

        for name in raw_names:
            if name in bone_names_tree:
                # it is a bone
                if group_item.include_children:
                    group_bone_names.extend(collect_children_recursive(name, bone_names_tree))
                else:
                    group_bone_names.append(name)
            elif name in kpt_list:
                # it is a keypoint
                group_kpt_names.append(name)
            else:
                # unknown → record error and skip just this name
                missing_names.append(name)

        # dedupe while preserving user’s order
        group_bone_names = _dedupe_preserve_order(group_bone_names)
        group_kpt_names  = _dedupe_preserve_order(group_kpt_names)

        # map to indices, skipping any that don’t resolve (shouldn’t happen after checks)
        bone_indices = []
        for bn in group_bone_names:
            try:
                bone_indices.append(bone_names_ordered.index(bn))
            except ValueError:
                missing_names.append(bn)

        kpt_indices = []
        for kn in group_kpt_names:
            try:
                kpt_indices.append(kpt_list.index(kn))
            except ValueError:
                missing_names.append(kn)

        bone_groups_out.append({
            "keypoints_names": group_kpt_names,
            "bone_names": group_bone_names,
            "keypoint_indices": kpt_indices,
            "bone_indices": bone_indices,
        })

    # If requested: raise error for unknown names but we’ve already skipped them in groups
    if missing_names:
        # Raise once with all missing names; comment out the next line if you prefer a warning-only behavior.
        raise ValueError(f"Unknown bone/keypoint name(s) in bone groups: {sorted(set(missing_names))}")

    # -- joints
    # Convert joint positions to lists (object-space coordinates)
    # joint positions was built i
    joints = [[float(c) for c in v] for v in joint_positions]
    joint_indices = list(range(len(joints)))
    kintree_unique_joints = [parent_indices, joint_indices]

    # -- geometry
    # CLAUDE FIX (A15): express the mesh in the same (armature) space as J and rest_rot.
    verts = [[float(c) for c in (obj_to_armature @ v.co)] for v in obj.data.vertices]
    faces = [list(p.vertices) for p in obj.data.polygons]

    # -- weights: include columns for virtual bones (zeros)
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

    # -- priors
    # set default to "no restriction" (180 Swing; 360 Twist) for every real bone
    # set default to "locked" (0 Swing; 0 Twist) for every virtual bone
    bone_name_2_prior = {
        name: {
            "swing_x": 3.14159,
            "swing_z": 3.14159,
            "twist_y": 2*3.14159,
        } if name not in virtual_bone_names else {
            "swing_x": 0.0,
            "swing_z": 0.0,
            "twist_y": 0.0,
        } for name in bone_names_ordered
    }

    if len(p.bone_priors_ui_item_collection) != len(bone_name_2_prior)-len(virtual_bone_names):
        raise ValueError("Bone priors UI collection length does not match number of physical bones. This likely means that the UI collection is not properly synced with the armature bones.")
    for bone_prior_ui_item in p.bone_priors_ui_item_collection:
        if bone_prior_ui_item.bone_name not in bone_name_2_index:
            raise ValueError(f"Bone prior item has unknown bone name '{bone_prior_ui_item.bone_name}'") 
        for angle_prior_name in ["swing_x", "swing_z", "twist_y"]:
            if not hasattr(bone_prior_ui_item, angle_prior_name):
                raise ValueError(f"Bone prior item is missing expected attribute '{angle_prior_name}'")
            bone_name_2_prior[bone_prior_ui_item.bone_name][angle_prior_name] = getattr(bone_prior_ui_item, angle_prior_name) / 180.0 * 3.14159

    out = {
        'V': verts,
        'F': faces,
        'J': joints,
        'vert2kpt': v2k,
        'weights': weights,
        'kpt_list': kpt_list,
        'n_bones': n_bone_groups,
        'bone_order': bone_names_ordered,        # bone order used for export
        'kintree_table': kintree_unique_joints,
        'bone_names_tree': {
            bone_name: {
                'p': data['p'],
                'c': data['c'],
                'joints': data['joints_idx'],
                # 'rest_rot' is the authoritative one (armature space, same space as J and V);
                # 'rest_rot_world' is kept for backwards compatibility with older consumers.
                'rest_rot': [[float(c) for c in row] for row in data['rest_rot']],
                'rest_rot_world': [[float(c) for c in row] for row in data['rest_rot_world']],
                'priors': bone_name_2_prior[bone_name] if bone_name in bone_name_2_prior else None,
            }
            for bone_name, data in bone_names_tree.items()
        },                                       # a tree-dict of parents, children and joint indices of bones
        'space': 'armature',                     # V, J and rest_rot all live in armature space
        'armature_matrix_world': [[float(c) for c in row] for row in arm_world],
        'virtual_bone_names': virtual_bone_names, # for identifying virtual bones
        'virtual_bone_mask': [1 if name in virtual_bone_names else 0 for name in bone_names_ordered], # 1 for virtual bones, 0 for physical bones
        'bone_groups': bone_groups_out,
        'bone_priors': bone_name_2_prior,
    }

    return out



def get_angle_of_bone():
    pass



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
            # CLAUDE FIX (B2): this loop was named `all_visible` and documented as "faces fully
            # visible", but it broke out on the FIRST unoccluded vertex, i.e. it implemented
            # *any*-visible. A face with a single visible corner contributed its whole area, so
            # `keypoint_visible_threshold` was far more permissive than intended and occluded
            # keypoints were labelled as fully visible.
            all_visible = True
            for coord in face_coords:
                if is_vertex_occluded(deps, cam_obj, Vector(coord)):
                    all_visible = False
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
    """Compute camera matrices for a Blender camera.

    Returns a dict containing both:
      * Matrices that operate directly on Blender-world coordinates (used by the
        helper utilities in this exporter), and
      * Pure CV extrinsics together with an explicit Blender→CV basis-change
        matrix for downstream tooling.
    """

    cam_name = cam_obj.name
    cached = cam_name_2_matrix.get(cam_name)
    if cached and cached.get('P_blender') is not None and cached.get('P') is not None:
        return cached

    f, KRT, K_mat, R_4x4, T_4x4, Rt_mat = get_3x4_P_matrix_Blendercam2Blenderimage(cam_obj, scene)
    K_np = np.array(K_mat, dtype=float)
    Rt_np = np.array(Rt_mat, dtype=float)

    R_world_to_blcam = Rt_np[:, :3]
    t_world_to_blcam = Rt_np[:, 3]

    blender_cam_2_cv = np.array(BLENDER_CAM_2_CV_CAM, dtype=float)
    R_blender = blender_cam_2_cv @ R_world_to_blcam
    t_blender = blender_cam_2_cv @ t_world_to_blcam
    Rt_blender_cv = np.concatenate([R_blender, t_blender[:, None]], axis=1)
    P_blender = K_np @ Rt_blender_cv

    F = BLENDERWORLD_2_CVWORLD
    F_inv = np.linalg.inv(F)

    R_cv = R_blender @ F_inv
    t_cv = t_blender
    Rt_cv = np.concatenate([R_cv, t_cv[:, None]], axis=1)
    P_cv = K_np @ Rt_cv

    cam_name_2_matrix[cam_name] = {
        'f': float(f) if f is not None else None,
        'K': Matrix(K_np.tolist()),
        'R': Matrix(R_cv.tolist()),
        't': Vector(t_cv.tolist()),
        'Rt': Matrix(Rt_cv.tolist()),
        'P': Matrix(P_cv.tolist()),
        'FROM_BLENDERWORLD': Matrix(F.tolist()),
        'R_blender': Matrix(R_blender.tolist()),
        't_blender': Vector(t_blender.tolist()),
        'Rt_blender': Matrix(Rt_blender_cv.tolist()),
        'P_blender': Matrix(P_blender.tolist()),
    }
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

    # CLAUDE FIX (B14): materials live on the MESH DATA, not on the object. The old code iterated
    # objects while mutating `o.data.materials`, so two objects sharing one mesh had that mesh's
    # slot list cleared and re-appended twice -- duplicating slots and, because clearing resets
    # every polygon's material_index to 0, destroying per-face material assignments. Snapshot and
    # restore per unique mesh datablock, including material_index.
    all_mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
    mesh_to_objects = defaultdict(list)
    for o in all_mesh_objects:
        mesh_to_objects[o.data].append(o)
    orig_materials = {
        me: ([slot for slot in me.materials], [poly.material_index for poly in me.polygons])
        for me in mesh_to_objects
    }

    # Save world and set black background (optional but ensures no stray background)
    orig_world = scene.world
    black_world = None
    try:
        black_world = bpy.data.worlds.new(name="SYNTH_tmp_black_world")
        black_world.use_nodes = True
        # clear nodes
        for nd in list(black_world.node_tree.nodes):
            black_world.node_tree.nodes.remove(nd)
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
        for me, objs in mesh_to_objects.items():
            is_target = any(o.name == target_obj.name for o in objs)
            if is_target and len(objs) > 1:
                print(f"SYNTH warning: mesh '{me.name}' is shared by the target object and "
                      f"{len(objs) - 1} other object(s); they cannot be masked separately.")
            me.materials.clear()
            me.materials.append(white_em if is_target else black_em)

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
        for me, (mats, face_indices) in orig_materials.items():
            try:
                me.materials.clear()
                for m in mats:
                    me.materials.append(m)
                # materials.clear() resets every polygon's material_index to 0
                for poly, idx in zip(me.polygons, face_indices):
                    poly.material_index = idx
            except Exception:
                pass

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
        for datablock, collection in ((white_em, bpy.data.materials),
                                      (black_em, bpy.data.materials),
                                      (black_world, bpy.data.worlds)):
            try:
                if datablock is not None and datablock.users == 0:
                    collection.remove(datablock, do_unlink=True)
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


def write_polygons_to_yolo(polygons, image_width, image_height, out_path, class_index: int = 0):
    """Write one YOLO-seg line per polygon.

    CLAUDE FIX (B3): the leading field of a YOLO-seg line is the CLASS index, but this function
    used to write the polygon's position in the list. A mask that yields three contours therefore
    declared classes 0, 1 and 2 while create_dataset_yaml() declares a single class, so training
    either failed or learned nonsense labels. The class is now passed in explicitly.
    """
    lines = []
    for polygon in polygons:
        norm_pts = []
        for x, y in polygon:
            norm_pts.append(f"{(x / image_width):.3f}")
            norm_pts.append(f"{(y / image_height):.3f}")
        lines.append(f"{int(class_index)} " + " ".join(norm_pts))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")



def draw_points_on_img(points, img_path, out_path, annot_radius = 1):
    img = cv2.imread(str(img_path))
    for point in points:
        # CLAUDE FIX (B4): filled circles are requested with thickness=-1; lineType=-1 is not a
        # valid line type and the marker was drawn as a 1-px outline.
        cv2.circle(img=img, center=(int(point[0]),int(point[1])), radius=annot_radius, color=(255, 255, 255), thickness=-1)
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
        # CLAUDE FIX (B4): thickness=-1 fills the circle; lineType=-1 is not a valid line type.
        cv2.circle(img=img, center=(int(x),int(y)), radius=1, color=color, thickness=-1) # center
        cv2.circle(img=img, center=(int(x),int(y)), radius=conf_scaled_annot_radius, color=color, thickness=-1)
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

        cam_objects = sync_camera_selections(scene)
        enabled_camera_names = {item.camera_name for item in p.camera_selections if item.enabled}

        modes = ["regular"] + (["binary"] if p.render_binary else [])

        skipped_count = 0
        for cam in cam_objects:
            if cam.name not in enabled_camera_names:
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


    # CLAUDE FIX (B5): one source of truth for the annotated image size. The out-of-bounds test
    # used the evaluated render resolution while the YOLO normalisation divided by the UI fields
    # `image_width_px` / `image_height_px`; any render_scale != 1.0, or a scene resolution that had
    # drifted from the UI, produced silently mis-scaled labels. Both now use this helper.
    @staticmethod
    def _annotation_image_size(scene):
        pct = scene.render.resolution_percentage / 100.0
        return int(round(scene.render.resolution_x * pct)), int(round(scene.render.resolution_y * pct))

    def handle_render_item(self, context, qitem):
        scene = context.scene

        def reset_render_settings():
            scene.node_tree.nodes["Alpha Over"].inputs[1].default_value = (0, 0, 0, 0)
            scene.node_tree.nodes["Brightness/Contrast"].inputs[1].default_value = 0
            scene.node_tree.nodes["Brightness/Contrast"].inputs[2].default_value = 0

        p = scene.synth_props

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

        img_w, img_h = self._annotation_image_size(scene)
        if (img_w, img_h) != (int(p.image_width_px), int(p.image_height_px)):
            self.report({'WARNING'},
                        f"scene render size {img_w}x{img_h} differs from the UI fields "
                        f"{int(p.image_width_px)}x{int(p.image_height_px)}; labels follow the "
                        f"rendered size. Press 'Apply And Save Settings' to sync them.")

        # CLAUDE FIX (B6): the queue holds one item per (camera, frame, mode), but the beauty
        # render and the whole keypoint/label pipeline used to run for BOTH modes -- every frame
        # was rendered and annotated twice, and the second pass overwrote the first. The beauty
        # pass and the annotations now run only for the 'regular' item.
        if mode == 'regular':
            scene.render.filepath = render_out_file_path_bl
            if p.use_compositor:
                reset_render_settings()
            bpy.ops.render.render(write_still=True)
            self._annotate_frame(context, cam_obj, img_w, img_h,
                                 render_out_file_path_os, kpt_annot_out_file_path,
                                 kpt_label_out_path)

        if p.render_binary and mode == 'binary':
            self._render_and_write_mask(context, img_w, img_h, render_prefix_cam_frame,
                                        render_out_file_path_os, mask_annot_out_file_path,
                                        mask_label_out_path)

    def _annotate_frame(self, context, cam_obj, img_w, img_h,
                        render_out_file_path_os, kpt_annot_out_file_path, kpt_label_out_path):
        scene = context.scene
        p = scene.synth_props
        kpt_list = [kp.strip() for kp in p.keypoint_list_csv.split(',') if kp.strip()]

        try:
            deps = bpy.context.evaluated_depsgraph_get()

            faces, vertices, normals, kpt_2_verts_list_world, kpt_2_faces_list_world = \
                get_deformed_mesh_data(deps, p.collection_name, p.object_name, kpt_list)

            kpt_2_visibility_pct, kpt_2_visible_faces = (
                get_keypoint_visibility_from_faces(deps, kpt_2_faces_list_world, cam_obj)
                if p.check_keypoint_visibility
                else (
                    {k: 1.0 for k in kpt_2_verts_list_world.keys()},
                    {
                        kpt: [face["coords"] for face in face_list]
                        for kpt, face_list in kpt_2_faces_list_world.items()
                    },
                )
            )

            # no filtering yet
            kpt_2_visible_vert_coords_list_world = {
                kpt: list(set([point for poly in kpt_2_visible_faces[kpt] for point in poly]))
                for kpt in kpt_2_visibility_pct
            }
            kpt_2_coords_list_world = {
                kpt: list(set([point for poly in kpt_2_faces_list_world[kpt] for point in poly["coords"]]))
                for kpt in kpt_2_visibility_pct
            }

            # avg over the visible vertices coords / over all the verts
            kpt_2_avg_coords_world_visible = get_avg_kpt_coords_3d(kpt_2_visible_vert_coords_list_world)
            kpt_2_avg_coords_world = get_avg_kpt_coords_3d(kpt_2_coords_list_world)

            cam_mats = get_cam_matrix_for_cam(cam_obj, scene)
            P = cam_mats['P_blender']

            EPS = 1e-8

            def is_outside_image_bounds(x, y):
                return x < EPS or x > (img_w - EPS) or y < EPS or y > (img_h - EPS)

            def project_world_keypoints(kpt_2_coords):
                kpt_2_projected = {}
                for kpt, coords in kpt_2_coords.items():
                    ph = P @ Vector((*coords, 1.0))
                    # require a meaningful positive depth (z)
                    if not (ph.z > EPS):
                        continue
                    kpt_2_projected[kpt] = (ph.x / ph.z, ph.y / ph.z)
                return kpt_2_projected

            kpt_2_coords_image_filtered_by_vis = project_world_keypoints(kpt_2_avg_coords_world_visible)
            kpt_2_coords_image_all_faces_count = project_world_keypoints(kpt_2_avg_coords_world)

            # 0: not labeled / out-of-view, 1: labeled but occluded, 2: labeled and visible
            occluded_status = 1 if p.keep_occluded_keypoints else 0
            kpt_2_vis_status = {
                kpt:
                    0 if is_outside_image_bounds(*coords)
                    else occluded_status if kpt_2_visibility_pct.get(kpt, 0.0) < p.keypoint_visible_threshold
                    else 2
                for kpt, coords in kpt_2_coords_image_filtered_by_vis.items()
            }
            for kpt in kpt_list:
                if kpt not in kpt_2_vis_status:
                    kpt_2_vis_status[kpt] = occluded_status

            # CLAUDE FIX (B10): a fully occluded keypoint is absent from
            # kpt_2_coords_image_filtered_by_vis, and its all-faces centroid may also have failed
            # to project (behind the camera), so the direct subscript raised KeyError and the whole
            # frame's labels were lost. Fall back to "missing" instead.
            for kpt, status in list(kpt_2_vis_status.items()):
                if status == 1:
                    fallback = kpt_2_coords_image_all_faces_count.get(kpt)
                    if fallback is None:
                        kpt_2_vis_status[kpt] = 0
                        kpt_2_coords_image_filtered_by_vis[kpt] = (0, 0)
                    else:
                        kpt_2_coords_image_filtered_by_vis[kpt] = fallback
                elif status == 0:
                    kpt_2_coords_image_filtered_by_vis[kpt] = (0, 0)

            img_annot_source_file_path = render_out_file_path_os

            if p.create_annotated_images and p.draw_every_keypoint_vertex:
                visible_verts = []
                for vertex_list in kpt_2_verts_list_world.values():
                    for vertex in vertex_list:
                        if (not is_vertex_occluded(deps, cam_obj, Vector(vertex["co"]))
                                if p.check_keypoint_visibility else True):
                            vertex_bl_cam = P @ Vector(tuple(vertex["co"]) + (1,))
                            if abs(vertex_bl_cam.z) < EPS:
                                continue
                            visible_verts.append((vertex_bl_cam.x / vertex_bl_cam.z,
                                                  vertex_bl_cam.y / vertex_bl_cam.z))
                draw_points_on_img(visible_verts, img_annot_source_file_path, kpt_annot_out_file_path)
                img_annot_source_file_path = kpt_annot_out_file_path

            if p.create_annotated_images and p.draw_every_keypoint_face:
                visible_faces = []
                for face_list in kpt_2_visible_faces.values():
                    for face in face_list:
                        projected_face = []
                        for vertex_world in face:
                            ph = P @ Vector((*vertex_world, 1))
                            if abs(ph.z) < EPS:
                                continue
                            projected_face.append((ph.x / ph.z, ph.y / ph.z))
                        if projected_face:
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
                img_w,
                img_h,
                [0],
                kpt_label_out_path
            )

        except Exception as e:
            self.report({'WARNING'}, f"Keypoint generation failed: {e}")

    def _render_and_write_mask(self, context, img_w, img_h, render_prefix_cam_frame,
                               render_out_file_path_os, mask_annot_out_file_path,
                               mask_label_out_path):
        scene = context.scene
        p = scene.synth_props
        if p.use_compositor:
            scene.node_tree.nodes["Alpha Over"].inputs[1].default_value = (0, 0, 0, 1)
            scene.node_tree.nodes["Brightness/Contrast"].inputs[1].default_value = 50
            scene.node_tree.nodes["Brightness/Contrast"].inputs[2].default_value = 100
            scene.render.filepath = mask_annot_out_file_path
            bpy.ops.render.render(write_still=True)
        else:
            render_binary_mask_keep_occluders_black(
                scene, get_target_object(scene), mask_annot_out_file_path)

        if os.path.exists(mask_annot_out_file_path):
            polygons = get_mask_polygons_from_binary_image(mask_annot_out_file_path)
            if p.create_annotated_images:
                draw_polygons(render_out_file_path_os, mask_annot_out_file_path, polygons)
            write_polygons_to_yolo(polygons, img_w, img_h, mask_label_out_path, class_index=0)
        else:
            try:
                img = cv2.imread(render_out_file_path_os)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                    tmp_mask = os.path.join(resolve(p.mask_label_dir),
                                            render_prefix_cam_frame + "_tmp_mask.png")
                    os.makedirs(os.path.dirname(tmp_mask), exist_ok=True)
                    cv2.imwrite(tmp_mask, thresh)
                    polygons = get_mask_polygons_from_binary_image(tmp_mask)
                    write_polygons_to_yolo(polygons, img_w, img_h, mask_label_out_path, class_index=0)
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
                # try:
                self.handle_render_item(context, qitem)
                # except Exception as e:
                #     self.report({'WARNING'}, f"Render failed for item: {e}")

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
        try:
            sync_bone_priors_ui_item_collection(scene)
        except Exception:
            pass

        # serialize bone groups (UI list) to a simple list of dicts
        bone_groups_cfg = [
            {
                "names_csv": item.names_csv,
                "include_children": bool(item.include_children),
            }
            for item in p.bone_groups
        ]
        bone_priors_cfg = [
            {
                "bone_name": item.bone_name,
                "swing_z": float(item.swing_z),
                "twist_y": float(item.twist_y),
                "swing_x": float(item.swing_x),
            }
            for item in p.bone_priors_ui_item_collection
        ]

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
            'BONE_GROUPS': bone_groups_cfg,
            'BONE_PRIORS': bone_priors_cfg,
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

        try:
            sync_bone_priors_ui_item_collection(scene)
        except Exception:
            pass


        # Load Bone Groups (if any)
        if 'BONE_GROUPS' in cfg and isinstance(cfg['BONE_GROUPS'], list):
            try:
                # clear existing list
                p.bone_groups.clear()
                # repopulate
                for item in cfg['BONE_GROUPS']:
                    # tolerate partial/old entries
                    names_csv = item.get('names_csv', '')
                    include_children = bool(item.get('include_children', False))
                    slot = p.bone_groups.add()
                    slot.names_csv = names_csv
                    slot.include_children = include_children
                # reset active index
                p.bone_groups_index = min(max(len(p.bone_groups) - 1, 0), len(p.bone_groups) - 1) if p.bone_groups else -1
            except Exception as e:
                self.report({'WARNING'}, f"Could not load bone groups: {e}")

        # Load Bone Priors (if any)
        if 'BONE_PRIORS' in cfg and isinstance(cfg['BONE_PRIORS'], list):
            try:
                prior_by_bone_name = {item.bone_name: item for item in p.bone_priors_ui_item_collection}
                for item in cfg['BONE_PRIORS']:
                    bone_name = item.get('bone_name', '')
                    if bone_name not in prior_by_bone_name:
                        continue
                    slot = prior_by_bone_name[bone_name]
                    for key in ("swing_z", "twist_y", "swing_x"):
                        if key in item:
                            setattr(slot, key, float(item[key]))
            except Exception as e:
                self.report({'WARNING'}, f"Could not load bone priors: {e}")


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



# --------------------------- Camera matrices export operator ----------------

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
        def mat_to_list(m):
            return [[float(v) for v in row] for row in m]

        entry = {
            'f': float(mats['f']) if mats.get('f') is not None else None,
            'K': mat_to_list(mats['K']),
            'R': mat_to_list(mats['R']),
            't': [float(v) for v in mats['t']],
            'Rt': mat_to_list(mats['Rt']),
            'P': mat_to_list(mats['P']),
            'FROM_BLENDERWORLD': mat_to_list(mats['FROM_BLENDERWORLD']),
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
    

class SYNTH_OT_export_camera_matrices(Operator):
    bl_idname = "synth.export_camera_matrices"
    bl_label = "Export Camera Matrices"
    bl_description = "Export computed camera parameters & matrices (f, K, R, t, P, Rt) for [Blender world -> CV image]-conversion for all scene cameras to cam_matrices.json in the annotation folder. -- NOTE -- f is specified in mm, K is specified in pixels -- f is in mm, K is in pixels and maps CV camera coordinates (x right, y down, z forward) to pixels -- CLAUDE FIX (B9): the exported R, t, Rt and P consume CV-WORLD coordinates, not Blender-world ones: get_cam_matrix_for_cam builds them as R_cv = R_blender @ FROM_BLENDERWORLD^-1. Convert a Blender-world point with FROM_BLENDERWORLD first (x_cv = FROM_BLENDERWORLD @ x_blender). The Blender-world variants exist internally as R_blender/Rt_blender/P_blender but are not written to this file."

    def execute(self, context):
        try:
            path_matrices_were_saved_to = export_cam_matrices(context)
            self.report({'INFO'}, f"Wrote camera parameters & matrices to {path_matrices_were_saved_to}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'WARNING'}, f"Failed to export camera matrices: {e}")
            return {'CANCELLED'}


# -------------------- Bone groups Operator -------------------------------------

class SYNTH_UL_bone_groups(UIList):
    """Draw one row per bone group with a text field + include_children toggle."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "names_csv", text="", emboss=True)
        row.prop(item, "include_children", text="children", emboss=True)

class SYNTH_OT_bone_group_add(Operator):
    bl_idname = "synth.bone_group_add"
    bl_label = "Add Bone Group"
    def execute(self, context):
        p = context.scene.synth_props
        item = p.bone_groups.add()
        item.names_csv = ""
        item.include_children = False
        p.bone_groups_index = len(p.bone_groups) - 1
        return {'FINISHED'}

class SYNTH_OT_bone_group_remove(Operator):
    bl_idname = "synth.bone_group_remove"
    bl_label = "Remove Bone Group"
    def execute(self, context):
        p = context.scene.synth_props
        idx = p.bone_groups_index
        if 0 <= idx < len(p.bone_groups):
            p.bone_groups.remove(idx)
            p.bone_groups_index = min(idx, len(p.bone_groups) - 1)
        return {'FINISHED'}


class SYNTH_OT_refresh_bone_priors_ui_item_collection(Operator):
    bl_idname = "synth.refresh_bone_priors_ui_item_collection"
    bl_label = "Refresh Priors"
    bl_description = "Rebuild UI rows for the priors from bones of the armature attached to the selected object"

    def execute(self, context):
        arm_obj, bone_names = sync_bone_priors_ui_item_collection(context.scene)
        if arm_obj is None:
            self.report({'WARNING'}, "No armature found on selected object.")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Synced priors for {len(bone_names)} bones from armature '{arm_obj.name}'")
        return {'FINISHED'}


class SYNTH_OT_toggle_rest_pose_articulated_pose(Operator):
    bl_idname = "synth.toggle_rest_pose_articulated_pose"
    bl_label = "Toggle Rest Pose / Articulated Pose"
    bl_description = "Toggle target armature between rest pose and cached articulated pose"

    def execute(self, context):
        scene = context.scene
        arm_obj = find_target_armature(scene)
        if arm_obj is None:
            self.report({'ERROR'}, "No armature found on selected object.")
            return {'CANCELLED'}

        global armature_pose_toggle_cache

        # Enter rest-like pose: cache current articulated pose first
        if not armature_pose_toggle_cache["is_rest_mode"]:
            # Cache editable pose state from the ORIGINAL armature object
            cached_basis_mats = {
                pb.name: pb.matrix_basis.copy()
                for pb in arm_obj.pose.bones
            }

            armature_pose_toggle_cache["is_rest_mode"] = True
            armature_pose_toggle_cache["armature_name"] = arm_obj.name
            armature_pose_toggle_cache["bone_mats"] = cached_basis_mats

            # Reset pose input to identity relative to rest pose
            for pb in arm_obj.pose.bones:
                pb.matrix_basis = Matrix.Identity(4)

            context.view_layer.update()

            self.report(
                {'INFO'},
                f"Set armature '{arm_obj.name}' to rest pose and cached articulated pose."
            )
            return {'FINISHED'}

        # Restore articulated pose from cache
        if armature_pose_toggle_cache["armature_name"] != arm_obj.name:
            self.report(
                {'ERROR'},
                "Cached articulated pose belongs to another armature. "
                "Toggle back with the original armature selected."
            )
            return {'CANCELLED'}

        cached_basis_mats = armature_pose_toggle_cache["bone_mats"]
        if not cached_basis_mats:
            self.report({'ERROR'}, "No cached articulated pose available to restore.")
            return {'CANCELLED'}

        for pb in arm_obj.pose.bones:
            mat = cached_basis_mats.get(pb.name)
            if mat is not None:
                pb.matrix_basis = mat.copy()

        context.view_layer.update()

        armature_pose_toggle_cache["is_rest_mode"] = False
        armature_pose_toggle_cache["armature_name"] = None
        armature_pose_toggle_cache["bone_mats"] = {}

        self.report(
            {'INFO'},
            f"Restored cached articulated pose for armature '{arm_obj.name}'."
        )
        return {'FINISHED'}


class SYNTH_OT_set_bone_prior_from_pose(Operator):
    """
    Button for setting an swing-twist prior for either swing_x, swing_z or twist angle of a single bone.
    Association between button and corresponding text input field is achieved via syncing 
    the two members bone_name and field_name to the corresponding members of the text field.
    Attention:
    1) The angle is computed from the current pose of the armature relative to the rest pose, so make sure the armature is in the desired pose before clicking the button.
    2) The calculated angle is only accurate if the bone is rotated purely about the corresponding local axis (X for swing_x, Z for swing_z, Y for twist).
    """
    bl_idname = "synth.set_bone_prior_from_pose"
    bl_label = "set"
    bl_description = "set angle from the current armature deformation"

    # these are set when instantiating the button
    bone_name: StringProperty(name="Bone Name", default="")
    field_name: StringProperty(name="Field Name", default="") # swing_x, swing_z, or twist_y

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props

        arm_obj, _ = sync_bone_priors_ui_item_collection(scene)
        if arm_obj is None:
            self.report({'ERROR'}, "No armature found on selected object.")
            return {'CANCELLED'}

        deps = context.evaluated_depsgraph_get()
        deps.update()
        arm_eval = arm_obj.evaluated_get(deps)
        arm_rest = arm_obj.data
        pb = arm_eval.pose.bones.get(self.bone_name)
        rb = arm_rest.bones.get(self.bone_name)
        if pb is None or rb is None:
            self.report({'ERROR'}, f"Bone '{self.bone_name}' not found in armature '{arm_obj.name}'")
            return {'CANCELLED'}

        # CLAUDE FIX (B8): `Bone.matrix` is a 3x3 that is ALREADY expressed in the bone's parent
        # space, so `rb.parent.matrix.inverted() @ rb.matrix` multiplied two matrices living in
        # different parents' spaces, and the result was then compared against `PoseBone.matrix`,
        # which is a 4x4 in ARMATURE space. Use `Bone.matrix_local` (armature space) on the rest
        # side so both sides are parent-relative in the same space.
        if rb.parent is not None:
            rest_mat = rb.parent.matrix_local.inverted() @ rb.matrix_local
            pose_mat = pb.parent.matrix.inverted() @ pb.matrix
        else:
            # root bone fallback: use armature-space orientation
            rest_mat = rb.matrix_local
            pose_mat = pb.matrix
        rel_mat = rest_mat.inverted().to_3x3() @ pose_mat.to_3x3()

        # swing, twist = rel_mat.to_quaternion().to_swing_twist('Y')
        # swing_axis_angle = swing.to_axis_angle()
        # self.report({'INFO'}, f"Bone '{self.bone_name}' swing axis: ({swing_axis_angle[0][0]:.2f}, {swing_axis_angle[0][1]:.2f}, {swing_axis_angle[0][2]:.2f}), swing angle: {swing_axis_angle[1]/3.14159*180:.2f} deg, twist angle: {twist/3.14159*180:.2f} deg")
        # swing_twist = {
        #     'swing_x': swing_axis_angle[1]/3.14159*180 if abs(swing_axis_angle[0][0]) > 0.95 else 0, # only set swing_x if swing axis is mostly aligned with local X
        #     'swing_z': swing_axis_angle[1]/3.14159*180 if abs(swing_axis_angle[0][2]) > 0.95 else 0, # only set swing_z if swing axis is mostly aligned with local Z
        #     'twist_y': twist/3.14159*180
        # }
        euler_x = rel_mat.to_euler('XYZ')
        euler_y = rel_mat.to_euler('YZX')
        euler_z = rel_mat.to_euler('ZXY')
        swing_twist = {
            'swing_x': euler_x.x / 3.14159 * 180,
            'swing_z': euler_z.z / 3.14159 * 180,
            'twist_y': euler_y.y / 3.14159 * 180,
        }
        self.report({'INFO'}, f"Bone '{self.bone_name}' {self.field_name}: x: {swing_twist['swing_x']:.2f} deg, z: {swing_twist['swing_z']:.2f} deg, y: {swing_twist['twist_y']:.2f} deg")

        if self.field_name == "swing_x":
            if np.isclose(swing_twist["swing_z"], 0) == False or np.isclose(swing_twist["twist_y"], 0) == False:
                self.report({'ERROR'}, f"Bone '{self.bone_name}' has non-zero swing_z or twist_y, please rotate bone only about its local X axis for accurate swing_x prior")
                return {'CANCELLED'}
            if 170 < swing_twist["swing_x"] < 190:
                self.report({'ERROR'}, f"Bone '{self.bone_name}' has a swing_x angle close to 180 degrees, which can be ambiguous for the swing-twist decomposition. Please rotate bone slightly away from 180 degrees for a more accurate swing_x prior")
                return {'CANCELLED'}
        if self.field_name == "swing_z":
            if np.isclose(swing_twist["swing_x"], 0) == False or np.isclose(swing_twist["twist_y"], 0) == False:
                self.report({'ERROR'}, f"Bone '{self.bone_name}' has non-zero swing_x or twist_y, please rotate bone only about its local Z axis for accurate swing_z prior")
                return {'CANCELLED'}
            if 170 < swing_twist["swing_z"] < 190:
                self.report({'ERROR'}, f"Bone '{self.bone_name}' has a swing_z angle close to 180 degrees, which can be ambiguous for the swing-twist decomposition. Please rotate bone slightly away from 180 degrees for a more accurate swing_z prior")
                return {'CANCELLED'}
        if self.field_name == "twist_y":
            if np.isclose(swing_twist["swing_x"], 0) == False or np.isclose(swing_twist["swing_z"], 0) == False:
                self.report({'ERROR'}, f"Bone '{self.bone_name}' has non-zero swing_x or swing_z, please rotate bone only about its local Y axis for accurate twist_y prior")
                return {'CANCELLED'}

        prior_ui_item = None
        # find corresponding set of text input fields for this button
        for bone_prior_ui_item in p.bone_priors_ui_item_collection:
            if bone_prior_ui_item.bone_name == self.bone_name:
                prior_ui_item = bone_prior_ui_item
                break
        if prior_ui_item is None:
            self.report({'ERROR'}, f"No prior row found for bone '{self.bone_name}'")
            return {'CANCELLED'}

        if not hasattr(prior_ui_item, self.field_name):
            self.report({'ERROR'}, f"Unknown prior field '{self.field_name}'")
            return {'CANCELLED'}

        # set the text input field with the correct name to the calculated value
        setattr(prior_ui_item, self.field_name, swing_twist[self.field_name])
        self.report({'INFO'}, f"{self.field_name} of {prior_ui_item.bone_name} was set to {swing_twist[self.field_name]:.2f} degrees based on current pose")
        return {'FINISHED'}


# --------------------------- Mesh export operator ----------------

class SYNTH_OT_export_mesh(Operator):
    bl_idname = "synth.export_mesh"
    bl_label = "Export Template Mesh JSON"
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



# --------------------------- Pose Time Series (schema v2) ------------------------------
#
# CLAUDE FIX (A1-A9): the exchange format between this add-on and the 4D-reconstruction
# module is now `pose_time_series/2`. The previous version exported
# `parent_pose_bone.matrix.inverted() @ pose_bone.matrix`, which at rest equals the bone's
# REST relative orientation rather than the identity, so it was not the quantity
# LBS_edit.LBS consumes (a delta-from-rest rotation in template axes). The importer then
# tried to repair that with a conjugation through `rest_rot_world`, walked a *linear* chain
# instead of the real (branching) bone tree, and rebuilt the translation from the wrong
# bone's rest vector.
#
# Convention, matching LBS_edit.LBS exactly:
#     D(b)      = R_pose(b) @ R_rest(b)^-1                 (armature space)
#     body_pose = expmap( D(parent)^-1 @ D(b) )            (== 0 in the rest pose)
#     head(b)   = head(parent) + D(parent) @ (H(b) - H(parent)) * L(parent)
#     world(b)  = Translation(head(b)) @ (D(b) @ R_rest(b))
# See pose_time_series_schema_v2.md for the full specification.

POSE_TIME_SERIES_SCHEMA = "pose_time_series/2"
_PTS_ANG_EPS = 1e-12
_PTS_VIRTUAL_WARN = 1e-3


def _rot3(mat):
    """Orthonormal 3x3 rotation part of a possibly scaled/sheared matrix."""
    return mat.to_3x3().to_quaternion().to_matrix()


def _expmap(rot3):
    return rot3.to_quaternion().to_exponential_map()


def _from_expmap(vec):
    v = Vector(vec)
    a = v.length
    if a < _PTS_ANG_EPS:
        return Matrix.Identity(3)
    return Matrix.Rotation(a, 3, v / a)


def _matrix3_from_rows(rows):
    """Build a 3x3 Matrix from nested list-of-rows as stored in get_mesh_json."""
    return Matrix(((rows[0][0], rows[0][1], rows[0][2]),
                   (rows[1][0], rows[1][1], rows[1][2]),
                   (rows[2][0], rows[2][1], rows[2][2])))


def _ensure_collection(name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _frame_from_y_and_ref(y_axis, ref):
    """Orthonormal frame with +Y along `y_axis`, rolled by `ref`.

    Same construction as get_virtual_bone_rest_matrix_from_bones, so a virtual bone's posed
    frame collapses onto its rest frame when the rig is in its rest pose.
    """
    y = y_axis.normalized()
    if abs(ref.normalized().dot(y)) > 0.999:
        ref = ref.orthogonal()
    x = ref.cross(y)
    if x.length < 1e-8:
        raise ValueError("degenerate x axis while building a virtual bone frame")
    x.normalize()
    z = y.cross(x)
    return Matrix((x, y, z)).transposed()          # columns = x, y, z


def _pts_rest_tables(mesh_info, arm_obj):
    """Rest geometry of every bone (real + virtual) in ARMATURE space."""
    order = list(mesh_info["bone_order"])
    tree = mesh_info["bone_names_tree"]
    virtual = set(mesh_info["virtual_bone_names"])
    joints = mesh_info["J"]
    joint_parent = mesh_info["kintree_table"][0]

    roots = [b for b in order if not tree[b]["p"]]
    if len(roots) != 1:
        raise ValueError(f"pose_time_series requires exactly one root bone, found {roots}")
    if order[0] != roots[0]:
        raise ValueError("bone_order[0] is not the root bone")

    aw_inv3 = arm_obj.matrix_world.inverted().to_3x3()
    rest_R, rest_head, rest_len = {}, {}, {}
    for b in order:
        tail_j = tree[b]["joints"][1]
        head_j = joint_parent[tail_j]
        rest_head[b] = Vector(joints[head_j])
        rest_len[b] = float((Vector(joints[tail_j]) - rest_head[b]).length) or 1.0
        rows = tree[b].get("rest_rot")
        if rows:
            rest_R[b] = _matrix3_from_rows(rows)                      # already armature space
        else:
            # legacy template: rest_rot_world was pre-multiplied by arm.matrix_world
            rest_R[b] = _rot3((aw_inv3 @ _matrix3_from_rows(tree[b]["rest_rot_world"])).to_4x4())
    return order, tree, virtual, rest_R, rest_head, rest_len


def _pts_posed_armature_space(order, tree, virtual, rest_R, pose_bones):
    """Return (P, D, seg) for one frame, all in armature space.

    P[b]   4x4 pose matrix of the bone (synthesised for virtual bones)
    D[b]   3x3 delta-from-rest rotation  (R_pose @ R_rest^-1)
    seg[b] posed head->tail length of the bone
    """
    P, D, seg = {}, {}, {}
    for b in order:                                     # BFS order: parents come first
        parent = tree[b]["p"]
        if b in virtual:
            child = tree[b]["c"][0]
            if parent not in pose_bones or child not in pose_bones:
                raise ValueError(f"virtual bone '{b}' references missing pose bones")
            head = Vector(pose_bones[parent].tail)
            gap = Vector(pose_bones[child].head) - head
            # CLAUDE FIX (B15): degenerate gaps are an error, not a silent fallback to the
            # parent's posed rotation (which disagreed with the rest-side construction).
            if gap.length < 1e-8:
                raise ValueError(f"virtual bone '{b}' is degenerate in this frame")
            # CLAUDE FIX (B16): the roll reference is the parent's POSED local Z, not its rest Z.
            # With the rest Z, a rigid rotation of the parent gave the virtual bone a spurious
            # twist, contradicting `virtual_bone_mask` (LBS forces virtual bones to identity).
            ref = D[parent] @ (rest_R[parent] @ Vector((0.0, 0.0, 1.0)))
            P[b] = Matrix.Translation(head) @ _frame_from_y_and_ref(gap, ref).to_4x4()
            seg[b] = float(gap.length)
        else:
            pb = pose_bones[b]
            P[b] = pb.matrix.copy()
            seg[b] = float((Vector(pb.tail) - Vector(pb.head)).length)
        D[b] = _rot3(P[b]) @ rest_R[b].inverted()
    return P, D, seg


def _pts_solve_frame(entry, order, tree, rest_R, rest_head, arm_world):
    """Armature-space pose matrices for every bone of one frame (normative chain)."""
    root = order[0]
    idx = {b: i - 1 for i, b in enumerate(order) if i}

    aw_inv = arm_world.inverted()
    awR = _rot3(arm_world)

    body_pose = entry.get("body_pose", [])
    body_len = entry.get("body_bone_length", [])
    if len(body_pose) != len(order) - 1 or len(body_len) != len(order) - 1:
        raise ValueError("body_pose / body_bone_length length does not match bone_order")

    L = {root: float(entry.get("root_bone_length", 1.0))}
    for b in order[1:]:
        L[b] = float(body_len[idx[b]])

    D = {root: awR.inverted() @ _from_expmap(entry["global_ori"])}
    head = {root: aw_inv @ Vector(entry["global_t"])}
    P = {}
    for b in order:
        parent = tree[b]["p"]
        if parent:
            D[b] = D[parent] @ _from_expmap(body_pose[idx[b]])
            head[b] = head[parent] + D[parent] @ ((rest_head[b] - rest_head[parent]) * L[parent])
        P[b] = Matrix.Translation(head[b]) @ (D[b] @ rest_R[b]).to_4x4()
    return P


def _pts_find_source(context):
    """(source mesh object, its armature) for the object selected in the UI."""
    p = context.scene.synth_props
    col = bpy.data.collections.get(p.collection_name)
    if col is None:
        raise ValueError(f"Collection '{p.collection_name}' not found")
    obj = col.objects.get(p.object_name)
    if obj is None:
        raise ValueError(f"Object '{p.object_name}' not found in '{p.collection_name}'")
    arm = None
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object:
            arm = mod.object
            break
    if arm is None:
        raise ValueError("Target object has no armature modifier.")
    return obj, arm


class SYNTH_OT_export_pose_time_series_json(Operator):
    bl_idname = "synth.export_pose_time_series_json"
    bl_label = "Export Pose Time Series (JSON)"
    bl_description = ("Export per-frame root position, delta-from-rest bone rotations "
                      "(exponential map, template axes) and bone length factors to a "
                      "pose_time_series/2 JSON file")

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props

        # CLAUDE FIX (B12): every failure path returns {'CANCELLED'}; the old code raised a bare
        # Exception out of execute() for some of them, which surfaced as an operator traceback.
        try:
            _, arm_obj = _pts_find_source(context)
            mesh_info = get_mesh_json(context)
            order, tree, virtual, rest_R, rest_head, rest_len = _pts_rest_tables(mesh_info, arm_obj)
        except Exception as exc:
            self.report({'ERROR'}, f"Template inspection failed: {exc}")
            return {'CANCELLED'}

        root = order[0]
        out_dir = resolve(p.annot_out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"pose_time_series_{p.collection_name}_{p.object_name}.json")

        deps = context.evaluated_depsgraph_get()
        frame_start, frame_end = scene.frame_start, scene.frame_end
        try:
            fps = float(scene.render.fps) / float(scene.render.fps_base)
        except Exception:
            fps = float(scene.render.fps)

        data = {
            "meta": {
                "schema": POSE_TIME_SERIES_SCHEMA,
                "producer": "synthetic_data_generator_ui.py",
                "armature": arm_obj.name,
                "bone_order": order,
                "body_pose_bone_order": order[1:],
                "virtual_bone_names": sorted(virtual),
                "virtual_bone_mask": [1 if b in virtual else 0 for b in order],
                "rotation": "axis_angle_exponential_map",
                "space": ("global_t/global_ori in Blender world; body_pose in template "
                          "(armature) axes, delta from rest"),
                "units": "meters",
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "fps": float(fps),
            },
            "frames": [],
        }

        original_frame = scene.frame_current
        virtual_warned = False
        try:
            for frame in range(frame_start, frame_end + 1):
                scene.frame_set(frame)
                deps.update()
                arm_eval = arm_obj.evaluated_get(deps)
                pose_bones = arm_eval.pose.bones

                try:
                    P, D, seg = _pts_posed_armature_space(order, tree, virtual, rest_R, pose_bones)
                except Exception as exc:
                    self.report({'ERROR'}, f"Frame {frame}: {exc}")
                    return {'CANCELLED'}

                arm_world = arm_eval.matrix_world
                global_t = arm_world @ Vector(P[root].translation)
                global_ori = _expmap(_rot3(arm_world) @ D[root])

                body_pose, body_len = [], []
                for b in order[1:]:
                    exp = _expmap(D[tree[b]["p"]].inverted() @ D[b])
                    if b in virtual and exp.length > _PTS_VIRTUAL_WARN and not virtual_warned:
                        virtual_warned = True
                        self.report({'WARNING'},
                                    f"virtual bone '{b}' rotates by {exp.length:.3f} rad at frame "
                                    f"{frame}; LBS forces virtual bones to identity, so the "
                                    f"reconstruction cannot reproduce this rig exactly.")
                    body_pose.append([float(exp.x), float(exp.y), float(exp.z)])
                    body_len.append(float(seg[b] / rest_len[b]))

                entry = {
                    "frame": int(frame),
                    "time": float((frame - frame_start) / fps) if fps else 0.0,
                    "global_t": [float(global_t.x), float(global_t.y), float(global_t.z)],
                    "global_ori": [float(global_ori.x), float(global_ori.y), float(global_ori.z)],
                    "body_pose": body_pose,
                    "body_bone_length": body_len,
                }
                root_len = float(seg[root] / rest_len[root])
                if abs(root_len - 1.0) > 1e-6:
                    # the reconstruction pins the root bone's length to 1.0 (fish_model prepends a
                    # 1.0), so this field is only honoured on a Blender -> Blender round trip
                    entry["root_bone_length"] = root_len
                data["frames"].append(entry)
        finally:
            scene.frame_set(original_frame)

        try:
            with open(out_path, 'w') as jf:
                json.dump(data, jf, indent=2)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not write {out_path}: {exc}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Wrote {len(data['frames'])} frames to {out_path}")
        return {'FINISHED'}


def _pts_disconnect_bones(context, arm_obj, report=None):
    """`use_connect` locks a pose bone's location channel; bone length factors need it free."""
    view_layer = context.view_layer
    prev_active = view_layer.objects.active
    try:
        view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        for eb in arm_obj.data.edit_bones:
            eb.use_connect = False
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception as exc:
        if report:
            report({'WARNING'}, f"Could not disconnect bones ({exc}); bone length factors other "
                                f"than 1.0 will be ignored.")
    finally:
        try:
            view_layer.objects.active = prev_active
        except Exception:
            pass


def create_animation_from_pose_time_series(context, timeseries_path, report=None):
    """Rebuild a Blender animation from a pose_time_series/2 JSON.

    Returns (new_arm_obj, new_mesh_obj).
    """
    scene = context.scene

    with open(timeseries_path, 'r') as f:
        ts = json.load(f)
    meta = ts.get("meta", {})
    if meta.get("schema") != POSE_TIME_SERIES_SCHEMA:
        raise ValueError(
            f"expected schema '{POSE_TIME_SERIES_SCHEMA}', got '{meta.get('schema')}'. "
            f"Files written before this fix fold each bone's rest orientation into the pose "
            f"channel and cannot be converted without the rig they came from; re-export them."
        )

    src_obj, src_arm = _pts_find_source(context)
    mesh_info = get_mesh_json(context)
    order, tree, virtual, rest_R, rest_head, rest_len = _pts_rest_tables(mesh_info, src_arm)

    # CLAUDE FIX (A8): a bone-order mismatch used to be a print. The importer indexes
    # `bone_names_tree`, `J` and `kintree_table` from the CURRENT template while walking the
    # FILE's order, so a mismatch silently produces garbage. Refuse instead.
    if list(meta.get("bone_order", [])) != order:
        raise ValueError("bone_order in the JSON does not match the current template; refusing to "
                         "import because the bone indices would be silently wrong.")

    # ---- duplicate object + armature into 'Reconstructions'
    recon = _ensure_collection("Reconstructions")
    new_obj = src_obj.copy()
    new_obj.data = src_obj.data.copy()
    new_obj.animation_data_clear()
    recon.objects.link(new_obj)

    new_arm = src_arm.copy()
    new_arm.data = src_arm.data.copy()
    new_arm.animation_data_clear()
    recon.objects.link(new_arm)

    for mod in new_obj.modifiers:
        if mod.type == 'ARMATURE':
            mod.object = new_arm
    # parent the mesh to the duplicated armature so that it inherits the per-frame `scale`
    new_obj.parent = new_arm
    new_obj.matrix_parent_inverse = Matrix.Identity(4)
    new_obj.matrix_local = src_arm.matrix_world.inverted() @ src_obj.matrix_world

    _pts_disconnect_bones(context, new_arm, report)

    action = bpy.data.actions.new(name=f"recon_action_{new_arm.name}")
    new_arm.animation_data_create()
    new_arm.animation_data.action = action

    pose_bones = new_arm.pose.bones
    data_bones = new_arm.data.bones
    base_world = src_arm.matrix_world.copy()
    new_arm.rotation_mode = 'QUATERNION'

    # CLAUDE FIX (A7): rotation_mode must be set BEFORE any matrix is written, otherwise the
    # decomposition lands in a different channel than the one that gets keyframed.
    for pb in pose_bones:
        pb.rotation_mode = 'QUATERNION'

    frames = ts["frames"]
    for entry in frames:
        f = int(entry["frame"])
        s = float(entry.get("scale", 1.0) or 1.0)
        arm_world = base_world @ Matrix.Scale(s, 4)
        new_arm.matrix_world = arm_world
        new_arm.keyframe_insert(data_path="location", frame=f)
        new_arm.keyframe_insert(data_path="rotation_quaternion", frame=f)
        new_arm.keyframe_insert(data_path="scale", frame=f)

        P = _pts_solve_frame(entry, order, tree, rest_R, rest_head, arm_world)

        for b in order:
            if b in virtual:
                continue                                   # chain-only, no Blender bone
            db = data_bones.get(b)
            if db is None:
                continue
            # Blender's parent, NOT the tree parent: virtual bones do not exist in the armature
            bl_parent = db.parent
            if bl_parent is None:
                basis = db.matrix_local.inverted() @ P[b]
            else:
                rest_rel = bl_parent.matrix_local.inverted() @ db.matrix_local
                basis = rest_rel.inverted() @ (P[bl_parent.name].inverted() @ P[b])
            pb = pose_bones[b]
            # CLAUDE FIX (A7): write matrix_basis, not pose_bone.matrix. The `matrix` setter
            # solves against the parent's *currently evaluated* matrix, so writing parents and
            # children in the same tick without a depsgraph update solved children against a
            # stale parent. matrix_basis is purely local and has no such dependency.
            pb.matrix_basis = basis
            pb.keyframe_insert(data_path="location", frame=f)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=f)
            pb.keyframe_insert(data_path="scale", frame=f)

    if frames:
        scene.frame_start = int(meta.get("frame_start", frames[0]["frame"]))
        scene.frame_end = int(meta.get("frame_end", frames[-1]["frame"]))

    context.view_layer.update()
    return new_arm, new_obj


class SYNTH_OT_create_animation_from_pose_time_series(Operator, ImportHelper):
    """Create an animation on a duplicated object from a pose_time_series/2 JSON"""
    bl_idname = "synth.create_animation_from_pose_time_series"
    bl_label = "Create Animation from Pose Time Series"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="pose_time_series_*.json;*.json", options={'HIDDEN'})

    def execute(self, context):
        try:
            new_arm, new_obj = create_animation_from_pose_time_series(
                context, self.filepath, report=self.report)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Failed to create animation: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Created '{new_obj.name}' + '{new_arm.name}' in 'Reconstructions'")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SYNTH_OT_verify_pose_time_series_roundtrip(Operator, ImportHelper):
    """Re-solve a pose_time_series JSON and compare it against the live armature"""
    bl_idname = "synth.verify_pose_time_series_roundtrip"
    bl_label = "Verify Pose Time Series Round Trip"
    bl_description = ("Recompute every pose bone matrix from the JSON and compare it, frame by "
                      "frame, against the source armature. Reports the max absolute error.")

    filename_ext = ".json"
    filter_glob: StringProperty(default="pose_time_series_*.json;*.json", options={'HIDDEN'})

    def execute(self, context):
        scene = context.scene
        try:
            _, src_arm = _pts_find_source(context)
            with open(self.filepath) as f:
                ts = json.load(f)
            if ts.get("meta", {}).get("schema") != POSE_TIME_SERIES_SCHEMA:
                raise ValueError(f"not a {POSE_TIME_SERIES_SCHEMA} file")
            mesh_info = get_mesh_json(context)
            order, tree, virtual, rest_R, rest_head, _ = _pts_rest_tables(mesh_info, src_arm)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        deps = context.evaluated_depsgraph_get()
        original = scene.frame_current
        worst, worst_bone, worst_frame = 0.0, "", -1
        try:
            for entry in ts["frames"]:
                f = int(entry["frame"])
                scene.frame_set(f)
                deps.update()
                arm_eval = src_arm.evaluated_get(deps)
                P_ref, _, _ = _pts_posed_armature_space(order, tree, virtual, rest_R,
                                                        arm_eval.pose.bones)
                P = _pts_solve_frame(entry, order, tree, rest_R, rest_head, arm_eval.matrix_world)
                for b in order:
                    err = max(abs(P[b][r][c] - P_ref[b][r][c]) for r in range(4) for c in range(4))
                    if err > worst:
                        worst, worst_bone, worst_frame = err, b, f
        except Exception as exc:
            self.report({'ERROR'}, f"Verification failed: {exc}")
            return {'CANCELLED'}
        finally:
            scene.frame_set(original)

        level = 'INFO' if worst < 1e-5 else 'WARNING'
        self.report({level}, f"round-trip max |err| = {worst:.3e} "
                             f"(bone '{worst_bone}', frame {worst_frame})")
        return {'FINISHED'}


# --------------------------- UI & Registration ----------------------------------

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
        box.label(text="Cameras")
        cam_objects = get_scene_cameras_sorted()
        selection_by_name = {item.camera_name: item for item in p.camera_selections}
        if not cam_objects:
            box.label(text="No cameras found in scene.")
        else:
            for cam in cam_objects:
                item = selection_by_name.get(cam.name)
                if item is not None:
                    box.prop(item, 'enabled', text=cam.name)
                else:
                    box.label(text=f"{cam.name} (will be added on render queue build)", icon='ERROR')
        box = layout.box()
        box.label(text="Target Object & Keypoints")
        box.prop(p, 'collection_name')
        box.prop(p, 'object_name')
        box.prop(p, 'keypoint_list_csv')
        box.operator('synth.export_keypoint_list', icon='EXPORT')
        box = layout.box()
        box.label(text="Bone Groups")

        row = box.row()
        row.template_list(
            "SYNTH_UL_bone_groups",          # list type
            "",                              # list id
            context.scene.synth_props, "bone_groups",  # data & prop
            context.scene.synth_props, "bone_groups_index",  # active index
            rows=3
        )

        col = row.column(align=True)
        col.operator("synth.bone_group_add", icon="ADD", text="")
        col.operator("synth.bone_group_remove", icon="REMOVE", text="")

        box = layout.box()
        header_row = box.row(align=True)
        header_row.label(text="Template Priors (degrees, relative to parent)")
        header_row.operator("synth.refresh_bone_priors_ui_item_collection", icon='FILE_REFRESH', text="Refresh bone list in GUI")
        toggle_text = "Restore Articulated Pose" if armature_pose_toggle_cache["is_rest_mode"] else "Set Rest Pose"
        header_row.operator("synth.toggle_rest_pose_articulated_pose", icon='ARMATURE_DATA', text=toggle_text)

        arm_obj, bone_names = get_target_armature_bone_names_sorted(scene)
        prior_ui_item_by_bone_name = {item.bone_name: item for item in p.bone_priors_ui_item_collection}
        if arm_obj is None:
            box.label(text="No armature found on selected object.", icon='ERROR')
        elif not bone_names:
            box.label(text="No bones found on armature.", icon='ERROR')
        else:
            if armature_pose_toggle_cache["is_rest_mode"]:
                box.label(text="Rest pose mode active. Press toggle again to restore cached articulated pose.")
            else:
                box.label(text="Articulated pose mode active. Press toggle to cache the current pose and temporarily set the model to rest pose.")

            box.label(text="")
            explanation_row = box.row(align=False)
            explanation_icon = 'TRIA_DOWN' if p.show_priors_explanation else 'TRIA_RIGHT'
            explanation_row.prop(p, 'show_priors_explanation', text="Show/Hide Explanation", icon=explanation_icon, emboss=False)
            if p.show_priors_explanation:
                box.label(text="Explanation: Three values are required to set a bone prior: swing about local X, swing about local Z, and twist about local Y. The GUI allows you to set these values based on the current pose. For each bone, click the 'set' button next to a prior to set that prior to the angle of the current pose about the corresponding axis.", icon='INFO')
                box.label(text="Note: the angles shown in the GUI are in degrees relative to the rest pose. (They will be converted to radians when exporting the template.)")
                box.label(text="Attention: For setting a prior for a certain axis:", icon='INFO')
                box.label(text="1) Set the model to rest pose via the toggle button (see above).")
                box.label(text="2) Go to pose mode and select the bone in question.")
                box.label(text="3) press 'R' and then press the name of the axis ('X', 'Y', or 'Z') *twice* in order to rotate about the bones local axis.")
                box.label(text="4) Do not rotate about any other axis.")
                box.label(text="4.5) Click the 'set' button next to the prior you want to set for that bone. (You can also type the angle manually without changing any bone pose. No need to press \"set\" then.)")
                box.label(text="5) Repeat for the other axes.")
                box.label(text="6) Restore the articulated pose by pressing the toggle button.")
                box.label(text="")
            box.label(text="Angle about bone local...")
            row = box.row(align=True)
            for angle_prior_name in ['', 'X', 'Z', 'Y']:
                row.label(text=angle_prior_name)

            box.label(text="Which will be assigned to be the maximum of...")
            row = box.row(align=True)
            for angle_prior_description in ['', 'Swing X', 'Swing Z', 'Twist Y']:
                row.label(text=angle_prior_description)
            row = box.row(align=True)
            for angle_prior_max_info in ['Maximum:', '180', '180', '360']:
                row.label(text=angle_prior_max_info)
            for bone_name in bone_names:
                # get corresponding ui item
                bone_prior_ui_row = prior_ui_item_by_bone_name.get(bone_name)
                if bone_prior_ui_row is None:
                    box.label(text=f"{bone_name} (missing row; click Refresh)", icon='ERROR')
                    continue

                row = box.row(align=True)
                row.label(text=bone_name)
                
                # create a property in the row for each of the prior angles
                # from the documentation:
                # bpy.types.UILayout.prop:
                # Parameters:
                #   data (AnyType, (never None)) – Data from which to take property
                #   property (string, (never None)) – Identifier of property in data
                #   text (string, (optional)) – Override automatic text of the item
                row.prop(bone_prior_ui_row, 'swing_x', text="Swing X")
                op = row.operator("synth.set_bone_prior_from_pose", text="set")
                op.bone_name = bone_name
                op.field_name = "swing_x"

                row.prop(bone_prior_ui_row, 'swing_z', text="Swing Z")
                op = row.operator("synth.set_bone_prior_from_pose", text="set")
                op.bone_name = bone_name
                op.field_name = "swing_z"

                row.prop(bone_prior_ui_row, 'twist_y', text="Twist Y")
                op = row.operator("synth.set_bone_prior_from_pose", text="set")
                op.bone_name = bone_name
                op.field_name = "twist_y"

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
        row = layout.row()
        row.operator('synth.export_camera_matrices', icon='FILE_FOLDER')
        row.operator('synth.export_mesh', icon='FILE_FOLDER')
        row = layout.row()
        row.operator('synth.export_pose_time_series_json', icon='SEQUENCE')
        row.operator('synth.create_animation_from_pose_time_series', icon='IMPORT')
        row = layout.row()
        row.operator('synth.verify_pose_time_series_roundtrip', icon='CHECKMARK')


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


classes = (
    SYNTH_BoneGroupItem,
    SYNTH_CameraSelectionItem,
    SYNTH_BonePriorItem,
    SYNTH_PropertyGroup,
    SYNTH_OT_apply_settings,
    SYNTH_OT_load_config,
    SYNTH_OT_export_keypoint_list,
    SYNTH_PT_main_panel,
    SYNTH_UL_bone_groups,
    SYNTH_OT_bone_group_add,
    SYNTH_OT_bone_group_remove,
    SYNTH_OT_refresh_bone_priors_ui_item_collection,
    SYNTH_OT_toggle_rest_pose_articulated_pose,
    SYNTH_OT_set_bone_prior_from_pose,
    SYNTH_OT_unregister_timed_render,
    TimedRender,
    SYNTH_OT_export_camera_matrices,
    SYNTH_OT_export_mesh,
    SYNTH_OT_create_videos,
    SYNTH_OT_export_pose_time_series_json,
    SYNTH_OT_create_animation_from_pose_time_series,
    SYNTH_OT_verify_pose_time_series_roundtrip,
)



def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.synth_props = PointerProperty(type=SYNTH_PropertyGroup)
    for scene in bpy.data.scenes:
        try:
            sync_camera_selections(scene)
        except Exception:
            pass
        try:
            sync_bone_priors_ui_item_collection(scene)
        except Exception:
            pass


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, 'synth_props'):
        del bpy.types.Scene.synth_props


if __name__ == '__main__':
    register()