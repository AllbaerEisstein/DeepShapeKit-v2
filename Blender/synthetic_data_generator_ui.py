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

# CONTENTS
#   GLOBALS
#   PROPERTY GROUP (scene.synth_props)
#   UTILITIES -- paths, scene lookups, camera intrinsics
#   MESH / KEYPOINT EXTRACTION (depsgraph-evaluated)
#   TEMPLATE EXPORT -- get_mesh_json (bones, joints, skinning weights)
#   VISIBILITY / OCCLUSION
#   PROJECTION HELPERS
#   MASK RENDERING & YOLO LABEL WRITERS
#   TIMEDRENDER OPERATOR (modal render + annotation queue)
#   YOLO DATASET HELPERS
#   SETTINGS OPERATORS (apply / load config / keypoint list)
#   CAMERA MATRIX EXPORT
#   BONE GROUPS & BONE PRIORS OPERATORS
#   MESH / TEMPLATE EXPORT OPERATOR
#   CREATE VIDEOS OPERATOR
#   POSE TIME SERIES (schema v2) -- export / import / verify
#   RECONSTRUCTION EVALUATION -- volumetric 3D IoU & keypoint distances
#   RECONSTRUCTION EVALUATION -- MPVE / MPJPE / per-bone SO(3) geodesic error
#   UI PANEL & REGISTRATION

import csv
import pickle
import bpy
import os
import json
import shutil
import re
import glob
import math
import time
from collections import defaultdict
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy.props import (
    StringProperty, BoolProperty, FloatProperty, IntProperty,
    PointerProperty, CollectionProperty, EnumProperty,
)
from bpy_extras.io_utils import ImportHelper
from bpy.types import Panel, Operator, PropertyGroup, UIList 
import cv2
import numpy as np


# =============================================================================
# GLOBALS
# =============================================================================

# cache for camera matrices (per-camera)
cam_name_2_matrix = {}

# cache for priors UI convenience toggle
armature_pose_toggle_cache = {
    "is_rest_mode": False,
    "armature_name": None,
    "bone_mats": {},
}

# SPEEDUP (S1): the evaluated-mesh extraction depends on the FRAME only, never on the camera,
# but the render queue visits every camera for every frame. Holding the last extraction lets all
# cameras of one frame share a single to_mesh()/vertex-group scan. Keyed by
# (collection, object, frame, keypoint tuple); a single slot is enough because the queue is
# ordered frame-major (see TimedRender.execute).
_deformed_mesh_cache = {"key": None, "value": None}


def invalidate_deformed_mesh_cache():
    _deformed_mesh_cache["key"] = None
    _deformed_mesh_cache["value"] = None


def get_deformed_mesh_data_cached(deps, collection_name, object_name, kpt_list, frame):
    """get_deformed_mesh_data() memoised on (object, frame, keypoints).

    `frame` must identify the evaluated state; TimedRender invalidates the cache whenever it
    calls frame_set, so a stale entry cannot outlive the frame it was built for.
    """
    key = (collection_name, object_name, int(frame), tuple(kpt_list))
    if _deformed_mesh_cache["key"] == key and _deformed_mesh_cache["value"] is not None:
        return _deformed_mesh_cache["value"]
    value = get_deformed_mesh_data(deps, collection_name, object_name, kpt_list)
    _deformed_mesh_cache["key"] = key
    _deformed_mesh_cache["value"] = value
    return value


# SPEEDUP (S4): datablocks for the material-override binary render. Creating two materials, a
# world and their node trees -- and removing them again -- for every single mask render is pure
# overhead; they are identical every time, so they are built once and reused for the whole run.
_mask_render_datablocks = {"white": None, "black": None, "world": None}

EMPTY_KPT_SET = frozenset()

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


# =============================================================================
# PROPERTY GROUP (scene.synth_props)
# =============================================================================
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

    seconds_per_timer_tick: FloatProperty(
        name="Work Per Tick (s)",
        description=(
            "How long the render queue is allowed to keep working before handing control back "
            "to Blender. The Timer Interval is idle time paid ONCE PER BATCH instead of once "
            "per queue item, so raising this cuts the fixed overhead of long queues. Lower it "
            "if the UI feels unresponsive or ESC reacts too slowly"
        ),
        default=2.0,
        min=0.0
    )

    use_persistent_render_data: BoolProperty(
        name="Persistent Render Data",
        description=(
            "Keep the synced scene in memory between renders (Cycles: Performance > Final "
            "Render > Persistent Data). Avoids re-syncing the whole scene for every frame and "
            "every camera, which is a large speedup for long queues, at the cost of higher "
            "memory use. Off by default: the binary pass swaps every material in the file twice "
            "per frame, so check a handful of masks against a non-persistent run before "
            "enabling it for a full dataset. Restored after the run"
        ),
        default=False
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

    # Reconstruction-dataset export: a second, parallel output tree in the schema the DSKv2
    # reconstruction pipeline consumes (the one extract_frames_edit.py writes), so the same
    # render pass that produces YOLO training labels also produces annotated ground truth that
    # can be pointed at `--ground-truth-dataset` or `--reconstruct-from-gt`.
    create_reconstruction_dataset: BoolProperty(
        name="Create Reconstruction Dataset",
        description=(
            "Additionally write a DSKv2-schema reconstruction dataset (origin/, cropped/, mask/, "
            "mask_full/, files.csv, files_crop.csv, keypoints_confs.pickle, keypoints_gt.pickle, "
            "index.json) alongside the YOLO export. Requires 'Render Binary Masks', since the "
            "binary mask render is the ground-truth segmentation"
        ),
        default=False
    )

    reconstruction_dataset_out_dir: StringProperty(
        name="Reconstruction Dataset Dir",
        description="Output root for the DSKv2-schema reconstruction dataset",
        subtype='DIR_PATH',
        default="//synthetic_data/dataset"
    )

    # Reconstruction evaluation (volumetric IoU / keypoint distances)
    iou_sample_count: IntProperty(
        name="IoU Samples / Frame",
        description=("Monte-Carlo occupancy samples drawn per frame from the union of the two "
                     "meshes' bounding boxes. Fish meshes are far simpler than ShapeNet objects, "
                     "so 20k-50k already gives a standard error well below 0.005 IoU"),
        default=30000,
        min=1000,
        max=1000000,
        step=1000
    )

    iou_random_seed: IntProperty(
        name="IoU Random Seed",
        description="Seed for the sample point RNG; the same seed is reused on every frame so "
                    "the per-frame IoU curve is not contaminated by sampling jitter",
        default=0,
        min=0
    )

    iou_recon_object_name: StringProperty(
        name="Reconstruction Object",
        description="Optional explicit name of the reconstruction mesh in the 'Reconstructions' "
                    "collection. Leave empty to auto-detect the newest copy of the target object",
        default=""
    )

    iou_with_keypoint_distances: BoolProperty(
        name="Also Compute Keypoint Distances",
        description="Compute the per-keypoint 3D distances inside the SAME frame loop as the IoU "
                    "and write the sibling keypoint_distances_*.json. Free: both metrics are "
                    "derived from one evaluated-mesh extraction per object per frame",
        default=True
    )

    kpt_dist_warn_threshold_bl: FloatProperty(
        name="Keypoint Dist Warn (BL)",
        description="If > 0, the final report is raised to WARNING when the largest per-keypoint "
                    "distance over the sequence exceeds this many GT body lengths. Regression "
                    "gate; 0 disables. Renamed from the old metres-valued "
                    "'kpt_dist_warn_threshold' when the 3D metrics moved to body lengths -- a "
                    "value carried over from a .blend saved before that change would silently "
                    "mean something ~10x different, so the old name is deliberately not reused",
        default=0.0,
        min=0.0
    )

    pts2_batch_dir: StringProperty(
        name="PTS2 Batch Dir",
        description="Directory of pose_time_series/2 JSONs -- typically the 'pts2_collected' "
                    "folder written by sweep_view_combinations.py's collect_results(). Every "
                    "*.json in it is imported, scored against the target mesh and deleted again; "
                    "the results are written to collected_3d_metrics.json one level up, next to "
                    "the folder itself",
        subtype='DIR_PATH',
        default="//"
    )

    # Reconstruction evaluation (MPVE / MPJPE / per-bone SO(3) geodesic error)
    mpve_normalization_mode: EnumProperty(
        name="MPVE Colour Scale",
        description="How the per-vertex error is mapped to the colour range of the heat map "
                    "and to the clipped-fraction figure recorded in mpve_*.json",
        items=[
            ('global_p95', "Global p95",
             "One scale across all frames, vmax = the 95th percentile of every (frame, vertex) "
             "error. Frames are comparable"),
            ('global_max', "Global Max",
             "One scale across all frames, vmax = the largest error anywhere. Frames are "
             "comparable, but one outlier frame flattens the rest"),
            ('fixed', "Fixed",
             "vmax taken from 'MPVE Fixed vmax', in the selected unit. Frames are comparable "
             "across RUNS as well, which is what a regression gate needs"),
            ('per_frame', "Per Frame (not comparable)",
             "vmax = that frame's own maximum. NOT comparable across frames: it makes an "
             "excellent frame look identical to a catastrophic one"),
        ],
        default='global_p95'
    )

    mpve_fixed_vmax: FloatProperty(
        name="MPVE Fixed vmax",
        description="Upper end of the colour scale when the normalisation mode is 'Fixed', "
                    "expressed in the selected MPVE unit. Errors above it are clamped to the top "
                    "colour and the clipped fraction is recorded per frame",
        default=0.0,
        min=0.0
    )

    mpve_units: EnumProperty(
        name="MPVE Units",
        description="Unit the MPVE colour scale and the normalised summaries are expressed in. "
                    "Body lengths divide by the per-frame GT L_body, which makes the number "
                    "comparable across fish of different size",
        items=[('meters', "Meters", "Blender world metres"),
               ('body_lengths', "Body Lengths", "Divided by the per-frame GT body length")],
        default='meters'
    )

    mpve_area_weighted: BoolProperty(
        name="Area-Weighted MPVE",
        description="Weight each vertex by 1/3 of the summed area of its incident faces, so the "
                    "mean is over surface area rather than over vertices. Off by default; the "
                    "JSON always records whether the template's tessellation is uniform enough "
                    "for the unweighted mean to be meaningful",
        default=False
    )

    mpjpe_pa_degeneracy_threshold: FloatProperty(
        name="PA Degeneracy Threshold",
        description="Smallest sigma_2/sigma_0 of the Procrustes cross-covariance still accepted. "
                    "With 6-10 keypoints on a fish that is frequently nearly straight the point "
                    "configuration is close to 1-D and the alignment is unstable; below this "
                    "ratio the PA term is reported as null with pa_degenerate: true instead of "
                    "a meaningless number",
        default=1e-3,
        min=0.0,
        max=1.0,
        precision=6
    )

    pck_tau_body_lengths: FloatProperty(
        name="PCK tau (body lengths)",
        description="Threshold of the 3D-PCK outlier rate, in GT body lengths. Deliberately "
                    "generous: a single tail-flip frame dominates the MPJPE mean, so the "
                    "fraction of points beyond tau is the number that exposes it",
        default=0.1,
        min=0.0,
        soft_max=1.0
    )

    bone_err_swing_twist: BoolProperty(
        name="Swing/Twist Decomposition",
        description="Additionally split each bone's rotation error into swing and twist about "
                    "the bone's local Y axis. Twist about a fish's body axis is far less "
                    "observable from silhouettes than swing, so the split shows which DoF the "
                    "multi-view rig actually constrains. Also flags bones whose GT pose sits on "
                    "its swing-twist prior, where the error is structurally floored",
        default=False
    )

    bone_err_prior_tolerance: FloatProperty(
        name="Prior Saturation Tol",
        description="A GT bone counts as sitting on its swing-twist limit when it is within this "
                    "relative tolerance of the prior (elliptical for swing, scalar for twist), "
                    "using the same decomposition as losses_edit.decompose_to_swing_twist",
        default=0.05,
        min=0.0,
        max=1.0
    )

    recon_roundtrip_json: StringProperty(
        name="Round-Trip JSON",
        description="The pose_time_series/2 file the reconstruction was created from. Every "
                    "evaluated frame is re-solved from it and compared against the reconstruction "
                    "armature inside the metric's own frame loop; a failure means the rotation "
                    "metric would be measuring a convention mismatch rather than reconstruction "
                    "error",
        subtype='FILE_PATH',
        default=""
    )

    bone_err_require_roundtrip: BoolProperty(
        name="Require Round-Trip Check",
        description="Refuse to compute the per-bone rotation error unless the round trip above "
                    "has been run and passed. Untick only when you know the two armatures share "
                    "a convention for another reason",
        default=True
    )

    bone_err_roundtrip_tol: FloatProperty(
        name="Round-Trip Tol",
        description="Largest max|err| on a 4x4 pose matrix still accepted by the round-trip "
                    "precondition",
        default=1e-4,
        min=0.0,
        precision=6
    )


# =============================================================================
# UTILITIES -- paths, scene lookups, camera intrinsics
# =============================================================================

def camera_name_to_view_name(cam_name):
    """
    Map a Blender camera name to the view/folder name used everywhere downstream.

    'Camera.003_Fish Top R' -> '003_Fish Top R_Camera'. This expression previously appeared
    verbatim in both TimedRender.make_prefix_cam_frame and export_cam_matrices; the
    reconstruction-dataset export needs the same keys as cam_matrices.json, so all three now
    share this one definition.
    """
    if '.' in cam_name:
        return cam_name.split('.', 1)[1] + '_' + cam_name.split('.', 1)[0]
    return cam_name


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


# --- camera intrinsic helpers ---

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


# =============================================================================
# MESH / KEYPOINT EXTRACTION (depsgraph-evaluated)
# =============================================================================

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

    n_verts = len(mesh_eval.vertices)
    n_polys = len(mesh_eval.polygons)

    # SPEEDUP (S2a): pull coordinates, normals and polygon areas out in bulk with foreach_get
    # instead of one Python attribute access per element, and derive the world-space coordinates
    # with a single (N,3) matrix product instead of one `Matrix @ Vector` per vertex.
    co_flat = np.empty(n_verts * 3, dtype=np.float64)
    mesh_eval.vertices.foreach_get('co', co_flat)
    co_local = co_flat.reshape(-1, 3)

    nrm_flat = np.empty(n_verts * 3, dtype=np.float64)
    mesh_eval.vertices.foreach_get('normal', nrm_flat)
    nrm_local = nrm_flat.reshape(-1, 3)

    m = np.array(obj2world, dtype=np.float64)                  # 4x4, row-major
    co_world = co_local @ m[:3, :3].T + m[:3, 3]
    co_world_t = [tuple(c) for c in co_world.tolist()]         # built once, shared below

    areas = np.empty(n_polys, dtype=np.float64)
    mesh_eval.polygons.foreach_get('area', areas)
    areas = areas.tolist()

    # Polygon vertex indices in bulk. loop_total/loop_start + the loop-vertex array reproduces
    # `poly.vertices` for n-gons without touching each polygon from Python.
    loop_total = np.empty(n_polys, dtype=np.int32)
    loop_start = np.empty(n_polys, dtype=np.int32)
    mesh_eval.polygons.foreach_get('loop_total', loop_total)
    mesh_eval.polygons.foreach_get('loop_start', loop_start)
    n_loops = len(mesh_eval.loops)
    loop_verts = np.empty(n_loops, dtype=np.int32)
    mesh_eval.loops.foreach_get('vertex_index', loop_verts)
    loop_verts_l = loop_verts.tolist()
    loop_total_l = loop_total.tolist()
    loop_start_l = loop_start.tolist()

    faces = []
    for i in range(n_polys):
        s = loop_start_l[i]
        faces.append({
            "id":    i,
            "area":  areas[i],
            "verts": loop_verts_l[s:s + loop_total_l[i]],
        })

    co_local_t = [tuple(c) for c in co_local.tolist()]
    vertices = [{"id": i, "co": co} for i, co in enumerate(co_local_t)]
    normals = [(co_local_t[i], tuple(nv)) for i, nv in enumerate(nrm_local.tolist())]

    # SPEEDUP (S2b): the old code scanned ALL vertices once per vertex group and ran an inner
    # `any(g.group == ...)` over each vertex's group memberships -- O(V * G * groups_per_vertex).
    # One pass over the vertices fills every group at once.
    wanted_group_index_2_name = {
        vg.index: vg.name for vg in obj_eval.vertex_groups if vg.name in kpt_list
    }
    kpt_2_verts_objco = {name: [] for name in wanted_group_index_2_name.values()}
    if wanted_group_index_2_name:
        for vi, v in enumerate(mesh_eval.vertices):
            for g in v.groups:
                name = wanted_group_index_2_name.get(g.group)
                if name is not None:
                    kpt_2_verts_objco[name].append(vi)

    kpt_2_verts_worldco = {
        kpt: [{"id": i, "co": co_world_t[i]} for i in idx_list]
        for kpt, idx_list in kpt_2_verts_objco.items()
    }

    # Keypoint -> world-space faces (strictly associated: every vertex of the face belongs to the
    # keypoint).
    # SPEEDUP (S2c): the old form was O(faces * keypoints) with a fresh `set()` allocation per
    # (face, keypoint) pair. Inverting the mapping to vertex -> owning keypoints and intersecting
    # the owner sets of a face's vertices makes it O(total face corners), independent of the
    # number of keypoints. World coordinates are looked up from the precomputed table instead of
    # being re-transformed per face corner (shared vertices were transformed once per incident
    # face before).
    vert_2_kpts = defaultdict(set)
    for kpt, idx_list in kpt_2_verts_objco.items():
        for vi in idx_list:
            vert_2_kpts[vi].add(kpt)

    kpt_2_faces_worldco = {kpt: [] for kpt in kpt_2_verts_objco}
    if vert_2_kpts:
        for face in faces:
            fverts = face["verts"]
            if not fverts:
                continue
            owners = vert_2_kpts.get(fverts[0])
            if not owners:
                continue
            for vi in fverts[1:]:
                owners = owners & vert_2_kpts.get(vi, EMPTY_KPT_SET)
                if not owners:
                    break
            if not owners:
                continue
            entry = {"coords": [co_world_t[i] for i in fverts], "area": face["area"]}
            for kpt in owners:
                kpt_2_faces_worldco[kpt].append(entry)

    # CLAUDE FIX (B1): `bm.free()` used to be called a second time here on a BMesh that had
    # already been freed above. The duplicate call raises ReferenceError, which propagated out of
    # this function and was swallowed by the broad `except` in TimedRender.handle_render_item --
    # silently dropping every keypoint label for every frame. The BMesh is gone entirely now: it
    # only ever fed `normals`, which foreach_get('normal') gives for free.
    # docs: The object owns the mesh data-block. To force free it use to_mesh_clear().
    obj_eval.to_mesh_clear()
    
    return faces, vertices, normals, kpt_2_verts_worldco, kpt_2_faces_worldco



# =============================================================================
# TEMPLATE EXPORT -- get_mesh_json (bones, joints, skinning weights)
# =============================================================================

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


# =============================================================================
# VISIBILITY / OCCLUSION
# =============================================================================

def is_vertex_occluded(deps, cam_obj, vertex_co_world, eps=1e-4, cache=None):
    """Ray-cast occlusion test for one world-space point.

    SPEEDUP (S3a): `cache` is an optional dict shared across one (camera, frame). Keypoint faces
    share their corner vertices with every neighbouring face of the same keypoint, so the very
    same point used to be ray-cast once per incident face -- and again by the
    `draw_every_keypoint_vertex` overlay. Memoising on the exact coordinate tuple removes those
    duplicates without changing a single result.
    """
    if cache is not None:
        key = tuple(vertex_co_world)
        hit_cached = cache.get(key)
        if hit_cached is not None:
            return hit_cached

    if deps is None:
        deps = bpy.context.evaluated_depsgraph_get()
    cam_co = cam_obj.matrix_world.translation
    dir_vec = (Vector(vertex_co_world) - cam_co)
    dist_to_pt = dir_vec.length
    if dist_to_pt < eps:
        result = False
    else:
        dir_vec.normalize()
        origin = cam_co + dir_vec * eps
        hit, hit_loc, _, _, hit_obj, _ = bpy.context.scene.ray_cast(deps, origin, dir_vec)
        if not hit:
            result = False
        else:
            dist_hit = (hit_loc - origin).length
            result = dist_hit < (dist_to_pt - eps)

    if cache is not None:
        cache[key] = result
    return result


def get_keypoint_visibility_from_faces(deps, kpt_2_faces_worldco, cam_obj, occlusion_cache=None):
    """
    For each keypoint, kpt_2_faces_worldco[kpt] is a list of faces,
    each face is a list of world-space (x,y,z) tuples.

    Returns:
      - kpt_2_visibility_pct: { kpt: visible_area/total_area }
      - kpt_2_visible_faces: { kpt: [ face_coords, … ] } for faces fully visible
    """

    kpt_2_visibility_pct = {}
    kpt_2_visible_faces  = defaultdict(list)

    if deps is None:
        deps = bpy.context.evaluated_depsgraph_get()
    if occlusion_cache is None:
        occlusion_cache = {}

    # SPEEDUP (S3b): hoist the invariants out of the inner loop -- the camera position, the eps
    # offset and the bound ray_cast method were re-resolved for every corner of every face.
    eps = 1e-4
    cam_co = cam_obj.matrix_world.translation.copy()
    ray_cast = bpy.context.scene.ray_cast

    def occluded(coord):
        cached = occlusion_cache.get(coord)
        if cached is not None:
            return cached
        dir_vec = Vector(coord) - cam_co
        dist_to_pt = dir_vec.length
        if dist_to_pt < eps:
            result = False
        else:
            dir_vec.normalize()
            origin = cam_co + dir_vec * eps
            hit, hit_loc, _, _, _, _ = ray_cast(deps, origin, dir_vec)
            result = bool(hit) and (hit_loc - origin).length < (dist_to_pt - eps)
        occlusion_cache[coord] = result
        return result

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
                if occluded(coord):
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


# =============================================================================
# PROJECTION HELPERS
# =============================================================================


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


# =============================================================================
# MASK RENDERING & YOLO LABEL WRITERS
# =============================================================================

def get_mask_render_datablocks():
    """Return the (white emission, black emission, black world) datablocks, creating them once.

    SPEEDUP (S4a): these were built from scratch -- four node trees plus their links -- and then
    removed again on every single mask render. They are constant, so they are created lazily and
    reused for the whole run; `free_mask_render_datablocks()` drops them when the queue drains.
    """
    cached = _mask_render_datablocks

    def _alive(db):
        try:
            _ = db.name if db is not None else None
            return db is not None
        except ReferenceError:
            return False

    if not _alive(cached.get("white")):
        white_em = bpy.data.materials.new(name="SYNTH_tmp_white_emission")
        white_em.use_nodes = True
        ntw = white_em.node_tree
        for n in list(ntw.nodes):
            ntw.nodes.remove(n)
        emis = ntw.nodes.new('ShaderNodeEmission')
        emis.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        outm = ntw.nodes.new('ShaderNodeOutputMaterial')
        ntw.links.new(emis.outputs['Emission'], outm.inputs['Surface'])
        cached["white"] = white_em

    if not _alive(cached.get("black")):
        black_em = bpy.data.materials.new(name="SYNTH_tmp_black_emission")
        black_em.use_nodes = True
        ntb = black_em.node_tree
        for n in list(ntb.nodes):
            ntb.nodes.remove(n)
        bemis = ntb.nodes.new('ShaderNodeEmission')
        bemis.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        outbm = ntb.nodes.new('ShaderNodeOutputMaterial')
        ntb.links.new(bemis.outputs['Emission'], outbm.inputs['Surface'])
        cached["black"] = black_em

    if not _alive(cached.get("world")):
        try:
            black_world = bpy.data.worlds.new(name="SYNTH_tmp_black_world")
            black_world.use_nodes = True
            for nd in list(black_world.node_tree.nodes):
                black_world.node_tree.nodes.remove(nd)
            bg = black_world.node_tree.nodes.new('ShaderNodeBackground')
            bg.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
            outw = black_world.node_tree.nodes.new('ShaderNodeOutputWorld')
            black_world.node_tree.links.new(bg.outputs['Background'], outw.inputs['Surface'])
            cached["world"] = black_world
        except Exception:
            cached["world"] = None

    return cached.get("white"), cached.get("black"), cached.get("world")


def free_mask_render_datablocks():
    """Drop the reusable mask datablocks once the render queue is done."""
    for key, collection in (("white", bpy.data.materials),
                            ("black", bpy.data.materials),
                            ("world", bpy.data.worlds)):
        db = _mask_render_datablocks.get(key)
        _mask_render_datablocks[key] = None
        try:
            if db is not None and db.users == 0:
                collection.remove(db, do_unlink=True)
        except Exception:
            pass


class binary_render_settings:
    """Temporarily switch the scene to the cheapest settings that still give an exact mask.

    SPEEDUP (S5): the binary pass renders flat emission shaders against a black world, so its
    result is identical at one sample and needs no denoising, no ray bounces and no colour
    management. It is also a 1-channel image, so writing it as 8-bit BW PNG with low compression
    saves both the encode and the subsequent OpenCV decode. Everything is restored on exit, so
    the beauty pass keeps the user's settings.
    """

    def __init__(self, scene, sampling=True):
        self.scene = scene
        # The compositor path renders the *shaded* scene and then blows it out to white, so a
        # one-sample render could leave noise holes inside the silhouette. Only the material
        # override path is guaranteed noise-free, hence the switch.
        self.sampling = sampling
        self.saved = []

    def _set(self, owner, attr, value):
        try:
            self.saved.append((owner, attr, getattr(owner, attr)))
            setattr(owner, attr, value)
        except Exception:
            if self.saved and self.saved[-1][0] is owner and self.saved[-1][1] == attr:
                self.saved.pop()

    def __enter__(self):
        scene = self.scene
        img = scene.render.image_settings
        self._set(img, 'file_format', 'PNG')
        self._set(img, 'color_mode', 'BW')
        self._set(img, 'color_depth', '8')
        self._set(img, 'compression', 15)

        if not self.sampling:
            return self

        engine = getattr(scene.render, 'engine', '')
        if engine == 'CYCLES' and hasattr(scene, 'cycles'):
            cy = scene.cycles
            self._set(cy, 'samples', 1)
            self._set(cy, 'use_denoising', False)
            self._set(cy, 'use_adaptive_sampling', False)
            self._set(cy, 'max_bounces', 0)
            self._set(cy, 'use_light_tree', False)
        elif engine.startswith('BLENDER_EEVEE') and hasattr(scene, 'eevee'):
            ee = scene.eevee
            self._set(ee, 'taa_render_samples', 1)
            self._set(ee, 'use_gtao', False)
            self._set(ee, 'use_bloom', False)
            self._set(ee, 'use_ssr', False)
        return self

    def __exit__(self, *exc):
        for owner, attr, value in reversed(self.saved):
            try:
                setattr(owner, attr, value)
            except Exception:
                pass
        self.saved.clear()
        return False


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

    white_em, black_em, black_world = get_mask_render_datablocks()

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
    # restore per unique mesh datablock.
    #
    # SPEEDUP (S4b): the fix above snapshotted and rewrote EVERY polygon's material_index of
    # EVERY mesh in the file, twice per mask render -- an O(total scene polygons) Python loop for
    # each frame and each camera. Overwriting the existing slots in place instead of clearing the
    # slot list keeps the slot count, so material_index is never touched at all and only the slot
    # pointers have to be restored. Meshes with no slot at all get one appended and removed.
    mesh_to_objects = defaultdict(list)
    for o in bpy.data.objects:
        if o.type == 'MESH':
            mesh_to_objects[o.data].append(o)
    orig_materials = {me: list(me.materials) for me in mesh_to_objects}

    target_mesh = target_obj.data

    # Save world and set black background (optional but ensures no stray background)
    orig_world = scene.world
    if black_world is not None:
        try:
            scene.world = black_world
        except Exception:
            pass

    # --- assign materials: target -> white, others -> black --------------------
    try:
        for me, objs in mesh_to_objects.items():
            is_target = me is target_mesh
            if is_target and len(objs) > 1:
                print(f"SYNTH warning: mesh '{me.name}' is shared by the target object and "
                      f"{len(objs) - 1} other object(s); they cannot be masked separately.")
            mat = white_em if is_target else black_em
            if len(me.materials) == 0:
                me.materials.append(mat)
            else:
                for i in range(len(me.materials)):
                    me.materials[i] = mat

        # ensure output dir exists
        os.makedirs(os.path.dirname(out_abspath), exist_ok=True)

        # set render settings and render
        scene.render.filepath = out_abspath
        try:
            scene.render.film_transparent = False
        except Exception:
            pass

        # SPEEDUP (S5): every surface in the scene is now a flat emission shader lit by nothing,
        # so the mask image is noise-free at one sample. Rendering it with the beauty pass's
        # sample count (and its denoiser) is wasted work -- often the single largest cost of the
        # whole run.
        with binary_render_settings(scene):
            bpy.ops.render.render(write_still=True)

    finally:
        # --- restore materials ------------------------------------------------
        for me, mats in orig_materials.items():
            try:
                if not mats:
                    me.materials.clear()
                    continue
                for i, m in enumerate(mats):
                    me.materials[i] = m
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


def get_mask_polygons_from_binary_image(img_path, mask=None):
    """SPEEDUP (S6): `mask` lets the caller pass an already-decoded grayscale image. The mask PNG
    used to be decoded twice per frame and view -- once here and once in
    recon_dataset_read_binary_mask -- for no reason."""
    if mask is None:
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


# =============================================================================
# TIMEDRENDER OPERATOR (modal render + annotation queue)
# =============================================================================

# =============================================================================
# RECONSTRUCTION DATASET EXPORT (DSKv2 schema)
#
# Writes the same folder layout `extract_frames_edit.py` produces and
# `dataloaders_edit.Multiview_Dataset` consumes, but populated with *annotated* ground truth
# instead of detector output.
#
# Two loader facts drive the design:
#   1. Multiview_Dataset treats a (frame, instance, view) as "mask present" only if the
#      `cropped`, `mask` AND `mask_full` rows all exist in files_crop.csv with a non-empty
#      file_loc and all three files are readable. Writing only mask_full/ makes the loader
#      silently report every frame as maskless, so the tight crop and the cropped image are
#      written too even though their pixel content is never read into the returned tensors.
#   2. Keypoint presence is driven purely by conf > 0 in keypoints_confs.pickle, which has no
#      room for a tri-state visibility flag. The lossless 0/1/2 flags therefore go into a
#      separate keypoints_gt.pickle that only the evaluation code reads.
# =============================================================================

RECON_VIEW_SUBDIRS = ('origin', 'cropped', 'mask', 'mask_full',
                      'bbox-masked_image', 'keypoints_results')

# Directories already created this session, so the per-frame writer can skip the makedirs calls.
_recon_created_view_dirs = set()

RECON_FILES_CSV_HEADER = ['frame', 'file_loc', 'category', 'sub_index', 'folder']
RECON_FILES_CROP_CSV_HEADER = ['frame', 'file_loc', 'category', 'sub_index', 'folder', 'bbox']


def recon_dataset_crop_and_pad(image, mask, bbox):
    """
    Square-pad crop, mirroring extract_frames_edit.predict_masks_yolo's local crop_and_pad so
    the two dataset producers agree. Returns (cropped_image, cropped_mask).
    """
    xmin, ymin, xmax, ymax = bbox
    crop_img = image[ymin:ymax, xmin:xmax]
    crop_mask = mask[ymin:ymax, xmin:xmax]

    h, w = crop_img.shape[:2]
    diff = abs(h - w)
    if h < w:
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        crop_img = np.pad(crop_img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant')
        crop_mask = np.pad(crop_mask, ((pad_top, pad_bottom), (0, 0)), mode='constant')
    elif w < h:
        pad_left = diff // 2
        pad_right = diff - pad_left
        crop_img = np.pad(crop_img, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
        crop_mask = np.pad(crop_mask, ((0, 0), (pad_left, pad_right)), mode='constant')
    return crop_img, crop_mask


def recon_dataset_read_binary_mask(mask_path, img=None):
    """Read the rendered GT mask as a 0/1 uint8 array, or None if it is unreadable.

    SPEEDUP (S6): accepts an already-decoded grayscale image to avoid a second PNG decode of the
    very same file."""
    if img is None:
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def recon_dataset_write_frame(
    root, view_name, frame, origin_image_path, gt_mask_binary,
    kpt_2_coords, kpt_2_vis_status, kpt_list, write_bbox_masked=True,
):
    """
    Write one (view, frame) into the reconstruction dataset and return the CSV rows and the
    two keypoint dictionaries for it. Nothing is appended to disk-resident CSVs here; the
    caller accumulates and the finalizer writes them, because TimedRender is a modal operator
    processing one queue item per tick and a partially-written run must stay re-runnable.

    An empty GT mask (object fully out of frame or fully occluded) yields no crop rows at all
    and the "no instance detected" keypoint sentinel, rather than a degenerate zero-area bbox
    that the loader would happily accept.
    """
    view_dir = os.path.join(root, view_name)
    # SPEEDUP (S8): the six directories are the same for every frame of a view; creating them per
    # frame cost six filesystem calls each time for nothing.
    if view_dir not in _recon_created_view_dirs:
        for sub in RECON_VIEW_SUBDIRS:
            os.makedirs(os.path.join(view_dir, sub), exist_ok=True)
        _recon_created_view_dirs.add(view_dir)

    origin_name = f"{view_name}_{frame}.png"
    origin_rel = f"{view_name}/origin/{origin_name}"
    shutil.copyfile(origin_image_path, os.path.join(view_dir, 'origin', origin_name))

    files_row = [frame, origin_rel, 'origin', 0, view_name]
    crop_rows = []

    # SPEEDUP (S7): `gt_mask_binary.sum()` accumulated the whole full-resolution array into an
    # int64 just to answer "is anything set". cv2.boundingRect already reports an empty mask as a
    # zero-sized rectangle and is the value we need anyway, so one pass replaces two.
    bbox = None
    if gt_mask_binary is not None:
        x, y, w, h = cv2.boundingRect(gt_mask_binary)
        if w > 0 and h > 0:
            bbox = [int(x), int(y), int(x + w), int(y + h)]
    has_instance = bbox is not None
    if has_instance:
        origin_img = cv2.imread(str(origin_image_path), cv2.IMREAD_COLOR)
        crop_img, crop_mask = recon_dataset_crop_and_pad(origin_img, gt_mask_binary, bbox)

        cv2.imwrite(os.path.join(view_dir, 'mask_full', f"image_{frame}_0_mask_full.png"),
                    (gt_mask_binary * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(view_dir, 'mask', f"image_{frame}_0_mask.png"),
                    (crop_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(view_dir, 'cropped', f"image_{frame}_0.png"), crop_img)

        bbox_str = f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
        crop_rows.append([frame, f"{view_name}/cropped/image_{frame}_0.png", 'cropped', 0, view_name, bbox_str])
        crop_rows.append([frame, f"{view_name}/mask/image_{frame}_0_mask.png", 'mask', 0, view_name, bbox_str])
        crop_rows.append([frame, f"{view_name}/mask_full/image_{frame}_0_mask_full.png", 'mask_full', 0, view_name, bbox_str])

        if write_bbox_masked:
            # Not needed by Multiview_Dataset, but it gives full schema parity so this tree can
            # also be fed to predict_masks_yolo / detect_keypoints_yolo as if it were real
            # extract_from_video output.
            bbox_masked = np.zeros_like(origin_img)
            bbox_masked[bbox[1]:bbox[3], bbox[0]:bbox[2]] = origin_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            name = f"image_{frame}_0_bbox-masked.png"
            cv2.imwrite(os.path.join(view_dir, 'bbox-masked_image', name), bbox_masked)
            crop_rows.append([frame, f"{view_name}/bbox-masked_image/{name}", 'bbox-masked', 0, view_name, bbox_str])

    # Pipeline-native, lossy: this is what the optimizer consumes under --reconstruct-from-gt.
    # Only genuinely visible keypoints get a confidence, because a real detector cannot see
    # what is occluded; occluded (vis == 1) is therefore correctly recorded as undetected.
    if has_instance:
        confs_entry = {}
        for kpt in kpt_list:
            if kpt_2_vis_status.get(kpt, 0) == 2:
                x_img, y_img = kpt_2_coords.get(kpt, (0.0, 0.0))
                confs_entry[kpt] = [float(x_img), float(y_img), 1.0]
            else:
                confs_entry[kpt] = [0.0, 0.0, 0.0]
    else:
        confs_entry = {kpt: [-1.0, -1.0, -1.0] for kpt in kpt_list}

    # Evaluation-only, lossless: the raw 0/1/2 flag is kept unconditionally so the metrics can
    # separate a miss (GT visible, detector silent) from a correct absence.
    gt_entry = {}
    for kpt in kpt_list:
        vis = int(kpt_2_vis_status.get(kpt, 0))
        x_img, y_img = kpt_2_coords.get(kpt, (0.0, 0.0)) if vis > 0 else (0.0, 0.0)
        gt_entry[kpt] = [float(x_img), float(y_img), float(vis)]

    return files_row, crop_rows, confs_entry, gt_entry


def _recon_read_existing_csv(path, key_fields, header):
    """Read an existing CSV into {key_tuple: row_list}; empty dict if absent or unreadable."""
    if not os.path.exists(path):
        return {}
    out = {}
    try:
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh, quotechar='"'):
                key = tuple(str(row.get(f, '')) for f in key_fields)
                out[key] = [row.get(col, '') for col in header]
    except Exception:
        return {}
    return out


def _recon_load_existing_pickle(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as fh:
            data = pickle.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def recon_dataset_finalize(root, state, kpt_list, report=None):
    """
    Merge this run's accumulated rows with whatever is already on disk and write the per-view
    CSVs, the two keypoint pickles and index.json.

    Merging rather than appending is what makes the export incrementally safe: TimedRender's
    queue builder skips (camera, frame) items whose renders and labels already exist, so a
    resumed run holds only the newly rendered frames in memory and must not drop the rest.
    """
    os.makedirs(root, exist_ok=True)
    index_path = os.path.join(root, 'index.json')

    index = {}
    if os.path.exists(index_path):
        try:
            with open(index_path) as fh:
                index = json.load(fh)
        except Exception:
            index = {}

    frame_folders = list(index.get('frame_folders', []))
    index_files = dict(index.get('index_files', {}))
    image_sizes = dict(index.get('image_sizes', {}))
    image_counts = dict(index.get('image_counts', {}))
    camera_matrices = dict(index.get('camera_matrices', {}))

    for view_name, view_state in state['views'].items():
        view_dir = os.path.join(root, view_name)
        for sub in RECON_VIEW_SUBDIRS:
            os.makedirs(os.path.join(view_dir, sub), exist_ok=True)

        files_csv_path = os.path.join(view_dir, 'files.csv')
        merged_files = _recon_read_existing_csv(files_csv_path, ['frame'], RECON_FILES_CSV_HEADER)
        for row in view_state['files_rows']:
            merged_files[(str(row[0]),)] = row

        crop_csv_path = os.path.join(view_dir, 'files_crop.csv')
        merged_crops = _recon_read_existing_csv(
            crop_csv_path, ['frame', 'category', 'sub_index'], RECON_FILES_CROP_CSV_HEADER)
        # Drop every stale crop row for a frame this run re-rendered before re-adding, so a
        # frame that changed from "has instance" to "no instance" does not keep its old rows.
        rerendered = {str(row[0]) for row in view_state['files_rows']}
        merged_crops = {k: v for k, v in merged_crops.items() if k[0] not in rerendered}
        for row in view_state['crop_rows']:
            merged_crops[(str(row[0]), str(row[2]), str(row[3]))] = row

        with open(files_csv_path, 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(RECON_FILES_CSV_HEADER)
            writer.writerows([merged_files[k] for k in sorted(merged_files, key=lambda k: int(k[0]))])

        with open(crop_csv_path, 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(RECON_FILES_CROP_CSV_HEADER)
            writer.writerows([
                merged_crops[k] for k in sorted(merged_crops, key=lambda k: (int(k[0]), k[1], int(k[2])))
            ])

        # 1:1 origin -> reconstruction frame map, listing only frames actually on disk.
        with open(os.path.join(view_dir, 'frame2video_1.csv'), 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['origin_frame', 'new_frame'])
            for key in sorted(merged_files, key=lambda k: int(k[0])):
                writer.writerow([int(key[0]), int(key[0])])

        kpt_dir = os.path.join(view_dir, 'keypoints_results')
        confs_path = os.path.join(kpt_dir, 'keypoints_confs.pickle')
        gt_path = os.path.join(kpt_dir, 'keypoints_gt.pickle')
        merged_confs = _recon_load_existing_pickle(confs_path)
        merged_confs.update(view_state['keypoints_confs'])
        merged_gt = _recon_load_existing_pickle(gt_path)
        merged_gt.update(view_state['keypoints_gt'])
        with open(confs_path, 'wb') as fh:
            pickle.dump(merged_confs, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(gt_path, 'wb') as fh:
            pickle.dump(merged_gt, fh, protocol=pickle.HIGHEST_PROTOCOL)

        if view_name not in frame_folders:
            frame_folders.append(view_name)
        index_files[view_name] = files_csv_path
        image_sizes[view_name] = [int(view_state['image_size'][0]), int(view_state['image_size'][1])]
        image_counts[view_name] = len(merged_files)
        if view_state.get('camera_matrix') is not None:
            camera_matrices[view_name] = view_state['camera_matrix']

    index.update({
        'frame_folders': frame_folders,
        'index_files': index_files,
        'image_sizes': image_sizes,
        'image_counts': image_counts,
        'camera_matrices': camera_matrices,
        'keypoint_list': list(kpt_list),
        'max_n_instances': 1,
        'status': 'keypoints_detected',
    })
    distinct_counts = set(image_counts.values())
    index['image_count'] = next(iter(distinct_counts)) if len(distinct_counts) == 1 else None

    with open(index_path, 'w') as fh:
        json.dump(index, fh, indent=2)

    # Multiview_Dataset refuses to load views with differing frame counts, since a view short by
    # one frame silently shifts every later frame against the others. Say so here rather than
    # letting it surface as an exception at reconstruction time.
    if len(distinct_counts) > 1 and report is not None:
        report({'WARNING'},
               "Reconstruction dataset views have different frame counts: "
               + ", ".join(f"{v}={c}" for v, c in image_counts.items())
               + ". Reconstruction will refuse to load it until every enabled camera has "
                 "rendered the same frame range.")
    return index_path


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
    # Accumulated reconstruction-dataset rows, keyed by view name. Held in memory across the
    # whole queue because the modal operator drains the queue in batches across timer ticks,
    # and index.json / the CSVs can only be written once every view is known.
    recon_state = None
    # Bridges _annotate_frame (which runs on the 'regular' queue item) to _render_and_write_mask
    # (which runs on the following 'binary' item for the same camera and frame). The GT mask and
    # the GT keypoints for one (view, frame) never exist inside the same call.
    recon_pending_keypoints = None
    # Ray-cast memo, valid for one (camera, frame); see _annotate_frame.
    _occlusion_cache = None
    _occlusion_cache_key = None
    # scene.render.use_persistent_data as it was before the run, or None if untouched.
    _saved_persistent_data = None
    # The render-size/UI mismatch warning is emitted at most once per run.
    _size_warning_reported = False
    # Whether the run has already forced a frame_set; see handle_render_item.
    _first_frame_set_done = False

    def make_prefix_cam_frame(self, cam_name, frame_number):
        return f"{camera_name_to_view_name(cam_name)}_{str(frame_number).zfill(4)}"

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props
        self.cancel_render = False
        self.rendering = False
        self.render_queue = []

        cam_objects = sync_camera_selections(scene)
        enabled_camera_names = {item.camera_name for item in p.camera_selections if item.enabled}

        self.recon_state = {'views': {}} if p.create_reconstruction_dataset else None
        self.recon_pending_keypoints = {}
        if p.create_reconstruction_dataset and not p.render_binary:
            # The binary render *is* the ground-truth segmentation; without it the dataset would
            # have no mask_full, and Multiview_Dataset would then treat every frame as maskless
            # while still loading successfully -- a silent, hard-to-trace failure.
            self.recon_state = None
            self.report({'WARNING'},
                        "'Create Reconstruction Dataset' needs 'Render Binary Masks' enabled; "
                        "skipping the reconstruction-dataset export for this run.")

        modes = ["regular"] + (["binary"] if p.render_binary else [])

        # `resolve()` hits bpy.path.abspath and was called three times per queue item; the four
        # directories are constant for the whole run.
        render_dir_os = resolve(p.render_out_dir)
        mask_dir_os = resolve(p.mask_label_dir)
        kpt_dir_os = resolve(p.kpt_label_dir)

        enabled_cams = [cam for cam in cam_objects if cam.name in enabled_camera_names]

        skipped_count = 0
        # SPEEDUP (S1b): the queue is built FRAME-major (frame -> camera -> mode) rather than
        # camera-major. The evaluated mesh, its vertex-group scan and its keypoint faces depend
        # only on the frame, so ordering this way lets every camera of a frame reuse one
        # extraction (see get_deformed_mesh_data_cached) and lets scene.frame_set -- a full
        # depsgraph re-evaluation -- run once per frame instead of once per (camera, frame).
        # Nothing downstream depends on the order: the CSVs are sorted in recon_dataset_finalize
        # and the YOLO/video steps look files up by name.
        for frame_index in range(scene.frame_start, scene.frame_end + 1):
            for cam in enabled_cams:
                for mode in modes:
                    render_prefix = self.make_prefix_cam_frame(cam.name, frame_index)
                    render_path_os = os.path.join(render_dir_os, render_prefix + ".png")
                    mask_label_path_os = os.path.join(mask_dir_os, render_prefix + ".txt")
                    kpt_label_path_os = os.path.join(kpt_dir_os, render_prefix + ".txt")
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
                            'mask_annot_path':          os.path.join(mask_dir_os, render_prefix + ".png"),
                            'kpt_annot_path':           os.path.join(kpt_dir_os,  render_prefix + ".png"),
                            'mask_label_path':          mask_label_path_os,
                            'kpt_label_path':           kpt_label_path_os,
                        })
                    else:
                        skipped_count += 1

        self.total = len(self.render_queue)
        self.report({'INFO'}, f"Queued {self.total} renders (skipped {skipped_count})")

        invalidate_deformed_mesh_cache()
        self._size_warning_reported = False
        self._first_frame_set_done = False

        # SPEEDUP (S9): Cycles re-syncs the whole scene for every bpy.ops.render.render call.
        # Persistent data keeps the synced scene between renders, which is the standard win for
        # rendering many frames of one scene -- at the cost of holding it in memory, hence the
        # toggle. Restored in cleanup().
        self._saved_persistent_data = None
        if p.use_persistent_render_data:
            try:
                self._saved_persistent_data = scene.render.use_persistent_data
                scene.render.use_persistent_data = True
            except Exception:
                self._saved_persistent_data = None

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

        # Release everything the speedups kept alive for the duration of the queue.
        if getattr(self, '_saved_persistent_data', None) is not None:
            try:
                context.scene.render.use_persistent_data = self._saved_persistent_data
            except Exception:
                pass
            self._saved_persistent_data = None
        invalidate_deformed_mesh_cache()
        self._occlusion_cache = None
        self._occlusion_cache_key = None
        free_mask_render_datablocks()


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

        # SPEEDUP (S10): frame_set() re-evaluates the entire depsgraph (armature, modifiers,
        # constraints, physics). With the frame-major queue the two modes and all cameras of one
        # frame ask for the same frame in a row, so it only has to run on an actual change --
        # which also keeps the cached mesh extraction valid for the whole frame. Assigning
        # scene.camera likewise tags the scene for an update, so it is guarded too.
        if scene.camera is not cam_obj:
            scene.camera = cam_obj
        if scene.frame_current != frame_index or not self._first_frame_set_done:
            # The very first item always forces the update, so a run can never start from a
            # depsgraph the scene happens to be sitting on but that was never evaluated.
            self._first_frame_set_done = True
            scene.frame_set(frame_index)
            invalidate_deformed_mesh_cache()
            self._occlusion_cache = None
            self._occlusion_cache_key = None

        img_w, img_h = self._annotation_image_size(scene)
        if (img_w, img_h) != (int(p.image_width_px), int(p.image_height_px)):
            # Reported once instead of once per queue item; the condition cannot change mid-run
            # without the user editing the scene, and thousands of identical warnings are their
            # own kind of slow.
            if not getattr(self, '_size_warning_reported', False):
                self._size_warning_reported = True
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
                                 kpt_label_out_path,
                                 view_name=camera_name_to_view_name(cam_name),
                                 frame_index=frame_index)

        if p.render_binary and mode == 'binary':
            self._render_and_write_mask(context, img_w, img_h, render_prefix_cam_frame,
                                        render_out_file_path_os, mask_annot_out_file_path,
                                        mask_label_out_path,
                                        cam_obj=cam_obj,
                                        view_name=camera_name_to_view_name(cam_name),
                                        frame_index=frame_index,
                                        image_size=(img_w, img_h))

    def _annotate_frame(self, context, cam_obj, img_w, img_h,
                        render_out_file_path_os, kpt_annot_out_file_path, kpt_label_out_path,
                        view_name=None, frame_index=None):
        scene = context.scene
        p = scene.synth_props
        kpt_list = [kp.strip() for kp in p.keypoint_list_csv.split(',') if kp.strip()]

        try:
            deps = bpy.context.evaluated_depsgraph_get()

            # SPEEDUP (S1c): shared across every camera of this frame -- see
            # get_deformed_mesh_data_cached and the frame-major queue in execute().
            faces, vertices, normals, kpt_2_verts_list_world, kpt_2_faces_list_world = \
                get_deformed_mesh_data_cached(deps, p.collection_name, p.object_name,
                                              kpt_list, scene.frame_current)

            # One ray-cast memo per (camera, frame): the visibility pass and the optional
            # per-vertex overlay below hit the same coordinates repeatedly.
            occl_key = (cam_obj.name, scene.frame_current)
            if getattr(self, '_occlusion_cache_key', None) != occl_key:
                self._occlusion_cache_key = occl_key
                self._occlusion_cache = {}
            occlusion_cache = self._occlusion_cache

            kpt_2_visibility_pct, kpt_2_visible_faces = (
                get_keypoint_visibility_from_faces(deps, kpt_2_faces_list_world, cam_obj,
                                                   occlusion_cache=occlusion_cache)
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
                        if (not is_vertex_occluded(deps, cam_obj, vertex["co"],
                                                   cache=occlusion_cache)
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

            # Hand the *same* dictionaries write_pose_labels_yolo just consumed to the
            # reconstruction-dataset export; visibility is never recomputed on a second path.
            if self.recon_state is not None and view_name is not None:
                self.recon_pending_keypoints[(view_name, frame_index)] = (
                    dict(kpt_2_coords_image_filtered_by_vis), dict(kpt_2_vis_status)
                )

        except Exception as e:
            self.report({'WARNING'}, f"Keypoint generation failed: {e}")

    def _recon_write_frame(self, context, view_name, frame_index, cam_obj,
                           render_out_file_path_os, gt_mask_binary, image_size):
        """Add one (view, frame) to the in-memory reconstruction-dataset accumulator."""
        scene = context.scene
        p = scene.synth_props
        kpt_list = [kp.strip() for kp in p.keypoint_list_csv.split(',') if kp.strip()]
        root = resolve(p.reconstruction_dataset_out_dir)

        coords, vis_status = self.recon_pending_keypoints.pop(
            (view_name, frame_index), ({}, {})
        )

        try:
            files_row, crop_rows, confs_entry, gt_entry = recon_dataset_write_frame(
                root, view_name, frame_index, render_out_file_path_os,
                gt_mask_binary, coords, vis_status, kpt_list,
            )
        except Exception as e:
            self.report({'WARNING'}, f"Reconstruction dataset write failed for "
                                     f"{view_name} frame {frame_index}: {e}")
            return

        view_state = self.recon_state['views'].setdefault(view_name, {
            'files_rows': [], 'crop_rows': [],
            'keypoints_confs': {}, 'keypoints_gt': {},
            'image_size': image_size, 'camera_matrix': None,
        })
        view_state['image_size'] = image_size
        view_state['files_rows'].append(files_row)
        view_state['crop_rows'].extend(crop_rows)
        # str(frame) -> instance "0" -> kpt -> [...], the shape Multiview_Dataset expects.
        view_state['keypoints_confs'][str(frame_index)] = {"0": confs_entry}
        view_state['keypoints_gt'][str(frame_index)] = {"0": gt_entry}

        if view_state['camera_matrix'] is None:
            try:
                view_state['camera_matrix'] = cam_matrix_json_entry(
                    cam_obj, get_cam_matrix_for_cam(cam_obj, scene))
            except Exception as e:
                self.report({'WARNING'}, f"Could not compute camera matrix for {cam_obj.name}: {e}")

    def _render_and_write_mask(self, context, img_w, img_h, render_prefix_cam_frame,
                               render_out_file_path_os, mask_annot_out_file_path,
                               mask_label_out_path,
                               cam_obj=None, view_name=None, frame_index=None, image_size=None):
        scene = context.scene
        p = scene.synth_props
        if p.use_compositor:
            scene.node_tree.nodes["Alpha Over"].inputs[1].default_value = (0, 0, 0, 1)
            scene.node_tree.nodes["Brightness/Contrast"].inputs[1].default_value = 50
            scene.node_tree.nodes["Brightness/Contrast"].inputs[2].default_value = 100
            scene.render.filepath = mask_annot_out_file_path
            # Single-channel 8-bit output only; the sample count is left alone because the
            # compositor path renders the shaded scene and noise would punch holes in the mask.
            with binary_render_settings(scene, sampling=False):
                bpy.ops.render.render(write_still=True)
        else:
            render_binary_mask_keep_occluders_black(
                scene, get_target_object(scene), mask_annot_out_file_path)

        if os.path.exists(mask_annot_out_file_path):
            # SPEEDUP (S6b): decode the freshly written mask PNG once and hand the array to both
            # consumers; it used to be read from disk twice per view and frame.
            mask_gray = cv2.imread(str(mask_annot_out_file_path), cv2.IMREAD_GRAYSCALE)
            polygons = get_mask_polygons_from_binary_image(mask_annot_out_file_path, mask=mask_gray)
            # NOTE: this must happen BEFORE the draw_polygons call below. When
            # create_annotated_images is on, draw_polygons *overwrites* mask_annot_out_file_path
            # with an annotated copy of the beauty render, destroying the binary GT mask in
            # place. Reading it afterwards would silently feed an RGB overlay to the exporter.
            if self.recon_state is not None and view_name is not None:
                self._recon_write_frame(
                    context, view_name, frame_index, cam_obj, render_out_file_path_os,
                    recon_dataset_read_binary_mask(mask_annot_out_file_path, img=mask_gray),
                    image_size or (img_w, img_h),
                )
            if p.create_annotated_images:
                draw_polygons(render_out_file_path_os, mask_annot_out_file_path, polygons)
            write_polygons_to_yolo(polygons, img_w, img_h, mask_label_out_path, class_index=0)
        else:
            # No binary render for this frame. The YOLO path falls back to thresholding the
            # beauty render, but that heuristic is exactly what the reconstruction metrics are
            # meant to stop relying on, so the frame is recorded as "no instance" instead: the
            # origin row is still written, which keeps the per-view frame counts equal.
            if self.recon_state is not None and view_name is not None:
                self._recon_write_frame(
                    context, view_name, frame_index, cam_obj, render_out_file_path_os,
                    None, image_size or (img_w, img_h),
                )
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
                if self.recon_state is not None and self.recon_state['views']:
                    try:
                        index_path = recon_dataset_finalize(
                            resolve(context.scene.synth_props.reconstruction_dataset_out_dir),
                            self.recon_state,
                            [kp.strip() for kp in context.scene.synth_props.keypoint_list_csv.split(',') if kp.strip()],
                            report=self.report,
                        )
                        self.report({'INFO'}, f"Wrote DSKv2 reconstruction dataset index to {index_path}")
                    except Exception as e:
                        self.report({'WARNING'}, f"Reconstruction dataset finalization failed: {e}")
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
                # SPEEDUP (S11): one item per timer tick meant `event_timer_interval` seconds of
                # pure idle waiting per item -- 0.35 s by default, i.e. ~35 min of nothing for a
                # 6-camera 500-frame run, and it dominates entirely once the cheap items (the
                # annotation-only pass, or resumed runs whose renders already exist) are counted.
                # Items are now drained until the tick's time budget is spent; ESC is still
                # handled between batches.
                budget = max(0.0, context.scene.synth_props.seconds_per_timer_tick)
                t0 = time.perf_counter()
                while self.render_queue and not self.cancel_render:
                    qitem = self.render_queue.pop(0)
                    # try:
                    self.handle_render_item(context, qitem)
                    # except Exception as e:
                    #     self.report({'WARNING'}, f"Render failed for item: {e}")
                    if time.perf_counter() - t0 >= budget:
                        break

        return {'PASS_THROUGH'}


# =============================================================================
# YOLO DATASET HELPERS
# =============================================================================

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


# =============================================================================
# SETTINGS OPERATORS (apply / load config / keypoint list)
# =============================================================================
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
            'SECONDS_PER_TIMER_TICK': p.seconds_per_timer_tick,
            'use_persistent_render_data': p.use_persistent_render_data,
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
            'create_reconstruction_dataset': p.create_reconstruction_dataset,
            'RECONSTRUCTION_DATASET_OUT_DIR': p.reconstruction_dataset_out_dir,
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
            'SECONDS_PER_TIMER_TICK': 'seconds_per_timer_tick',
            'use_persistent_render_data': 'use_persistent_render_data',
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
            'create_reconstruction_dataset': 'create_reconstruction_dataset',
            'RECONSTRUCTION_DATASET_OUT_DIR': 'reconstruction_dataset_out_dir',
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



# =============================================================================
# CAMERA MATRIX EXPORT
# =============================================================================

def cam_matrix_json_entry(cam, mats):
    """
    Serialise get_cam_matrix_for_cam's output into the exact dict schema DSKv2 parses
    (parse_cams_json.CameraSet). Shared by cam_matrices.json and by the reconstruction
    dataset's index.json['camera_matrices'] so the two can never drift apart.

    'distortion' is deliberately absent: synthetic renders carry no lens distortion and the
    downstream parser defaults a missing key to zero.
    """
    def mat_to_list(m):
        return [[float(v) for v in row] for row in m]

    return {
        'f': float(mats['f']) if mats.get('f') is not None else None,
        'K': mat_to_list(mats['K']),
        'R': mat_to_list(mats['R']),
        't': [float(v) for v in mats['t']],
        'Rt': mat_to_list(mats['Rt']),
        'P': mat_to_list(mats['P']),
        'FROM_BLENDERWORLD': mat_to_list(mats['FROM_BLENDERWORLD']),
        'camera_name': cam.name
    }


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
        out[camera_name_to_view_name(cam.name)] = cam_matrix_json_entry(cam, mats)
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


# =============================================================================
# BONE GROUPS & BONE PRIORS OPERATORS
# =============================================================================

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


# =============================================================================
# MESH / TEMPLATE EXPORT OPERATOR
# =============================================================================

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
        

# =============================================================================
# CREATE VIDEOS OPERATOR
# =============================================================================

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



# =============================================================================
# POSE TIME SERIES (schema v2) -- export / import / verify
# =============================================================================
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


# --- blocked frames ---------------------------------------------------------
#
# A blocked frame is one the reconstruction pipeline emitted but did NOT produce by
# fitting the optimizer to observations of that frame -- currently only gap-filled
# interpolated poses, but the reason is carried per record so future ones need no
# change here. multiview_reconstruction_edit.py writes the list into both the metrics
# JSON and the pose_time_series/2 meta; this add-on reads the pts2 copy (the file it
# already opens) and passes it through UNINTERPRETED to the 3D metrics output.
#
# Blocked frames are still scored. Excluding them is the analysis step's decision, not
# this one's: the per-frame numbers are cheap to produce, genuinely interesting when
# asking how bad interpolation actually is, and impossible to recover later if dropped
# here. So every frame is measured, and the flag rides along beside the measurements.

_RECON_BLOCKED_PROP = "dsk_blocked_frames"
_RECON_BLOCKED_FIELDS = ("frame_number", "frame_index_in_this_reconstruction_run",
                         "reason_blocked")


def _pts_blocked_records(meta, frames, report=None):
    """meta['blocked_frames'] as a validated list of records, or [] if there is none.

    Kept permissive on purpose: an unreadable or absent list means 'nothing is known to be
    blocked', which is the correct reading of a file written before the field existed, and a
    malformed entry is dropped individually rather than voiding the whole list. Everything
    dropped is reported, because a silently empty blocked list looks exactly like a clean run.
    """
    raw = meta.get("blocked_frames")
    if raw is None:
        if report:
            report({'WARNING'},
                   "this pose time series has no 'blocked_frames' in its meta (written before "
                   "the field existed); no frame will be flagged as blocked. Re-run the "
                   "reconstruction to record it.")
        return []
    if not isinstance(raw, list):
        if report:
            report({'WARNING'}, f"'blocked_frames' is {type(raw).__name__}, not a list; "
                                f"ignoring it.")
        return []

    known = {int(f["frame"]) for f in frames if isinstance(f.get("frame"), (int, float))}
    records, dropped = [], 0
    for entry in raw:
        if not isinstance(entry, dict) or not all(k in entry for k in _RECON_BLOCKED_FIELDS):
            dropped += 1
            continue
        try:
            record = {
                "frame_number": int(entry["frame_number"]),
                "frame_index_in_this_reconstruction_run":
                    int(entry["frame_index_in_this_reconstruction_run"]),
                "reason_blocked": str(entry["reason_blocked"]),
            }
        except (TypeError, ValueError):
            dropped += 1
            continue
        records.append(record)
    if dropped and report:
        report({'WARNING'}, f"{dropped} malformed entry/entries in 'blocked_frames' were "
                            f"ignored; each needs {list(_RECON_BLOCKED_FIELDS)}.")

    # The frame NUMBER is what this add-on keys on -- it scores an animation by frame, not by
    # array position -- so a blocked number with no matching keyframe would silently flag
    # nothing. Worth a warning: it means the list and the frames came from different runs.
    orphans = sorted({r["frame_number"] for r in records} - known)
    if orphans and report:
        report({'WARNING'}, f"blocked frame(s) {orphans[:5]} are not in this file's frame list; "
                            f"they cannot be flagged. The blocked list may be from another run.")
    return records


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

    # Carried through verbatim from the pts2 meta, then read back by _recon_pair_context() for
    # every 3D metric operator (single-run and batch alike) and copied into the 3D metrics file.
    # This add-on does not decide what is blocked or why -- the reconstruction pipeline does,
    # and re-deriving it here (e.g. by looking for the per-frame "interpolated" flag) would give
    # a second, silently diverging definition. Frames are still SCORED normally; the list only
    # records which of them were not optimizer-fitted, so a consumer can aggregate accordingly.
    #
    # Stamped on the MESH, not the armature, because _iou_find_reconstruction only ever searches
    # MESH objects in 'Reconstructions' -- that is the one place a reader will look for it. As a
    # JSON string rather than a native array because Blender's ID-property arrays cannot hold
    # dicts, and reject an empty list besides (the common case: most runs block nothing).
    blocked_records = _pts_blocked_records(meta, frames, report)
    new_obj[_RECON_BLOCKED_PROP] = json.dumps(blocked_records)
    if blocked_records and report:
        by_reason = {}
        for entry in blocked_records:
            key = entry["reason_blocked"]
            by_reason[key] = by_reason.get(key, 0) + 1
        report({'INFO'}, f"{len(blocked_records)} of {len(frames)} frame(s) in "
                         f"'{new_obj.name}' are blocked ({by_reason}); they are still scored, "
                         f"and flagged as blocked in the 3D metrics output.")

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


# =============================================================================
# RECONSTRUCTION EVALUATION -- volumetric 3D IoU & keypoint distances
# =============================================================================
#
# Two metrics that quantify how well a reconstruction (an object in the
# 'Reconstructions' collection, produced by create_animation_from_pose_time_series)
# recovers the ground-truth synthetic animation:
#
#   IoU(frame)       = Vol(GT n R) / Vol(GT u R)
#                      Monte-Carlo occupancy estimate: N points drawn uniformly from
#                      the AABB enclosing both meshes, each classified inside/outside
#                      by a BVHTree ray-parity test (odd hit count == inside).
#                      Standard definition from the reconstruction literature
#                      (Occupancy Networks, Mescheder et al. 2019, sec. 4).
#   dist(kpt, frame) = || centroid_GT(kpt) - centroid_R(kpt) ||_2
#                      centroid = mean world-space position of a keypoint's vertex
#                      group members, i.e. exactly get_avg_kpt_coords_3d.
#
# UNITS. Every length in this section is reported BOTH in metres and in GT body
# lengths, and the body-length figure is the one to compare across runs. A metre
# is meaningless as a quality score here: it depends on how large the artist
# happened to model the fish, so two sweeps of the same pipeline on differently
# scaled scenes are not comparable in metres, and neither is a 3D distance
# against the 2D keypoint error, which analyze_metrics.py already normalises by
# gt_body_length_px. L_body is the SAME quantity the MPVE/MPJPE section uses --
# _recon_body_length(), || centroid('mouth tip') - centroid('caudal peduncle') ||
# on the GT mesh -- so a body length means one thing across the whole file.
# It is measured per frame and ALWAYS on the ground truth: taken from the
# reconstruction, an over-scaled fit would divide its own error away.
#
#   dist_bl(kpt, frame) = dist(kpt, frame) / L_body(frame)
#   vol_*_bl3(frame)    = vol_*(frame)     / L_body(frame)^3
#
# The IoU itself is a ratio of two volumes and is therefore ALREADY scale free;
# normalising it would be wrong, so it is left exactly as it is. Same for the
# occupancy counts. Metres are kept beside the body lengths rather than replaced,
# because L_body can be unmeasurable on a frame (see body_length_source) and a
# null normalised value with no absolute value beside it would be unreadable.
#
# Ray parity is preferred over bmesh.ops.intersect_boolean / the boolean modifier:
# LBS-skinned meshes self-intersect at sharp bends, where exact CSG fails outright
# while parity only misclassifies the small doubly-covered region.
#
# Both metrics are derived from ONE evaluated-mesh extraction per object per frame
# (_recon_frame_payload), because obj.evaluated_get(deps).to_mesh() is by far the
# most expensive step of the loop.

# /2: every length gained its body-length-normalised counterpart, and the frame
# records gained body_length_m / body_length_source. A reader must be able to tell
# a file with those fields from one without, hence the version bump.
VOLUMETRIC_IOU_SCHEMA = "volumetric_iou/2"
KEYPOINT_DISTANCE_SCHEMA = "keypoint_distances/2"

# Fixed, deliberately non-axis-aligned ray direction: fish rigs are modelled on the world
# axes, so an axis-aligned ray grazes coplanar fin/body geometry and breaks the parity count.
_IOU_RAY_DIR = Vector((0.5773502692, 0.3574067444, 0.7341827546)).normalized()
_IOU_RAY_EPS = 1e-6      # offset past a hit so the same face is not re-hit
_IOU_MAX_HITS = 256      # parity loop guard (a fish silhouette needs <10 along any ray)


# --- GT / reconstruction pairing -------------------------------------------

def _recon_kpt_list(context):
    """Same parse as TimedRender._annotate_frame."""
    p = context.scene.synth_props
    return [kp.strip() for kp in p.keypoint_list_csv.split(',') if kp.strip()]


def _iou_find_reconstruction(src_obj, explicit_name=""):
    """Locate the mesh that create_animation_from_pose_time_series produced for `src_obj`.

    That function does `new_obj = src_obj.copy()` and links the copy into 'Reconstructions'.
    Blender uniquifies copies, so the reconstruction is NOT called `src_obj.name` -- it is
    '<name>.001', '<name>.002', ... Match that pattern plus an identical vertex count (the copy
    is topology-identical by construction) and take the highest suffix = most recent import.
    Returns (obj, n_candidates).
    """
    col = bpy.data.collections.get("Reconstructions")
    if col is None:
        raise ValueError("No 'Reconstructions' collection; run "
                         "'Create Animation from Pose Time Series' first.")
    if explicit_name:
        obj = col.objects.get(explicit_name)
        if obj is None or obj.type != 'MESH':
            raise ValueError(f"Mesh '{explicit_name}' not found in 'Reconstructions'.")
        return obj, 1

    n_src = len(src_obj.data.vertices)
    pat = re.compile(r"^" + re.escape(src_obj.name) + r"(?:\.(\d+))?$")
    cands = []
    for ob in col.objects:
        if ob.type != 'MESH' or ob is src_obj:
            continue
        m = pat.match(ob.name)
        if m is None or len(ob.data.vertices) != n_src:
            continue
        cands.append((int(m.group(1) or 0), ob))
    if not cands:
        raise ValueError(f"No reconstruction of '{src_obj.name}' in 'Reconstructions' "
                         f"(looked for '{src_obj.name}.NNN' with {n_src} vertices).")
    cands.sort(key=lambda t: t[0])
    return cands[-1][1], len(cands)


def _iou_action_range(obj):
    """Keyframed frame range of `obj`'s action, or None."""
    if obj is None:
        return None
    ad = getattr(obj, "animation_data", None)
    act = ad.action if ad else None
    if act is None:
        return None
    try:
        lo, hi = act.frame_range
        return int(math.floor(lo)), int(math.ceil(hi))
    except Exception:
        return None


def _recon_blocked_frames(rec_obj, report=None):
    """The blocked-frame records stamped on `rec_obj` by the pts2 importer.

    Returns (records, stamp_present). A reconstruction imported before the stamp existed
    reports once and yields ([], False) -- 'not recorded', which the caller keeps distinct
    from the benign ([], True) 'recorded, and nothing was blocked'. The two look identical in
    the output otherwise, and conflating them would let a run with unknown provenance pass as
    a clean one.
    """
    raw = rec_obj.get(_RECON_BLOCKED_PROP)
    if raw is None:
        if report:
            report({'WARNING'},
                   f"'{rec_obj.name}' carries no blocked-frame stamp (imported before this was "
                   f"recorded); no frame can be flagged as blocked. Re-run 'Create Animation "
                   f"from Pose Time Series' to record it.")
        return [], False
    try:
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError(f"expected a list, got {type(records).__name__}")
        return [dict(r) for r in records], True
    except (TypeError, ValueError) as exc:
        if report:
            report({'WARNING'}, f"'{rec_obj.name}': unreadable blocked-frame stamp ({exc}); "
                                f"no frame is flagged.")
        return [], False


def _recon_pair_context(context, report=None, rec_obj=None):
    """Common preamble of both metric operators: GT/reconstruction pairing + frame range.

    Returns dict(src_obj, src_arm, rec_obj, rec_arm, kpt_list, frame_lo, frame_hi,
    blocked_records, blocked_by_frame, blocked_known).
    Raises ValueError on anything the caller must turn into {'CANCELLED'}.

    `rec_obj` bypasses the '<name>.NNN, newest wins' search of _iou_find_reconstruction and the
    'Reconstruction Object' property with an object the caller already holds a reference to.
    Only the batch operator uses it: it imports one pts2 file at a time and therefore knows
    exactly which mesh it just created, so name-based auto-detection would be both redundant and
    wrong (a stale 'Reconstruction Object' override would silently score the wrong mesh).

    `blocked_records` are the reconstruction's blocked frames, restricted to the evaluated
    range. NOTHING is skipped because of them -- every frame in [frame_lo, frame_hi] is still
    measured -- they are carried so each metric can mark its per-frame rows and so the run-level
    list can be copied into the output. Resolving them HERE rather than inside each metric's
    loop is deliberate: this is the one preamble every 3D metric operator shares, so a single
    read-back covers the IoU, the keypoint distances, MPVE, MPJPE, the bone-rotation error and
    the batch collector, and none of them can disagree about which frames were blocked.
    """
    scene = context.scene
    p = scene.synth_props

    src_obj, src_arm = _pts_find_source(context)
    if rec_obj is None:
        rec_obj, n_cands = _iou_find_reconstruction(src_obj, p.iou_recon_object_name.strip())
        if n_cands > 1 and report:
            report({'WARNING'}, f"{n_cands} reconstructions of '{src_obj.name}' found; using the "
                                f"newest ('{rec_obj.name}'). Set 'Reconstruction Object' to "
                                f"override.")
    rec_arm = rec_obj.parent if (rec_obj.parent and rec_obj.parent.type == 'ARMATURE') else None

    # Mismatched frame ranges: intersect the scene range with both actions' keyed ranges.
    f_start, f_end = int(scene.frame_start), int(scene.frame_end)
    lo, hi = f_start, f_end
    for owner in (src_arm, rec_arm, rec_obj):
        r = _iou_action_range(owner)
        if r is not None:
            lo, hi = max(lo, r[0]), min(hi, r[1])
    if lo > hi:
        raise ValueError(f"GT and reconstruction animations do not overlap inside the scene "
                         f"range [{f_start}, {f_end}].")
    if (lo, hi) != (f_start, f_end) and report:
        report({'WARNING'}, f"Frame range mismatch: evaluating the overlap [{lo}, {hi}] instead "
                            f"of the scene range [{f_start}, {f_end}].")

    records, known = _recon_blocked_frames(rec_obj, report)
    # Restricted to the evaluated range so the run-level list in the output describes exactly
    # the frames the file reports on; a record outside [lo, hi] belongs to no row there.
    in_range = [r for r in records
                if isinstance(r.get("frame_number"), int) and lo <= r["frame_number"] <= hi]
    by_frame = {r["frame_number"]: r["reason_blocked"] for r in in_range}
    if in_range and report:
        n_range = hi - lo + 1
        report({'INFO'},
               f"{len(in_range)} of {n_range} frame(s) in [{lo}, {hi}] are blocked; all are "
               f"still scored and flagged as blocked in the output.")

    return {"src_obj": src_obj, "src_arm": src_arm, "rec_obj": rec_obj, "rec_arm": rec_arm,
            "kpt_list": _recon_kpt_list(context), "frame_lo": lo, "frame_hi": hi,
            "blocked_records": sorted(in_range, key=lambda r: r["frame_number"]),
            "blocked_by_frame": by_frame, "blocked_known": known}


def _recon_vertex_group_members(obj):
    """{vertex_group_name: set(vertex indices)} on the base (pre-modifier) mesh."""
    idx2name = {vg.index: vg.name for vg in obj.vertex_groups}
    out = {name: set() for name in idx2name.values()}
    for v in obj.data.vertices:
        for g in v.groups:
            name = idx2name.get(g.group)
            if name is not None:
                out[name].add(v.index)
    return out


def _recon_check_keypoint_correspondence(src_obj, rec_obj, kpt_list):
    """Verify the 'no correspondence search needed' assumption, loudly.

    create_animation_from_pose_time_series builds the reconstruction as `src_obj.copy()` with
    `new_obj.data = src_obj.data.copy()`, so vertex count, vertex order AND vertex groups are
    identical by construction and keypoint k on GT is keypoint k on R by definition. That is an
    assumption about how the reconstruction was produced, not an invariant -- a hand-built or
    re-imported object would break it silently and yield distances between different anatomy.
    Checked once, O(V), before any frame is touched. Returns the list of keypoints that exist on
    both objects but have no members (handled per-frame as 'missing', not fatal).
    """
    n_src, n_rec = len(src_obj.data.vertices), len(rec_obj.data.vertices)
    if n_src != n_rec:
        raise ValueError(f"vertex count mismatch: GT '{src_obj.name}' has {n_src}, reconstruction "
                         f"'{rec_obj.name}' has {n_rec}; keypoint correspondence is not defined.")
    g_src = _recon_vertex_group_members(src_obj)
    g_rec = _recon_vertex_group_members(rec_obj)
    miss_src = [k for k in kpt_list if k not in g_src]
    miss_rec = [k for k in kpt_list if k not in g_rec]
    if miss_src or miss_rec:
        raise ValueError(
            f"keypoint vertex groups missing -- on GT '{src_obj.name}': {miss_src or 'none'}; "
            f"on reconstruction '{rec_obj.name}': {miss_rec or 'none'}. Fix 'Keypoint List' or "
            f"re-create the reconstruction.")
    differing = [k for k in kpt_list if g_src[k] != g_rec[k]]
    if differing:
        raise ValueError(
            f"vertex group membership differs between GT and reconstruction for {differing}; "
            f"refusing to compute distances between mismatched keypoints.")
    return [k for k in kpt_list if not g_src[k]]


# --- per-frame evaluation (shared by both metrics) --------------------------

def _recon_frame_payload(deps, collection_name, obj, kpt_list, need_bvh, report=None,
                         need_verts=False, need_face_areas=False):
    """ONE evaluated-mesh extraction per object per frame, feeding every metric.

    Everything comes out of a single get_deformed_mesh_data() call:
      (a) `vertices` + `faces` -> world-space BVHTree + AABB for the IoU occupancy sampling,
      (b) `kpt_2_verts_worldco` -> get_avg_kpt_coords_3d -> keypoint centroids,
      (c) `need_verts` -> a contiguous (N_v, 3) float64 world-space vertex array for MPVE,
      (d) `need_face_areas` -> the evaluated face areas + face vertex indices, for the
          optional area-weighted MPVE mean and the tessellation-uniformity report.
    to_mesh()/to_mesh_clear() handling stays inside get_deformed_mesh_data; the BVHTree owns its
    own copy of the geometry, so nothing survives the frame except what is returned.

    Pass kpt_list=[] to skip the vertex-group scan, need_bvh=False to skip the tree build.
    """
    obj_eval = obj.evaluated_get(deps)
    # An object in an excluded/hidden collection is absent from the depsgraph: evaluated_get()
    # then returns the original and to_mesh() yields the UNDEFORMED rest mesh -- a constant,
    # meaningless metric rather than an error.
    if report is not None and not obj_eval.is_evaluated:
        report({'WARNING'}, f"'{obj.name}' is not in the depsgraph (collection excluded or "
                            f"disabled in the view layer); its mesh will not be deformed.")

    faces, vertices, _normals, kpt_2_verts_worldco, _kpt_faces = get_deformed_mesh_data(
        deps, collection_name, obj.name, kpt_list)

    # get_avg_kpt_coords_3d wants {kpt: [(x, y, z), ...]}, but get_deformed_mesh_data returns
    # [{"id": i, "co": (x, y, z)}, ...]; feeding it the raw dicts raises inside
    # np.array(..., dtype=float). The 2D annotation path hands it bare coord tuples for the
    # same reason. Unwrap "co" here rather than duplicating the centroid logic.
    payload = {
        "n_verts": len(vertices),
        "n_faces": len(faces),
        "kpt_centroids": get_avg_kpt_coords_3d(
            {k: [e["co"] for e in ents] for k, ents in kpt_2_verts_worldco.items()}),
        "bvh": None,
        "box": None,
        "verts_world": None,
    }

    # CLAUDE (MPVE): `vertices[i]["co"]` is OBJECT space and GT and reconstruction have
    # different matrix_world (the reconstruction armature carries the per-frame scale S), so
    # the world-space array is built here, per object, as one (N, 3) matrix product -- never a
    # Python loop over `Matrix @ Vector`.
    if need_verts:
        mw = np.array(obj_eval.matrix_world, dtype=np.float64)
        if vertices:
            co_local = np.array([v["co"] for v in vertices], dtype=np.float64).reshape(-1, 3)
            payload["verts_world"] = np.ascontiguousarray(co_local @ mw[:3, :3].T + mw[:3, 3])
        else:
            payload["verts_world"] = np.zeros((0, 3), dtype=np.float64)
    if need_face_areas:
        payload["face_areas"] = np.fromiter((f["area"] for f in faces), dtype=np.float64,
                                            count=len(faces))
        payload["face_verts"] = [f["verts"] for f in faces]

    if not need_bvh:
        return payload

    mw = obj_eval.matrix_world
    world = [mw @ Vector(v["co"]) for v in vertices]
    polys = [f["verts"] for f in faces if len(f["verts"]) >= 3]
    payload["n_faces"] = len(polys)
    if not world or not polys:
        return payload                       # degenerate: caller skips the frame

    xs = [c.x for c in world]
    ys = [c.y for c in world]
    zs = [c.z for c in world]
    payload["box"] = (Vector((min(xs), min(ys), min(zs))),
                      Vector((max(xs), max(ys), max(zs))))
    # FromPolygons tessellates ngons internally, which is what the parity test needs (a
    # non-planar LBS-deformed ngon would otherwise be entered and left through one face).
    payload["bvh"] = BVHTree.FromPolygons(world, polys, all_triangles=False, epsilon=0.0)
    return payload


# --- volumetric IoU ---------------------------------------------------------

def _iou_point_inside(bvh, point, direction, max_dist):
    """Ray-parity inside/outside test (odd hit count == inside).

    mathutils' BVHTree.ray_cast only returns the FIRST hit, so the ray is restarted just past
    each hit.
    """
    origin = point
    remaining = max_dist
    hits = 0
    while hits < _IOU_MAX_HITS:
        loc, _nrm, idx, dist = bvh.ray_cast(origin, direction, remaining)
        if idx is None:
            break
        hits += 1
        origin = loc + direction * _IOU_RAY_EPS
        remaining -= (dist + _IOU_RAY_EPS)
        if remaining <= 0.0:
            break
    return (hits & 1) == 1


def _iou_frame(bvh_a, box_a, bvh_b, box_b, n_samples, seed):
    """Monte-Carlo occupancy IoU between two world-space BVHTrees.

    Samples uniformly from the AABB enclosing both meshes. Padding the box is IoU-neutral --
    extra points fall outside both meshes and enter neither numerator nor denominator -- it only
    guards against samples landing exactly on a mesh-tangent box face.
    Returns a dict, or None if the box or the union is degenerate.
    """
    lo = np.array([min(box_a[0][i], box_b[0][i]) for i in range(3)], dtype=np.float64)
    hi = np.array([max(box_a[1][i], box_b[1][i]) for i in range(3)], dtype=np.float64)
    ext = hi - lo
    diag = float(np.linalg.norm(ext))
    if diag <= 0.0 or not np.all(np.isfinite(ext)):
        return None
    pad = 1e-3 * diag
    lo -= pad
    hi += pad
    ext = hi - lo
    box_vol = float(ext[0] * ext[1] * ext[2])
    if box_vol <= 0.0:
        return None

    rng = np.random.default_rng(seed)
    pts = rng.random((n_samples, 3)) * ext + lo

    # Cheap vectorised AABB rejection: a point outside a mesh's own bbox cannot be inside it,
    # which removes most of the ray casts when the two meshes are offset from each other.
    a_lo = np.array(box_a[0], dtype=np.float64)
    a_hi = np.array(box_a[1], dtype=np.float64)
    b_lo = np.array(box_b[0], dtype=np.float64)
    b_hi = np.array(box_b[1], dtype=np.float64)
    cand_a = np.all((pts >= a_lo) & (pts <= a_hi), axis=1)
    cand_b = np.all((pts >= b_lo) & (pts <= b_hi), axis=1)

    pts_list = pts.tolist()
    ray_len = diag * 2.0 + 1.0
    inside_a = np.zeros(n_samples, dtype=bool)
    inside_b = np.zeros(n_samples, dtype=bool)
    for i in np.nonzero(cand_a)[0]:
        inside_a[i] = _iou_point_inside(bvh_a, Vector(pts_list[i]), _IOU_RAY_DIR, ray_len)
    for i in np.nonzero(cand_b)[0]:
        inside_b[i] = _iou_point_inside(bvh_b, Vector(pts_list[i]), _IOU_RAY_DIR, ray_len)

    n_a = int(inside_a.sum())
    n_b = int(inside_b.sum())
    n_i = int((inside_a & inside_b).sum())
    n_u = int((inside_a | inside_b).sum())
    if n_u == 0:
        return None                      # both meshes zero-volume w.r.t. this sample set

    iou = n_i / n_u
    scale = box_vol / float(n_samples)
    return {
        "iou": float(iou),
        # binomial standard error of the ratio over the union samples; ~0.003 at n_u = 20k
        "iou_stderr": float(math.sqrt(max(iou * (1.0 - iou), 0.0) / n_u)),
        "vol_gt": float(n_a * scale),
        "vol_recon": float(n_b * scale),
        "vol_intersection": float(n_i * scale),
        "vol_union": float(n_u * scale),
        "n_inside_gt": n_a,
        "n_inside_recon": n_b,
        "n_intersection": n_i,
        "n_union": n_u,
    }


# --- body-length normalisation ----------------------------------------------
#
# _recon_body_length() and _RECON_BODY_LENGTH_KPTS live in the MPVE/MPJPE section
# further down. Calling forward into it is deliberate: one definition of L_body for
# the whole file is worth more than locality, because two definitions would drift
# and silently make the 3D distances here incomparable with the MPJPE ones there.

def _iou_kpt_payload_kpt_list(src_obj, kpt_list):
    """`kpt_list` plus whichever body-axis keypoints the GT mesh actually carries.

    L_body must be measurable even when the user turned the keypoint distances off, so
    the payload always extracts the two body-axis groups. They are appended, never
    substituted, and _kpt_frame_distances() iterates the caller's `kpt_list`, so the extra
    centroids widen the normaliser's coverage without adding a measured keypoint.
    Filtering against the GT's actual vertex groups keeps get_deformed_mesh_data from
    being asked for a group that does not exist on this template.
    """
    names = list(kpt_list)
    present = {vg.name for vg in src_obj.vertex_groups}
    for k in _RECON_BODY_LENGTH_KPTS:
        if k in present and k not in names:
            names.append(k)
    return names


def _iou_kpt_body_length(pay_gt):
    """(L_body in metres, source) for one frame, from the GT payload alone.

    payload['box'] is the world-space AABB of exactly the vertices _recon_body_length()
    would reduce to a diagonal, so feeding it the two corners reproduces that fallback
    bit for bit without asking _recon_frame_payload for the full (N, 3) vertex array.
    """
    box = pay_gt.get("box")
    corners = None
    if box is not None:
        corners = np.asarray([tuple(box[0]), tuple(box[1])], dtype=np.float64)
    return _recon_body_length(pay_gt.get("kpt_centroids") or {}, corners, None)


def _bl_norm(value, l_body, power=1):
    """`value` in body lengths (or BL^power), or None when L_body is unusable.

    None rather than NaN: these land in JSON, where null is the file format's own
    missing value and survives a round trip through every reader.
    """
    if value is None or l_body is None or not (l_body > _RECON_EPS):
        return None
    try:
        out = float(value) / (float(l_body) ** power)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _body_length_block(lengths, sources):
    """Sequence-level summary of the normaliser itself.

    Reported next to every aggregate that was divided by it: a body-length number is only
    as trustworthy as L_body, and a run that silently fell back to the AABB diagonal on
    most frames must be visible as such rather than inferred from a suspiciously smooth curve.
    """
    vals = [float(v) for v in lengths if v is not None and math.isfinite(float(v))]
    by_source = {}
    for s in sources:
        by_source[s] = by_source.get(s, 0) + 1
    block = {
        "definition": (f"|| centroid('{_RECON_BODY_LENGTH_KPTS[0]}') - "
                       f"centroid('{_RECON_BODY_LENGTH_KPTS[1]}') || on the GT mesh"),
        "measured_on": "ground_truth",
        "units": "meters",
        "n_frames_measured": len(vals),
        "n_frames_unavailable": len(sources) - len(vals),
        "sources": by_source,
    }
    if vals:
        ordered = sorted(vals)
        mid = len(ordered) // 2
        block.update({
            "median": float(ordered[mid] if len(ordered) % 2
                            else 0.5 * (ordered[mid - 1] + ordered[mid])),
            "min": float(ordered[0]),
            "max": float(ordered[-1]),
        })
    else:
        block.update({"median": None, "min": None, "max": None})
    return block


# --- keypoint distances -----------------------------------------------------

def _kpt_frame_distances(cent_gt, cent_rc, kpt_list, missing, report=None):
    """{kpt: || c_GT - c_R ||_2} for the keypoints present on both meshes this frame.

    A keypoint whose vertex group resolved to no vertices on the evaluated mesh is dropped by
    get_avg_kpt_coords_3d; it is excluded from this frame's aggregate and warned about ONCE
    (`missing` is the operator-lifetime memo), never per frame.
    """
    out = {}
    for k in kpt_list:
        a, b = cent_gt.get(k), cent_rc.get(k)
        if a is None or b is None:
            if k not in missing:
                where = ("both meshes" if a is None and b is None
                         else ("the GT mesh" if a is None else "the reconstruction"))
                missing[k] = where
                if report:
                    report({'WARNING'}, f"keypoint '{k}' has no vertices on {where}; excluded "
                                        f"from every frame it is missing on.")
            continue
        out[k] = float((Vector(a) - Vector(b)).length)
    return out


def _kpt_record(frame, per_keypoint, l_body=None, l_source="unavailable", blocked_reason=None):
    """One frames[] entry: per-keypoint distances plus this frame's mean/max.

    Every distance appears twice, in metres and in body lengths; the '_bl' fields are null
    on a frame whose L_body could not be measured. mean_bl is the mean of the per-keypoint
    RATIOS, not mean(distance)/L_body -- identical here because L_body is one number per
    frame, but it stays correct if L_body ever becomes per-keypoint.
    """
    entry = {"frame": int(frame), "per_keypoint": per_keypoint}
    per_keypoint_bl = {k: _bl_norm(v, l_body) for k, v in per_keypoint.items()}
    per_keypoint_bl = {k: v for k, v in per_keypoint_bl.items() if v is not None}
    entry["per_keypoint_bl"] = per_keypoint_bl
    if per_keypoint:
        worst = max(per_keypoint, key=per_keypoint.get)
        entry["mean"] = float(sum(per_keypoint.values()) / len(per_keypoint))
        entry["max"] = float(per_keypoint[worst])
        entry["max_keypoint"] = worst
    else:
        entry["mean"] = None
        entry["max"] = None
        entry["max_keypoint"] = None
    if per_keypoint_bl:
        worst_bl = max(per_keypoint_bl, key=per_keypoint_bl.get)
        entry["mean_bl"] = float(sum(per_keypoint_bl.values()) / len(per_keypoint_bl))
        entry["max_bl"] = float(per_keypoint_bl[worst_bl])
        entry["max_bl_keypoint"] = worst_bl
    else:
        entry["mean_bl"] = None
        entry["max_bl"] = None
        entry["max_bl_keypoint"] = None
    entry["n_keypoints"] = len(per_keypoint)
    entry["body_length_m"] = None if l_body is None else float(l_body)
    entry["body_length_source"] = l_source
    # Measured like any other frame; the flag only says the pose was not optimizer-fitted.
    entry["blocked"] = blocked_reason is not None
    entry["reason_blocked"] = blocked_reason
    return entry


def _kpt_dist_meta(p, ctx, lo, hi):
    return {
        "schema": KEYPOINT_DISTANCE_SCHEMA,
        "producer": "synthetic_data_generator_ui.py",
        "metric": "l2_distance_between_vertex_group_centroids_world_space",
        "gt_collection": p.collection_name,
        "gt_object": ctx["src_obj"].name,
        "gt_armature": ctx["src_arm"].name if ctx["src_arm"] else None,
        "recon_collection": "Reconstructions",
        "recon_object": ctx["rec_obj"].name,
        "recon_armature": ctx["rec_arm"].name if ctx["rec_arm"] else None,
        "keypoint_list": list(ctx["kpt_list"]),
        "frame_start": int(lo),
        "frame_end": int(hi),
        "blocked_frames": [dict(r) for r in (ctx.get("blocked_records") or [])],
        "blocked_stamp_present": bool(ctx.get("blocked_known")),
        "blocked_policy": ("every frame in [frame_start, frame_end] is measured; 'blocked' on a "
                           "frame row means its pose was not produced by fitting the optimizer "
                           "to that frame, and the consumer decides whether to aggregate it"),
        "units": "meters; every field suffixed '_bl' is the same quantity in GT body lengths",
        "primary_units": "body_lengths",
        "normalisation": ("divided by the per-frame GT body length L_body; see summary."
                          "body_length for its definition, provenance and spread"),
    }


def _kpt_dist_summarize(records, per_kpt, per_kpt_bl, missing, blocked_records=()):
    """Aggregate per keypoint / per frame. Pure: builds the summary dict, touches no file.

    Split out of _kpt_dist_finalize so the batch operator can put the same summary into an
    in-memory collection without also writing a per-run keypoint_distances_*.json.

    `per_kpt` and `per_kpt_bl` are the metre and body-length series of the same keypoints.
    They are aggregated independently rather than dividing the metre aggregate by a mean
    L_body: the mean of the per-frame ratios is the quantity that is comparable across runs,
    and a frame whose L_body was unmeasurable must drop out of the body-length aggregate
    while still counting towards the metre one.
    """
    per_kpt_summary = {}
    for k, series in per_kpt.items():
        vals = [d for _f, d in series]
        i_max = max(range(len(vals)), key=lambda j: vals[j])
        entry = {
            "n_frames": len(vals),
            "mean": float(sum(vals) / len(vals)),
            "max": float(vals[i_max]),
            "max_frame": int(series[i_max][0]),
        }
        series_bl = per_kpt_bl.get(k, [])
        vals_bl = [d for _f, d in series_bl]
        if vals_bl:
            j_max = max(range(len(vals_bl)), key=lambda j: vals_bl[j])
            entry.update({
                "n_frames_bl": len(vals_bl),
                "mean_bl": float(sum(vals_bl) / len(vals_bl)),
                "max_bl": float(vals_bl[j_max]),
                "max_bl_frame": int(series_bl[j_max][0]),
            })
        else:
            entry.update({"n_frames_bl": 0, "mean_bl": None, "max_bl": None,
                          "max_bl_frame": None})
        per_kpt_summary[k] = entry

    all_vals = [(f["frame"], k, d) for f in records for k, d in f["per_keypoint"].items()]
    all_bl = [(f["frame"], k, d) for f in records
              for k, d in f.get("per_keypoint_bl", {}).items()]
    overall = {
        "n_frames": len(records),
        "n_blocked": len(list(blocked_records)),
        "blocked_frames": [dict(r) for r in blocked_records],
        "n_keypoints": len(per_kpt_summary),
        "skipped_keypoints": dict(missing),
        "body_length": _body_length_block([f.get("body_length_m") for f in records],
                                          [f.get("body_length_source", "unavailable")
                                           for f in records]),
        "per_keypoint": per_kpt_summary,
    }
    if all_vals:
        mx = max(all_vals, key=lambda t: t[2])
        overall.update({
            "overall_mean": float(sum(t[2] for t in all_vals) / len(all_vals)),
            "overall_max": float(mx[2]),
            "overall_max_frame": int(mx[0]),
            "overall_max_keypoint": mx[1],
        })
    if all_bl:
        mx = max(all_bl, key=lambda t: t[2])
        overall.update({
            "overall_mean_bl": float(sum(t[2] for t in all_bl) / len(all_bl)),
            "overall_max_bl": float(mx[2]),
            "overall_max_bl_frame": int(mx[0]),
            "overall_max_bl_keypoint": mx[1],
        })
    else:
        overall.update({"overall_mean_bl": None, "overall_max_bl": None,
                        "overall_max_bl_frame": None, "overall_max_bl_keypoint": None})
    return overall


def _kpt_dist_finalize(records, per_kpt, per_kpt_bl, missing, meta, out_path,
                       blocked_records=()):
    """Aggregate per keypoint / per frame, write the JSON, return the summary dict."""
    overall = _kpt_dist_summarize(records, per_kpt, per_kpt_bl, missing,
                                  blocked_records)
    data = {"meta": meta, "summary": overall, "frames": records}
    with open(out_path, 'w') as jf:
        json.dump(data, jf, indent=2)
    return overall


# --- shared IoU / keypoint frame loop ---------------------------------------
#
# The loop below is the single implementation behind THREE entry points:
#   synth.compute_volumetric_iou     need_bvh=True,  kpt_list optionally non-empty
#   synth.compute_keypoint_distances need_bvh=False, kpt_list non-empty
#   synth.batch_3d_metrics           need_bvh=True,  once per imported pts2 file
# It was factored out of the first two verbatim -- same order of operations, same
# payload extraction, same warn-once policy -- so their JSON output is unchanged.
# Exceptions propagate: each caller owns its own error wording and return value.

def _iou_kpt_eval_pass(context, ctx, kpt_list, need_bvh, n_samples, seed, report=None):
    """One frame_set/to_mesh pass producing the IoU frames and/or the keypoint distances.

    Returns dict(iou_frames, iou_skipped, kpt_records, per_kpt, missing).
    `kpt_list` empty disables the keypoint block; `need_bvh` False disables the IoU block
    (and, with it, the BVHTree build -- the cheap ~2 to_mesh()-per-frame path).
    The scene's current frame is restored on every exit path.
    """
    scene = context.scene
    p = scene.synth_props
    src_obj, rec_obj = ctx["src_obj"], ctx["rec_obj"]
    lo, hi = int(ctx["frame_lo"]), int(ctx["frame_hi"])
    want_kpts = bool(kpt_list)
    # widened only for the payload extraction: L_body must be measurable even in IoU-only mode
    payload_kpts = _iou_kpt_payload_kpt_list(src_obj, kpt_list)
    # Blocked frames are measured like any other. The flag rides along on each row so the
    # analysis step can drop them; dropping them here would make the per-frame numbers
    # unrecoverable and hide how bad the gap-filled poses actually are.
    blocked_by_frame = ctx.get("blocked_by_frame") or {}

    iou_frames, iou_skipped = [], []
    kpt_records, per_kpt, per_kpt_bl, missing = [], defaultdict(list), defaultdict(list), {}
    degenerate_warned = False

    deps = context.evaluated_depsgraph_get()
    original_frame = scene.frame_current
    try:
        for frame in range(lo, hi + 1):
            blocked_reason = blocked_by_frame.get(int(frame))
            scene.frame_set(frame)
            deps.update()

            pay_gt = pay_rc = None
            try:
                # the 'not in the depsgraph' warning is a property of the setup, not of the
                # frame, so it is only ever raised on the first one
                rep = report if frame == lo else None
                pay_gt = _recon_frame_payload(deps, p.collection_name, src_obj,
                                              payload_kpts, need_bvh, rep)
                pay_rc = _recon_frame_payload(deps, "Reconstructions", rec_obj,
                                              payload_kpts, need_bvh, rep)

                # GT-side only, per frame: the normaliser for every length below
                l_body, l_source = _iou_kpt_body_length(pay_gt)

                if want_kpts:
                    d = _kpt_frame_distances(pay_gt["kpt_centroids"], pay_rc["kpt_centroids"],
                                             kpt_list, missing, report)
                    rec = _kpt_record(frame, d, l_body, l_source, blocked_reason)
                    kpt_records.append(rec)
                    for k, v in d.items():
                        per_kpt[k].append((frame, v))
                    for k, v in rec["per_keypoint_bl"].items():
                        per_kpt_bl[k].append((frame, v))

                if not need_bvh:
                    continue

                # a degenerate frame still contributes its keypoint distances above: the
                # centroids are well defined even where the tessellation is not
                if pay_gt["bvh"] is None or pay_rc["bvh"] is None:
                    if not degenerate_warned:
                        degenerate_warned = True
                        if report:
                            report({'WARNING'},
                                   f"Frame {frame}: degenerate mesh (GT faces="
                                   f"{pay_gt['n_faces']}, recon faces={pay_rc['n_faces']}); "
                                   f"frame skipped.")
                    iou_skipped.append(frame)
                    iou_frames.append({"frame": int(frame), "iou": None,
                                       "reason": "degenerate_mesh",
                                       "body_length_m": (None if l_body is None
                                                         else float(l_body)),
                                       "body_length_source": l_source,
                                       "blocked": blocked_reason is not None,
                                       "reason_blocked": blocked_reason})
                    continue

                res = _iou_frame(pay_gt["bvh"], pay_gt["box"],
                                 pay_rc["bvh"], pay_rc["box"], n_samples, seed)
                if res is None:
                    if not degenerate_warned:
                        degenerate_warned = True
                        if report:
                            report({'WARNING'}, f"Frame {frame}: zero-volume occupancy; "
                                                f"frame skipped.")
                    iou_skipped.append(frame)
                    iou_frames.append({"frame": int(frame), "iou": None,
                                       "reason": "zero_volume",
                                       "body_length_m": (None if l_body is None
                                                         else float(l_body)),
                                       "body_length_source": l_source,
                                       "blocked": blocked_reason is not None,
                                       "reason_blocked": blocked_reason})
                    continue
                res["frame"] = int(frame)
                # 'iou' and the occupancy counts are ratios and stay as they are; only the
                # four volumes carry a unit, and theirs is a cubed length
                for key in ("vol_gt", "vol_recon", "vol_intersection", "vol_union"):
                    res[key + "_bl3"] = _bl_norm(res.get(key), l_body, power=3)
                res["body_length_m"] = None if l_body is None else float(l_body)
                res["body_length_source"] = l_source
                res["blocked"] = blocked_reason is not None
                res["reason_blocked"] = blocked_reason
                iou_frames.append(res)
            finally:
                # BVHTrees hold their own geometry copy; drop both payloads every frame so
                # nothing accumulates over a several-hundred-frame sequence.
                del pay_gt
                del pay_rc
    finally:
        scene.frame_set(original_frame)

    return {"iou_frames": iou_frames, "iou_skipped": iou_skipped,
            "kpt_records": kpt_records, "per_kpt": per_kpt, "per_kpt_bl": per_kpt_bl,
            "missing": missing, "blocked_records": list(ctx.get("blocked_records") or [])}


def _iou_meta(p, ctx, lo, hi, n_samples, seed):
    return {
        "schema": VOLUMETRIC_IOU_SCHEMA,
        "producer": "synthetic_data_generator_ui.py",
        "method": "monte_carlo_occupancy_bvh_ray_parity",
        "gt_collection": p.collection_name,
        "gt_object": ctx["src_obj"].name,
        "recon_object": ctx["rec_obj"].name,
        "samples_per_frame": int(n_samples),
        "random_seed": int(seed),
        "ray_direction": [float(_IOU_RAY_DIR.x), float(_IOU_RAY_DIR.y), float(_IOU_RAY_DIR.z)],
        "frame_start": int(lo),
        "frame_end": int(hi),
        "blocked_frames": [dict(r) for r in (ctx.get("blocked_records") or [])],
        "blocked_stamp_present": bool(ctx.get("blocked_known")),
        "blocked_policy": ("every frame in [frame_start, frame_end] is measured; 'blocked' on a "
                           "frame row means its pose was not produced by fitting the optimizer "
                           "to that frame, and the consumer decides whether to aggregate it"),
        "units": {
            "iou": "dimensionless ratio of volumes -- already scale free, NOT normalised",
            "iou_stderr": "dimensionless",
            "vol_*": "cubic meters",
            "vol_*_bl3": "cubic GT body lengths (vol_* / L_body^3)",
            "n_*": "sample counts, dimensionless",
            "body_length_m": "meters",
        },
    }


def _iou_summary(frames, skipped, blocked_records=()):
    """Aggregate the IoU frame records. None when no frame produced a valid IoU.

    The IoU needs no body-length normalisation, but the run's L_body is summarised here
    anyway: it is the scale the sibling keypoint distances were divided by, and the two
    blocks are read together.

    Blocked frames ARE included in every aggregate here, because they were measured like any
    other frame: this summary describes the whole evaluated range. `blocked_records` is carried
    alongside so a consumer that wants accuracy-over-fitted-frames-only can recompute from
    frames[] -- which is why every row keeps its own 'blocked' flag rather than relying on this
    list. n_blocked is reported so the two populations are never confused.
    """
    vals = [f["iou"] for f in frames if f.get("iou") is not None]
    if not vals:
        return None
    i_min = min(range(len(vals)), key=lambda k: vals[k])
    i_max = max(range(len(vals)), key=lambda k: vals[k])
    valid_frames = [f["frame"] for f in frames if f.get("iou") is not None]
    summary = {
        "n_frames": len(frames),
        "n_valid": len(vals),
        "n_skipped": len(skipped),
        "skipped_frames": list(skipped),
        "n_blocked": len(list(blocked_records)),
        "blocked_frames": [dict(r) for r in blocked_records],
        "mean_iou": float(sum(vals) / len(vals)),
        "min_iou": float(vals[i_min]),
        "min_iou_frame": int(valid_frames[i_min]),
        "max_iou": float(vals[i_max]),
        "max_iou_frame": int(valid_frames[i_max]),
        "body_length": _body_length_block(
            [f.get("body_length_m") for f in frames],
            [f.get("body_length_source", "unavailable") for f in frames]),
    }
    for key in ("vol_gt_bl3", "vol_recon_bl3", "vol_intersection_bl3", "vol_union_bl3"):
        series = [f[key] for f in frames if f.get(key) is not None]
        summary["mean_" + key] = float(sum(series) / len(series)) if series else None
    return summary


# =============================================================================
# RECONSTRUCTION EVALUATION -- MPVE / MPJPE / per-bone SO(3) geodesic error
# =============================================================================
#
# Three further metrics, computed from the SAME frame loop and the SAME two to_mesh()
# extractions per frame as the IoU / keypoint-distance pass above:
#
#   MPVE(f)  = (1 / N_v) sum_v || x^_{f,v} - x_{f,v} ||_2, world space, matched by VERTEX
#              INDEX. create_animation_from_pose_time_series builds the reconstruction as
#              src_obj.copy() with new_obj.data = src_obj.data.copy(), so vertex count,
#              order and groups are identical by construction (asserted once by
#              _recon_check_keypoint_correspondence, and per frame on the array shapes).
#              Exact correspondence is free: there is no Chamfer distance and no
#              nearest-neighbour search anywhere in this file.
#              Pavlakos et al., CVPR 2018; Kolotouros et al., SPIN, ICCV 2019;
#              Zuffi et al., SMAL, CVPR 2017.
#
#   MPJPE(f) = (1 / K) sum_i || y^_i - y_i ||_2 over the keypoint centroids and,
#              separately, over the armature bone heads, reported as the triple
#              global / root_relative / pa so that the three error sources can be told
#              apart:  global - root_relative ~ root-localisation error (MRPE),
#              root_relative - pa ~ global-orientation + scale error, pa = residual
#              articulation/shape error.
#              Ionescu et al., Human3.6M, TPAMI 2014; Kabsch 1976 / Gower 1975;
#              Umeyama, TPAMI 1991; Kanazawa et al., CVPR 2018 (PA-MPJPE);
#              Moon et al., RootNet, ICCV 2019 (root localisation).
#
#   theta_b  = arccos( clip( (tr(R_b^T R^_b) - 1) / 2, -1, 1 ) ) = || log(R_b^T R^_b) ||,
#              per bone and frame, in armature space (global) AND parent-relative (local,
#              the quantity the optimiser parameterises as `body_pose`). Both are needed:
#              a proximal error propagates down the chain, so a small local error can be a
#              large global one, and two compensating local errors can leave the global
#              frames nearly correct.
#              Huynh, Metrics for 3D rotations, JMIV 2009; Mahmood et al., AMASS, ICCV
#              2019; Zuffi et al., SMALR, CVPR 2018.
#
# Every distance metric reports a median beside its mean and an explicit outlier rate
# (3D-PCK at tau = pck_tau_body_lengths * L_body), because the failure mode of this
# pipeline is heavy-tailed -- one tail-flip frame dominates a mean, and a median alone
# hides exactly those frames. Every file carries a coverage block (n_frames, n_valid,
# skipped_frames with reasons, per-keypoint/per-bone valid counts): an accuracy number
# without its coverage flatters a pipeline that silently dropped its hard frames.

MPVE_SCHEMA = "mpve/1"
MPJPE_SCHEMA = "mpjpe/1"
BONE_ROTATION_ERROR_SCHEMA = "bone_rotation_error/1"

# Keypoints spanning the fish's body axis. L_body normalises every length in this section
# and is ALWAYS measured on the GT mesh -- measured on the reconstruction, an over-scaled
# fit would divide its own error away and get a free pass.
_RECON_BODY_LENGTH_KPTS = ("mouth tip", "caudal peduncle")
_RECON_UNGROUPED = "__ungrouped__"
_RECON_VIRTUAL_GROUP = "__virtual__"
_RECON_P = 95.0
_RECON_EPS = 1e-12
# max |R^T R - I| tolerated before a "rotation" matrix is rejected (see _recon_mat3)
_RECON_ORTHO_TOL = 1e-6
# max angle between the two armatures' world-space rotations before the armature-space
# comparison is reported as questionable (create_animation_from_pose_time_series sets
# rec_arm.matrix_world = src_arm.matrix_world @ Scale(S), so this should be ~0)
_RECON_ARM_ROT_TOL = math.radians(1e-3)


class _ReconFatal(RuntimeError):
    """A condition that invalidates the whole run, not just one frame.

    Vertex-count mismatch, a bone-name/convention mismatch between the two armatures or a
    failed pose round trip mean the numbers would measure the wrong thing entirely, so the
    operator aborts instead of emitting a plausible-looking file.
    """


# --- small numeric helpers (shared by all three metrics) --------------------

def _recon_num(x):
    """float(x), or None for None / NaN / inf.

    json.dump would happily write a bare `NaN`, which is not JSON. Every scalar that
    reaches a metric file goes through here, so a missing value is `null` with a `reason`
    beside it and never a silent 0.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _recon_nums(seq):
    return [_recon_num(v) for v in seq]


def _recon_stats(values, labels=None, label_key="argmax", weights=None):
    """mean / median / p95 / max / argmax of a 1-D sample, NaN-safe.

    NaN is the "missing on this frame, for a recorded reason" encoding used throughout
    this section; such entries are dropped and `n` reports how many samples survived, so
    an aggregate is never published without its coverage.

    `weights` weights the MEAN only. The median and p95 stay unweighted: a weighted
    quantile is a different estimator, and the area-weighting flag exists to test the
    mean's sensitivity to tessellation, not to redefine the quantiles.
    """
    a = np.asarray(values, dtype=np.float64).ravel()
    out = {"n": 0, "mean": None, "median": None, "p95": None, "max": None, label_key: None}
    if a.size == 0:
        return out
    m = np.isfinite(a)
    if not m.any():
        return out
    v = a[m]
    idx = np.nonzero(m)[0]
    i = int(idx[int(np.argmax(v))])
    if weights is None:
        mean = float(v.mean())
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()[m]
        sw = float(w.sum())
        mean = float((v * w).sum() / sw) if sw > _RECON_EPS else float(v.mean())
    out.update({
        "n": int(v.size),
        "mean": mean,
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, _RECON_P)),
        "max": float(v.max()),
        label_key: (labels[i] if labels is not None else i),
    })
    return out


def _recon_coverage(requested, valid, skipped, blocked_records=()):
    """Coverage rho = n_valid / n_frames, plus every skipped frame WITH its reason.

    Blocked frames are inside both terms: they were requested and they scored, so rho keeps its
    meaning -- 'of everything we set out to measure, how much produced a number'. Whether to
    aggregate them is a separate question, answered per row by frames[].blocked and summarised
    by n_blocked here, so a reader can see at a glance how much of a run's coverage is made up
    of poses that were not optimizer-fitted.
    """
    n = int(len(requested))
    nv = int(len(valid))
    blocked = [dict(r) for r in blocked_records]
    return {
        "n_frames": n,
        "n_valid": nv,
        "n_skipped": int(len(skipped)),
        "coverage_rho": (float(nv) / n) if n else None,
        "skipped_frames": [{"frame": int(f), "reason": str(r)} for f, r in skipped],
        "n_blocked": len(blocked),
        "blocked_frames": blocked,
        "blocked_policy": (
            "blocked frames are measured and counted in n_frames/n_valid/coverage_rho; the "
            "flag records that their pose was not produced by fitting the optimizer to that "
            "frame, leaving it to the consumer to exclude them from accuracy aggregates"
        ),
    }


def _recon_pck(errors, tau):
    """3D-PCK: fraction of valid entries within `tau`. None when tau is undefined."""
    if tau is None or not math.isfinite(tau) or tau <= 0.0:
        return None
    a = np.asarray(errors, dtype=np.float64).ravel()
    m = np.isfinite(a)
    if not m.any():
        return None
    return float((a[m] <= tau).mean())


def _recon_mat3(mat, check=False):
    """mathutils 3x3/4x4 -> (3,3) float64, optionally verified orthonormal.

    P[b] carries the reconstruction armature's per-frame scale S, and a scaled matrix
    makes the trace formula silently wrong (tr(R^T R^) is inflated by the scale, so the
    geodesic angle comes out too small). Callers pass matrices that have already been
    orthonormalised by _rot3 (to_quaternion().to_matrix()); `check` re-verifies that on
    the first frame instead of trusting it.
    """
    a = np.array(mat, dtype=np.float64)[:3, :3]
    if check:
        err = float(np.abs(a.T @ a - np.eye(3)).max())
        if err > _RECON_ORTHO_TOL:
            raise _ReconFatal(f"non-orthonormal rotation matrix (max |R^T R - I| = {err:.3e}); "
                              f"the geodesic trace formula would be silently wrong")
    return a


# --- SO(3) ------------------------------------------------------------------

def _so3_geodesic(R_a, R_b):
    """theta = arccos((tr(R_a^T R_b) - 1) / 2) = || log(R_a^T R_b) ||, batched (...,3,3).

    tr(A^T B) is the Frobenius inner product, hence the einsum rather than an explicit
    transpose-and-multiply. Huynh, JMIV 2009, eq. (23). Both operands MUST be orthonormal
    (see _recon_mat3).
    """
    tr = np.einsum('...ij,...ij->...', R_a, R_b)
    return np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))


def _so3_relative(R_a, R_b):
    """R_a^T R_b, batched."""
    return np.einsum('...ji,...jk->...ik', R_a, R_b)


def _so3_quat(R):
    """(...,3,3) rotation matrices -> (...,4) unit quaternions (w, x, y, z), w >= 0.

    Shepperd's branch selection (largest denominator wins) rather than the naive
    trace-only form, which loses all its precision near theta = pi -- exactly where a
    tail-flip frame lives.
    """
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    cand = np.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
                     1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], axis=-1)
    k = np.argmax(cand, axis=-1)
    s = 2.0 * np.sqrt(np.clip(np.take_along_axis(cand, k[..., None], axis=-1)[..., 0], 0.0, None))
    s = np.where(s < _RECON_EPS, 1.0, s)
    d21 = R[..., 2, 1] - R[..., 1, 2]
    d02 = R[..., 0, 2] - R[..., 2, 0]
    d10 = R[..., 1, 0] - R[..., 0, 1]
    a21 = R[..., 2, 1] + R[..., 1, 2]
    a02 = R[..., 0, 2] + R[..., 2, 0]
    a10 = R[..., 1, 0] + R[..., 0, 1]
    q0 = np.stack([0.25 * s, d21 / s, d02 / s, d10 / s], axis=-1)
    q1 = np.stack([d21 / s, 0.25 * s, a10 / s, a02 / s], axis=-1)
    q2 = np.stack([d02 / s, a10 / s, 0.25 * s, a21 / s], axis=-1)
    q3 = np.stack([d10 / s, a02 / s, a21 / s, 0.25 * s], axis=-1)
    ke = k[..., None]
    q = np.where(ke == 0, q0, np.where(ke == 1, q1, np.where(ke == 2, q2, q3)))
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), _RECON_EPS)
    return np.where(q[..., :1] < 0.0, -q, q)


def _so3_swing_twist_y(q):
    """Swing/twist magnitudes of (...,4) quaternions about the local +Y axis.

    Y is Blender's bone axis and the twist axis of the optimiser's angle priors
    (losses_edit.decompose_to_swing_twist). Twist about a fish's body axis is far less
    observable from silhouettes than swing, so splitting the error shows which DoF the
    multi-view rig actually constrains.

    q = q_swing * q_twist with q_twist = normalize((w, 0, y, 0)); with |q| = 1 the scalar
    part of q_swing is exactly n = sqrt(w^2 + y^2) and its vector part is
    ((wx + zy), 0, (wz - xy)) / n, so both angles follow without forming the product.
    Returns (swing, twist) in radians: twist signed about +Y, swing unsigned.
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    nn = w * w + y * y
    n = np.sqrt(nn)
    deg = n < 1e-8                              # exact 180 deg swing: the twist is undefined
    twist = 2.0 * np.arctan2(np.where(deg, 0.0, y), np.where(deg, 1.0, w))
    twist = (twist + np.pi) % (2.0 * np.pi) - np.pi
    v = np.hypot(w * x + z * y, w * z - x * y)
    # atan2 form, not 2*arccos(n): arccos loses half its significant digits at swing ~ 0
    swing = np.where(deg, 2.0 * np.arccos(np.clip(np.abs(w), 0.0, 1.0)),
                     2.0 * np.arctan2(v, nn))
    return swing, twist


def _so3_swing_twist_components(q):
    """(swing_x, twist_y, swing_z) in radians -- the optimiser's own decomposition.

    Numpy port of losses_edit.decompose_to_swing_twist, including its degeneracy gate, so
    that "this bone is sitting on its prior" is decided by exactly the criterion the
    fitting stage enforced. For a pure swing this returns the axis-angle vector's X and Z
    components; for a pure twist it returns the signed angle about Y.
    """
    q = np.asarray(q, dtype=np.float64)
    q = np.where(q[..., :1] < 0.0, -q, q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xz = np.sqrt(x * x + z * z + _RECON_EPS)
    yw = np.sqrt(y * y + w * w + _RECON_EPS)
    beta = np.arctan2(xz, yw)                                   # half the swing angle
    gate_ref = math.cos(math.radians(150.0) / 2.0)
    t = np.clip(yw / gate_ref, 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)                              # smoothstep, as in losses_edit
    sing = (np.abs(w) < 1e-6) & (np.abs(y) < 1e-6)
    twist = gate * (2.0 * np.arctan2(np.where(sing, 0.0, y), np.where(sing, 1.0, w)))
    g = twist * 0.5
    sinc = np.where(np.abs(beta) < 1e-8, 1.0, np.sin(beta) / np.where(beta == 0.0, 1.0, beta))
    scale = 2.0 / sinc
    swing_x = scale * (np.cos(g) * x - np.sin(g) * z)
    swing_z = scale * (np.sin(g) * x + np.cos(g) * z)
    return swing_x, twist, swing_z


# --- Procrustes -------------------------------------------------------------

def _recon_umeyama(src, dst, with_scale=True):
    """min_{s,R,t} sum_i || s R src_i + t - dst_i ||^2, R constrained to det(R) = +1.

    Kabsch 1976 / Gower 1975 orthogonal Procrustes in the closed form of Umeyama (TPAMI
    1991): SVD of the cross-covariance, with the sign of the last singular vector flipped
    when det(U V^T) < 0 so that the solution stays a rotation instead of a reflection.
    UNIFORM scale -- the scale-free variant is a different metric reported under the same
    name in the literature, which is why `meta` says which one this is.

    Returns the transform, the aligned points, the singular values of the cross-covariance
    and its condition number, or None for fewer than 3 points.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = int(src.shape[0])
    if n < 3 or dst.shape != src.shape:
        return None
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    X, Y = src - mu_s, dst - mu_d
    C = (Y.T @ X) / n
    U, S, Vt = np.linalg.svd(C)
    reflection = float(np.linalg.det(U @ Vt)) < 0.0
    dsign = np.array([1.0, 1.0, -1.0 if reflection else 1.0])
    R = U @ np.diag(dsign) @ Vt
    var_src = float((X ** 2).sum() / n)
    s = float((S * dsign).sum() / var_src) if (with_scale and var_src > _RECON_EPS) else 1.0
    t = mu_d - s * (R @ mu_s)
    s0, s2 = float(S[0]), float(S[2])
    gt_sv = np.linalg.svd(Y, compute_uv=False)
    return {
        "R": R, "t": t, "s": s,
        "aligned": s * (src @ R.T) + t,
        "singular_values": [float(v) for v in S],
        "condition_number": (s0 / s2) if s2 > _RECON_EPS else float('inf'),
        # sigma_2 / sigma_0: with K ~ 6-10 keypoints and a fish that is frequently nearly
        # straight the configuration is close to 1-D and the alignment is unstable.
        "sigma_ratio": (s2 / s0) if s0 > _RECON_EPS else 0.0,
        "gt_singular_values": [float(v) for v in gt_sv],
        "gt_sigma_ratio": float(gt_sv[2] / gt_sv[0]) if float(gt_sv[0]) > _RECON_EPS else 0.0,
        "reflection_corrected": bool(reflection),
        "n_points": n,
    }


def _recon_point_block(Y_gt, Y_rc, root_gt, root_rc, tau, sigma_min):
    """global / root_relative / pa error vectors for ONE frame's point set.

    Y_gt, Y_rc are (K, 3) world-space arrays whose rows are NaN where the point is missing
    on that mesh this frame; the same valid subset is used on both sides by construction.
    `root_gt` / `root_rc` are each set's own reference point (None -> the centroid of that
    set's valid points). PA is set to None -- never to a meaningless number -- when the
    configuration is degenerate.
    """
    K = int(Y_gt.shape[0])
    nan = np.full(K, np.nan)
    out = {"global": nan.copy(), "root_relative": nan.copy(), "pa": nan.copy(),
           "valid": np.zeros(K, dtype=bool), "pa_info": None,
           "pa_degenerate": True, "pa_reason": "not_computed"}
    valid = np.isfinite(Y_gt).all(axis=1) & np.isfinite(Y_rc).all(axis=1)
    out["valid"] = valid
    if not valid.any():
        out["pa_reason"] = "no_valid_points"
        return out
    A, B = Y_gt[valid], Y_rc[valid]
    out["global"][valid] = np.linalg.norm(B - A, axis=1)

    ra = np.asarray(root_gt, dtype=np.float64) if root_gt is not None else None
    rb = np.asarray(root_rc, dtype=np.float64) if root_rc is not None else None
    if ra is None or not np.isfinite(ra).all():
        ra = A.mean(axis=0)
    if rb is None or not np.isfinite(rb).all():
        rb = B.mean(axis=0)
    out["root_relative"][valid] = np.linalg.norm((B - rb) - (A - ra), axis=1)

    info = _recon_umeyama(B, A, with_scale=True)
    if info is None:
        out["pa_reason"] = f"fewer_than_3_valid_points ({int(valid.sum())})"
    else:
        out["pa_info"] = {
            "singular_values": info["singular_values"],
            "condition_number": _recon_num(info["condition_number"]),
            "sigma_ratio": info["sigma_ratio"],
            "gt_singular_values": info["gt_singular_values"],
            "gt_sigma_ratio": info["gt_sigma_ratio"],
            "scale": info["s"],
            "reflection_corrected": info["reflection_corrected"],
            "n_points": info["n_points"],
        }
        if info["sigma_ratio"] < sigma_min:
            out["pa_reason"] = (f"degenerate_configuration: sigma2/sigma0 = "
                                f"{info['sigma_ratio']:.3e} < {sigma_min:.3e}")
        else:
            out["pa"][valid] = np.linalg.norm(info["aligned"] - A, axis=1)
            out["pa_degenerate"] = False
            out["pa_reason"] = None
    if out["pa_degenerate"]:
        out["pa"][:] = np.nan
    out["tau"] = tau
    return out


def _recon_point_frame_stats(block, names, frame, tau, label_key="argmax_item"):
    """Per-frame mean/median/p95/max/argmax + PCK for each term of a point block."""
    entry = {"frame": int(frame), "n_valid": int(block["valid"].sum()),
             "pa_degenerate": bool(block["pa_degenerate"]),
             "pa_reason": block["pa_reason"], "alignment": block["pa_info"]}
    for term in ("global", "root_relative", "pa"):
        st = _recon_stats(block[term], labels=names, label_key=label_key)
        st["n_valid"] = st.pop("n")
        st["pck"] = _recon_pck(block[term], tau)
        entry[term] = st
    # the reportable differences: what each alignment step removed
    g, r, a = entry["global"]["mean"], entry["root_relative"]["mean"], entry["pa"]["mean"]
    entry["decomposition"] = {
        "root_localisation_error": _recon_num(None if (g is None or r is None) else g - r),
        "orientation_and_scale_error": _recon_num(None if (r is None or a is None) else r - a),
    }
    return entry


# --- template / mesh side helpers -------------------------------------------

def _recon_body_length(cent_gt, verts_gt, joints_gt):
    """Per-frame GT body length L_body and where it came from.

    Primary: || centroid(mouth tip) - centroid(caudal peduncle) || on the GT mesh.
    Fallbacks (flagged, never silent): the GT mesh AABB diagonal, then the GT keypoint or
    joint AABB diagonal when the mesh vertices were not extracted for this pass.
    """
    a = cent_gt.get(_RECON_BODY_LENGTH_KPTS[0])
    b = cent_gt.get(_RECON_BODY_LENGTH_KPTS[1])
    if a is not None and b is not None:
        L = float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))
        if math.isfinite(L) and L > _RECON_EPS:
            return L, "keypoints"
    for arr, src in ((verts_gt, "gt_mesh_aabb_diagonal"),
                     (np.asarray([c for c in cent_gt.values()], dtype=np.float64)
                      if cent_gt else None, "gt_keypoint_aabb_diagonal"),
                     (joints_gt, "gt_joint_aabb_diagonal")):
        if arr is None or len(arr) < 2:
            continue
        d = np.asarray(arr, dtype=np.float64)
        d = d[np.isfinite(d).all(axis=1)]
        if d.shape[0] < 2:
            continue
        L = float(np.linalg.norm(d.max(axis=0) - d.min(axis=0)))
        if math.isfinite(L) and L > _RECON_EPS:
            return L, src
    return None, "unavailable"


def _recon_vertex_area_incidence(face_verts, n_verts):
    """Flat (vertex, face) incidence arrays, built ONCE from the constant topology.

    Lets the per-frame vertex areas be a single np.bincount over the evaluated face areas
    instead of a Python loop over faces on every frame.
    """
    v_idx, f_idx = [], []
    for fi, verts in enumerate(face_verts):
        for vi in verts:
            v_idx.append(vi)
            f_idx.append(fi)
    return (np.asarray(v_idx, dtype=np.int64), np.asarray(f_idx, dtype=np.int64), int(n_verts))


def _recon_vertex_areas(incidence, face_areas):
    """Vertex area = 1/3 of the summed area of the incident faces (Deliverable 1)."""
    v_idx, f_idx, n_verts = incidence
    if v_idx.size == 0:
        return np.zeros(n_verts, dtype=np.float64)
    return np.bincount(v_idx, weights=face_areas[f_idx] / 3.0, minlength=n_verts)


def _recon_tessellation_report(face_areas):
    """Is the template uniform enough for an UNWEIGHTED per-vertex mean to be meaningful?

    An unweighted MPVE is a mean over vertices, not over surface area, so a densely
    tessellated fin counts far more than its area warrants. Recorded in every mpve JSON so
    the reader can decide whether `area_weighted` should have been on.
    """
    a = np.asarray(face_areas, dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0.0)]
    if a.size == 0:
        return {"n_faces": 0, "uniform_enough": None, "reason": "no faces"}
    mean, med = float(a.mean()), float(np.median(a))
    cv = float(a.std() / mean) if mean > _RECON_EPS else None
    ratio = float(np.percentile(a, 95) / med) if med > _RECON_EPS else None
    return {
        "n_faces": int(a.size),
        "area_mean": mean,
        "area_median": med,
        "area_cv": cv,
        "area_p95_over_median": ratio,
        "criterion": "uniform_enough := area_cv < 0.5 and area_p95/area_median < 3",
        "uniform_enough": bool(cv is not None and ratio is not None and cv < 0.5 and ratio < 3.0),
    }


def _recon_bone_group_table(mesh_info, order, virtual, report=None):
    """{group: [bones]} from get_mesh_json()["bone_groups"] -- the optimiser's own partition.

    Read from the template rather than from p.bone_groups so the metric aggregates on
    exactly the groups the optimiser scheduled on. Groups may overlap: a bone can belong to
    several, so this is NOT a partition and no code here assumes it is. Real bones in no
    group land in "__ungrouped__"; virtual bones (chain-only, forced to identity by LBS,
    with no Blender bone) are collected in "__virtual__" and left out of the headline
    summary.
    """
    groups, membership = {}, {b: [] for b in order}
    for i, g in enumerate(mesh_info.get("bone_groups", []) or []):
        names = [b for b in (g.get("bone_names") or []) if b in membership]
        if not names:
            continue
        name = f"group_{i:02d}"
        groups[name] = names
        for b in names:
            membership[b].append(name)
    ungrouped = [b for b in order if not membership[b] and b not in virtual]
    if ungrouped:
        groups[_RECON_UNGROUPED] = ungrouped
        for b in ungrouped:
            membership[b].append(_RECON_UNGROUPED)
        if report:
            report({'WARNING'}, f"{len(ungrouped)} bone(s) belong to no bone group; "
                                f"aggregated under '{_RECON_UNGROUPED}'.")
    if virtual:
        vs = [b for b in order if b in virtual]
        if vs:
            groups[_RECON_VIRTUAL_GROUP] = vs
            for b in vs:
                membership[b].append(_RECON_VIRTUAL_GROUP)
    return groups, membership


def _recon_heads_world(P, bones, arm_matrix_world):
    """(len(bones), 3) world-space bone heads from the armature-space pose matrices.

    P[b][:3, 3] is rest_head[b] mapped through the bone's pose matrix; the two armatures
    have different matrix_world (the reconstruction's carries the per-frame scale S), so
    each is mapped through its own.
    """
    m = np.array(arm_matrix_world, dtype=np.float64)
    H = np.array([[P[b][0][3], P[b][1][3], P[b][2][3]] for b in bones], dtype=np.float64)
    if H.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return H @ m[:3, :3].T + m[:3, 3]


def _recon_meta_base(schema, metric, p, ctx, lo, hi, aggregation_order, extra=None):
    """The meta block every metric file in this section shares.

    States the unit, the world convention, the body-length definition, the aggregation
    order and the coverage contract once, in every file, so no number is ambiguous.
    """
    meta = {
        "schema": schema,
        "producer": "synthetic_data_generator_ui.py",
        "metric": metric,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gt_collection": p.collection_name,
        "gt_object": ctx["src_obj"].name,
        "gt_armature": ctx["src_arm"].name if ctx["src_arm"] else None,
        "recon_collection": "Reconstructions",
        "recon_object": ctx["rec_obj"].name,
        "recon_armature": ctx["rec_arm"].name if ctx["rec_arm"] else None,
        "keypoint_list": list(ctx["kpt_list"]),
        "frame_start": int(lo),
        "frame_end": int(hi),
        "units": "meters",
        "normalised_units": "body_lengths (divided by the per-frame GT L_body)",
        "world_convention": {
            "space": "blender_world",
            "note": ("all coordinates are Blender world space in metres. Distances and "
                     "geodesic angles are invariant under the rigid change of basis to the "
                     "CV convention, so BLENDERWORLD_2_CVWORLD is recorded but NOT applied"),
            "blenderworld_2_cvworld": BLENDERWORLD_2_CVWORLD.tolist(),
        },
        "body_length": {
            "definition": (f"|| centroid('{_RECON_BODY_LENGTH_KPTS[0]}') - "
                           f"centroid('{_RECON_BODY_LENGTH_KPTS[1]}') || on the GT mesh"),
            "measured_on": "ground_truth_only",
            "rationale": ("measured on the reconstruction, an over-scaled fit would divide "
                          "its own error away"),
            "fallback": "GT mesh AABB diagonal, flagged per frame in body_length_source",
        },
        "aggregation_order": aggregation_order,
        "coverage_contract": ("n_frames / n_valid / skipped_frames (with reasons) and "
                              "per-item valid counts accompany every aggregate; missing or "
                              "degenerate values are null with a reason, never 0"),
        "correspondence": ("vertex index and vertex-group identity, guaranteed by "
                           "create_animation_from_pose_time_series (src_obj.copy() + "
                           "data.copy()) and asserted by "
                           "_recon_check_keypoint_correspondence; no correspondence search"),
    }
    if extra:
        meta.update(extra)
    return meta


# --- the shared frame loop ---------------------------------------------------

def _recon_local_delta(D, tree, b):
    """Parent-relative delta-from-rest rotation of bone `b` in armature space.

    Exactly the quantity the optimiser parameterises as `body_pose`:
    D(parent)^-1 @ D(b) with D = R_pose @ R_rest^-1 (see the pose_time_series/2 header).
    For the root there is no parent, so its local delta IS its global delta.
    """
    parent = tree[b]["p"]
    if not parent:
        return D[b]
    return D[parent].inverted() @ D[b]


def _recon_eval_pass(context, ctx, want, opts, report=None):
    """ONE frame loop feeding MPVE, MPJPE and the per-bone SO(3) metric.

    `want` is a subset of {'mpve', 'mpjpe', 'bone_rot'}. Whatever combination is asked
    for, the loop does exactly one scene.frame_set() + deps.update() and at most two
    to_mesh() extractions per frame (GT and reconstruction, both through
    _recon_frame_payload): the mesh extraction dominates the cost of the pass, so a third
    would be a bug. scene.frame_current is restored in a finally block and both payloads
    are dropped every frame so nothing accumulates over several hundred frames.

    Returns the raw per-frame series; the JSON shaping lives in the _mpve_build /
    _mpjpe_build / _bone_rot_build functions so the operators stay thin.
    """
    scene = context.scene
    p = scene.synth_props
    src_obj, src_arm = ctx["src_obj"], ctx["src_arm"]
    rec_obj, rec_arm = ctx["rec_obj"], ctx["rec_arm"]
    kpt_list = list(ctx["kpt_list"])
    lo, hi = int(ctx["frame_lo"]), int(ctx["frame_hi"])
    warnings = []

    # ---- guards, all before the first frame is touched ----------------------
    empty_kpts = _recon_check_keypoint_correspondence(src_obj, rec_obj, kpt_list) if kpt_list else []
    if empty_kpts:
        warnings.append(f"keypoint vertex groups with no members: {empty_kpts}")

    need_verts = 'mpve' in want
    pose_ok = src_arm is not None and rec_arm is not None
    if 'bone_rot' in want and not pose_ok:
        raise _ReconFatal("the per-bone rotation metric needs both armatures; the "
                          "reconstruction mesh has no armature parent.")
    if not pose_ok:
        warnings.append("no reconstruction armature: the MPJPE joint block is skipped and "
                        "MPVE's root-relative variant falls back to the mesh centroid.")

    order = tree = virtual = rest_R = rest_head = None
    bones_all, bones_real, groups, membership, rest_np, priors = [], [], {}, {}, None, {}
    ts_by_frame, roundtrip = {}, {"performed": False, "reason": "not requested",
                                  "max_abs_err": None, "tolerance": None,
                                  "n_frames_checked": 0, "verified_armature": None}
    if pose_ok:
        mesh_info = get_mesh_json(context)
        order, tree, virtual, rest_R, rest_head, _rest_len = _pts_rest_tables(mesh_info, src_arm)
        bones_all = list(order)
        bones_real = [b for b in order if b not in virtual]
        groups, membership = _recon_bone_group_table(mesh_info, order, virtual, report)
        priors = mesh_info.get("bone_priors", {}) or {}
        rest_np = np.stack([_recon_mat3(rest_R[b]) for b in bones_all])

        # Precondition: a bone-name mismatch would make the metric measure a convention
        # mismatch instead of reconstruction error.
        gt_names = {pb.name for pb in src_arm.pose.bones}
        rc_names = {pb.name for pb in rec_arm.pose.bones}
        if gt_names != rc_names:
            only_gt = sorted(gt_names - rc_names)
            only_rc = sorted(rc_names - gt_names)
            raise _ReconFatal(f"bone names differ between the armatures -- only on GT: "
                              f"{only_gt or 'none'}; only on the reconstruction: "
                              f"{only_rc or 'none'}. Refusing to compute.")
        missing = [b for b in bones_real if b not in gt_names]
        if missing:
            raise _ReconFatal(f"template bones absent from the armatures: {missing}")

        # Precondition: the pose_time_series round trip (the inner comparison of
        # SYNTH_OT_verify_pose_time_series_roundtrip), folded into the frame loop below so
        # it costs no extra frame_set.
        path = (opts.get("roundtrip_json") or "").strip()
        tol = float(opts.get("roundtrip_tol", 1e-4))
        roundtrip["tolerance"] = tol
        if path:
            with open(bpy.path.abspath(path)) as f:
                ts = json.load(f)
            if ts.get("meta", {}).get("schema") != POSE_TIME_SERIES_SCHEMA:
                raise _ReconFatal(f"'{path}' is not a {POSE_TIME_SERIES_SCHEMA} file")
            if list(ts.get("meta", {}).get("bone_order", [])) != order:
                raise _ReconFatal("bone_order in the round-trip JSON does not match the "
                                  "current template; the bone indices would be wrong.")
            ts_by_frame = {int(e["frame"]): e for e in ts.get("frames", [])}
            roundtrip.update({"performed": True, "reason": None,
                              "verified_armature": rec_arm.name, "source": path})
        elif opts.get("roundtrip_required"):
            raise _ReconFatal(
                "no pose time series JSON configured. The per-bone rotation metric refuses "
                "to run without the round-trip check, because without it the metric may be "
                "measuring a convention mismatch rather than reconstruction error. Set "
                "'Round-Trip JSON' to the file the reconstruction was created from, or "
                "untick 'Require Round-Trip Check'.")
        else:
            roundtrip["reason"] = "no pose time series JSON configured"
            warnings.append("round-trip check skipped: no pose time series JSON configured.")

    pa_sigma = float(opts.get("pa_sigma_min", 1e-3))
    tau_bl = float(opts.get("pck_tau_body_lengths", 0.1))
    area_weighted = bool(opts.get("area_weighted", False)) and need_verts
    swing_twist = bool(opts.get("swing_twist", False))
    prior_tol = float(opts.get("prior_saturation_tol", 0.05))

    # Blocked frames are evaluated like any other -- 'requested' is the full range, and rho
    # therefore still means 'of everything we set out to score, how much scored'. The blocked
    # list is carried through so the file can report which rows were not optimizer-fitted.
    blocked_records = list(ctx.get("blocked_records") or [])
    blocked_by_frame = ctx.get("blocked_by_frame") or {}
    evaluated_range = list(range(lo, hi + 1))

    res = {
        "frames": [], "requested": evaluated_range, "skipped": [],
        "blocked_records": blocked_records, "blocked": [],
        "body_length": [], "body_length_source": [], "warnings": warnings,
        "roundtrip": roundtrip, "empty_keypoints": empty_kpts,
        "arm_rotation_offset_deg": [],
        "mpve": {"err": [], "frames": [], "n_verts": None, "tessellation": None,
                 "area_weighted": area_weighted,
                 "root_reference": ("armature_root_bone_head" if pose_ok else "mesh_centroid"),
                 "root_reference_secondary": "mesh_centroid"},
        "mpjpe": {"keypoint": [], "joint": [], "kpt_names": kpt_list,
                  "joint_names": bones_real, "joint_available": pose_ok},
        "bone": {"names": bones_all, "real": bones_real,
                 "virtual": sorted(virtual) if virtual else [],
                 "groups": groups, "membership": membership,
                 "global": [], "local": [], "swing_global": [], "twist_global": [],
                 "swing_local": [], "twist_local": [], "prior_saturated": [],
                 "swing_twist": swing_twist, "priors": priors},
    }

    deps = context.evaluated_depsgraph_get()
    original_frame = scene.frame_current
    incidence = None
    try:
        # evaluated_range, not range(lo, hi + 1): interpolated frames never enter the loop, so
        # they cost no frame_set/to_mesh and cannot reach any per-frame array. `first` already
        # keys off res["frames"] being empty, so the one-shot setup (n_verts, tessellation,
        # area incidence) and the first-frame-only warnings still land on the first frame that
        # is actually evaluated rather than on a skipped one.
        for frame in evaluated_range:
            scene.frame_set(frame)
            deps.update()
            first = not res["frames"]
            rep = report if first else None
            pay_gt = pay_rc = None
            try:
                need_areas = area_weighted or first
                pay_gt = _recon_frame_payload(deps, p.collection_name, src_obj, kpt_list,
                                              False, rep, need_verts=need_verts,
                                              need_face_areas=need_areas)
                pay_rc = _recon_frame_payload(deps, "Reconstructions", rec_obj, kpt_list,
                                              False, rep, need_verts=need_verts)

                V_gt = V_rc = None
                if need_verts:
                    V_gt, V_rc = pay_gt["verts_world"], pay_rc["verts_world"]
                    if V_gt is None or V_rc is None:
                        raise _ReconFatal("world-space vertices were not returned by "
                                          "_recon_frame_payload (need_verts)")
                    if V_gt.shape != V_rc.shape:
                        raise _ReconFatal(
                            f"frame {frame}: vertex array shape mismatch "
                            f"{V_gt.shape} vs {V_rc.shape}; the per-vertex correspondence "
                            f"assumption is broken, refusing to continue.")
                    if V_gt.shape[0] == 0:
                        raise ValueError("empty evaluated mesh")

                # ---- armature side (once per frame, per armature)
                P_gt = D_gt = P_rc = D_rc = None
                Mw_gt = Mw_rc = None
                if pose_ok:
                    gt_arm_eval = src_arm.evaluated_get(deps)
                    rc_arm_eval = rec_arm.evaluated_get(deps)
                    P_gt, D_gt, _ = _pts_posed_armature_space(order, tree, virtual, rest_R,
                                                              gt_arm_eval.pose.bones)
                    P_rc, D_rc, _ = _pts_posed_armature_space(order, tree, virtual, rest_R,
                                                              rc_arm_eval.pose.bones)
                    Mw_gt = gt_arm_eval.matrix_world
                    Mw_rc = rc_arm_eval.matrix_world
                    # create_animation_from_pose_time_series sets
                    # rec_arm.matrix_world = src_arm.matrix_world @ Scale(S), so the two
                    # armature frames differ by a pure scale and armature-space rotations are
                    # directly comparable. Verified rather than assumed.
                    off = float(_so3_geodesic(_recon_mat3(_rot3(Mw_gt), check=first),
                                              _recon_mat3(_rot3(Mw_rc), check=first)))
                    res["arm_rotation_offset_deg"].append(math.degrees(off))
                    if off > _RECON_ARM_ROT_TOL and first:
                        warnings.append(
                            f"the two armatures' world rotations differ by "
                            f"{math.degrees(off):.4f} deg; armature-space rotation errors "
                            f"are offset by that constant rotation.")
                    if ts_by_frame:
                        entry = ts_by_frame.get(frame)
                        if entry is not None:
                            P_ref = _pts_solve_frame(entry, order, tree, rest_R, rest_head, Mw_rc)
                            err = max(abs(P_ref[b][r][c] - P_rc[b][r][c])
                                      for b in order for r in range(4) for c in range(4))
                            roundtrip["n_frames_checked"] += 1
                            if roundtrip["max_abs_err"] is None or err > roundtrip["max_abs_err"]:
                                roundtrip["max_abs_err"] = float(err)
                            if err > roundtrip["tolerance"]:
                                raise _ReconFatal(
                                    f"pose round trip failed at frame {frame}: max |err| = "
                                    f"{err:.3e} > {roundtrip['tolerance']:.3e}. The metric "
                                    f"would measure a convention mismatch, not reconstruction "
                                    f"error.")

                # ---- body length (GT only)
                joints_gt = (_recon_heads_world(P_gt, bones_real, Mw_gt)
                             if (pose_ok and bones_real) else None)
                L_body, L_src = _recon_body_length(pay_gt["kpt_centroids"], V_gt, joints_gt)
                tau = (tau_bl * L_body) if (L_body and tau_bl > 0.0) else None

                # ---- MPVE
                mpve_entry = None
                err_row = None
                if need_verts:
                    if first:
                        res["mpve"]["n_verts"] = int(V_gt.shape[0])
                        res["mpve"]["tessellation"] = _recon_tessellation_report(
                            pay_gt.get("face_areas", np.zeros(0)))
                        if area_weighted:
                            incidence = _recon_vertex_area_incidence(
                                pay_gt.get("face_verts") or [], V_gt.shape[0])
                    w = None
                    if area_weighted and incidence is not None:
                        w = _recon_vertex_areas(incidence, pay_gt.get("face_areas",
                                                                     np.zeros(0)))
                    # fully vectorised: no Python loop over vertices anywhere below
                    e_glob = np.linalg.norm(V_rc - V_gt, axis=1)
                    c_gt, c_rc = V_gt.mean(axis=0), V_rc.mean(axis=0)
                    e_cent = np.linalg.norm((V_rc - c_rc) - (V_gt - c_gt), axis=1)
                    if pose_ok:
                        r_gt = _recon_heads_world(P_gt, [order[0]], Mw_gt)[0]
                        r_rc = _recon_heads_world(P_rc, [order[0]], Mw_rc)[0]
                        e_root = np.linalg.norm((V_rc - r_rc) - (V_gt - r_gt), axis=1)
                    else:
                        e_root = e_cent
                    err_row = e_glob.astype(np.float32)

                    st_glob = _recon_stats(e_glob, label_key="argmax_vertex")
                    st_glob["pck"] = _recon_pck(e_glob, tau)
                    variants = {
                        "global": st_glob,
                        "root_relative": _recon_stats(e_root, label_key="argmax_vertex"),
                        "root_relative_centroid": _recon_stats(e_cent, label_key="argmax_vertex"),
                    }
                    if L_body:
                        variants["body_length_normalised"] = {
                            k: (v / L_body if isinstance(v, float) else v)
                            for k, v in st_glob.items()
                            if k in ("mean", "median", "p95", "max")
                        }
                        variants["body_length_normalised"]["argmax_vertex"] = st_glob["argmax_vertex"]
                    else:
                        variants["body_length_normalised"] = {
                            "mean": None, "median": None, "p95": None, "max": None,
                            "argmax_vertex": None, "reason": f"body_length {L_src}"}
                    mpve_entry = {
                        "frame": int(frame),
                        "n_verts": int(V_gt.shape[0]),
                        "body_length": _recon_num(L_body),
                        "body_length_source": L_src,
                        "variants": variants,
                        "area_weighted_mean": (_recon_stats(e_glob, weights=w)["mean"]
                                               if w is not None else None),
                    }

                # ---- MPJPE (keypoints, then joints)
                kpt_entry = joint_entry = None
                if 'mpjpe' in want:
                    if kpt_list:
                        Y_gt = np.full((len(kpt_list), 3), np.nan, dtype=np.float64)
                        Y_rc = np.full_like(Y_gt, np.nan)
                        for i, k in enumerate(kpt_list):
                            a = pay_gt["kpt_centroids"].get(k)
                            b = pay_rc["kpt_centroids"].get(k)
                            if a is not None:
                                Y_gt[i] = a
                            if b is not None:
                                Y_rc[i] = b
                        blk = _recon_point_block(Y_gt, Y_rc, None, None, tau, pa_sigma)
                        kpt_entry = _recon_point_frame_stats(blk, kpt_list, frame, tau,
                                                             "argmax_keypoint")
                        kpt_entry["_err"] = {t: blk[t] for t in ("global", "root_relative", "pa")}
                    if pose_ok and bones_real:
                        H_gt = joints_gt
                        H_rc = _recon_heads_world(P_rc, bones_real, Mw_rc)
                        root_i = bones_real.index(order[0]) if order[0] in bones_real else None
                        r_gt = H_gt[root_i] if root_i is not None else None
                        r_rc = H_rc[root_i] if root_i is not None else None
                        blk = _recon_point_block(H_gt, H_rc, r_gt, r_rc, tau, pa_sigma)
                        joint_entry = _recon_point_frame_stats(blk, bones_real, frame, tau,
                                                               "argmax_bone")
                        joint_entry["_err"] = {t: blk[t] for t in ("global", "root_relative", "pa")}

                # ---- per-bone SO(3)
                bone_vals = None
                if 'bone_rot' in want:
                    Rg_gt = np.stack([_recon_mat3(_rot3(P_gt[b]), check=first) for b in bones_all])
                    Rg_rc = np.stack([_recon_mat3(_rot3(P_rc[b]), check=first) for b in bones_all])
                    Dl_gt = np.stack([_recon_mat3(_recon_local_delta(D_gt, tree, b)) for b in bones_all])
                    Dl_rc = np.stack([_recon_mat3(_recon_local_delta(D_rc, tree, b)) for b in bones_all])
                    rel_glob = _so3_relative(Rg_gt, Rg_rc)       # already in bone-local axes
                    rel_loc = _so3_relative(Dl_gt, Dl_rc)        # armature axes
                    bone_vals = {
                        "global": _so3_geodesic(Rg_gt, Rg_rc),
                        "local": _so3_geodesic(Dl_gt, Dl_rc),
                    }
                    if swing_twist:
                        # conjugate the local error into each bone's own rest space, which
                        # is where losses_edit's priors (and their Y twist axis) live
                        rel_loc_bone = np.einsum('bji,bjk,bkl->bil', rest_np, rel_loc, rest_np)
                        sg, tg = _so3_swing_twist_y(_so3_quat(rel_glob))
                        sl, tl = _so3_swing_twist_y(_so3_quat(rel_loc_bone))
                        bone_vals.update({"swing_global": sg, "twist_global": tg,
                                          "swing_local": sl, "twist_local": tl})
                        # GT's own articulation vs its prior: an error at a bone that is
                        # pinned against its swing-twist limit is structurally floored and
                        # must not be pooled naively with a free bone's error.
                        gt_bone = np.einsum('bji,bjk,bkl->bil', rest_np, Dl_gt, rest_np)
                        sx, ty, sz = _so3_swing_twist_components(_so3_quat(gt_bone))
                        sat = np.zeros(len(bones_all), dtype=bool)
                        for i, b in enumerate(bones_all):
                            pri = priors.get(b) or {}
                            lx = abs(float(pri.get("swing_x", 0.0)))
                            lz = abs(float(pri.get("swing_z", 0.0)))
                            ly = abs(float(pri.get("twist_y", 0.0)))
                            ell = 0.0
                            if lx > _RECON_EPS:
                                ell += (sx[i] / lx) ** 2
                            if lz > _RECON_EPS:
                                ell += (sz[i] / lz) ** 2
                            hit = ell >= (1.0 - prior_tol) ** 2 if (lx > _RECON_EPS or lz > _RECON_EPS) else False
                            if ly > _RECON_EPS:
                                hit = hit or (abs(ty[i]) >= (1.0 - prior_tol) * ly)
                            sat[i] = bool(hit)
                        bone_vals["prior_saturated"] = sat

            except _ReconFatal:
                raise
            except Exception as exc:
                res["skipped"].append((frame, f"{type(exc).__name__}: {exc}"))
                if report and len(res["skipped"]) == 1:
                    report({'WARNING'}, f"frame {frame} skipped: {exc}")
                continue
            finally:
                # both payloads (and the vertex arrays they own) are dropped every frame
                del pay_gt
                del pay_rc

            # ---- commit the frame only once every requested metric succeeded
            res["frames"].append(int(frame))
            # parallel to res["frames"]: position i answers 'was frame i blocked'
            res["blocked"].append(blocked_by_frame.get(int(frame)))
            res["body_length"].append(L_body if L_body else float('nan'))
            res["body_length_source"].append(L_src)
            if mpve_entry is not None:
                res["mpve"]["err"].append(err_row)
                res["mpve"]["frames"].append(mpve_entry)
            if kpt_entry is not None:
                res["mpjpe"]["keypoint"].append(kpt_entry)
            if joint_entry is not None:
                res["mpjpe"]["joint"].append(joint_entry)
            if bone_vals is not None:
                for k, v in bone_vals.items():
                    res["bone"][k].append(v)
    finally:
        scene.frame_set(original_frame)

    if ts_by_frame and roundtrip["n_frames_checked"] == 0:
        raise _ReconFatal("the round-trip JSON covers none of the evaluated frames "
                          f"[{lo}, {hi}]; the convention check did not actually run.")
    if not res["frames"]:
        raise _ReconFatal(f"no frame in [{lo}, {hi}] could be evaluated: "
                          f"{res['skipped'][:3]}")
    return res


# --- aggregation & file builders --------------------------------------------

def _recon_sequence_summary(frame_values, frames, pooled=None, label_key="argmax_frame"):
    """Aggregate per-frame scalars over the sequence, plus the pooled distribution.

    The headline pair is the mean AND the median of the per-frame means; the pooled
    median over every raw sample is reported beside them because the two answer different
    questions and the difference between them is the size of the heavy tail.
    """
    s = _recon_stats(frame_values, labels=frames, label_key=label_key)
    out = {
        "n_frames": s["n"],
        "mean_of_frame_means": s["mean"],
        "median_of_frame_means": s["median"],
        "p95_of_frame_means": s["p95"],
        "max_frame_mean": s["max"],
        label_key: s[label_key],
    }
    if pooled is not None:
        a = np.asarray(pooled, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            out.update({"pooled_n": int(a.size), "pooled_mean": float(a.mean()),
                        "pooled_median": float(np.median(a)),
                        "pooled_p95": float(np.percentile(a, _RECON_P)),
                        "pooled_max": float(a.max())})
    return out


# --- MPVE --------------------------------------------------------------------
#
# The heat map of Deliverable 4 is driven entirely by the arrays this section already
# produces: the (N_f, N_v) error matrix, the per-frame body lengths and the resolved
# vmin/vmax. They are cached in _MPVE_HEATMAP_CACHE and written to the .npz, so the
# visualisation never re-evaluates a mesh.

_MPVE_HEATMAP_CACHE = {
    "key": None,          # (gt collection, gt object, reconstruction object)
    "err": None,          # (N_f, N_v) float32, metres, world space
    "frames": None,       # (N_f,) int32
    "body_length": None,  # (N_f,) float32, GT
    "scale": None,        # dict from _mpve_resolve_scale
    "npz_path": None,
}


def _mpve_cache_key(p, ctx):
    return (p.collection_name, p.object_name, ctx["rec_obj"].name)


def _mpve_cache_store(key, err, frames, body_length, scale, npz_path):
    _MPVE_HEATMAP_CACHE.update({"key": key, "err": err, "frames": frames,
                                "body_length": body_length, "scale": scale,
                                "npz_path": npz_path})


def _mpve_cache_clear():
    for k in _MPVE_HEATMAP_CACHE:
        _MPVE_HEATMAP_CACHE[k] = None


def _mpve_units_array(err, body_length, units):
    """(N_f, N_v) error in the requested display unit; body lengths divide by GT L_body."""
    if units != 'body_lengths':
        return err
    bl = np.asarray(body_length, dtype=np.float64).reshape(-1, 1)
    bl = np.where(np.isfinite(bl) & (bl > _RECON_EPS), bl, np.nan)
    return (err / bl).astype(np.float32)


def _mpve_resolve_scale(err_units, mode, fixed_vmax, units):
    """vmin/vmax for the colour mapping, in `units`.

    `global_p95` (default) and `global_max` are one scale across all frames and are the
    only modes in which frames are visually comparable; `per_frame` is resolved by the
    consumer per row and is explicitly marked non-comparable.
    """
    a = err_units[np.isfinite(err_units)] if err_units.size else np.zeros(0)
    vmax, source = None, mode
    if mode == 'fixed':
        vmax = float(fixed_vmax)
        if not (math.isfinite(vmax) and vmax > 0.0):
            vmax, source = (float(np.percentile(a, _RECON_P)) if a.size else None), 'global_p95'
    elif mode == 'global_max':
        vmax = float(a.max()) if a.size else None
    elif mode == 'per_frame':
        vmax = float(a.max()) if a.size else None      # informational only for this mode
    else:
        vmax = float(np.percentile(a, _RECON_P)) if a.size else None
    return {
        "mode": mode,
        "vmax_source": source,
        "units": units,
        "vmin": 0.0,
        "vmax": _recon_num(vmax),
        "frames_comparable": mode in ('global_p95', 'global_max', 'fixed'),
        "note": ("per_frame normalisation makes an excellent frame look identical to a "
                 "catastrophic one; it is not comparable across frames"
                 if mode == 'per_frame' else None),
    }


def _mpve_build(res, p, ctx, lo, hi, opts):
    """(json dict, npz payload dict) for the MPVE metric."""
    frames = np.asarray(res["frames"], dtype=np.int32)
    rows = res["mpve"]["err"]
    E = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
    bl = np.asarray(res["body_length"], dtype=np.float32)
    entries = res["mpve"]["frames"]
    # res["blocked"] is parallel to res["frames"], which is parallel to mpve["frames"] --
    # all three are appended in the same commit step of _recon_eval_pass.
    for entry, reason in zip(entries, res.get("blocked", [])):
        entry["blocked"] = reason is not None
        entry["reason_blocked"] = reason

    units = opts.get("mpve_units", 'meters')
    E_u = _mpve_units_array(E, bl, units)
    scale = _mpve_resolve_scale(E_u, opts.get("mpve_normalization_mode", 'global_p95'),
                                opts.get("mpve_fixed_vmax", 0.0), units)
    vmax = scale["vmax"]
    if E_u.size and vmax:
        clipped = (E_u > vmax).mean(axis=1)
    else:
        clipped = np.full(len(entries), np.nan)
    for i, e in enumerate(entries):
        e["heatmap_clipped_fraction"] = _recon_num(clipped[i]) if i < len(clipped) else None

    with np.errstate(invalid='ignore'):
        pv_mean = E.mean(axis=0) if E.size else np.zeros(0, dtype=np.float32)
        pv_p95 = (np.percentile(E, _RECON_P, axis=0) if E.size else np.zeros(0, dtype=np.float32))

    def series(term, key="mean"):
        return [e["variants"][term].get(key) for e in entries]

    summary = {
        "global": _recon_sequence_summary(series("global"), res["frames"], pooled=E),
        "root_relative": _recon_sequence_summary(series("root_relative"), res["frames"]),
        "root_relative_centroid": _recon_sequence_summary(series("root_relative_centroid"),
                                                          res["frames"]),
        "body_length_normalised": _recon_sequence_summary(series("body_length_normalised"),
                                                          res["frames"]),
        "per_vertex": {
            "n_vertices": int(pv_mean.size),
            "mean_of_per_vertex_means": _recon_num(pv_mean.mean() if pv_mean.size else None),
            "median_of_per_vertex_means": _recon_num(np.median(pv_mean) if pv_mean.size else None),
            "max_per_vertex_mean": _recon_num(pv_mean.max() if pv_mean.size else None),
            "argmax_vertex": int(np.argmax(pv_mean)) if pv_mean.size else None,
            "max_per_vertex_p95": _recon_num(pv_p95.max() if pv_p95.size else None),
            "arrays": "see the sibling .npz (per_vertex_mean, per_vertex_p95)",
        },
        "outlier_rate": {
            "tau_body_lengths": opts.get("pck_tau_body_lengths"),
            "pck_mean_over_frames": _recon_num(np.nanmean([
                e["variants"]["global"].get("pck") for e in entries
                if e["variants"]["global"].get("pck") is not None]) if entries else None),
            "definition": "fraction of vertices within tau = tau_body_lengths * L_body(f)",
        },
        "area_weighted": {
            "enabled": bool(res["mpve"]["area_weighted"]),
            "mean_of_frame_means": _recon_stats(
                [e.get("area_weighted_mean") for e in entries])["mean"],
            "definition": "vertex weight = 1/3 * sum(area of incident faces), GT mesh, per frame",
            "tessellation": res["mpve"]["tessellation"],
        },
        "coverage": _recon_coverage(res["requested"], res["frames"], res["skipped"],
                                     res.get("blocked_records", ())),
        "heatmap_scale": scale,
    }
    meta = _recon_meta_base(
        MPVE_SCHEMA, "mean_per_vertex_error_world_space_l2", p, ctx, lo, hi,
        aggregation_order=("vertex -> frame -> sequence: the per-frame value is the mean "
                           "over vertices; the sequence value is the mean AND the median of "
                           "those per-frame means. pooled_* are over all (frame, vertex) "
                           "samples and are reported separately"),
        extra={
            "n_vertices": res["mpve"]["n_verts"],
            "variants": {
                "global": "raw world-space difference",
                "root_relative": (f"each mesh's own {res['mpve']['root_reference']} subtracted "
                                  f"(primary definition)"),
                "root_relative_centroid": "each mesh's own vertex centroid subtracted (secondary)",
                "body_length_normalised": "global error divided by the per-frame GT L_body",
            },
            "per_vertex_arrays": os.path.basename(opts.get("npz_path", "")) or None,
            "references": ["Pavlakos et al., CVPR 2018", "Kolotouros et al., SPIN, ICCV 2019",
                           "Zuffi et al., SMAL, CVPR 2017"],
            "warnings": res["warnings"],
        })
    data = {"meta": meta, "summary": summary, "frames": entries}
    npz = {"err": E, "frames": frames, "body_length": bl,
           "per_vertex_mean": pv_mean.astype(np.float32),
           "per_vertex_p95": np.asarray(pv_p95, dtype=np.float32),
           "vmax": np.asarray([vmax if vmax is not None else np.nan], dtype=np.float32),
           "units": np.asarray([units])}
    return data, npz, scale


# --- MPJPE -------------------------------------------------------------------

def _mpjpe_block(entries, names, frames, tau_bl):
    """One of the two MPJPE blocks (keypoint centroids / bone heads)."""
    out = {"names": list(names), "n_items": len(names)}
    for term in ("global", "root_relative", "pa"):
        pooled = (np.concatenate([e["_err"][term] for e in entries]) if entries else None)
        per_item = {}
        for i, name in enumerate(names):
            vals = [e["_err"][term][i] for e in entries]
            st = _recon_stats(vals, labels=frames, label_key="argmax_frame")
            st["n_valid_frames"] = st.pop("n")
            per_item[name] = st
        block = {
            "overall": _recon_sequence_summary([e[term]["mean"] for e in entries],
                                               frames, pooled=pooled),
            "per_item": per_item,
            "pck_mean_over_frames": _recon_num(np.nanmean(
                [e[term]["pck"] for e in entries if e[term]["pck"] is not None])
                if any(e[term]["pck"] is not None for e in entries) else None),
        }
        if term == "pa":
            deg = [e["frame"] for e in entries if e["pa_degenerate"]]
            block["degenerate_frames"] = deg
            block["n_degenerate_frames"] = len(deg)
            block["degenerate_reasons"] = sorted({str(e["pa_reason"]) for e in entries
                                                  if e["pa_degenerate"]})
        out[term] = block
    out["decomposition"] = {
        "root_localisation_error": _recon_sequence_summary(
            [e["decomposition"]["root_localisation_error"] for e in entries], frames),
        "orientation_and_scale_error": _recon_sequence_summary(
            [e["decomposition"]["orientation_and_scale_error"] for e in entries], frames),
        "definition": ("global - root_relative ~ root localisation (MRPE, Moon et al. 2019); "
                       "root_relative - pa ~ global orientation + scale error"),
    }
    out["pck"] = {"tau_body_lengths": tau_bl,
                  "definition": "fraction of items within tau = tau_body_lengths * L_body(f)"}
    out["frames"] = [{k: v for k, v in e.items() if k != "_err"} for e in entries]
    return out


def _mpjpe_build(res, p, ctx, lo, hi, opts):
    frames = res["frames"]
    tau_bl = opts.get("pck_tau_body_lengths")
    data = {"meta": None, "summary": {}, "keypoint": None, "joint": None}
    kpt_entries = res["mpjpe"]["keypoint"]
    joint_entries = res["mpjpe"]["joint"]
    if kpt_entries:
        data["keypoint"] = _mpjpe_block(kpt_entries, res["mpjpe"]["kpt_names"],
                                        [e["frame"] for e in kpt_entries], tau_bl)
    if joint_entries:
        data["joint"] = _mpjpe_block(joint_entries, res["mpjpe"]["joint_names"],
                                     [e["frame"] for e in joint_entries], tau_bl)
    else:
        data["joint"] = {"reason": "no reconstruction armature; joint block not computed"}
    data["summary"] = {
        "keypoint": {t: (data["keypoint"][t]["overall"] if data["keypoint"] else None)
                     for t in ("global", "root_relative", "pa")},
        "joint": {t: (data["joint"][t]["overall"] if joint_entries else None)
                  for t in ("global", "root_relative", "pa")},
        "coverage": _recon_coverage(res["requested"], frames, res["skipped"],
                                     res.get("blocked_records", ())),
    }
    data["meta"] = _recon_meta_base(
        MPJPE_SCHEMA, "mean_per_joint_position_error_world_space_l2", p, ctx, lo, hi,
        aggregation_order=("item -> frame -> sequence: the per-frame value is the mean over "
                           "the K valid items; the sequence value is the mean AND median of "
                           "those per-frame means; pooled_* are over all (frame, item) samples"),
        extra={
            "terms": {
                "global": "no alignment removed; isolates global localisation error (theta)",
                "root_relative": "translation removed; isolates global orientation + articulation",
                "pa": "similarity transform removed (Kabsch/Umeyama); residual articulation/shape",
            },
            "pa_scale": "UNIFORM scale is included in the alignment (Umeyama); the scale-free "
                        "variant is a different metric reported under the same name",
            "pa_degeneracy": {
                "test": "sigma_2 / sigma_0 of the cross-covariance",
                "threshold": opts.get("pa_sigma_min"),
                "action": "pa is null with pa_degenerate: true, never a meaningless number",
                "why": "K ~ 6-10 points on a frequently near-straight fish is close to a 1-D "
                       "configuration, where the rotation is unstable",
            },
            "root_relative_reference": {
                "keypoint": "centroid of each set's valid keypoints",
                "joint": "each armature's own root bone head",
            },
            "joint_set": {
                "bones": res["mpjpe"]["joint_names"],
                "excluded_virtual_bones": res["bone"]["virtual"],
                "source": "P[b][:3,3] of _pts_posed_armature_space, mapped through each "
                          "armature's own matrix_world",
            },
            "references": ["Ionescu et al., Human3.6M, TPAMI 2014", "Kabsch 1976 / Gower 1975",
                           "Umeyama, TPAMI 1991", "Kanazawa et al., CVPR 2018",
                           "Moon et al., RootNet, ICCV 2019"],
            "warnings": res["warnings"],
        })
    return data


# --- per-bone SO(3) ----------------------------------------------------------

def _bone_rot_build(res, p, ctx, lo, hi, opts):
    b = res["bone"]
    frames = res["frames"]
    names = b["names"]
    virtual = set(b["virtual"])
    G = np.degrees(np.stack(b["global"])) if b["global"] else np.zeros((0, len(names)))
    L = np.degrees(np.stack(b["local"])) if b["local"] else np.zeros((0, len(names)))
    have_st = bool(b["swing_twist"]) and bool(b["swing_global"])
    SG = np.degrees(np.stack(b["swing_global"])) if have_st else None
    TG = np.degrees(np.stack(b["twist_global"])) if have_st else None
    SL = np.degrees(np.stack(b["swing_local"])) if have_st else None
    TL = np.degrees(np.stack(b["twist_local"])) if have_st else None
    SAT = np.stack(b["prior_saturated"]) if (have_st and b["prior_saturated"]) else None

    per_bone = {}
    for i, name in enumerate(names):
        item = {
            "is_virtual": name in virtual,
            "groups": list(b["membership"].get(name, [])),
            "global": _recon_stats(G[:, i], labels=frames, label_key="argmax_frame"),
            "local": _recon_stats(L[:, i], labels=frames, label_key="argmax_frame"),
        }
        for _k in ("global", "local"):
            item[_k]["n_valid_frames"] = item[_k].pop("n")
        if have_st:
            item["swing_global"] = _recon_stats(SG[:, i], labels=frames, label_key="argmax_frame")
            item["twist_global"] = _recon_stats(np.abs(TG[:, i]), labels=frames,
                                                label_key="argmax_frame")
            item["swing_local"] = _recon_stats(SL[:, i], labels=frames, label_key="argmax_frame")
            item["twist_local"] = _recon_stats(np.abs(TL[:, i]), labels=frames,
                                               label_key="argmax_frame")
        if SAT is not None:
            n_sat = int(SAT[:, i].sum())
            item["prior_saturated_frames"] = n_sat
            item["prior_saturated_fraction"] = float(n_sat) / SAT.shape[0] if SAT.shape[0] else None
            item["prior_note"] = ("errors on frames where the GT bone sits on its swing-twist "
                                  "limit are structurally floored by the prior and should not be "
                                  "pooled naively" if n_sat else None)
        per_bone[name] = item

    def group_summary(bones, key):
        """Mean of the per-bone means (NOT a flat pool, which weights big groups by count)."""
        vals = [per_bone[x][key]["mean"] for x in bones]
        st = _recon_stats(vals, labels=list(bones), label_key="argmax_bone")
        idx = [names.index(x) for x in bones]
        arr = (G if key == "global" else L)[:, idx] if idx else np.zeros(0)
        flat = arr[np.isfinite(arr)] if arr.size else arr
        st["pooled_mean"] = _recon_num(flat.mean() if flat.size else None)
        st["pooled_median"] = _recon_num(np.median(flat) if flat.size else None)
        st["n_bones_with_data"] = st.pop("n")
        st["n_bones"] = len(bones)
        return st

    per_group = {}
    for g, bl_names in b["groups"].items():
        per_group[g] = {"bone_names": list(bl_names), "n_bones": len(bl_names),
                        "global": group_summary(bl_names, "global"),
                        "local": group_summary(bl_names, "local")}

    frame_entries = []
    for fi, f in enumerate(frames):
        entry = {"frame": int(f),
                 "global": _recon_stats(G[fi], labels=names, label_key="argmax_bone"),
                 "local": _recon_stats(L[fi], labels=names, label_key="argmax_bone"),
                 "per_bone": {n: {"global": _recon_num(G[fi, i]), "local": _recon_num(L[fi, i])}
                              for i, n in enumerate(names)}}
        if res["arm_rotation_offset_deg"]:
            entry["armature_world_rotation_offset_deg"] = _recon_num(
                res["arm_rotation_offset_deg"][fi])
        frame_entries.append(entry)

    real = [n for n in names if n not in virtual]
    summary = {
        "global": group_summary(real, "global") if real else None,
        "local": group_summary(real, "local") if real else None,
        "per_group": {g: {"global": per_group[g]["global"]["mean"],
                          "local": per_group[g]["local"]["mean"]} for g in per_group},
        "worst_bone_global": max(real, key=lambda n: (per_bone[n]["global"]["mean"] or -1.0),
                                 default=None),
        "coverage": _recon_coverage(res["requested"], frames, res["skipped"],
                                     res.get("blocked_records", ())),
        "roundtrip": res["roundtrip"],
        "armature_world_rotation_offset_deg": _recon_stats(
            res["arm_rotation_offset_deg"], labels=frames, label_key="argmax_frame"),
    }
    meta = _recon_meta_base(
        BONE_ROTATION_ERROR_SCHEMA, "per_bone_geodesic_so3_error", p, ctx, lo, hi,
        aggregation_order=("bone -> frame -> group: a bone's value is aggregated over frames "
                           "first, and a group's value is the mean of its per-bone means, not a "
                           "flat pool (which would weight large groups by bone count). "
                           "pooled_* are the flat pool, reported separately"),
        extra={
            "units": "degrees",
            "definition": "theta = arccos(clip((tr(R^T R^) - 1)/2, -1, 1)) = || log(R^T R^) ||",
            "variants": {
                "global": "armature-space bone frames, _rot3(P[b]) of _pts_posed_armature_space",
                "local": "parent-relative delta-from-rest, D(parent)^-1 D(b) -- the quantity the "
                         "optimiser parameterises as body_pose. Its geodesic angle is invariant "
                         "to the rest-frame conjugation, so it equals the angle of the same "
                         "quantity expressed in the bone's own rest space",
            },
            "orthonormalisation": ("P[b] carries the reconstruction armature's scale S; every "
                                   "matrix is passed through _rot3 (to_quaternion().to_matrix()) "
                                   "and verified against |R^T R - I| < "
                                   f"{_RECON_ORTHO_TOL:g} before the trace"),
            "rest_pose": "the GT template's rest_R is used for BOTH armatures",
            "root_bone": "the root has no parent, so its local error equals its global error",
            "bone_groups": {"source": "get_mesh_json()['bone_groups']",
                            "overlapping": True,
                            "ungrouped_bucket": _RECON_UNGROUPED,
                            "virtual_bucket": _RECON_VIRTUAL_GROUP,
                            "virtual_excluded_from_summary": True},
            "swing_twist": {
                "enabled": bool(have_st),
                "axis": "bone local +Y (Blender's bone axis, the optimiser's twist axis)",
                "note": "twist about the body axis is far less observable from silhouettes than "
                        "swing, so the split shows which DoF the multi-view rig constrains",
                "prior_saturation_tolerance": opts.get("prior_saturation_tol"),
            },
            "references": ["Huynh, Metrics for 3D rotations, JMIV 2009",
                           "Mahmood et al., AMASS, ICCV 2019",
                           "Zuffi et al., SMALR, CVPR 2018"],
            "warnings": res["warnings"],
        })
    return {"meta": meta, "summary": summary, "per_bone": per_bone,
            "per_group": per_group, "frames": frame_entries}


# --- operators --------------------------------------------------------------

def _kpt_dist_report(ks):
    """One-line keypoint-distance report: body lengths first, metres in parentheses.

    Body lengths lead because they are the comparable number; the metre value stays visible
    so that a reader who knows the scene scale can sanity-check it. A run whose L_body was
    never measurable reports 'n/a BL' rather than a silently absolute number.
    """
    mean_bl, max_bl = ks.get("overall_mean_bl"), ks.get("overall_max_bl")
    if mean_bl is None:
        head = (f"kpt dist mean n/a BL, max n/a BL (L_body unavailable on every frame; "
                f"source {ks.get('body_length', {}).get('sources')})")
    else:
        head = (f"kpt dist mean {mean_bl:.4f} BL, max {max_bl:.4f} BL @ frame "
                f"{ks.get('overall_max_bl_frame')} ('{ks.get('overall_max_bl_keypoint')}')")
    mean_m, max_m = ks.get("overall_mean"), ks.get("overall_max")
    if mean_m is not None:
        head += f" [{mean_m:.5f} m / {max_m:.5f} m]"
    return head


class SYNTH_OT_compute_volumetric_iou(Operator):
    """Per-frame volumetric 3D IoU between the GT mesh and its reconstruction"""
    bl_idname = "synth.compute_volumetric_iou"
    bl_label = "Compute Volumetric 3D IoU"
    bl_description = ("Monte-Carlo occupancy IoU, Vol(GT n R) / Vol(GT u R), between the target "
                      "mesh and its counterpart in 'Reconstructions', for every frame in the "
                      "scene range. Optionally computes the per-keypoint 3D distances in the "
                      "same pass. Writes JSON next to the pose time series export")

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props

        try:
            ctx = _recon_pair_context(context, self.report)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        src_obj, rec_obj = ctx["src_obj"], ctx["rec_obj"]
        lo, hi = ctx["frame_lo"], ctx["frame_hi"]
        with_kpts = bool(p.iou_with_keypoint_distances)
        kpt_list = ctx["kpt_list"] if with_kpts else []
        if with_kpts and not kpt_list:
            with_kpts = False
            self.report({'WARNING'}, "'Keypoint List' is empty; computing the IoU only.")
        if with_kpts:
            try:
                empty = _recon_check_keypoint_correspondence(src_obj, rec_obj, kpt_list)
            except Exception as exc:
                self.report({'ERROR'}, f"Keypoint correspondence check failed: {exc}")
                return {'CANCELLED'}
            if empty:
                self.report({'WARNING'}, f"keypoint vertex groups with no members: {empty}")

        n_samples, seed = int(p.iou_sample_count), int(p.iou_random_seed)
        out_dir = resolve(p.annot_out_dir)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not create {out_dir}: {exc}")
            return {'CANCELLED'}
        iou_path = os.path.join(out_dir,
                                f"volumetric_iou_{p.collection_name}_{p.object_name}.json")
        kpt_path = os.path.join(out_dir,
                                f"keypoint_distances_{p.collection_name}_{p.object_name}.json")

        data = {"meta": _iou_meta(p, ctx, lo, hi, n_samples, seed), "frames": []}

        try:
            res_pass = _iou_kpt_eval_pass(context, ctx, kpt_list, True, n_samples, seed,
                                          self.report)
        except Exception as exc:
            self.report({'ERROR'}, f"Computation failed at frame {scene.frame_current}: {exc}")
            return {'CANCELLED'}
        data["frames"] = res_pass["iou_frames"]
        kpt_records, per_kpt, per_kpt_bl, missing = (
            res_pass["kpt_records"], res_pass["per_kpt"], res_pass["per_kpt_bl"],
            res_pass["missing"])

        summary = _iou_summary(data["frames"], res_pass["iou_skipped"],
                               res_pass["blocked_records"])
        if summary is None:
            self.report({'ERROR'}, "No frame produced a valid IoU.")
            return {'CANCELLED'}
        data["summary"] = summary

        try:
            with open(iou_path, 'w') as jf:
                json.dump(data, jf, indent=2)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not write {iou_path}: {exc}")
            return {'CANCELLED'}

        s = data["summary"]
        msg = (f"3D IoU vs '{rec_obj.name}': mean {s['mean_iou']:.4f}, "
               f"min {s['min_iou']:.4f} @ {s['min_iou_frame']}, "
               f"max {s['max_iou']:.4f} @ {s['max_iou_frame']} "
               f"({s['n_valid']}/{s['n_frames']} frames, {n_samples} samples)")
        level = 'INFO'
        if with_kpts and kpt_records:
            try:
                ks = _kpt_dist_finalize(kpt_records, per_kpt, per_kpt_bl, missing,
                                        _kpt_dist_meta(p, ctx, lo, hi), kpt_path,
                                        res_pass["blocked_records"])
            except Exception as exc:
                self.report({'WARNING'}, f"IoU written, but keypoint distances failed: {exc}")
            else:
                msg += "; " + _kpt_dist_report(ks)
                thr = float(p.kpt_dist_warn_threshold_bl)
                if thr > 0.0 and (ks.get("overall_max_bl") or 0.0) > thr:
                    level = 'WARNING'
        self.report({level}, msg + f" -> {out_dir}")
        return {'FINISHED'}


class SYNTH_OT_compute_keypoint_distances(Operator):
    """Per-frame, per-keypoint 3D distance between GT and reconstruction"""
    bl_idname = "synth.compute_keypoint_distances"
    bl_label = "Compute Keypoint 3D Distances"
    bl_description = ("For every frame, the L2 distance between each keypoint's world-space "
                      "vertex-group centroid on the target mesh and on its counterpart in "
                      "'Reconstructions'. Writes keypoint_distances_<collection>_<object>.json")

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props

        try:
            ctx = _recon_pair_context(context, self.report)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        kpt_list = ctx["kpt_list"]
        if not kpt_list:
            self.report({'ERROR'}, "'Keypoint List' is empty; nothing to measure.")
            return {'CANCELLED'}

        src_obj, rec_obj = ctx["src_obj"], ctx["rec_obj"]
        try:
            empty = _recon_check_keypoint_correspondence(src_obj, rec_obj, kpt_list)
        except Exception as exc:
            self.report({'ERROR'}, f"Keypoint correspondence check failed: {exc}")
            return {'CANCELLED'}
        if empty:
            self.report({'WARNING'}, f"keypoint vertex groups with no members: {empty}")

        out_dir = resolve(p.annot_out_dir)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not create {out_dir}: {exc}")
            return {'CANCELLED'}
        out_path = os.path.join(out_dir,
                                f"keypoint_distances_{p.collection_name}_{p.object_name}.json")

        lo, hi = ctx["frame_lo"], ctx["frame_hi"]
        try:
            # need_bvh=False: no tree build, no occupancy sampling -- this pass is ~2
            # to_mesh() calls per frame and nothing else.
            res_pass = _iou_kpt_eval_pass(context, ctx, kpt_list, False, 0, 0, self.report)
        except Exception as exc:
            self.report({'ERROR'}, f"Keypoint distances failed at frame "
                                   f"{scene.frame_current}: {exc}")
            return {'CANCELLED'}
        records, per_kpt, per_kpt_bl, missing = (
            res_pass["kpt_records"], res_pass["per_kpt"], res_pass["per_kpt_bl"],
            res_pass["missing"])

        if not per_kpt:
            self.report({'ERROR'}, "No keypoint could be measured on any frame.")
            return {'CANCELLED'}

        try:
            summary = _kpt_dist_finalize(records, per_kpt, per_kpt_bl, missing,
                                         _kpt_dist_meta(p, ctx, lo, hi), out_path,
                                         res_pass["blocked_records"])
        except Exception as exc:
            self.report({'ERROR'}, f"Could not write {out_path}: {exc}")
            return {'CANCELLED'}

        thr = float(p.kpt_dist_warn_threshold_bl)
        level = 'WARNING' if (thr > 0.0 and (summary.get("overall_max_bl") or 0.0) > thr) \
            else 'INFO'
        bl = summary["body_length"]
        if bl["n_frames_unavailable"]:
            self.report({'WARNING'},
                        f"L_body unmeasurable on {bl['n_frames_unavailable']} of "
                        f"{summary['n_frames']} frame(s); those frames have no body-length "
                        f"value. Sources: {bl['sources']}")
        self.report({level},
                    f"vs '{rec_obj.name}': {_kpt_dist_report(summary)} over "
                    f"{summary['n_frames']} frames, {summary['n_keypoints']} keypoints, "
                    f"L_body median {_recon_fmt(bl['median'], ' m')} ({bl['sources']}) "
                    f"-> {out_path}")
        return {'FINISHED'}


# --- batch: one pts2 file per view combination ------------------------------
#
# sweep_view_combinations.py's collect_results() writes, per view combination `leaf`:
#   <out_root>/metrics_collected/metrics_{leaf}.json   (+ collected_metrics.json)
#   <out_root>/pts2_collected/pts2_{leaf}.json
# The metrics files hold the 2D, image-space scores; the 3D scores need Blender, because
# they need the GT mesh the sweep never sees. This operator closes that gap: point it at
# pts2_collected/, and it produces <out_root>/collected_3d_metrics.json keyed by exactly the
# same `leaf` run keys, so analyze_metrics.py's RUN_KEY_PATTERN and grouping apply unchanged.

# Batch output file name; sibling of pts2_collected/, mirroring collected_metrics.json.
COLLECTED_3D_METRICS_NAME = "collected_3d_metrics.json"


def _pts2_run_key(filename):
    """'pts2_k4__v0-1-2-5__Top_L....json' -> 'k4__v0-1-2-5__Top_L...'.

    collect_results() names the copies 'pts2_{leaf}.json' where leaf == combo_folder_name(),
    so stripping the prefix and the extension recovers the key collected_metrics.json is
    keyed by and analyze_metrics.RUN_KEY_PATTERN ('^k(\\d+)__') parses the view count from.
    Both affixes are stripped defensively: a file that was renamed by hand still yields a
    usable key rather than an exception.
    """
    stem = os.path.basename(filename)
    if stem.lower().endswith(".json"):
        stem = stem[:-len(".json")]
    if stem.startswith("pts2_"):
        stem = stem[len("pts2_"):]
    return stem


def _recon_collection_object_names():
    """Names currently in 'Reconstructions', or an empty set if it does not exist yet."""
    col = bpy.data.collections.get("Reconstructions")
    return {ob.name for ob in col.objects} if col else set()


def _recon_remove_if_orphan(datablocks, block):
    """Delete `block` from `datablocks` once nothing references it any more.

    A fake user is dropped first: bpy.data.actions.new() leaves use_fake_user set in some
    Blender builds, which would keep every imported action alive for the whole batch and
    defeat the point of purging.
    """
    if block is None:
        return
    try:
        if getattr(block, "use_fake_user", False):
            block.use_fake_user = False
        if block.users == 0:
            datablocks.remove(block)
    except (ReferenceError, RuntimeError):
        pass


def _recon_purge_imported(context, before_names):
    """Remove everything create_animation_from_pose_time_series added to 'Reconstructions'.

    Diffing against a name snapshot taken before the import, rather than deleting the two
    returned objects, also cleans up after a PARTIAL import: that function links the mesh and
    the armature before it keyframes them, so an exception thrown half-way through would
    otherwise leak both objects into the next iteration -- where _iou_find_reconstruction's
    'highest .NNN suffix wins' rule would happily pick the wreckage.

    Object data is only removed once the objects that used it are gone and its user count has
    actually dropped to zero, so a mesh or armature shared with the GT (it never is, both are
    .copy()s, but the check costs nothing) survives.
    """
    col = bpy.data.collections.get("Reconstructions")
    if col is None:
        return 0
    victims = [ob for ob in col.objects if ob.name not in before_names]
    if not victims:
        return 0

    actions, meshes, armatures = [], [], []
    for ob in victims:
        ad = getattr(ob, "animation_data", None)
        if ad is not None and ad.action is not None:
            actions.append(ad.action)
        data = ob.data
        if isinstance(data, bpy.types.Mesh):
            meshes.append(data)
        elif isinstance(data, bpy.types.Armature):
            armatures.append(data)
        try:
            ob.animation_data_clear()      # drops this object's user of the action
        except Exception:
            pass

    n = 0
    for ob in victims:
        try:
            bpy.data.objects.remove(ob, do_unlink=True)
            n += 1
        except (ReferenceError, RuntimeError):
            pass
    for act in actions:
        _recon_remove_if_orphan(bpy.data.actions, act)
    for mesh in meshes:
        _recon_remove_if_orphan(bpy.data.meshes, mesh)
    for arm in armatures:
        _recon_remove_if_orphan(bpy.data.armatures, arm)

    try:
        context.view_layer.update()
    except Exception:
        pass
    return n


def _batch_3d_metrics_one(context, rec_obj, n_samples, seed, with_kpts, report=None):
    """Score ONE already-imported reconstruction against the GT.

    The batch output keeps the existing per-metric ``{meta, summary, frames}``
    block structure used by ``collected_3d_metrics.json``. In addition to the
    volumetric/keypoint pass, MPVE and MPJPE are computed through the shared
    ``_recon_eval_pass`` so the batch evaluator uses exactly the same metric
    definitions as the standalone operators.
    """
    p = context.scene.synth_props
    ctx = _recon_pair_context(context, report, rec_obj=rec_obj)
    lo, hi = int(ctx["frame_lo"]), int(ctx["frame_hi"])

    # ``with_kpts`` controls only the existing IoU-side keypoint-distance block.
    # MPJPE has its own use of the configured keypoint list and is always included
    # in the batch output when those keypoints exist.
    kpt_list = ctx["kpt_list"] if with_kpts else []
    if kpt_list:
        # raises on a vertex-count or vertex-group mismatch -> this file is skipped
        _recon_check_keypoint_correspondence(ctx["src_obj"], ctx["rec_obj"], kpt_list)

    # Existing 3D IoU + optional keypoint-distance batch pass.
    res_iou = _iou_kpt_eval_pass(context, ctx, kpt_list, True, n_samples, seed, report)

    summary = _iou_summary(res_iou["iou_frames"], res_iou["iou_skipped"],
                           res_iou["blocked_records"])
    if summary is None:
        raise ValueError("no frame produced a valid IoU (degenerate or zero-volume throughout)")

    entry = {
        "volumetric_iou": {
            "meta": _iou_meta(p, ctx, lo, hi, n_samples, seed),
            "summary": summary,
            "frames": res_iou["iou_frames"],
        },
    }
    if kpt_list and res_iou["per_kpt"]:
        entry["keypoint_distances"] = {
            "meta": _kpt_dist_meta(p, ctx, lo, hi),
            "summary": _kpt_dist_summarize(res_iou["kpt_records"], res_iou["per_kpt"],
                                           res_iou["per_kpt_bl"], res_iou["missing"],
                                           res_iou["blocked_records"]),
            "frames": res_iou["kpt_records"],
        }

    # ------------------------------------------------------------------
    # MPVE + MPJPE
    # ------------------------------------------------------------------
    # These metrics do not need the SO(3) round-trip precondition. Disable
    # round-trip loading here even if the corresponding UI property is set,
    # because the batch feature is deliberately limited to MPVE/MPJPE.
    metric_want = {"mpve", "mpjpe"}
    opts = _recon_metric_opts(p, metric_want)
    opts["roundtrip_json"] = ""
    opts["roundtrip_required"] = False
    # No per-vertex NPZ is written by the batch collector. _mpve_build only
    # records the basename in meta, so keep that reference empty rather than
    # implying that a sibling NPZ exists.
    opts["npz_path"] = ""

    # Use the configured keypoint list for MPJPE itself, independently of the
    # IoU keypoint-distance toggle.
    mp_ctx = dict(ctx)
    mp_ctx["kpt_list"] = list(ctx["kpt_list"])

    res_metrics = _recon_eval_pass(context, mp_ctx, metric_want, opts, report)

    mpve_data, _mpve_npz, _mpve_scale = _mpve_build(
        res_metrics, p, mp_ctx, lo, hi, opts
    )
    mpjpe_data = _mpjpe_build(res_metrics, p, mp_ctx, lo, hi, opts)

    # _mpjpe_build stores the raw metric in metres. For the common collector
    # schema, add the same GT-body-length normalisation used by MPVE so the
    # analyzer can consume MPJPE as a comparable 3D length metric.
    body_lengths = {
        int(frame): length
        for frame, length in zip(res_metrics["frames"], res_metrics["body_length"])
    }
    blocked_by_frame = {
        int(frame): reason
        for frame, reason in zip(res_metrics["frames"], res_metrics["blocked"])
        if reason is not None
    }
    for block_name in ("keypoint", "joint"):
        block = mpjpe_data.get(block_name)
        if not isinstance(block, dict) or not isinstance(block.get("frames"), list):
            continue
        for frame_entry in block["frames"]:
            frame = int(frame_entry["frame"])
            l_body = body_lengths.get(frame)
            frame_entry["body_length"] = _recon_num(l_body)
            frame_entry["blocked"] = frame in blocked_by_frame
            frame_entry["reason_blocked"] = blocked_by_frame.get(frame)
            frame_entry["body_length_normalised"] = {}
            for term in ("global", "root_relative", "pa"):
                stats = frame_entry.get(term)
                mean_m = stats.get("mean") if isinstance(stats, dict) else None
                frame_entry["body_length_normalised"][term] = _bl_norm(mean_m, l_body)

    entry["mpve"] = mpve_data
    entry["mpjpe"] = mpjpe_data
    # The run-level copy of the pipeline's blocked list, carried through unchanged from the
    # pts2 meta. Run-level because it is a property of the reconstruction, not of any one
    # metric, so analyze_metrics.py reads it once here instead of picking a block and hoping
    # it was present -- while each metric block still keeps its own copy and each frame row
    # its own flag, so no block depends on this one to be interpretable.
    entry["blocked_frames"] = {
        "records": [dict(r) for r in (ctx.get("blocked_records") or [])],
        "stamp_present": bool(ctx.get("blocked_known")),
        "frame_start": int(lo),
        "frame_end": int(hi),
        "policy": ("every frame in [frame_start, frame_end] was measured; these are the ones "
                   "whose pose the reconstruction pipeline did not obtain by fitting the "
                   "optimizer to that frame"),
    }
    return entry


class SYNTH_OT_batch_3d_metrics(Operator):
    """3D IoU + keypoint distances for every pose time series in a directory"""
    bl_idname = "synth.batch_3d_metrics"
    bl_label = "Batch 3D Metrics from PTS2 Dir"
    bl_description = ("For every pose_time_series/2 JSON in 'PTS2 Batch Dir': rebuild the "
                      "reconstruction, compute volumetric IoU/keypoint distances plus MPVE/MPJPE "
                      "with the same metric implementations as the single-run operators, then "
                      "delete it again. Writes collected_3d_metrics.json one level above the "
                      "directory, keyed by view combination exactly like collected_metrics.json")

    def execute(self, context):
        scene = context.scene
        p = scene.synth_props

        batch_dir = resolve(p.pts2_batch_dir)
        if not batch_dir or not os.path.isdir(batch_dir):
            self.report({'ERROR'}, f"'PTS2 Batch Dir' is not a directory: {batch_dir}")
            return {'CANCELLED'}
        # sorted() so the run order (and therefore the report) is reproducible
        files = sorted(glob.glob(os.path.join(batch_dir, "*.json")))
        if not files:
            self.report({'ERROR'}, f"No *.json in {batch_dir}")
            return {'CANCELLED'}

        # sibling of the directory, i.e. <out_root>/collected_3d_metrics.json when batch_dir is
        # <out_root>/pts2_collected -- next to metrics_collected/collected_metrics.json's root
        out_path = os.path.normpath(os.path.join(batch_dir, os.pardir,
                                                 COLLECTED_3D_METRICS_NAME))

        # The GT side is resolved once, before the loop: an unusable target ('Object' not set,
        # no armature modifier, ...) is a fatal setup error, not a per-file one, and reporting
        # it once beats repeating it for every file in the directory.
        try:
            _pts_find_source(context)
        except Exception as exc:
            self.report({'ERROR'}, f"Ground truth object unavailable: {exc}")
            return {'CANCELLED'}

        n_samples, seed = int(p.iou_sample_count), int(p.iou_random_seed)
        with_kpts = bool(p.iou_with_keypoint_distances)
        if with_kpts and not _recon_kpt_list(context):
            with_kpts = False
            self.report({'WARNING'}, "'Keypoint List' is empty; computing the IoU only.")

        # create_animation_from_pose_time_series moves the scene range to each file's own
        # range; restore the user's range once the batch is done.
        saved_range = (int(scene.frame_start), int(scene.frame_end))
        collected, skipped = {}, []

        for path in files:
            name = os.path.basename(path)
            before = _recon_collection_object_names()
            try:
                # Cheap gate before the expensive import: a directory of pts2 files may also
                # hold the sweep's generated configs or a stray metrics copy.
                with open(path, 'r') as fp:
                    schema = json.load(fp).get("meta", {}).get("schema")
                if schema != POSE_TIME_SERIES_SCHEMA:
                    raise ValueError(f"schema is '{schema}', expected "
                                     f"'{POSE_TIME_SERIES_SCHEMA}'")

                _new_arm, new_obj = create_animation_from_pose_time_series(
                    context, path, report=self.report)
                entry = _batch_3d_metrics_one(context, new_obj, n_samples, seed, with_kpts,
                                              self.report)

                run_key = _pts2_run_key(name)
                if run_key in collected:
                    self.report({'WARNING'}, f"duplicate run key '{run_key}' from '{name}'; "
                                             f"the later file wins.")
                collected[run_key] = entry
            except Exception as exc:
                # One bad file must not cost the other N-1 runs: report and carry on.
                import traceback
                traceback.print_exc()
                self.report({'WARNING'}, f"Skipped '{name}': {exc}")
                skipped.append(name)
            finally:
                # unconditional: bounded memory across a directory of several hundred files,
                # and a clean 'Reconstructions' for the next iteration's auto-detection
                _recon_purge_imported(context, before)

        scene.frame_start, scene.frame_end = saved_range

        try:
            with open(out_path, 'w') as jf:
                json.dump(collected, jf, indent=2)
        except Exception as exc:
            self.report({'ERROR'}, f"Scored {len(collected)} run(s) but could not write "
                                   f"{out_path}: {exc}")
            return {'CANCELLED'}

        level = 'WARNING' if skipped and not collected else 'INFO'
        self.report({level}, f"Batch 3D metrics: {len(collected)} processed, {len(skipped)} "
                             f"skipped, of {len(files)} file(s) -> {out_path}")
        return {'FINISHED'}


# --- MPVE / MPJPE / SO(3) operators -----------------------------------------

def _recon_metric_opts(p, want):
    """Scene properties -> the options dict the metric pass reads."""
    return {
        "pa_sigma_min": float(p.mpjpe_pa_degeneracy_threshold),
        "pck_tau_body_lengths": float(p.pck_tau_body_lengths),
        "area_weighted": bool(p.mpve_area_weighted),
        "mpve_units": p.mpve_units,
        "mpve_normalization_mode": p.mpve_normalization_mode,
        "mpve_fixed_vmax": float(p.mpve_fixed_vmax),
        "swing_twist": bool(p.bone_err_swing_twist),
        "prior_saturation_tol": float(p.bone_err_prior_tolerance),
        "roundtrip_json": p.recon_roundtrip_json,
        "roundtrip_tol": float(p.bone_err_roundtrip_tol),
        # the round trip is a precondition of the ROTATION metric specifically: without it
        # that metric may be measuring a convention mismatch rather than reconstruction error
        "roundtrip_required": bool(p.bone_err_require_roundtrip) and ('bone_rot' in want),
    }


def _recon_fmt(x, suffix="", prec=5):
    return f"{x:.{prec}f}{suffix}" if isinstance(x, float) and math.isfinite(x) else "n/a"


def _recon_metrics_execute(op, context, want):
    """Shared body of the four metric operators: one pass, then the requested files.

    Keeping the operators thin means MPVE + MPJPE + the rotation error can be produced from
    a single frame loop (synth.compute_recon_metrics) at the cost of one.
    """
    scene = context.scene
    p = scene.synth_props

    try:
        ctx = _recon_pair_context(context, op.report)
    except Exception as exc:
        op.report({'ERROR'}, str(exc))
        return {'CANCELLED'}

    if 'mpjpe' in want and not ctx["kpt_list"]:
        op.report({'WARNING'}, "'Keypoint List' is empty; the MPJPE keypoint block will be "
                               "skipped and only bone heads measured.")

    out_dir = resolve(p.annot_out_dir)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as exc:
        op.report({'ERROR'}, f"Could not create {out_dir}: {exc}")
        return {'CANCELLED'}

    stem = f"{p.collection_name}_{p.object_name}"
    paths = {
        "mpve": os.path.join(out_dir, f"mpve_{stem}.json"),
        "mpve_npz": os.path.join(out_dir, f"mpve_per_vertex_{stem}.npz"),
        "mpjpe": os.path.join(out_dir, f"mpjpe_{stem}.json"),
        "bone_rot": os.path.join(out_dir, f"bone_rotation_error_{stem}.json"),
    }
    opts = _recon_metric_opts(p, want)
    opts["npz_path"] = paths["mpve_npz"]
    lo, hi = int(ctx["frame_lo"]), int(ctx["frame_hi"])

    try:
        res = _recon_eval_pass(context, ctx, want, opts, op.report)
    except _ReconFatal as exc:
        op.report({'ERROR'}, str(exc))
        return {'CANCELLED'}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        op.report({'ERROR'}, f"Metric pass failed at frame {scene.frame_current}: {exc}")
        return {'CANCELLED'}

    msgs, written = [], []
    try:
        if 'mpve' in want:
            data, npz, scale = _mpve_build(res, p, ctx, lo, hi, opts)
            with open(paths["mpve"], 'w') as jf:
                json.dump(data, jf, indent=2)
            np.savez_compressed(paths["mpve_npz"], **npz)
            # the heat map of Deliverable 4 reads this cache; it never re-evaluates a mesh
            _mpve_cache_store(_mpve_cache_key(p, ctx), npz["err"], npz["frames"],
                              npz["body_length"], scale, paths["mpve_npz"])
            written += [paths["mpve"], paths["mpve_npz"]]
            g = data["summary"]["global"]
            n = data["summary"]["body_length_normalised"]
            msgs.append(f"MPVE mean {_recon_fmt(g['mean_of_frame_means'], ' m')} / median "
                        f"{_recon_fmt(g['median_of_frame_means'], ' m')} / p95 "
                        f"{_recon_fmt(g['p95_of_frame_means'], ' m')} "
                        f"({_recon_fmt(n['mean_of_frame_means'], ' BL', 4)}), worst frame "
                        f"{g.get('argmax_frame')}")

        if 'mpjpe' in want:
            data = _mpjpe_build(res, p, ctx, lo, hi, opts)
            with open(paths["mpjpe"], 'w') as jf:
                json.dump(data, jf, indent=2)
            written.append(paths["mpjpe"])
            k = data["summary"]["keypoint"]
            if k and k["global"]:
                pa = k["pa"]["mean_of_frame_means"] if k["pa"] else None
                msgs.append(f"MPJPE(kpt) global {_recon_fmt(k['global']['mean_of_frame_means'], ' m')}"
                            f" / root-rel "
                            f"{_recon_fmt(k['root_relative']['mean_of_frame_means'], ' m')}"
                            f" / PA {_recon_fmt(pa, ' m')}")
            j = data["summary"]["joint"]
            if j and j.get("global"):
                msgs.append(f"MPJPE(joint) global "
                            f"{_recon_fmt(j['global']['mean_of_frame_means'], ' m')}")
            if data["keypoint"] and data["keypoint"]["pa"]["n_degenerate_frames"]:
                op.report({'WARNING'}, f"PA alignment was degenerate on "
                                       f"{data['keypoint']['pa']['n_degenerate_frames']} frame(s) "
                                       f"(near-1D keypoint configuration); those frames report "
                                       f"pa: null.")

        if 'bone_rot' in want:
            data = _bone_rot_build(res, p, ctx, lo, hi, opts)
            with open(paths["bone_rot"], 'w') as jf:
                json.dump(data, jf, indent=2)
            written.append(paths["bone_rot"])
            s = data["summary"]
            if s["global"] and s["local"]:
                msgs.append(f"SO(3) global {_recon_fmt(s['global']['mean'], ' deg', 3)} / local "
                            f"{_recon_fmt(s['local']['mean'], ' deg', 3)} (worst bone "
                            f"'{s.get('worst_bone_global')}')")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        op.report({'ERROR'}, f"Metrics computed but writing failed: {exc}")
        return {'CANCELLED'}

    cov = _recon_coverage(res["requested"], res["frames"], res["skipped"],
                                     res.get("blocked_records", ()))
    level = 'INFO'
    if cov["n_skipped"]:
        level = 'WARNING'
        msgs.append(f"{cov['n_skipped']} frame(s) skipped, see skipped_frames")
    for w in res["warnings"]:
        op.report({'WARNING'}, w)
    op.report({level}, "; ".join(msgs) + f" | rho = {cov['n_valid']}/{cov['n_frames']} -> {out_dir}")
    return {'FINISHED'}


class SYNTH_OT_compute_mpve(Operator):
    """Mean Per-Vertex Error between the GT mesh and its reconstruction"""
    bl_idname = "synth.compute_mpve"
    bl_label = "Compute MPVE"
    bl_description = ("Per-frame mean/median/p95/max world-space L2 distance between "
                      "corresponding vertices of the target mesh and its counterpart in "
                      "'Reconstructions', in three variants (global, root-relative, "
                      "body-length-normalised). Writes mpve_<collection>_<object>.json and "
                      "the (N_f, N_v) per-vertex array as .npz")

    def execute(self, context):
        return _recon_metrics_execute(self, context, {'mpve'})


class SYNTH_OT_compute_mpjpe(Operator):
    """MPJPE with global / root-relative / Procrustes-aligned decomposition"""
    bl_idname = "synth.compute_mpjpe"
    bl_label = "Compute MPJPE"
    bl_description = ("Per-frame MPJPE over the keypoint centroids and over the armature "
                      "bone heads, each reported as the triple global / root_relative / pa "
                      "(Umeyama similarity, uniform scale) with a degeneracy gate and 3D-PCK. "
                      "Writes mpjpe_<collection>_<object>.json")

    def execute(self, context):
        return _recon_metrics_execute(self, context, {'mpjpe'})


class SYNTH_OT_compute_bone_rotation_error(Operator):
    """Per-bone geodesic SO(3) error, aggregated by bone group"""
    bl_idname = "synth.compute_bone_rotation_error"
    bl_label = "Compute Bone Rotation Error"
    bl_description = ("Per-bone, per-frame geodesic angle between the GT and reconstruction "
                      "bone frames, in armature space and parent-relative, aggregated over the "
                      "optimiser's own bone groups. Refuses to run unless the pose time series "
                      "round trip passes. Writes bone_rotation_error_<collection>_<object>.json")

    def execute(self, context):
        return _recon_metrics_execute(self, context, {'bone_rot'})


class SYNTH_OT_compute_recon_metrics(Operator):
    """MPVE + MPJPE + per-bone SO(3) error from a single frame loop"""
    bl_idname = "synth.compute_recon_metrics"
    bl_label = "Compute All Metrics"
    bl_description = ("Compute MPVE, MPJPE and the per-bone rotation error in ONE pass "
                      "(one frame_set and two to_mesh() extractions per frame) and write all "
                      "three JSON files plus the per-vertex .npz")

    def execute(self, context):
        return _recon_metrics_execute(self, context, {'mpve', 'mpjpe', 'bone_rot'})


# =============================================================================
# UI PANEL & REGISTRATION
# =============================================================================

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
        box.prop(p, 'create_reconstruction_dataset')
        if p.create_reconstruction_dataset:
            box.prop(p, 'reconstruction_dataset_out_dir')
            if not p.render_binary:
                box.label(text="Needs 'Render Binary Masks' enabled.", icon='ERROR')
        box = layout.box()
        box.label(text="Performance")
        box.prop(p, 'event_timer_interval')
        box.prop(p, 'seconds_per_timer_tick')
        box.prop(p, 'use_persistent_render_data')
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

        box = layout.box()
        box.label(text="Reconstruction Evaluation")
        row = box.row(align=True)
        row.prop(p, 'iou_sample_count')
        row.prop(p, 'iou_random_seed')
        box.prop(p, 'iou_recon_object_name')
        box.prop(p, 'iou_with_keypoint_distances')
        box.prop(p, 'kpt_dist_warn_threshold_bl')
        row = box.row(align=True)
        row.operator('synth.compute_volumetric_iou', icon='MESH_CUBE')
        row.operator('synth.compute_keypoint_distances', icon='EMPTY_AXIS')

        box.prop(p, 'pts2_batch_dir')
        row = box.row(align=True)
        row.operator('synth.batch_3d_metrics', icon='FILE_REFRESH')

        box.separator()
        box.label(text="MPVE / MPJPE / Bone Rotation")
        row = box.row(align=True)
        row.prop(p, 'mpve_units', text="")
        row.prop(p, 'mpve_normalization_mode', text="")
        if p.mpve_normalization_mode == 'fixed':
            box.prop(p, 'mpve_fixed_vmax')
        elif p.mpve_normalization_mode == 'per_frame':
            box.label(text="Per-frame scale: NOT comparable across frames", icon='ERROR')
        box.prop(p, 'mpve_area_weighted')
        row = box.row(align=True)
        row.prop(p, 'pck_tau_body_lengths')
        row.prop(p, 'mpjpe_pa_degeneracy_threshold')
        box.prop(p, 'bone_err_swing_twist')
        if p.bone_err_swing_twist:
            box.prop(p, 'bone_err_prior_tolerance')
        box.prop(p, 'recon_roundtrip_json')
        row = box.row(align=True)
        row.prop(p, 'bone_err_require_roundtrip')
        row.prop(p, 'bone_err_roundtrip_tol')
        if p.bone_err_require_roundtrip and not p.recon_roundtrip_json.strip():
            box.label(text="Bone rotation error needs a round-trip JSON", icon='ERROR')
        row = box.row(align=True)
        row.operator('synth.compute_mpve', icon='MESH_DATA')
        row.operator('synth.compute_mpjpe', icon='EMPTY_AXIS')
        row = box.row(align=True)
        row.operator('synth.compute_bone_rotation_error', icon='BONE_DATA')
        row.operator('synth.compute_recon_metrics', icon='SEQUENCE')


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
    SYNTH_OT_compute_volumetric_iou,
    SYNTH_OT_compute_keypoint_distances,
    SYNTH_OT_batch_3d_metrics,
    SYNTH_OT_compute_mpve,
    SYNTH_OT_compute_mpjpe,
    SYNTH_OT_compute_bone_rotation_error,
    SYNTH_OT_compute_recon_metrics,
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