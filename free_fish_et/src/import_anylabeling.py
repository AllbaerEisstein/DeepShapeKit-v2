"""
Import manually created AnyLabeling (labelme-schema) annotations as a DSKv2 ground-truth
dataset, bypassing the YOLO11-seg / YOLO11-pose inference stages.

The module writes *exactly* the on-disk artifacts that
`extract_frames_edit.predict_masks_yolo` + `extract_frames_edit.detect_keypoints_yolo`
produce, so `dataloaders_edit.Multiview_Dataset` and everything downstream of it
(`pose_optimizer_edit`, `multiview_edit`, `multiview_reconstruction_edit`) consume the result
without a single change:

    <dataset_path>/
    ├── index.json                       # status -> 'keypoints_detected', max_n_instances, keypoint_list
    └── <view>/
        ├── origin/                      # owned by extract_frames -- read only, never rewritten
        ├── files.csv                    # owned by extract_frames -- read only, never rewritten
        ├── cropped/                     # written here
        ├── mask/                        # written here
        ├── mask_full/                   # written here
        ├── bbox-masked_image/           # written here
        ├── keypoints_results/
        │   └── keypoints_confs.pickle   # written here
        └── files_crop.csv               # written here

Ownership boundary
------------------
Frame extraction, frame numbering and the frame range are exclusively owned by
`extract_frames_edit.extract_from_video`. This module *only* consumes the existing `origin/`
frames and the frame numbers already recorded in each view's `files.csv`; it never extracts,
re-indexes or renumbers anything. A frame that extraction did not produce cannot be imported,
and an annotation for such a frame is reported and skipped rather than silently inventing a
frame.

Keypoint confidence semantics (must match the detector path exactly, because the loss and
masking logic in `losses_edit` / `dataloaders_edit` keys off it):

    [x, y, 1.0]        keypoint annotated by hand           (detector: detected, conf>0)
    [0.0, 0.0, 0.0]    instance present, keypoint unlabeled (detector: `make_zero_dict`)
    [-1.0, -1.0, -1.0] instance not present in this frame   (detector: `make_no_instance_detected_dict`)

The distinction between the last two is the "missed instance" vs "missing keypoint"
distinction; collapsing them would make a hand-annotated dataset score differently from a
detected one for reasons that have nothing to do with the annotations.

Partial imports
---------------
`import_masks` and `import_keypoints` select which half of the annotation is used. Clearing one
of them keeps the detector's existing output for that modality, which is what the GUI's "Keep
keypoints from pose model detections" / "Keep segmasks from mask segmentation" checkboxes do:

    import_masks=True,  import_keypoints=True   both come from the annotations (default)
    import_masks=True,  import_keypoints=False  masks annotated, keypoints kept from YOLO-pose
    import_masks=False, import_keypoints=True   keypoints annotated, masks kept from YOLO-seg

A partial import mixes two independently produced instance numberings in one dataset, and
`Multiview_Dataset` pairs a mask row with a keypoint entry purely by `sub_index`. Nothing in the
schema would reveal fish 0's mask being fitted against fish 1's keypoints, so partial imports
are checked geometrically before anything is written (see `_check_instance_alignment`).
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.dsk_types import InstancesKeypointsDict
from src.extract_frames_edit import (
    crop_and_pad,
    get_image_np_from_path,
    draw_kpts_on_img,
    make_no_instance_detected_kpt_dict,
    make_zero_kpt_dict,
    polygon_to_binary_mask,
    save_crops,
)

# `infer_mask` pulls in ultralytics at import time. The whole point of this feature is to be
# able to reconstruct on a machine that has no detector installed, so the import is tolerated
# to fail here and only raised if `img2bbx` is actually needed (i.e. at write time).
try:  # pragma: no cover - depends on the environment, not on the logic
    from src.infer_mask import img2bbx as _img2bbx

    _IMG2BBX_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as exc:  # noqa: BLE001 - ultralytics raises non-ImportError types too
    _img2bbx = None
    _IMG2BBX_IMPORT_ERROR = exc


# --------------------------------------------------------------------------------------
# Constants shared with the detector path
# --------------------------------------------------------------------------------------

#: `index.json` states this import is allowed to start from. Anything else means
#: `extract_frames` has not run (or has not finished) for this dataset.
ALLOWED_INPUT_STATUS = ('origin', 'masks_detected', 'keypoints_detected')

#: Identical to the header `predict_masks_yolo` writes.
FILES_CROP_CSV_HEADER = ['frame', 'file_loc', 'category', 'sub_index', 'folder', 'bbox']

#: Same padding `predict_masks_yolo` passes to `infer_mask.img2bbx`, so the bbox-masked images
#: this import writes are framed exactly like the detector's.
BBOX_MASKED_PADDING = 20

CROP_SUBDIRS = ('cropped', 'mask', 'mask_full', 'bbox-masked_image')

#: Fraction of an instance's retained keypoints that must fall inside the counterpart bbox for
#: the two numberings to be considered the same instance. Keypoints legitimately sit slightly
#: outside a tight silhouette (fin tips, mouth tip), hence not 1.0.
ALIGNMENT_MIN_INSIDE_FRACTION = 0.6

#: Bbox tolerance for that test, as a fraction of the bbox's own size.
ALIGNMENT_BBOX_TOLERANCE = 0.15

#: Above this share of mismatched instances a partial import is refused rather than warned about.
ALIGNMENT_MAX_MISMATCH_RATE = 0.25

IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

#: labelme/AnyLabeling shape types this importer understands. Anything else is reported and
#: skipped -- silently ignoring e.g. a `rectangle` the annotator used instead of a polygon
#: would produce a dataset that is quietly missing instances.
SUPPORTED_SHAPE_TYPES = ('polygon', 'point')


# --------------------------------------------------------------------------------------
# Result / reporting types
# --------------------------------------------------------------------------------------


@dataclass
class ViewImportStats:
    """Per-view counters, mirroring what the detector stages print at the end of a run."""

    view: str
    annotation_dir: Optional[Path] = None
    target_frames: int = 0
    frames_with_instance: int = 0
    missed_frames: int = 0
    instances_written: int = 0
    keypoints_annotated: int = 0
    keypoints_expected: int = 0

    @property
    def detection_percentage(self) -> float:
        return (100.0 * self.frames_with_instance / self.target_frames) if self.target_frames else 0.0

    @property
    def keypoint_percentage(self) -> float:
        return (100.0 * self.keypoints_annotated / self.keypoints_expected) if self.keypoints_expected else 0.0


@dataclass
class ImportSummary:
    """Everything the GUI/CLI needs to report the outcome of an import."""

    dataset_path: Path
    anylabeling_root: Path
    views: List[str] = field(default_factory=list)
    kpt_list: List[str] = field(default_factory=list)
    max_n_instances: int = 0
    imported_masks: bool = True
    imported_keypoints: bool = True
    group_id_to_sub_index: Dict[str, int] = field(default_factory=dict)
    stats: Dict[str, ViewImportStats] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    dry_run: bool = False

    def text_report(self) -> str:
        sources = (
            f"masks: {'annotations' if self.imported_masks else 'segmentation network (kept)'}, "
            f"keypoints: {'annotations' if self.imported_keypoints else 'pose network (kept)'}"
        )
        lines = [
            f"AnyLabeling import{' (dry run)' if self.dry_run else ''}: "
            f"{self.anylabeling_root} -> {self.dataset_path}",
            f"  {sources}",
            f"  keypoints ({len(self.kpt_list)}): {', '.join(self.kpt_list)}",
            f"  instances per frame (max_n_instances): {self.max_n_instances}",
        ]
        for view in self.views:
            s = self.stats[view]
            parts = [f"{s.frames_with_instance}/{s.target_frames} frames annotated "
                     f"({s.detection_percentage:.2f}%)"]
            if self.imported_masks:
                parts.append(f"{s.instances_written} instances")
            if self.imported_keypoints:
                parts.append(f"{s.keypoints_annotated}/{s.keypoints_expected} keypoints "
                             f"({s.keypoint_percentage:.2f}%)")
            lines.append(f"  {view}: " + ', '.join(parts))
        if self.warnings:
            lines.append(f"  warnings: {len(self.warnings)}")
        return "\n".join(lines)


class AnyLabelingImportError(ValueError):
    """
    Raised for conditions that make the produced dataset unusable or silently wrong.

    Carries *all* offending items rather than only the first, because these annotations are
    made by hand: reporting one typo per run turns fixing a label set into a dozen round trips.
    """

    def __init__(self, message: str, problems: Optional[Sequence[str]] = None):
        self.problems = list(problems or [])
        if self.problems:
            shown = self.problems[:25]
            more = len(self.problems) - len(shown)
            message = (
                message
                + "\n  - "
                + "\n  - ".join(shown)
                + (f"\n  ... and {more} more" if more > 0 else "")
            )
        super().__init__(message)


# --------------------------------------------------------------------------------------
# Parsed annotation containers
# --------------------------------------------------------------------------------------


@dataclass
class _FrameAnnotation:
    """One AnyLabeling JSON, resolved against one extracted frame."""

    frame: int
    json_path: Path
    #: normalized group id -> list of polygons, each a list of [x, y]
    polygons: Dict[str, List[List[List[float]]]] = field(default_factory=lambda: defaultdict(list))
    #: normalized group id -> {keypoint name: (x, y)}
    points: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: defaultdict(dict))

    def group_ids(self) -> List[str]:
        return sorted(set(self.polygons) | set(self.points), key=_group_sort_key)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _normalize_group_id(raw) -> str:
    """
    labelme writes `group_id: null` for an ungrouped shape, which is by far the common case in
    single-fish footage. It is normalized to the reserved token '' so that a dataset that mixes
    ungrouped and grouped frames still yields one stable instance slot for the ungrouped ones.
    """
    if raw is None:
        return ''
    if isinstance(raw, bool):  # bool is an int subclass; a bool group id is a mistake, not a group
        return str(int(raw))
    if isinstance(raw, (int, np.integer)):
        return str(int(raw))
    text = str(raw).strip()
    return text


def _group_sort_key(gid: str) -> Tuple[int, float, str]:
    """Ungrouped first, then numeric ids in numeric order, then anything else alphabetically."""
    if gid == '':
        return (0, 0.0, '')
    try:
        return (1, float(gid), '')
    except ValueError:
        return (2, 0.0, gid)


def _annotation_stem(json_path: Path) -> str:
    """
    Filename stem of the *image* an annotation belongs to.

    AnyLabeling saves `<image stem>.json`, but some export paths produce `<image name>.json`
    (i.e. `view_12.png.json`); both resolve to the same origin frame.
    """
    stem = json_path.stem
    lowered = stem.lower()
    for suffix in IMAGE_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _read_labelme_json(json_path: Path) -> Optional[dict]:
    """
    Read a labelme/AnyLabeling annotation, or return None when the file is not one.

    A labelme file is identified by carrying a `shapes` list; that is the only key this
    importer depends on, and it is what distinguishes an annotation from the other JSONs that
    live in a DSKv2 dataset (index.json, camera matrices, template meshes).
    """
    try:
        with json_path.open() as jf:
            data = json.load(jf)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get('shapes'), list):
        return None
    return data


def _callback_arity(callback: Callable) -> int:
    try:
        params = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return 2
    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return 2
    return len(positional)


class _Progress:
    """
    Adapter around whatever progress callback the caller supplies.

    DSKv2_demo's step handlers pass their own progress hook and the exact signature differs
    between the mask and keypoint handlers; probing the arity once keeps this module usable
    from the GUI, from the CLI wrapper (no callback) and from tests alike.
    """

    def __init__(self, callback: Optional[Callable] = None):
        self._callback = callback
        self._arity = _callback_arity(callback) if callback is not None else 0

    def __call__(self, message: str, fraction: Optional[float] = None) -> None:
        if self._callback is None:
            return
        try:
            if self._arity >= 2:
                self._callback(message, fraction)
            elif self._arity == 1:
                self._callback(message)
            else:
                self._callback()
        except Exception as exc:  # noqa: BLE001 - a broken UI hook must not abort an import
            tqdm.write(f"   progress callback raised {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# Template / keypoint list
# --------------------------------------------------------------------------------------


def load_kpt_list(template_json_path: Path) -> List[str]:
    """
    Read the canonical keypoint list from a template mesh JSON (e.g.
    `Bluegill_Body_mesh_pts2.json`).

    This is the same list that ends up in `index.json['keypoint_list']` and therefore the order
    that `fish_model_edit`'s `vert2kpt` and the bone groups' `keypoint_indices` are defined
    against. It must come from the template, never from the annotation files: taking the order
    from whatever the annotator happened to click first would permute keypoint identities and
    the fit would converge confidently onto the wrong correspondences.
    """
    template_json_path = Path(template_json_path)
    with template_json_path.open() as jf:
        template = json.load(jf)
    kpt_list = template.get('kpt_list')
    if not isinstance(kpt_list, list) or not kpt_list or not all(isinstance(k, str) for k in kpt_list):
        raise AnyLabelingImportError(
            f"Template {template_json_path} does not contain a usable 'kpt_list' entry."
        )
    return list(kpt_list)


# --------------------------------------------------------------------------------------
# Dataset-side preconditions
# --------------------------------------------------------------------------------------


def check_import_preconditions(
    dataset_path: Path,
    import_masks: bool = True,
    import_keypoints: bool = True,
) -> Tuple[bool, str]:
    """
    Cheap, side-effect-free check the GUI can call to decide whether to enable the import
    control. Returns (ok, human readable reason).

    A modality that is *not* being imported has to already exist on disk, so keeping the
    detector's masks requires the mask step to have run, and keeping its keypoints requires the
    keypoint step to have run. Checking that here is what lets the GUI grey the control out with
    a specific reason instead of failing halfway through a run.
    """
    dataset_path = Path(dataset_path)
    if not import_masks and not import_keypoints:
        return False, (
            "Both 'keep' options are ticked, so there is nothing left to import. Untick one of "
            "them, or skip this step entirely."
        )

    index_path = dataset_path / 'index.json'
    if not index_path.is_file():
        return False, f"No index.json in {dataset_path} - run 'extract frames' first."
    try:
        with index_path.open() as jf:
            index_json = json.load(jf)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"index.json could not be read ({exc}) - re-run 'extract frames'."

    status = index_json.get('status')
    if status not in ALLOWED_INPUT_STATUS:
        return False, (
            f"Dataset status is '{status}'; expected one of {', '.join(ALLOWED_INPUT_STATUS)}. "
            "Run 'extract frames' first."
        )

    views = index_json.get('frame_folders')
    if not isinstance(views, list) or not views:
        return False, "index.json lists no frame folders - run 'extract frames' first."

    for view in views:
        if not (dataset_path / view / 'files.csv').is_file():
            return False, f"View '{view}' has no files.csv - re-run 'extract frames'."
        if not (dataset_path / view / 'origin').is_dir():
            return False, f"View '{view}' has no origin/ folder - re-run 'extract frames'."
        if view not in index_json.get('image_sizes', {}):
            return False, f"index.json has no image size for view '{view}' - re-run 'extract frames'."

        if not import_masks:
            if not (dataset_path / view / 'files_crop.csv').is_file():
                return False, (
                    f"import_masks=False but view '{view}' has no files_crop.csv - run mask "
                    "detection first, or import the masks from the annotations instead."
                )
        if not import_keypoints:
            if not (dataset_path / view / 'keypoints_results' / 'keypoints_confs.pickle').is_file():
                return False, (
                    f"import_keypoints=False but view '{view}' has no keypoints_confs.pickle - "
                    "run keypoint detection first, or import the keypoints from the annotations "
                    "instead."
                )

    if not import_keypoints and not index_json.get('keypoint_list'):
        return False, (
            "The existing keypoints carry no keypoint_list in index.json, so their order cannot "
            "be verified against the template. Re-run keypoint detection, or untick 'Keep "
            "keypoints from pose model detections'."
        )

    parts = []
    if import_masks:
        parts.append('masks')
    if import_keypoints:
        parts.append('keypoints')
    return True, f"Ready: {len(views)} view(s) extracted; will import {' and '.join(parts)}."


def _read_view_frames(dataset_path: Path, view: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Return (frame number by origin-image stem, origin-image stem by frame number) for one view,
    taken from that view's `files.csv` -- the authoritative record of which frames exist.
    """
    frame_by_stem: Dict[str, int] = {}
    stem_by_frame: Dict[int, str] = {}
    with (dataset_path / view / 'files.csv').open() as f:
        for row in csv.DictReader(f, quotechar='"'):
            stem = Path(row['file_loc']).stem
            frame = int(row['frame'])
            frame_by_stem[stem] = frame
            stem_by_frame[frame] = stem
    return frame_by_stem, stem_by_frame


def resolve_view_annotation_dir(anylabeling_root: Path, view: str) -> Optional[Path]:
    """
    Map a dataset view to its folder of annotation JSONs.

    The layout AnyLabeling produces by default is "the JSON sits next to the image it annotates,
    same basename", so annotating the extracted frames in place yields
    `<dataset>/<view>/origin/<view>_<frame>.json`. That is the primary layout supported here,
    and it is why `anylabeling_root` may legitimately *be* the dataset path itself.

    Accepted layouts, in order of preference:
      1. `<root>/<view>/origin/`                 - labels next to the extracted frames
      2. `<root>/<view>/`                        - labels collected per view (searched recursively)
      3. the same two with '_undistorted' stripped - annotated under the pre-undistortion name
      4. `<root>/<view>` matched case-insensitively
      5. `<root>/` itself, when the root holds JSONs directly (single-view datasets)

    Layout 2 is searched recursively, so a root that mirrors the dataset tree resolves to the
    same files whether or not the annotator kept them in `origin/`.
    """
    anylabeling_root = Path(anylabeling_root)
    names = [view]
    if view.endswith('_undistorted'):
        names.append(view[: -len('_undistorted')])

    for name in names:
        origin_dir = anylabeling_root / name / 'origin'
        if origin_dir.is_dir() and any(origin_dir.glob('*.json')):
            return origin_dir
    for name in names:
        view_dir = anylabeling_root / name
        if view_dir.is_dir():
            return view_dir

    lowered = view.lower()
    for child in sorted(p for p in anylabeling_root.iterdir() if p.is_dir()):
        if child.name.lower() == lowered:
            return child

    if any(p.suffix.lower() == '.json' for p in anylabeling_root.iterdir() if p.is_file()):
        return anylabeling_root
    return None


# --------------------------------------------------------------------------------------
# Pass 1: parse + validate
# --------------------------------------------------------------------------------------


def _parse_view_annotations(
    view: str,
    annotation_dir: Path,
    frame_by_stem: Dict[str, int],
    image_size: Sequence[int],
    kpt_list: Sequence[str],
    target_frames: Sequence[int],
    problems: List[str],
    warnings: List[str],
    mask_labels: Optional[Sequence[str]] = None,
    import_keypoints: bool = True,
) -> Dict[int, _FrameAnnotation]:
    """
    Parse every annotation JSON of one view.

    Hard problems (appended to `problems`, abort the import before anything is written):
      * a point label that is not in the template's `kpt_list`
      * annotated image dimensions that disagree with `index.json['image_sizes'][view]`
      * unreadable / malformed JSON
    Soft problems (appended to `warnings`, import continues):
      * unsupported shape types, degenerate polygons, duplicate labels within one instance,
        annotations for frames that were never extracted, and extracted frames with no
        annotation at all.
    """
    kpt_name_set = set(kpt_list)
    mask_label_set = set(mask_labels) if mask_labels else None
    width, height = int(image_size[0]), int(image_size[1])
    annotations: Dict[int, _FrameAnnotation] = {}
    unknown_labels: Dict[str, int] = defaultdict(int)
    polygon_labels: Dict[str, int] = defaultdict(int)
    orphan_files: List[str] = []

    json_paths = sorted(p for p in annotation_dir.rglob('*.json') if p.is_file())
    for json_path in json_paths:
        # The annotation folder is frequently the dataset's own `<view>/origin/`, and the root
        # may be the dataset itself, so the sweep will meet JSONs that are not annotations
        # (index.json, camera matrices, template meshes). Resolve the frame first and read the
        # file only once it is known to belong to an extracted frame: an unrelated JSON is then
        # skipped instead of being reported as a malformed annotation.
        frame = frame_by_stem.get(_annotation_stem(json_path))
        data = None
        if frame is None:
            data = _read_labelme_json(json_path)
            if data is None:
                continue  # not a labelme file at all -- not this feature's business
            # AnyLabeling records the image it annotates; honour it when the JSON was renamed.
            image_path = data.get('imagePath')
            if isinstance(image_path, str) and image_path:
                frame = frame_by_stem.get(Path(image_path).stem)
            if frame is None:
                orphan_files.append(json_path.name)
                continue

        if data is None:
            data = _read_labelme_json(json_path)
            if data is None:
                problems.append(
                    f"[{view}] {json_path.name}: matches extracted frame {frame} but is not a "
                    "readable labelme/AnyLabeling annotation"
                )
                continue

        if frame in annotations:
            warnings.append(
                f"[{view}] frame {frame} is annotated by more than one file; "
                f"ignoring {json_path.name}"
            )
            continue

        ann_w = data.get('imageWidth')
        ann_h = data.get('imageHeight')
        if isinstance(ann_w, (int, float)) and isinstance(ann_h, (int, float)):
            if int(ann_w) != width or int(ann_h) != height:
                # Coordinates are stored in absolute pixels, so a size mismatch means every
                # point and polygon in this file lands somewhere else in the extracted frame.
                problems.append(
                    f"[{view}] {json_path.name}: annotated on {int(ann_w)}x{int(ann_h)} but the "
                    f"extracted frames are {width}x{height}"
                )
                continue
        else:
            warnings.append(
                f"[{view}] {json_path.name}: no imageWidth/imageHeight, cannot verify that it "
                f"was annotated at {width}x{height}"
            )

        shapes = data['shapes']  # presence and type guaranteed by _read_labelme_json
        frame_ann = _FrameAnnotation(frame=frame, json_path=json_path)
        for shape_index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                warnings.append(f"[{view}] {json_path.name}: shape #{shape_index} is not an object")
                continue
            shape_type = str(shape.get('shape_type', '')).strip().lower()
            label = shape.get('label')
            label = str(label) if label is not None else ''
            gid = _normalize_group_id(shape.get('group_id'))
            points = shape.get('points')

            if shape_type not in SUPPORTED_SHAPE_TYPES:
                warnings.append(
                    f"[{view}] {json_path.name}: ignoring shape #{shape_index} of unsupported "
                    f"type '{shape_type or 'missing'}' (label '{label}')"
                )
                continue
            if not isinstance(points, list) or not points:
                warnings.append(
                    f"[{view}] {json_path.name}: ignoring shape #{shape_index} ('{label}') "
                    "with no points"
                )
                continue
            try:
                coords = [(float(p[0]), float(p[1])) for p in points]
            except (TypeError, ValueError, IndexError):
                problems.append(
                    f"[{view}] {json_path.name}: shape #{shape_index} ('{label}') has "
                    "non-numeric point coordinates"
                )
                continue

            if shape_type == 'polygon':
                if len(coords) < 3:
                    warnings.append(
                        f"[{view}] {json_path.name}: ignoring polygon #{shape_index} with only "
                        f"{len(coords)} point(s)"
                    )
                    continue
                if mask_label_set is not None and label not in mask_label_set:
                    warnings.append(
                        f"[{view}] {json_path.name}: ignoring polygon #{shape_index} labelled "
                        f"'{label}', which is not one of the requested mask labels"
                    )
                    continue
                if label in kpt_name_set:
                    # A keypoint name on a polygon is almost always a shape_type slip in the
                    # annotation tool. Rasterizing it as a segmentation mask would silently
                    # merge a keypoint into the fish silhouette, so it is called out.
                    warnings.append(
                        f"[{view}] {json_path.name}: polygon #{shape_index} is labelled with the "
                        f"keypoint name '{label}'; it is being used as a segmentation mask"
                    )
                polygon_labels[label] += 1
                frame_ann.polygons[gid].append([[x, y] for x, y in coords])
            else:  # 'point'
                if label not in kpt_name_set:
                    unknown_labels[label] += 1
                    continue
                if label in frame_ann.points[gid]:
                    warnings.append(
                        f"[{view}] {json_path.name}: keypoint '{label}' annotated more than once "
                        f"for instance '{gid or 'ungrouped'}'; keeping the first"
                    )
                    continue
                x, y = coords[0]
                if not (0 <= x < width and 0 <= y < height):
                    warnings.append(
                        f"[{view}] {json_path.name}: keypoint '{label}' at ({x:.1f}, {y:.1f}) "
                        f"lies outside the {width}x{height} frame"
                    )
                frame_ann.points[gid][label] = (x, y)

        annotations[frame] = frame_ann

    if unknown_labels:
        # Fatal when the points are the import's output, because a typo'd name would silently
        # drop a keypoint. Only a warning when the keypoints are the detector's and these points
        # are being ignored wholesale anyway.
        sink = problems if import_keypoints else warnings
        sink.extend(
            f"[{view}] point label '{label}' ({count} occurrence(s)) is not in the template's "
            f"kpt_list: {', '.join(kpt_list)}"
            + ('' if import_keypoints else " (ignored: keypoints are kept from the pose network)")
            for label, count in sorted(unknown_labels.items())
        )
    if orphan_files:
        shown = ', '.join(sorted(orphan_files)[:5])
        warnings.append(
            f"[{view}] {len(orphan_files)} annotation file(s) do not match any extracted frame "
            f"and were ignored (e.g. {shown}). Frame numbering is owned by extract_frames."
        )

    missing = [frame for frame in target_frames if frame not in annotations]
    if missing:
        shown = ', '.join(str(frame) for frame in missing[:10])
        warnings.append(
            f"[{view}] {len(missing)} of {len(target_frames)} extracted frame(s) have no "
            f"annotation file and are treated as 'no instance detected' (e.g. {shown})"
        )
    return annotations


# --------------------------------------------------------------------------------------
# Pass 2: write
# --------------------------------------------------------------------------------------


def _read_retained_keypoints(dataset_path: Path, view: str) -> Dict[str, Dict[str, List[float]]]:
    """Load the keypoints the detector already wrote for one view, or {} if unreadable."""
    path = dataset_path / view / 'keypoints_results' / 'keypoints_confs.pickle'
    try:
        with path.open('rb') as handle:
            data = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_retained_mask_bboxes(dataset_path: Path, view: str) -> Dict[int, Dict[int, List[int]]]:
    """
    Load the detector's mask bboxes for one view as {frame: {sub_index: [x1, y1, x2, y2]}}.

    Read from files_crop.csv rather than from the mask images, because the CSV is what
    `Multiview_Dataset` actually pairs with the keypoints; a bbox that disagreed with its own
    mask would be the loader's problem, not this check's.
    """
    path = dataset_path / view / 'files_crop.csv'
    bboxes: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
    try:
        with path.open() as f:
            for row in csv.DictReader(f, quotechar='"'):
                if row.get('category') != 'mask':
                    continue
                try:
                    frame = int(row['frame'])
                    sub_index = int(row['sub_index'])
                    bbox = [int(v) for v in row['bbox'].strip('[]').split(',')]
                except (KeyError, ValueError):
                    continue
                if len(bbox) == 4:
                    bboxes[frame][sub_index] = bbox
    except OSError:
        return {}
    return dict(bboxes)


def _fraction_inside(points: Sequence[Tuple[float, float]], bbox: Sequence[int]) -> Optional[float]:
    """Fraction of `points` inside `bbox`, widened by ALIGNMENT_BBOX_TOLERANCE. None if empty."""
    if not points:
        return None
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * ALIGNMENT_BBOX_TOLERANCE
    pad_y = (y2 - y1) * ALIGNMENT_BBOX_TOLERANCE
    inside = sum(
        1 for x, y in points
        if (x1 - pad_x) <= x <= (x2 + pad_x) and (y1 - pad_y) <= y <= (y2 + pad_y)
    )
    return inside / len(points)


def _check_instance_alignment(
    view: str,
    imported_bboxes: Dict[int, Dict[int, List[int]]],
    retained_points: Dict[int, Dict[int, List[Tuple[float, float]]]],
    problems: List[str],
    warnings: List[str],
    label: str,
) -> None:
    """
    Verify that the imported half and the retained half of a partial import refer to the same
    instances, by testing whether each instance's keypoints land inside its counterpart's bbox.

    This is the check a partial import stands or falls on. `Multiview_Dataset` pairs a mask row
    with a keypoint entry by `sub_index` alone, and the annotator's `group_id` ordering has no
    reason to match the detector's detection ordering. Without this test, fitting fish 0's
    silhouette to fish 1's keypoints produces a confident, entirely wrong reconstruction with
    nothing anywhere in the artifacts to indicate it.

    A single-instance dataset cannot be permuted, but the test still runs there: a low score then
    means the annotations and the detections are looking at different frames or different fish.
    """
    checked = 0
    mismatched: List[str] = []
    for frame, per_instance in sorted(imported_bboxes.items()):
        retained_frame = retained_points.get(frame)
        if not retained_frame:
            continue
        for sub_index, bbox in sorted(per_instance.items()):
            points = retained_frame.get(sub_index)
            if not points:
                continue
            fraction = _fraction_inside(points, bbox)
            if fraction is None:
                continue
            checked += 1
            if fraction < ALIGNMENT_MIN_INSIDE_FRACTION:
                mismatched.append(f"frame {frame} instance {sub_index} ({fraction:.0%} inside)")

    if not checked:
        warnings.append(
            f"[{view}] {label}: no frame has both an imported and a retained instance, so their "
            "instance numbering could not be cross-checked. Verify the result before trusting it."
        )
        return

    rate = len(mismatched) / checked
    if not mismatched:
        return
    detail = f"{len(mismatched)}/{checked} instances disagree ({rate:.0%}): " + ', '.join(mismatched[:8])
    if rate > ALIGNMENT_MAX_MISMATCH_RATE:
        problems.append(
            f"[{view}] {label}: the imported and retained instance numbering do not match - "
            f"{detail}. Reconstruction would fit one fish's silhouette to another's keypoints. "
            "Renumber the group_ids to match the detector's instance order, or import both "
            "masks and keypoints."
        )
    else:
        warnings.append(f"[{view}] {label}: {detail}")


def _active_group_ids(
    ann: '_FrameAnnotation',
    import_masks: bool,
    import_keypoints: bool,
) -> List[str]:
    """
    Group ids in one frame that contribute something to the modalities being imported.

    A group id with only points contributes nothing when just the masks are imported, and one
    with only a polygon still contributes when the keypoints are imported (it marks the instance
    as present with no localized keypoints, i.e. [0,0,0] rather than [-1,-1,-1]).
    """
    active = []
    for gid in ann.group_ids():
        has_polygon = bool(ann.polygons.get(gid))
        has_points = bool(ann.points.get(gid))
        if import_masks and has_polygon:
            active.append(gid)
        elif import_keypoints and (has_points or has_polygon):
            active.append(gid)
    return active


def _validate_partial_import(
    dataset_path: Path,
    selected_views: Sequence[str],
    parsed: Dict[str, Dict[int, '_FrameAnnotation']],
    index_json: dict,
    group_id_to_sub_index: Dict[str, int],
    kpt_list: Sequence[str],
    import_masks: bool,
    problems: List[str],
    warnings: List[str],
) -> None:
    """
    Cross-check the annotated half of a partial import against the detected half that is being
    kept, in whichever direction applies.

    Importing masks  -> imported bboxes tested against the retained detector keypoints.
    Importing points -> retained detector bboxes tested against the imported keypoints.
    """
    label = ('annotated masks vs kept keypoints' if import_masks
             else 'kept masks vs annotated keypoints')

    for view in selected_views:
        image_size = index_json['image_sizes'][view]
        imported_bboxes: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
        retained_points: Dict[int, Dict[int, List[Tuple[float, float]]]] = defaultdict(dict)

        if import_masks:
            for frame, ann in parsed[view].items():
                for gid, polygons in ann.polygons.items():
                    if not polygons or gid not in group_id_to_sub_index:
                        continue
                    rasterized = _rasterize_instance(polygons, image_size)
                    if rasterized is not None:
                        imported_bboxes[frame][group_id_to_sub_index[gid]] = rasterized[1]
            for frame_str, instances in _read_retained_keypoints(dataset_path, view).items():
                try:
                    frame = int(frame_str)
                except (TypeError, ValueError):
                    continue
                for inst_str, kpt_dict in (instances or {}).items():
                    try:
                        sub_index = int(inst_str)
                    except (TypeError, ValueError):
                        continue  # the '-1' no-instance sentinel
                    points = [
                        (float(v[0]), float(v[1]))
                        for v in kpt_dict.values()
                        if len(v) >= 3 and float(v[2]) > 0.0
                    ]
                    if points:
                        retained_points[frame][sub_index] = points
        else:
            imported_bboxes = defaultdict(dict, _read_retained_mask_bboxes(dataset_path, view))
            for frame, ann in parsed[view].items():
                for gid, point_dict in ann.points.items():
                    if not point_dict or gid not in group_id_to_sub_index:
                        continue
                    retained_points[frame][group_id_to_sub_index[gid]] = [
                        (float(x), float(y)) for x, y in point_dict.values()
                    ]

        _check_instance_alignment(
            view=view,
            imported_bboxes=dict(imported_bboxes),
            retained_points=dict(retained_points),
            problems=problems,
            warnings=warnings,
            label=label,
        )


def _rasterize_instance(
    polygons: Sequence[Sequence[Sequence[float]]],
    image_size: Sequence[int],
) -> Optional[Tuple[np.ndarray, List[int]]]:
    """
    Rasterize all polygons of one instance into a single full-frame 0/1 mask and return it with
    its bbox, or None when the result is empty.

    Several polygons for one group id (a fish split by an occluder, say) are unioned rather
    than treated as separate instances, because the group id -- not the shape count -- is what
    identifies an instance.
    """
    width, height = int(image_size[0]), int(image_size[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        # Reuses extract_frames_edit.polygon_to_binary_mask so hand-drawn and detector-derived
        # masks are rasterized by the exact same code (same rounding, same fill rule).
        mask = np.maximum(mask, polygon_to_binary_mask(polygon, image_size=(width, height)))

    x, y, w, h = cv2.boundingRect(mask)
    if w <= 0 or h <= 0:
        return None
    return mask, [int(x), int(y), int(x + w), int(y + h)]


def _write_instance(
    dataset_path: Path,
    view: str,
    frame: int,
    sub_index: int,
    origin_path: Path,
    origin_rgb: np.ndarray,
    mask_full: np.ndarray,
    bbox: List[int],
) -> List[list]:
    """Write the four artifacts for one instance and return its four files_crop.csv rows."""
    if _img2bbx is None:
        raise AnyLabelingImportError(
            "infer_mask.img2bbx could not be imported, so the bbox-masked images this dataset "
            f"needs cannot be written: {_IMG2BBX_IMPORT_ERROR}"
        )

    bbox_masked_fname = f"image_{frame}_{sub_index}_bbox-masked.png"
    _img2bbx(
        img_path=origin_path,
        bbox=bbox,
        padding=BBOX_MASKED_PADDING,
        out_dir=dataset_path / view / 'bbox-masked_image',
        out_filename=bbox_masked_fname,
    )

    crop_img, crop_mask = crop_and_pad(origin_rgb, mask_full, bbox)
    save_crops(dataset_path, view, frame, sub_index, crop_img, crop_mask, mask_full)

    return [
        [frame, f"{view}/cropped/image_{frame}_{sub_index}.png", 'cropped', sub_index, view, bbox],
        [frame, f"{view}/mask/image_{frame}_{sub_index}_mask.png", 'mask', sub_index, view, bbox],
        [frame, f"{view}/mask_full/image_{frame}_{sub_index}_mask_full.png", 'mask_full', sub_index, view, bbox],
        [frame, f"{view}/bbox-masked_image/{bbox_masked_fname}", 'bbox-masked', sub_index, view, bbox],
    ]


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def import_gt_from_anylabeling(
    dataset_path: Path,
    anylabeling_root: Optional[Path],
    kpt_list: List[str],
    views: Optional[Sequence[str]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    mask_labels: Optional[Sequence[str]] = None,
    import_masks: bool = True,
    import_keypoints: bool = True,
    write_keypoint_overlays: bool = True,
    progress_callback: Optional[Callable] = None,
    dry_run: bool = False,
) -> ImportSummary:
    """
    Turn a directory of AnyLabeling annotations into the mask + keypoint artifacts DSKv2's
    reconstruction consumes, replacing `predict_masks_yolo` and `detect_keypoints_yolo`.

    Args:
        dataset_path:      dataset root written by `extract_from_video` (contains `index.json`).
        anylabeling_root:  where the annotation JSONs live. Pass None for the normal workflow,
                           where the extracted frames were annotated in place and each JSON sits
                           in `<dataset>/<view>/origin/` next to the PNG of the same basename.
        kpt_list:          canonical keypoint names from the active template mesh JSON, in
                           template order (see `load_kpt_list`).
        views:             restrict the import to these views; defaults to every view in
                           `index.json['frame_folders']`.
        frame_indices:     restrict the import to these frame numbers; defaults to every frame
                           listed in each view's `files.csv`.
        mask_labels:       only rasterize polygons carrying these labels (e.g. ['fish']);
                           defaults to accepting every polygon label.
        import_masks:      write masks/crops from the annotated polygons. Set False to keep the
                           masks the segmentation network already produced.
        import_keypoints:  write keypoints from the annotated points. Set False to keep the
                           keypoints the pose network already produced.
                           At least one of the two must be True.
        write_keypoint_overlays: also write the `keypoints_results/keypoints_<frame>_<inst>.png`
                           visualisations the detector path writes.
        progress_callback: called as (message, fraction in [0,1]); arity is probed, so a
                           one-argument or zero-argument hook works too.
        dry_run:           parse and validate everything, write nothing.

    Returns:
        ImportSummary with per-view statistics and the collected warnings.

    Raises:
        AnyLabelingImportError: preconditions unmet, or any hard validation failure. Nothing is
        written in that case -- validation runs to completion over every view first, so one run
        reports every problem.
    """
    dataset_path = Path(dataset_path)
    # Annotating the extracted frames in place is the normal workflow, so the dataset itself is
    # the default annotation root.
    anylabeling_root = Path(anylabeling_root) if anylabeling_root is not None else dataset_path
    progress = _Progress(progress_callback)

    ok, reason = check_import_preconditions(dataset_path, import_masks, import_keypoints)
    if not ok:
        raise AnyLabelingImportError(f"Cannot import AnyLabeling annotations: {reason}")
    if not anylabeling_root.is_dir():
        raise AnyLabelingImportError(f"Annotation root does not exist: {anylabeling_root}")
    if not kpt_list:
        raise AnyLabelingImportError("kpt_list is empty; load the template mesh JSON first.")
    duplicate_kpts = {name for name in kpt_list if kpt_list.count(name) > 1}
    if duplicate_kpts:
        raise AnyLabelingImportError(
            "Template kpt_list contains duplicate names, which would make keypoints ambiguous: "
            + ', '.join(sorted(duplicate_kpts))
        )

    with (dataset_path / 'index.json').open() as jf:
        index_json = json.load(jf)

    available_views = list(index_json['frame_folders'])
    if views is None:
        selected_views = available_views
    else:
        missing_views = [v for v in views if v not in available_views]
        if missing_views:
            raise AnyLabelingImportError(
                "Requested view(s) are not in this dataset: " + ', '.join(missing_views)
            )
        selected_views = [v for v in available_views if v in set(views)]

    summary = ImportSummary(
        dataset_path=dataset_path,
        anylabeling_root=anylabeling_root,
        views=list(selected_views),
        kpt_list=list(kpt_list),
        imported_masks=bool(import_masks),
        imported_keypoints=bool(import_keypoints),
        dry_run=dry_run,
    )

    if not import_keypoints:
        # The retained keypoints are stored by name but consumed by *index*: the loader reads
        # them in index.json['keypoint_list'] order, which the template's vert2kpt and bone
        # groups are defined against. If the detector was run against a different template, the
        # retained keypoints mean something else than this import's masks do.
        retained_kpt_list = list(index_json.get('keypoint_list') or [])
        if retained_kpt_list != list(kpt_list):
            raise AnyLabelingImportError(
                "The keypoints being kept were produced for a different keypoint list than the "
                "active template's, so keeping them would silently re-label them.\n"
                f"  existing: {retained_kpt_list}\n"
                f"  template: {list(kpt_list)}\n"
                "Re-run keypoint detection with this template, or untick 'Keep keypoints from "
                "pose model detections'."
            )

    frame_indices_set = set(int(f) for f in frame_indices) if frame_indices is not None else None

    # ---------------------------------------------------------------- pass 1: parse+validate
    problems: List[str] = []
    parsed: Dict[str, Dict[int, _FrameAnnotation]] = {}
    target_frames_by_view: Dict[str, List[int]] = {}
    stem_by_frame_by_view: Dict[str, Dict[int, str]] = {}

    progress("Validating AnyLabeling annotations...", 0.0)
    for view_index, view in enumerate(selected_views):
        stats = ViewImportStats(view=view)
        summary.stats[view] = stats

        frame_by_stem, stem_by_frame = _read_view_frames(dataset_path, view)
        stem_by_frame_by_view[view] = stem_by_frame
        available_frames = sorted(stem_by_frame)
        target_frames = (
            [f for f in available_frames if f in frame_indices_set]
            if frame_indices_set is not None
            else available_frames
        )
        target_frames_by_view[view] = target_frames
        stats.target_frames = len(target_frames)

        annotation_dir = resolve_view_annotation_dir(anylabeling_root, view)
        stats.annotation_dir = annotation_dir
        if annotation_dir is None:
            # Hard error, not a warning. A view whose annotations simply are not there would
            # otherwise import as "no instance in any frame", which is a valid dataset the
            # reconstruction will happily consume and fit nothing to. The likely causes -- the
            # wrong folder picked, annotations never saved, saved next to the videos instead of
            # the frames -- are all things the user has to fix, so say so and stop.
            problems.append(
                f"[{view}] no AnyLabeling annotation files found. Expected one JSON per frame, "
                f"named after the frame it annotates, in "
                f"{anylabeling_root / view / 'origin'} "
                f"(e.g. {view}_0.json next to {view}_0.png)."
            )
            parsed[view] = {}
        else:
            parsed[view] = _parse_view_annotations(
                view=view,
                annotation_dir=annotation_dir,
                frame_by_stem=frame_by_stem,
                image_size=index_json['image_sizes'][view],
                kpt_list=kpt_list,
                target_frames=target_frames,
                problems=problems,
                warnings=summary.warnings,
                mask_labels=mask_labels,
                import_keypoints=import_keypoints,
            )
            if not parsed[view]:
                problems.append(
                    f"[{view}] {annotation_dir} contains no annotation file that matches an "
                    f"extracted frame. The JSON basename must equal the frame's, e.g. "
                    f"{view}_0.json for origin/{view}_0.png."
                )

        if not (dataset_path / view / 'frame2video_1.csv').is_file():
            # Multiview_Dataset maps reconstruction indices to per-view frame numbers through
            # this file. It is written by extract_from_video(also_create_frame2video_csv=True)
            # and is not this feature's to create -- flag it rather than fabricate it.
            summary.warnings.append(
                f"[{view}] frame2video_1.csv is missing; Multiview_Dataset needs it. Re-run "
                "frame extraction with the frame map enabled."
            )

        progress(
            f"Validated {view} ({view_index + 1}/{len(selected_views)})",
            0.5 * (view_index + 1) / max(1, len(selected_views)),
        )

    if problems:
        raise AnyLabelingImportError(
            "AnyLabeling annotations are inconsistent with the dataset/template; nothing was "
            "written:",
            problems,
        )

    # Instance identity must be stable across frames *and* views: instance k in view A has to
    # be the same fish as instance k in view B, or the triangulation fuses two different
    # animals. The mapping is therefore built once from every group id observed anywhere,
    # instead of renumbering per frame. Only the group ids that contribute to a modality being
    # imported are counted -- a points-only group id does not occupy a mask slot when the masks
    # are the annotations', and vice versa.
    observed_gids = sorted(
        {
            gid
            for view_ann in parsed.values()
            for ann in view_ann.values()
            for gid in _active_group_ids(ann, import_masks, import_keypoints)
        },
        key=_group_sort_key,
    )
    if not observed_gids:
        observed_gids = ['']
    group_id_to_sub_index = {gid: i for i, gid in enumerate(observed_gids)}
    summary.group_id_to_sub_index = dict(group_id_to_sub_index)
    summary.max_n_instances = len(observed_gids)

    if not (import_masks and import_keypoints):
        # A partial import mixes the annotator's numbering with the detector's. Verify they
        # agree before writing, and carry the retained side's instance count forward so the
        # retained artifacts do not fall outside max_n_instances.
        retained_max = int(index_json.get('max_n_instances') or 0)
        if retained_max > summary.max_n_instances:
            summary.max_n_instances = retained_max
        _validate_partial_import(
            dataset_path=dataset_path,
            selected_views=selected_views,
            parsed=parsed,
            index_json=index_json,
            group_id_to_sub_index=group_id_to_sub_index,
            kpt_list=kpt_list,
            import_masks=import_masks,
            problems=problems,
            warnings=summary.warnings,
        )
        if problems:
            raise AnyLabelingImportError(
                "The annotations and the detections being kept do not line up; nothing was "
                "written:",
                problems,
            )

    if dry_run:
        for view in selected_views:
            stats = summary.stats[view]
            for frame in target_frames_by_view[view]:
                ann = parsed[view].get(frame)
                if ann is None:
                    stats.missed_frames += 1
                    continue
                gids = _active_group_ids(ann, import_masks, import_keypoints)
                if gids:
                    stats.frames_with_instance += 1
                else:
                    stats.missed_frames += 1
                stats.instances_written += len(gids)
                stats.keypoints_expected += len(gids) * len(kpt_list)
                stats.keypoints_annotated += sum(len(ann.points.get(gid, {})) for gid in gids)
        progress("Dry run complete - nothing written.", 1.0)
        return summary

    # ------------------------------------------------------------------------ pass 2: write
    for view_index, view in enumerate(selected_views):
        stats = summary.stats[view]
        image_size = index_json['image_sizes'][view]
        target_frames = target_frames_by_view[view]
        stem_by_frame = stem_by_frame_by_view[view]
        view_annotations = parsed[view]

        print(f"processing frames for video {view}...")
        if import_masks:
            for sub_dir in CROP_SUBDIRS:
                os.makedirs(dataset_path / view / sub_dir, exist_ok=True)
        if import_keypoints:
            os.makedirs(dataset_path / view / 'keypoints_results', exist_ok=True)

        csv_rows: List[list] = []
        frame2prediction: Dict[str, InstancesKeypointsDict] = {}

        frame_range_label = (
            f"{target_frames[0]}-{target_frames[-1]}" if target_frames else 'no-frames'
        )
        pbar = tqdm(
            total=len(target_frames),
            desc=f"import for frame - of video {view} [{frame_range_label}]",
            dynamic_ncols=True,
        )
        info_bar = tqdm(total=0, bar_format='{desc}', position=1, leave=False)

        for frame in target_frames:
            frame_str = str(frame)
            instances_kpts = InstancesKeypointsDict()
            ann = view_annotations.get(frame)

            active_gids = _active_group_ids(ann, import_masks, import_keypoints) if ann else []

            if not active_gids:
                # Same sentinel `detect_keypoints_yolo` writes when it finds no instance at all,
                # so a hand-annotated gap is indistinguishable from a detector miss downstream.
                # When the keypoints are the detector's, its own entry for this frame is left
                # alone: an unannotated frame means "no polygon here", not "no fish here".
                if import_keypoints:
                    instances_kpts['-1'] = make_no_instance_detected_kpt_dict(list(kpt_list))
                    frame2prediction[frame_str] = instances_kpts
                stats.missed_frames += 1
                info_bar.set_description_str(f"last frame {frame}: no instance annotated")
                tqdm.write(
                    f"   annotation missing for frame {frame} of view {view}"
                    if ann is None
                    else f"   frame {frame} of view {view} has an annotation file but no usable shapes"
                )
                pbar.set_description_str(f"import for frame {frame} of video {view} [{frame_range_label}]")
                pbar.update(1)
                continue

            origin_path = dataset_path / view / 'origin' / f"{stem_by_frame[frame]}.png"
            origin_rgb = None
            wrote_any_instance = False
            annotated_counts: List[str] = []

            for gid in active_gids:
                sub_index = group_id_to_sub_index[gid]
                polygons = ann.polygons.get(gid, [])
                point_dict = ann.points.get(gid, {})

                rasterized = (
                    _rasterize_instance(polygons, image_size)
                    if (polygons and import_masks) else None
                )
                if polygons and import_masks and rasterized is None:
                    summary.warnings.append(
                        f"[{view}] frame {frame}, instance '{gid or 'ungrouped'}': polygon(s) "
                        "rasterize to an empty mask and were skipped"
                    )

                if rasterized is not None:
                    if origin_rgb is None:
                        if not origin_path.is_file():
                            summary.warnings.append(
                                f"[{view}] frame {frame}: origin image {origin_path.name} is "
                                "missing; instance skipped"
                            )
                            break
                        # Loaded once per frame rather than once per instance: the same helper
                        # `predict_masks_yolo` uses, just not re-decoded for each fish.
                        origin_rgb = get_image_np_from_path(str(origin_path))
                    mask_full, bbox = rasterized
                    csv_rows.extend(
                        _write_instance(
                            dataset_path=dataset_path,
                            view=view,
                            frame=frame,
                            sub_index=sub_index,
                            origin_path=origin_path,
                            origin_rgb=origin_rgb,
                            mask_full=mask_full,
                            bbox=bbox,
                        )
                    )
                    stats.instances_written += 1
                    wrote_any_instance = True

                if not import_keypoints:
                    continue

                # Keypoints: an instance that appears in this frame at all is "present", so its
                # unlabeled keypoints are recorded as missing-keypoint ([0,0,0]) rather than as
                # a missed instance ([-1,-1,-1]).
                kpt_dict = make_zero_kpt_dict(list(kpt_list))
                for name, (x, y) in point_dict.items():
                    kpt_dict[name] = [float(x), float(y), 1.0]
                instances_kpts[str(sub_index)] = kpt_dict
                stats.keypoints_expected += len(kpt_list)
                stats.keypoints_annotated += len(point_dict)
                annotated_counts.append(f"{sub_index}:{len(point_dict)}/{len(kpt_list)}")

                # The overlay is drawn on the bbox-masked image, which this run writes when the
                # masks are imported and the detector wrote otherwise; either way it has to
                # exist before it can be drawn on.
                bbox_masked_path = (dataset_path / view / 'bbox-masked_image'
                                    / f"image_{frame}_{sub_index}_bbox-masked.png")
                if write_keypoint_overlays and point_dict and bbox_masked_path.is_file():
                    draw_kpts_on_img(
                        kpt_dict,
                        bbox_masked_path,
                        dataset_path / view / 'keypoints_results'
                        / f"keypoints_{frame}_{sub_index}.png",
                    )

            if import_keypoints:
                frame2prediction[frame_str] = instances_kpts
            if wrote_any_instance or (not import_masks and active_gids):
                stats.frames_with_instance += 1
            else:
                stats.missed_frames += 1

            info_bar.set_description_str(
                f"last frame {frame} - keypoints per instance: [{', '.join(annotated_counts)}]"
            )
            pbar.set_description_str(f"import for frame {frame} of video {view} [{frame_range_label}]")
            pbar.update(1)

        info_bar.close()
        pbar.close()

        if import_masks:
            csv_rows.sort(key=lambda row: (row[0], row[3]))
            with (dataset_path / view / 'files_crop.csv').open('w', newline='') as csv_out_file:
                csvwriter = csv.writer(
                    csv_out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                csvwriter.writerow(FILES_CROP_CSV_HEADER)
                csvwriter.writerows(csv_rows)

        if import_keypoints:
            with (dataset_path / view / 'keypoints_results'
                  / 'keypoints_confs.pickle').open('wb') as handle:
                pickle.dump(frame2prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('******* import complete ****')
        print(f"  {view} - Percentage of frames with an annotated instance: {stats.detection_percentage:.2f}%")
        print(f"  {view} - Number of frames without any annotated instance: {stats.missed_frames}")
        if import_masks:
            print(f"  {view} - Instances written: {stats.instances_written}")
        else:
            print(f"  {view} - Masks kept from the segmentation network")
        if import_keypoints:
            print(f"  {view} - Annotated keypoints: {stats.keypoint_percentage:.2f}% "
                  f"({stats.keypoints_annotated}/{stats.keypoints_expected})")
        else:
            print(f"  {view} - Keypoints kept from the pose network")

        progress(
            f"Imported {view} ({view_index + 1}/{len(selected_views)})",
            0.5 + 0.5 * (view_index + 1) / max(1, len(selected_views)),
        )

    # ------------------------------------------------------------------------ index.json
    # Both modalities are on disk now, whichever produced each of them.
    index_json['status'] = 'keypoints_detected'
    index_json['max_n_instances'] = summary.max_n_instances
    if import_keypoints:
        index_json['keypoint_list'] = list(kpt_list)
    # Provenance, per modality: without it a hand-annotated dataset is indistinguishable from a
    # detected one, and a half-and-half dataset is indistinguishable from either. Metrics that
    # compare "GT" against "detected" silently lose their meaning otherwise.
    previous_source = index_json.get('annotation_source')
    if not isinstance(previous_source, dict):
        previous_source = {}
    index_json['annotation_source'] = {
        'masks': 'anylabeling' if import_masks else previous_source.get('masks', 'yolo'),
        'keypoints': 'anylabeling' if import_keypoints else previous_source.get('keypoints', 'yolo'),
    }
    index_json['anylabeling_root'] = str(anylabeling_root)
    with (dataset_path / 'index.json').open('w') as jf:
        json.dump(index_json, jf, indent=2)

    for warning in summary.warnings:
        print(f"Warning: {warning}")
    print(summary.text_report())
    progress("AnyLabeling import complete.", 1.0)
    return summary