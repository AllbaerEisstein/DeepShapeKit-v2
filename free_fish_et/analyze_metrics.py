#!/usr/bin/env python3
"""
Analyse the metrics collected by `sweep_view_combinations.py --collect-results`.

The sweep reconstructs the same sequence from every camera-view combination of
size >= 2 and writes one metrics file per run; the collector merges them into
`<out_path>/metrics_collected/collected_metrics.json`, a mapping

    run key (see combo_folder_name) -> raw metrics_instance_N.json content

Every per-view metric that `multiview_reconstruction.reconstruct()` wrote is
analysed, discovered from the data rather than hard-coded, so the GT-referenced
metrics that only exist when `gt_dataset_dir` was set (keypoint_PCK_AUC_to_gt,
contour_HD95_to_gt, keypoint_L2_distance_to_gt, keypoint_detection_coverage,
gt_body_length_px and the GT-visibility-conditioned hit / miss / hallucination /
correct-absence rates) need no separate code path here, and a metric added later
is picked up with a generic label instead of being dropped silently.
METRIC_REGISTRY supplies units, orientation and provenance for the metrics known
at the time of writing.

Raw per-frame values are pooled -- never per-run pre-averages -- over four
groupings: all runs, per view, per (view, #views) and per #views. Missing or
undefined frames are encoded NaN (float metrics) or null (keypoint metrics) and
are excluded from every statistic rather than coerced to zero, following the
convention documented in multiview_reconstruction.py: a 0 would read as a
perfect score for a comparison that is in fact undefined.

Usage:
    python analyze_metrics.py out/metrics_collected/collected_metrics.json
    python analyze_metrics.py --primary-color '#1B9E77' --font 'TeX Gyre Heros'
    python analyze_metrics.py --metrics contour_HD95_to_gt keypoint_PCK_AUC_to_gt
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import itertools
import json
import math
import random
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, quantiles, stdev
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

# --------------------------------------------------------------------------
# Metric taxonomy
# --------------------------------------------------------------------------

SCALAR = "scalar"  # metric[view] -> [value per frame]
KEYPOINT = "keypoint"  # metric[view][keypoint] -> [value per frame]

# Top-level keys of a metrics file that are not per-view frame series. Blocked
# explicitly because optimizer_losses is structurally indistinguishable from a
# scalar metric: it maps loss name -> frame series, not view -> frame series.
NON_METRIC_KEYS = frozenset(
    {
        "optimizer_losses",
        "interpolated_frames",
        "blocked_frames",
        "timing",
        "total_duration_min",
        "seconds_per_frame",
    }
)

# --------------------------------------------------------------------------
# Blocked frames
# --------------------------------------------------------------------------
#
# A blocked frame is one the reconstruction pipeline emitted -- it has a full row
# in every per-frame metric array -- but whose pose it did not obtain by fitting
# the optimizer to observations of that frame. Currently that means a pose
# gap-filled by interpolation across a detection gap, but the reason travels with
# each record, so a new one needs no change here.
#
# They are EXCLUDED from every statistic by default. A straight-line guess across
# a gap is not something the reconstruction produced, so scoring it measures the
# interpolation instead. Worse, it is not neutral: the runs with the most
# gap-filled frames are exactly the view combinations whose detections failed
# most, so including them flatters the weakest combinations and biases the
# sweep's central question.
#
# The list comes from the metrics file itself, as "blocked_frames":
#
#   [{"frame_number": 72,
#     "frame_index_in_this_reconstruction_run": 7,
#     "reason_blocked": "interpolated_pose_gap_fill"}, ...]
#
# `frame_index_in_this_reconstruction_run` is what this script uses: the per-view
# arrays in collected_metrics.json are positional and carry no frame numbers, so
# the index is the only key that addresses them. It is recorded by the producer at
# emission time rather than re-derived here, because it is not generally
# recoverable after the fact -- the arrays are dense over PROCESSED frames, and a
# frame dropped for an unavailable sample leaves no trace in them.
#
# Excluded frames still count as AVAILABLE in the coverage ratio rho: they were
# requested and the pipeline produced a row for them, so rho keeps meaning "of
# everything this run set out to measure, how much reached the analysis", and a
# run that had to interpolate half its frames does not silently report the same
# rho as one that fitted every frame.
#
# A run whose blocked_frames field is missing is analysed unfiltered, with one
# warning naming it -- never silently, and never by guessing an alignment.

# One-element list rather than a bare bool: build_report() needs to describe the policy the
# run actually used, and this keeps that out of its signature without a rebinding global.
EXCLUDE_BLOCKED = [True]
BLOCKED_FRAMES_KEY = "blocked_frames"
BLOCKED_INDEX_FIELD = "frame_index_in_this_reconstruction_run"


# The metric whose keys define the run's view set; any discovered entry not
# keyed by those views is rejected rather than analysed as if it were.
REFERENCE_METRIC = "IoU_reconstruction_and_gt"


@dataclass(frozen=True)
class MetricSpec:
    """
    Presentation and provenance of one metric. `label` is the axis label,
    `description` the figure caption, and `orientation` states the direction of
    improvement so that no plot can be read backwards.
    """

    label: str
    description: str
    orientation: str = "higher is better"
    # True when a single (frame, view) value cannot depend on which views entered
    # the reconstruction, i.e. it scores the detector or the GT alone. Its
    # variation across #views is then group composition, not a fitting effect.
    combination_invariant: bool = False
    # False for the 3D metrics: they are measured on the fused reconstruction and
    # have one value per (run, frame), so 'which view' is not a question that can
    # be asked of them. Figures keyed by view are skipped for such a metric.
    view_axis: bool = True
    # What one member of a sub-divided metric is called, for the figure titles and
    # file names. 'keypoint' for the per-keypoint metrics, 'bone group' for the
    # per-bone-group MPJPE: the second axis of a measure is structural, so calling
    # a bone group a keypoint would misname it in every title it appears in.
    member_noun: str = "keypoint"


# Registry order is report order: reconstruction quality first, then the
# detector-side controls, then the scale diagnostic.
METRIC_REGISTRY: Dict[str, MetricSpec] = {
    "IoU_reconstruction_and_gt": MetricSpec(
        label="IoU(reconstruction, GT)",
        description=(
            "Binary Jaccard index between the reprojected reconstruction silhouette and the "
            "ground-truth mask, per frame and view."
        ),
    ),
    "contour_HD95_to_gt": MetricSpec(
        label="contour HD95 vs GT [GT body lengths]",
        description=(
            "Symmetric 95th-percentile Hausdorff distance between the reconstruction and GT mask "
            "contours, divided by the per-frame, per-view GT body length. Unlike IoU it is "
            "sensitive to fin and caudal-tail geometry; NaN, never 0, when either contour is empty."
        ),
        orientation="lower is better",
    ),
    "keypoint_PCK_AUC_to_gt": MetricSpec(
        label="PCK-AUC vs GT",
        description=(
            "Area under the PCK(alpha) curve over alpha in [0, 0.5] (21 steps, trapezoidal, "
            "rescaled to [0, 1] by the sweep width), thresholds alpha x GT body length, pooled "
            "over the GT-visible keypoints of a single (frame, view)."
        ),
    ),
    "keypoint_L2_distance_to_gt": MetricSpec(
        label="keypoint L2 vs GT [GT body lengths]",
        description=(
            "Distance from each reprojected template keypoint to its GT keypoint, normalised by "
            "the GT body length. Defined only where the GT marks the keypoint visible."
        ),
        orientation="lower is better",
    ),
    "IoU_reconstruction_and_mask_detection": MetricSpec(
        label="IoU(reconstruction, detector mask)",
        description=(
            "Binary Jaccard index between the reprojected reconstruction silhouette and the "
            "detector's mask. Ground truth is not involved: this measures agreement with the "
            "signal the fit was actually driven by."
        ),
    ),
    "keypoint_L2_distance": MetricSpec(
        label="keypoint L2 vs detector [px]",
        description=(
            "Pixel distance between the reprojected template keypoint and the detector's keypoint "
            "in the padded canvas, both rounded to integer pixels. Ground truth is not involved."
        ),
        orientation="lower is better",
    ),
    "IoU_mask_detection_and_gt": MetricSpec(
        label="IoU(detector mask, GT)",
        description=(
            "Binary Jaccard index between the detector's mask and the GT mask: the ceiling the "
            "mask term of the fit could reach."
        ),
        combination_invariant=True,
    ),
    "keypoint_detection_coverage": MetricSpec(
        label="detector coverage rho",
        description=(
            "Fraction of the GT-visible keypoints of a (frame, view) that the detector produced at "
            "all. Reported alongside the accuracy metrics because those are computed over detected "
            "keypoints only, so failing on the hard frames would otherwise raise them."
        ),
        combination_invariant=True,
    ),
    "keypoint_hit_rate_vs_gt": MetricSpec(
        label="hit rate | GT visible",
        description=(
            "Detector produced a keypoint the GT marks visible. Conditioned on GT visibility: NaN "
            "wherever the GT marks the keypoint absent, so structural zeros never dilute the mean."
        ),
        combination_invariant=True,
    ),
    "keypoint_miss_rate_vs_gt": MetricSpec(
        label="miss rate | GT visible",
        description=(
            "Detector produced no keypoint where the GT marks one visible; the complement of the "
            "hit rate over the same GT-visible denominator."
        ),
        orientation="lower is better",
        combination_invariant=True,
    ),
    "keypoint_hallucination_rate_vs_gt": MetricSpec(
        label="hallucination rate | GT absent",
        description=(
            "Detector produced a keypoint the GT marks absent. Conditioned on GT absence, which is "
            "rare in a fully labelled synthetic sequence, so this metric is sparse by construction."
        ),
        orientation="lower is better",
        combination_invariant=True,
    ),
    "keypoint_correct_absence_rate_vs_gt": MetricSpec(
        label="correct-absence rate | GT absent",
        description=(
            "Detector correctly produced nothing where the GT marks the keypoint absent; the "
            "complement of the hallucination rate over the same denominator."
        ),
        combination_invariant=True,
    ),
    "gt_body_length_px": MetricSpec(
        label="GT body length [px]",
        description=(
            "The GT-only scale normaliser itself: projected mouth tip to caudal peduncle distance, "
            "falling back to the GT mask bounding-box diagonal where either landmark is unlabelled. "
            "A diagnostic, not a quality score: it reports how much foreshortening each view sees."
        ),
        orientation="diagnostic, no preferred direction",
        combination_invariant=True,
    ),
    # --- 3D, world space (collected_3d_metrics.json) ------------------------
    # Registered after the 2D block so that registry order stays report order and
    # the 2D metrics keep supplying build_report()'s reference view set.
    "IoU_3d_volumetric": MetricSpec(
        label="volumetric 3D IoU(reconstruction, GT)",
        description=(
            "Monte-Carlo occupancy estimate of Vol(GT and R) / Vol(GT or R) between the fitted "
            "template and the ground-truth mesh, both in world space. Unlike the 2D silhouette "
            "IoU this cannot be satisfied by a fit that only looks right from the cameras: it "
            "penalises depth and thickness errors no view constrains. A ratio of two volumes, so "
            "it is scale free and is deliberately not body-length normalised."
        ),
        view_axis=False,
    ),
    "keypoint_distance_3d_bl": MetricSpec(
        label="3D keypoint distance [body lengths]",
        description=(
            "Euclidean distance between each keypoint's world-space vertex-group centroid on the "
            "reconstruction and on the GT mesh, divided by the per-frame GT body length. This is "
            "the 3D counterpart of keypoint_L2_distance_to_gt and is normalised the same way, so "
            "the two are on one scale; it is an absolute world-space error, not a reprojection, "
            "so a fit that is wrong in depth cannot hide behind a camera."
        ),
        orientation="lower is better",
        view_axis=False,
    ),
    "body_length_3d_m": MetricSpec(
        label="GT body length [m]",
        description=(
            "The 3D scale normaliser itself: world-space mouth tip to caudal peduncle distance on "
            "the GT mesh, with the GT mesh AABB diagonal as a flagged fallback. A diagnostic, not "
            "a quality score. It should be near constant across a sweep, since every run is scored "
            "against the same ground-truth animation; drift or a step means some runs fell back to "
            "the AABB and their body-length figures are not on the same scale as the rest."
        ),
        orientation="diagnostic, no preferred direction",
        combination_invariant=True,
        view_axis=False,
    ),
    "volume_ratio_3d_recon_over_gt": MetricSpec(
        label="volume ratio recon / GT",
        description=(
            "Occupied volume of the reconstruction over that of the GT mesh, per frame. Scale "
            "fidelity: 1.0 is correct, below 1 the fit is shrunken and above 1 inflated. Read it "
            "beside the IoU, which a systematically over- or under-scaled fit depresses without "
            "saying in which direction. Dimensionless, so the body lengths cancel."
        ),
        orientation="1.0 is correct, either direction is worse",
        view_axis=False,
    ),
    "MPVE_3d_bl": MetricSpec(
        label="MPVE [body lengths]",
        description=(
            "Mean per-vertex Euclidean error of the fused reconstruction in world space, divided "
            "by the per-frame GT body length. Uses exact vertex-index correspondence."
        ),
        orientation="lower is better",
        view_axis=False,
    ),
    # MPJPE_3d_{keypoint,joint}[_root_relative|_pa]_bl[_by_bone_group]: see
    # _mpjpe_registry_entries() below, appended to this dict after it is built.
}


# --------------------------------------------------------------------------
# MPJPE registry entries (generated: see _mpjpe_registry_entries)
# --------------------------------------------------------------------------
#
# synthetic_data_generator_ui.py's _mpjpe_block computes each MPJPE variant under
# three alignment terms -- global (no alignment removed), root_relative
# (translation removed) and pa (a similarity transform removed) -- for each of
# two point sets -- keypoint centroids and bone heads -- and, when the template
# defines bone groups, a further breakdown of each of those by bone group. That
# is 2 point sets x 3 terms x 2 granularities = 12 MetricSpec entries, differing
# only in which point set, which term, and whether it is grouped; they are
# generated here rather than written out by hand so the twelve descriptions stay
# in lockstep with each other and with _mpjpe_block instead of drifting apart
# under independent edits.

# (metric-key infix, prose label, JSON key under 'mpjpe') for each point set.
_MPJPE_ITEMS: Tuple[Tuple[str, str], ...] = (
    ("keypoint", "keypoint centroids"),
    ("joint", "bone heads"),
)

# (metric-key infix, prose label, what the term isolates) for each alignment term,
# in _mpjpe_block's own order.
_MPJPE_TERMS: Tuple[Tuple[str, str, str], ...] = (
    ("", "global", "with no alignment removed"),
    (
        "_root_relative", "root-relative",
        "with each set's own translation removed, isolating orientation and articulation error",
    ),
    (
        "_pa", "Procrustes-aligned",
        "with a similarity transform (Kabsch/Umeyama, scale included) removed, isolating "
        "residual articulation and shape error; null wherever the point configuration is too "
        "close to one dimension for the rotation to be stable",
    ),
)


def _mpjpe_registry_entries() -> Dict[str, MetricSpec]:
    """
    The 12 MPJPE entries: {keypoint centroids, bone heads} x {global,
    root-relative, Procrustes-aligned} x {whole fish, per bone group}.

    Both granularities read a per-frame series -- the whole-fish ones from
    THREE_D_NESTED_SCALARS, the per-bone-group ones one level finer from
    THREE_D_GROUP_SCALARS -- and are otherwise ordinary view_axis=False metrics:
    synthetic_data_generator_ui.py's _mpjpe_block writes each frame's own
    per-group mean beside its whole-block one, so this script pools raw
    per-frame values here exactly like every other metric.
    """
    entries: Dict[str, MetricSpec] = {}
    for item_infix, item_label in _MPJPE_ITEMS:
        for term_infix, term_label, term_text in _MPJPE_TERMS:
            whole_key = f"MPJPE_3d_{item_infix}{term_infix}_bl"
            entries[whole_key] = MetricSpec(
                label=f"MPJPE ({item_label}, {term_label}) [body lengths]",
                description=(
                    f"Mean per-frame Euclidean error over {item_label}, {term_text}, divided by "
                    "the per-frame GT body length."
                ),
                orientation="lower is better",
                view_axis=False,
            )
            entries[f"{whole_key}_by_bone_group"] = MetricSpec(
                label=f"MPJPE per bone group ({item_label}, {term_label}) [body lengths]",
                description=(
                    f"{whole_key} restricted to one bone group's {item_label}, {term_text}: "
                    "per frame, the mean over just that group's items, divided by the same "
                    "frame's GT body length. The bone groups are the template's own partition "
                    "of the skeleton -- the body parts the optimizer schedules its stages on -- "
                    "so this says which part of the fish a view combination failed on, which "
                    f"{whole_key} above cannot. Groups may overlap and are unequally sized, so "
                    f"the measure pooled over all of them is a pool of group values, not "
                    f"{whole_key}."
                ),
                orientation="lower is better",
                view_axis=False,
                member_noun="bone group",
            )
    return entries


METRIC_REGISTRY.update(_mpjpe_registry_entries())


def spec_for(metric: str) -> MetricSpec:
    """Registry entry for a metric, or a neutral placeholder for a new one."""
    return METRIC_REGISTRY.get(
        metric,
        MetricSpec(
            label=metric.replace("_", " "),
            description="Not in METRIC_REGISTRY; units and orientation unknown to this script.",
            orientation="unknown",
        ),
    )


def metric_sort_key(metric: str) -> Tuple[int, str]:
    """Registry order first, unregistered metrics alphabetically after it."""
    known = list(METRIC_REGISTRY)
    return (known.index(metric), "") if metric in known else (len(known), metric)


# --------------------------------------------------------------------------
# Run keys, groupings, defaults
# --------------------------------------------------------------------------

# combo_folder_name() emits 'k{n}__v{i0-i1-...}' with an optional descriptive
# suffix that is dropped when the leaf name would exceed 200 chars, so the view
# count is taken from the prefix only.
RUN_KEY_PATTERN = re.compile(r"^k(\d+)__")

# --------------------------------------------------------------------------
# The 3D metrics in collected_3d_metrics.json
# --------------------------------------------------------------------------
#
# synthetic_data_generator_ui.py's 'Batch 3D Metrics from PTS2 Dir' operator
# writes a second collection beside the one this script's positional argument
# points at:
#
#   <out_root>/metrics_collected/collected_metrics.json   2D, image space
#   <out_root>/collected_3d_metrics.json                  3D, world space
#
# Both are keyed by the same combo_folder_name() leaves, so RUN_KEY_PATTERN,
# parse_n_views() and every grouping below apply to the 3D file unchanged and a
# 3D run lines up one-to-one with its 2D counterpart. The 3D file is picked up
# automatically when it sits at that default location, and its metrics enter the
# same cell table, summary and plots as the 2D ones.
#
# TWO STRUCTURAL DIFFERENCES, both handled explicitly further down:
#
# (1) NO VIEW AXIS. A volumetric IoU between two world-space meshes is one number
#     per (run, frame), not per (run, view, frame): the 3D metrics score the
#     reconstruction itself, after all views have been fused, so there is no view
#     to attribute a value to. They are tagged with the sentinel view VIEW_3D and
#     MetricSpec.view_axis=False, and the figures keyed by view are skipped for
#     them rather than drawn with one meaningless column. Everything that varies
#     with the number of views -- which is the actual question a view-combination
#     sweep asks -- still works.
#
# (2) NESTED FRAME RECORDS. A 2D run is metric -> view -> [value per frame]. A 3D
#     run is two blocks, each {meta, summary, frames}, whose frames[] are lists of
#     dicts keyed by "frame". THREE_D_SCALARS / THREE_D_KEYPOINTS below flatten
#     them into the flat per-frame arrays the rest of this script expects; the
#     precomputed "summary" in the file is deliberately ignored, because this
#     script pools raw frame values across runs and would otherwise be averaging
#     averages over unequal frame counts. The per-bone-group MPJPE entries
#     (THREE_D_GROUP_SCALARS) read one level finer than THREE_D_NESTED_SCALARS but
#     are the same shape of thing: synthetic_data_generator_ui.py's _mpjpe_block
#     writes each frame's OWN per-group mean beside its whole-block one, so this
#     script pools those raw per-frame values exactly like every other metric,
#     never a Blender-side pre-average.
#
# UNITS: every 3D length is read in GT BODY LENGTHS, never metres. A metre depends
# on how large the fish was modelled and is not comparable across scenes, nor with
# the 2D keypoint error, which is already normalised by gt_body_length_px. The IoU
# is a ratio of volumes and is scale free to begin with.

# Mirrors sweep_view_combinations.MAX_LEAF_NAME_LEN, for the same reason: stay
# clear of the 255-byte ext4 filename limit.
MAX_PLOT_NAME_LEN = 200

DEFAULT_COLLECTED_PATH = Path("metrics_collected/collected_metrics.json")
DEFAULT_OUT_DIR = Path("analysis_output")

GROUPING_OVERALL = "overall"
GROUPING_PER_VIEW = "per_view_overall"
GROUPING_PER_VIEW_PER_N = "per_view_per_n_views"
GROUPING_PER_N = "per_n_views"

DISTRIBUTION_POINTS = "points"  # one marker per raw frame value
DISTRIBUTION_BOXES = "boxes"  # one box per colour group instead of the markers

DEFAULT_PRIMARY_COLOR = "#40E0D0"  # turquoise
# Debian's fonts-linuxlibertine installs the family as 'Linux Biolinum O'; the
# bare name is tried first so a differently packaged install also resolves.
DEFAULT_FONT = "Linux Biolinum O,Linux Biolinum"
STRIP_RNG_SEED = 0  # thinning of the point strips is reproducible


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Tidy sample extraction
# --------------------------------------------------------------------------


class Sample(NamedTuple):
    """
    One frame's value of one metric, tagged with everything it groups by. The
    value is kept raw -- missing frames stay NaN and are filtered per group, so
    a group whose frames were all missing still exists and reports NaN.
    """

    metric: str
    keypoint: Optional[str]  # None for the scalar metrics
    view: str
    n_views: int
    value: float


# A measure is one plottable quantity: a scalar metric, a keypoint metric pooled
# over its keypoints (keypoint None), or one single keypoint of it.
MeasureKey = Tuple[str, Optional[str]]
# A cell is the finest grouping; every reported grouping is a union of cells.
CellKey = Tuple[str, int]  # (view, n_views)
CellTable = Dict[MeasureKey, Dict[CellKey, List[float]]]


def parse_n_views(run_key: str) -> Optional[int]:
    """Number of views in a run, from the run key prefix. None if unparseable."""
    match = RUN_KEY_PATTERN.match(run_key)
    return int(match.group(1)) if match else None


def _to_float(value: Any) -> float:
    """Frame value as float; None, bools and junk become NaN, i.e. 'missing'."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def _finite(values: Sequence[Any]) -> List[float]:
    """Drop None and NaN (missing/undefined frames) and cast the rest to float."""
    return [
        float(v)
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]


def classify_metric(metric_data: Any, view_names: Sequence[str]) -> Optional[str]:
    """
    SCALAR, KEYPOINT or None for a top-level metrics entry, decided from its
    structure. The keys must be view names, so a future non-view entry is
    skipped instead of being analysed as though its keys were cameras.
    """
    if not isinstance(metric_data, dict) or not metric_data:
        return None
    if not set(metric_data).intersection(view_names):
        return None
    sample_value = next(iter(metric_data.values()))
    if isinstance(sample_value, list):
        return SCALAR
    if isinstance(sample_value, dict):
        return KEYPOINT
    return None


def blocked_positions(
    run_metrics: Dict[str, Any], run_key: str
) -> Tuple[Optional[frozenset], Dict[str, int]]:
    """(positions to drop, count per reason) for one run's per-frame arrays.

    Returns (None, {}) when the run cannot be filtered -- the metrics file predates
    blocked_frames -- which the caller must treat as 'analyse unfiltered and say so', NOT as
    'nothing was blocked'. An empty frozenset is the different, benign answer: the field is
    present and no frame was blocked.
    """
    raw = run_metrics.get(BLOCKED_FRAMES_KEY)
    if raw is None:
        return None, {}
    if not isinstance(raw, list):
        warn(f"{run_key}: '{BLOCKED_FRAMES_KEY}' is {type(raw).__name__}, not a list; "
             f"treating the run as unfilterable.")
        return None, {}

    positions, by_reason, malformed = set(), {}, 0
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get(BLOCKED_INDEX_FIELD), int):
            malformed += 1
            continue
        positions.add(entry[BLOCKED_INDEX_FIELD])
        reason = str(entry.get("reason_blocked", "unspecified"))
        by_reason[reason] = by_reason.get(reason, 0) + 1
    if malformed:
        warn(f"{run_key}: {malformed} blocked-frame record(s) lack an integer "
             f"'{BLOCKED_INDEX_FIELD}' and were ignored; those frames stay in the statistics.")
    return frozenset(positions), by_reason


def _keep(values: Sequence[Any], drop: Optional[frozenset]) -> Iterator[Any]:
    """Yield the values whose position is not excluded."""
    if not drop:
        yield from values
        return
    for position, value in enumerate(values):
        if position not in drop:
            yield value


def _iter_scalar_samples(
    metric: str, metric_data: Dict[str, Any], n_views: int, run_key: str,
    drop: Optional[frozenset] = None,
) -> Iterator[Sample]:
    for view, values in metric_data.items():
        if not isinstance(values, list):
            warn(f"{run_key}: '{metric}' / '{view}' is not a frame list; skipping it.")
            continue
        view_name = sys.intern(str(view))
        for value in _keep(values, drop):
            yield Sample(metric, None, view_name, n_views, _to_float(value))


def _iter_keypoint_samples(
    metric: str, metric_data: Dict[str, Any], n_views: int, run_key: str,
    drop: Optional[frozenset] = None,
) -> Iterator[Sample]:
    for view, keypoints in metric_data.items():
        if not isinstance(keypoints, dict):
            warn(f"{run_key}: '{metric}' / '{view}' is not a keypoint mapping; skipping it.")
            continue
        view_name = sys.intern(str(view))
        for keypoint, values in keypoints.items():
            if not isinstance(values, list):
                warn(f"{run_key}: '{metric}' / '{view}' / '{keypoint}' is not a frame list; skipping it.")
                continue
            keypoint_name = sys.intern(str(keypoint))
            for value in _keep(values, drop):
                yield Sample(metric, keypoint_name, view_name, n_views, _to_float(value))


def iter_samples(
    collected: Dict[str, Any], wanted_metrics: Optional[Sequence[str]] = None,
    exclude_blocked: bool = True, availability: Optional[Dict[str, Dict[str, int]]] = None,
) -> Iterator[Sample]:
    """
    Stream every run's per-frame values as tidy samples. A run with an
    unparseable key or a malformed payload is warned about and skipped rather
    than aborting the analysis, as is an individual malformed metric entry.

    Blocked frames are dropped by position unless `exclude_blocked` is False; see the
    'Blocked frames' block near the top of this file. `availability`, if given, is filled
    with the per-run frame accounting rho is computed from -- excluded frames stay counted
    as available there, which is the whole point of tracking it separately from the samples.
    """
    n_runs = 0
    n_dropped = 0
    reasons: Dict[str, int] = {}
    unfilterable = []
    for run_key, run_metrics in collected.items():
        n_views = parse_n_views(run_key)
        if n_views is None:
            warn(f"{run_key}: run key does not start with 'k<N>__'; skipping run.")
            continue
        if not isinstance(run_metrics, dict):
            warn(f"{run_key}: metrics payload is not an object; skipping run.")
            continue

        reference = run_metrics.get(REFERENCE_METRIC)
        view_names = list(reference) if isinstance(reference, dict) else []
        if not view_names:
            warn(f"{run_key}: no '{REFERENCE_METRIC}' to establish the view set; skipping run.")
            continue

        n_runs += 1

        # The run's frame count, from the reference metric: every per-view array is the same
        # length by construction, and this is the denominator rho is built on.
        n_available = max(
            (len(v) for v in reference.values() if isinstance(v, list)), default=0
        )

        drop: Optional[frozenset] = frozenset()
        run_reasons: Dict[str, int] = {}
        if exclude_blocked:
            drop, run_reasons = blocked_positions(run_metrics, run_key)
            if drop is None:
                # Predates blocked_frames. Analysed unfiltered, named in one summary warning
                # after the loop rather than once per metric.
                unfilterable.append(run_key)
                drop = frozenset()
            else:
                n_dropped += len(drop)
                for reason, count in run_reasons.items():
                    reasons[reason] = reasons.get(reason, 0) + count

        if availability is not None:
            availability[run_key] = {
                "n_available": int(n_available),
                "n_blocked_excluded": int(len(drop)),
                "n_used": int(max(n_available - len(drop), 0)),
                "blocked_filterable": run_key not in unfilterable,
            }

        for metric in sorted(set(run_metrics) - NON_METRIC_KEYS, key=metric_sort_key):
            if wanted_metrics is not None and metric not in wanted_metrics:
                continue
            kind = classify_metric(run_metrics[metric], view_names)
            if kind is None:
                warn(f"{run_key}: '{metric}' is not a per-view frame series; skipping it.")
                continue
            metric_name = sys.intern(metric)
            if kind == SCALAR:
                yield from _iter_scalar_samples(metric_name, run_metrics[metric], n_views,
                                                run_key, drop)
            else:
                yield from _iter_keypoint_samples(metric_name, run_metrics[metric], n_views,
                                                  run_key, drop)

    log(f"Read {n_runs} run(s).")
    if exclude_blocked and n_dropped:
        log(f"Excluded {n_dropped} blocked frame position(s) across {n_runs} run(s) "
            f"({', '.join(f'{k}: {v}' for k, v in sorted(reasons.items()))}).")
    if unfilterable:
        warn(
            f"{len(unfilterable)} run(s) have no '{BLOCKED_FRAMES_KEY}' field, so blocked "
            f"frames CANNOT be located in their per-frame arrays and are included in the "
            f"statistics: {', '.join(unfilterable[:3])}"
            f"{' ...' if len(unfilterable) > 3 else ''}. Re-run the reconstruction to record "
            f"it, or pass --no-exclude-blocked to make every run's treatment uniform."
        )


# --------------------------------------------------------------------------
# 3D sample extraction (collected_3d_metrics.json)
# --------------------------------------------------------------------------

# Sentinel view for the metrics that have no view axis (see the block above).
# Short and not a legal camera stem, so it can never collide with a real view.
VIEW_3D = "3d"

# Default location relative to the 2D file: collect_results() writes
# metrics_collected/collected_metrics.json, the batch operator writes
# collected_3d_metrics.json one level up from it.
DEFAULT_3D_NAME = "collected_3d_metrics.json"


@dataclass(frozen=True)
class ThreeDScalar:
    """One scalar metric read out of a per-frame record of a 3D block.

    `block` is the top-level key of the run entry, `field` the per-frame key, and
    `divide_by` an optional second field the value is divided by, which is how the
    volume ratio is derived without the writer having to store it.
    """

    block: str
    field: str
    divide_by: Optional[str] = None


# Adding a metric that is already in the file is one line here plus a
# METRIC_REGISTRY entry. The body-length-cubed volumes (vol_gt_bl3,
# vol_recon_bl3, vol_intersection_bl3, vol_union_bl3) are written by the operator
# and can be surfaced the same way; they are left out by default because the IoU
# and the ratio below already answer what they would be read for.
THREE_D_SCALARS: Dict[str, ThreeDScalar] = {
    "IoU_3d_volumetric": ThreeDScalar("volumetric_iou", "iou"),
    "volume_ratio_3d_recon_over_gt": ThreeDScalar(
        "volumetric_iou", "vol_recon", divide_by="vol_gt"
    ),
    "body_length_3d_m": ThreeDScalar("volumetric_iou", "body_length_m"),
}

# metric -> (block, per-frame field holding {keypoint: value}). '_bl' by
# construction: metres are in the file too but must not be plotted, since they are
# not comparable across scenes.
THREE_D_KEYPOINTS: Dict[str, Tuple[str, str]] = {
    "keypoint_distance_3d_bl": ("keypoint_distances", "per_keypoint_bl"),
}

# Batch MPVE/MPJPE are stored using the same nested metric blocks as the
# standalone evaluators. These sources flatten the relevant per-frame scalar
# into the same Sample shape used by the rest of the analyzer. The batch
# operator adds body_length_normalised to every frame of both MPJPE blocks with
# all three alignment terms (global/root_relative/pa), so each of the six
# whole-fish MPJPE entries in _mpjpe_registry_entries() reads one of them here.
# Path format:
#   (metric block, optional sub-block, frame field path...)
THREE_D_NESTED_SCALARS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "MPVE_3d_bl": (
        ("mpve",),
        ("variants", "body_length_normalised", "mean"),
    ),
    **{
        f"MPJPE_3d_{item_infix}{term_infix}_bl": (
            ("mpjpe", item_infix),
            ("body_length_normalised", term),
        )
        for item_infix, _item_label in _MPJPE_ITEMS
        for term_infix, term in (("", "global"), ("_root_relative", "root_relative"), ("_pa", "pa"))
    },
}

# metric -> (item kind, alignment term) of the per-bone-group MPJPE entries. The
# item kind ("keypoint" or "joint") and term ("global"/"root_relative"/"pa")
# together address one frame of synthetic_data_generator_ui.py's
# run["mpjpe"][item_kind]["frames"][i]["body_length_normalised"]["per_group"][term],
# one level finer than THREE_D_NESTED_SCALARS reads -- _mpjpe_block writes this
# frame-wise, beside the frame's whole-block value, so it is read the same way:
# one Sample per (run, frame, group), pooled here rather than inside Blender. The
# group name enters the Sample on the KEYPOINT axis, so every by-keypoint
# grouping, summary, CSV row and figure applies to a bone group unchanged;
# MetricSpec.member_noun is what keeps the titles from calling it a keypoint.
THREE_D_GROUP_SCALARS: Dict[str, Tuple[str, str]] = {
    f"MPJPE_3d_{item_infix}{term_infix}_bl_by_bone_group": (item_infix, term)
    for item_infix, _item_label in _MPJPE_ITEMS
    for term_infix, term in (("", "global"), ("_root_relative", "root_relative"), ("_pa", "pa"))
}


def _nested_value(mapping: Any, path: Sequence[str]) -> Any:
    """Read a nested dictionary field, returning None for malformed/missing data."""
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _3d_frames(run: Dict[str, Any], block: str, run_key: str) -> List[Dict[str, Any]]:
    """The frames[] of one block of one run, or [] with a warning if unusable."""
    payload = run.get(block)
    if payload is None:
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        warn(f"{run_key}: 3D block '{block}' has no frames list; skipping it.")
        return []
    return [f for f in payload["frames"] if isinstance(f, dict)]


def _ratio(numerator: Any, denominator: Any) -> float:
    """numerator / denominator as a float, NaN on anything non-finite or zero."""
    num, den = _to_float(numerator), _to_float(denominator)
    if math.isnan(num) or math.isnan(den) or den == 0.0:
        return float("nan")
    value = num / den
    return value if math.isfinite(value) else float("nan")


def _3d_blocked(frame: Dict[str, Any]) -> bool:
    """Whether one 3D per-frame record is flagged blocked by the Blender evaluator."""
    return bool(frame.get("blocked"))


def iter_3d_samples(
    collected: Dict[str, Any], wanted_metrics: Optional[Sequence[str]] = None,
    exclude_blocked: bool = True,
) -> Iterator[Sample]:
    """Stream collected_3d_metrics.json as tidy samples on the sentinel view.

    Frames the operator could not score are present in the file as explicit nulls
    (a degenerate mesh, a zero-volume occupancy, a frame whose L_body was
    unmeasurable). They are emitted as NaN rather than dropped, so that a run's
    frame count stays honest and _finite() filters them exactly like a missing 2D
    frame.
    """
    n_runs = 0
    n_blocked = 0
    no_stamp = []
    group_missing: Dict[str, List[str]] = defaultdict(list)
    group_present: Dict[str, int] = defaultdict(int)
    group_partial: Dict[str, List[str]] = defaultdict(list)
    for run_key, run in collected.items():
        n_views = parse_n_views(run_key)
        if n_views is None:
            warn(f"{run_key}: 3D run key does not start with 'k<N>__'; skipping run.")
            continue
        if not isinstance(run, dict):
            warn(f"{run_key}: 3D payload is not an object; skipping run.")
            continue
        n_runs += 1

        # The batch operator SCORES every frame and flags the blocked ones, so the filtering
        # happens here, per row, exactly as it does for the 2D metrics -- one policy, one flag,
        # applied at one place. The run-level record is read only for the accounting and to
        # notice a run whose provenance is unknown.
        marker = run.get("blocked_frames")
        if isinstance(marker, dict):
            n_blocked += len(marker.get("records") or [])
            if not marker.get("stamp_present", True):
                no_stamp.append(run_key)
        elif marker is None:
            no_stamp.append(run_key)

        for metric, source in THREE_D_SCALARS.items():
            if wanted_metrics is not None and metric not in wanted_metrics:
                continue
            metric_name = sys.intern(metric)
            for frame in _3d_frames(run, source.block, run_key):
                if exclude_blocked and _3d_blocked(frame):
                    continue
                if source.divide_by is None:
                    value = _to_float(frame.get(source.field))
                else:
                    value = _ratio(frame.get(source.field), frame.get(source.divide_by))
                yield Sample(metric_name, None, VIEW_3D, n_views, value)

        for metric, (block, frame_field) in THREE_D_KEYPOINTS.items():
            if wanted_metrics is not None and metric not in wanted_metrics:
                continue
            metric_name = sys.intern(metric)
            for frame in _3d_frames(run, block, run_key):
                if exclude_blocked and _3d_blocked(frame):
                    continue
                per_keypoint = frame.get(frame_field)
                if not isinstance(per_keypoint, dict):
                    continue
                for keypoint, value in per_keypoint.items():
                    yield Sample(
                        metric_name, sys.intern(str(keypoint)), VIEW_3D, n_views,
                        _to_float(value),
                    )
        
        for metric, (block_path, field_path) in THREE_D_NESTED_SCALARS.items():
            if wanted_metrics is not None and metric not in wanted_metrics:
                continue
            metric_name = sys.intern(metric)

            payload = run
            for key in block_path:
                payload = payload.get(key) if isinstance(payload, dict) else None
            if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
                # MPJPE's keypoint/joint block may legitimately be absent (for
                # example when the configured keypoint list is empty), so this
                # is a quiet absence rather than a malformed-file warning.
                continue

            for frame in payload["frames"]:
                if not isinstance(frame, dict):
                    continue
                if exclude_blocked and _3d_blocked(frame):
                    continue
                value = _nested_value(frame, field_path)
                yield Sample(metric_name, None, VIEW_3D, n_views, _to_float(value))

        for metric, (item_kind, term) in THREE_D_GROUP_SCALARS.items():
            if wanted_metrics is not None and metric not in wanted_metrics:
                continue
            metric_name = sys.intern(metric)

            payload = run.get("mpjpe")
            payload = payload.get(item_kind) if isinstance(payload, dict) else None
            if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
                # MPJPE's keypoint/joint block may legitimately be absent (for
                # example when the configured keypoint list is empty), so this
                # is a quiet absence rather than a malformed-file warning, same
                # as THREE_D_NESTED_SCALARS treats it.
                continue

            matched = missing = 0
            for frame in payload["frames"]:
                if not isinstance(frame, dict):
                    continue
                if exclude_blocked and _3d_blocked(frame):
                    continue
                per_group = _nested_value(frame, ("body_length_normalised", "per_group", term))
                if not isinstance(per_group, dict):
                    # A generator that predates the per-bone-group patch, or a
                    # template with no bone groups defined, writes no per_group
                    # table on this frame: absent, not a fault.
                    missing += 1
                    continue
                matched += 1
                for group, value in per_group.items():
                    yield Sample(metric_name, sys.intern(str(group)), VIEW_3D, n_views,
                                 _to_float(value))

            if matched:
                group_present[metric] += 1
                if missing:
                    group_partial[metric].append(
                        f"{run_key} ({missing} of {matched + missing} frame(s))"
                    )
            elif missing:
                group_missing[metric].append(run_key)

    if collected:
        log(f"Read {n_runs} 3D run(s).")
        if n_blocked:
            log(f"{n_blocked} blocked frame(s) recorded across the 3D runs"
                f"{' and excluded' if exclude_blocked else ', all included'}.")
        if no_stamp:
            warn(
                f"{len(no_stamp)} 3D run(s) carry no blocked-frame record, so it is not known "
                f"whether any of their frames were blocked: {', '.join(no_stamp[:3])}"
                f"{' ...' if len(no_stamp) > 3 else ''}. Their per-frame rows are used as-is, "
                f"which may pool unfiltered 3D values beside filtered 2D ones."
            )
        for metric, missing in group_missing.items():
            present = group_present[metric]
            if present and missing:
                warn(
                    f"{metric}: {len(missing)} of {present + len(missing)} 3D run(s) carry no "
                    f"per-bone-group table on any frame, e.g. {missing[0]}; only the runs that "
                    "have one contribute to it."
                )
        for metric, partial in group_partial.items():
            warn(
                f"{metric}: {len(partial)} run(s) carry a per-bone-group table on only some of "
                f"their frames, e.g. {partial[0]}; the frames that have one are pooled as they "
                "are, which may rest on fewer frames than the run's whole-fish MPJPE does."
            )


def build_cell_table(samples: Iterator[Sample]) -> CellTable:
    """
    Pivot the tidy sample stream into measure -> (view, #views) -> values in one
    pass. Nothing rescans the raw JSON afterwards, and the per-frame values stay
    available for the distribution plots.
    """
    cells: Dict[MeasureKey, Dict[CellKey, List[float]]] = defaultdict(lambda: defaultdict(list))
    n_values = n_missing = 0

    for sample in samples:
        n_values += 1
        if math.isnan(sample.value):
            n_missing += 1
        cell: CellKey = (sample.view, sample.n_views)
        cells[(sample.metric, None)][cell].append(sample.value)
        if sample.keypoint is not None:
            # A keypoint sample feeds both its own measure and the pooled one.
            cells[(sample.metric, sample.keypoint)][cell].append(sample.value)

    log(
        f"Tabulated {n_values} frame value(s) into {len(cells)} measure(s); "
        f"{n_missing} of them missing (NaN/null)."
    )
    return {measure: dict(table) for measure, table in cells.items()}


# --------------------------------------------------------------------------
# Cell-table accessors
# --------------------------------------------------------------------------


def views_of(table: Dict[CellKey, List[float]]) -> List[str]:
    """The views this measure was observed in, sorted by name."""
    return sorted({view for view, _k in table})


def n_views_of(table: Dict[CellKey, List[float]]) -> List[int]:
    """The combination sizes this measure was observed at, ascending."""
    return sorted({k for _view, k in table})


def pooled(
    table: Dict[CellKey, List[float]],
    view: Optional[str] = None,
    n_views: Optional[int] = None,
) -> List[float]:
    """Finite values of every cell matching the given constraints."""
    out: List[float] = []
    for (cell_view, cell_k), values in table.items():
        if view is not None and cell_view != view:
            continue
        if n_views is not None and cell_k != n_views:
            continue
        out.extend(_finite(values))
    return out


def keypoints_of(cells: CellTable, metric: str) -> List[str]:
    """The keypoints of a keypoint metric; empty for a scalar metric."""
    return sorted(kp for (m, kp) in cells if m == metric and kp is not None)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _mean_median(values: Sequence[Any], metric_name: str) -> Dict[str, float]:
    """
    Mirrors DSKv2_demo._mean_median: missing frames are filtered out, never
    coerced to zero. Deviation: a group with no finite values yields NaN
    instead of raising, because sweep groupings are sparse by construction.
    """
    finite_values = _finite(values)
    if not finite_values:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(mean(finite_values)), "median": float(median(finite_values))}


def _stats(values: Sequence[Any], metric_name: str) -> Dict[str, float]:
    """
    Location, spread and support of one group. q1/q3 are the inclusive-method
    quartiles, i.e. exactly the box drawn by the distribution plots; sd is the
    sample standard deviation and stays NaN for a single observation.
    """
    finite_values = _finite(values)
    nan = float("nan")
    summary: Dict[str, float] = {
        "n_samples": len(finite_values),
        **_mean_median(finite_values, metric_name),
        "sd": nan,
        "q1": nan,
        "q3": nan,
        "iqr": nan,
        "min": nan,
        "max": nan,
    }
    if not finite_values:
        return summary
    summary["min"] = float(min(finite_values))
    summary["max"] = float(max(finite_values))
    if len(finite_values) >= 2:
        summary["sd"] = float(stdev(finite_values))
        q1, _q2, q3 = quantiles(finite_values, n=4, method="inclusive")
        summary.update({"q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)})
    return summary


def summarize_measure(table: Dict[CellKey, List[float]], metric: str) -> Dict[str, Any]:
    """The four groupings of one measure, all pooled from the raw frame values."""
    views = views_of(table)
    n_views_values = n_views_of(table)
    return {
        GROUPING_OVERALL: _stats(pooled(table), metric),
        GROUPING_PER_VIEW: {view: _stats(pooled(table, view=view), metric) for view in views},
        GROUPING_PER_VIEW_PER_N: {
            view: {
                str(k): _stats(table[(view, k)], metric)
                for k in n_views_values
                if (view, k) in table
            }
            for view in views
        },
        GROUPING_PER_N: {
            str(k): _stats(pooled(table, n_views=k), metric) for k in n_views_values
        },
    }


def summarize(cells: CellTable) -> Dict[str, Any]:
    """Full nested summary: metric -> groupings, plus per-keypoint sub-blocks."""
    metrics = sorted({metric for metric, _kp in cells}, key=metric_sort_key)
    summary: Dict[str, Any] = {}
    for metric in metrics:
        spec = spec_for(metric)
        block: Dict[str, Any] = {
            "label": spec.label,
            "description": spec.description,
            "orientation": spec.orientation,
            "combination_invariant": spec.combination_invariant,
            **summarize_measure(cells[(metric, None)], metric),
        }
        keypoints = keypoints_of(cells, metric)
        if keypoints:
            block["by_keypoint"] = {
                keypoint: summarize_measure(cells[(metric, keypoint)], metric)
                for keypoint in keypoints
            }
        summary[metric] = block
    return summary


def coverage_report(availability: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """Frame accounting and the coverage ratio rho, overall and per run.

    rho = n_used / n_available. Blocked frames stay in the DENOMINATOR: they were requested
    and the pipeline produced a row for them, so rho keeps meaning 'of everything this run set
    out to measure, how much reached the analysis'. Were they dropped from both terms instead,
    every run would report rho = 1.0 and a run that had to interpolate half its frames would be
    indistinguishable from one that fitted all of them -- which is exactly the difference
    between view combinations that this sweep exists to measure.
    """
    if not availability:
        return {}
    available = sum(r["n_available"] for r in availability.values())
    used = sum(r["n_used"] for r in availability.values())
    blocked = sum(r["n_blocked_excluded"] for r in availability.values())
    unfilterable = sorted(k for k, r in availability.items() if not r["blocked_filterable"])
    return {
        "definition": "coverage_rho = n_used / n_available; blocked frames stay in n_available",
        "n_available": available,
        "n_used": used,
        "n_blocked_excluded": blocked,
        "coverage_rho": (used / available) if available else None,
        "n_runs_without_blocked_field": len(unfilterable),
        "runs_without_blocked_field": unfilterable,
        "per_run": {
            key: {**row, "coverage_rho": (row["n_used"] / row["n_available"])
                  if row["n_available"] else None}
            for key, row in sorted(availability.items())
        },
    }


def build_report(summary: Dict[str, Any], cells: CellTable, source: Path) -> Dict[str, Any]:
    """Summary plus the provenance needed to read it without the source file."""
    # meta['views'] must describe the sweep's cameras, so the reference table is
    # taken from a metric that actually has a view axis. Falling back to the first
    # metric keeps a 3D-only run working: its view set is then [VIEW_3D], which is
    # the honest answer rather than a fabricated camera list.
    reference_metric = next(
        (m for m in summary if spec_for(m).view_axis), next(iter(summary))
    )
    reference_table = cells[(reference_metric, None)]
    view_less = sorted(m for m in summary if not spec_for(m).view_axis)
    return {
        "meta": {
            "source": str(source),
            "views": views_of(reference_table),
            "n_views_values": n_views_of(reference_table),
            "metrics": list(summary),
            "keypoints": sorted({kp for (_m, kp) in cells if kp is not None}),
            "groupings": [GROUPING_OVERALL, GROUPING_PER_VIEW, GROUPING_PER_VIEW_PER_N, GROUPING_PER_N],
            "nan_policy": (
                "missing/undefined frames are excluded from every statistic, never zero-filled"
            ),
            "blocked_frame_policy": (
                "excluded: frames the reconstruction did not obtain by fitting the optimizer "
                "to that frame are dropped from every statistic, but still count as available "
                "in coverage_rho"
                if EXCLUDE_BLOCKED[0] else
                "included: every frame is aggregated, blocked or not"
            ),
            "view_less_metrics": view_less,
            "view_less_note": (
                f"these are measured on the fused 3D reconstruction and have one value per "
                f"(run, frame), not per view; they carry the sentinel view '{VIEW_3D}' and their "
                f"'{GROUPING_PER_VIEW}' / '{GROUPING_PER_VIEW_PER_N}' groupings are that single "
                f"sentinel, not a comparison across cameras"
            ),
        },
        "metrics": summary,
    }


# --------------------------------------------------------------------------
# Numeric output
# --------------------------------------------------------------------------

CSV_COLUMNS = [
    "metric",
    "keypoint",
    "grouping",
    "view",
    "n_views",
    "n_samples",
    "mean",
    "median",
    "sd",
    "q1",
    "q3",
    "iqr",
    "min",
    "max",
    "orientation",
]


def _rows_for_measure(
    metric: str, keypoint: str, block: Dict[str, Any], orientation: str
) -> Iterator[Dict[str, Any]]:
    common = {"metric": metric, "keypoint": keypoint, "orientation": orientation}
    yield {
        **common,
        "grouping": GROUPING_OVERALL,
        "view": "",
        "n_views": "",
        **block[GROUPING_OVERALL],
    }
    for view, stats in block[GROUPING_PER_VIEW].items():
        yield {**common, "grouping": GROUPING_PER_VIEW, "view": view, "n_views": "", **stats}
    for view, by_n in block[GROUPING_PER_VIEW_PER_N].items():
        for n_views, stats in by_n.items():
            yield {
                **common,
                "grouping": GROUPING_PER_VIEW_PER_N,
                "view": view,
                "n_views": n_views,
                **stats,
            }
    for n_views, stats in block[GROUPING_PER_N].items():
        yield {**common, "grouping": GROUPING_PER_N, "view": "", "n_views": n_views, **stats}


def summary_to_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the nested summary into one row per (measure, grouping, group)."""
    rows: List[Dict[str, Any]] = []
    for metric, block in summary.items():
        orientation = block["orientation"]
        rows.extend(_rows_for_measure(metric, "", block, orientation))
        for keypoint, kp_block in block.get("by_keypoint", {}).items():
            rows.extend(_rows_for_measure(metric, keypoint, kp_block, orientation))
    return rows


def write_summary(report: Dict[str, Any], out_dir: Path) -> Tuple[Path, Path]:
    """Write metrics_summary.json and its flat metrics_summary.csv counterpart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics_summary.json"
    csv_path = out_dir / "metrics_summary.csv"

    with json_path.open("w") as fp:
        json.dump(report, fp, indent=2)

    rows = summary_to_rows(report["metrics"])
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Wrote {json_path}")
    log(f"Wrote {csv_path} ({len(rows)} rows)")
    return json_path, csv_path


# --------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------

RGB = Tuple[float, float, float]


def parse_color(spec: str) -> RGB:
    """'#40E0D0', '40E0D0', '64,224,208' (0-255) or '0.25,0.88,0.82' (0-1) -> RGB."""
    text = spec.strip()
    hex_text = text[1:] if text.startswith("#") else text
    if re.fullmatch(r"[0-9A-Fa-f]{6}", hex_text):
        return (
            int(hex_text[0:2], 16) / 255.0,
            int(hex_text[2:4], 16) / 255.0,
            int(hex_text[4:6], 16) / 255.0,
        )

    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    if len(parts) == 3:
        try:
            values = [float(p) for p in parts]
        except ValueError:
            raise SystemExit(f"--primary-color: cannot parse {spec!r} as three numbers.")
        if any(v < 0 for v in values):
            raise SystemExit(f"--primary-color: negative component in {spec!r}.")
        if max(values) > 255.0:
            raise SystemExit(f"--primary-color: component above 255 in {spec!r}.")
        if max(values) > 1.0:
            values = [v / 255.0 for v in values]
        return (values[0], values[1], values[2])

    raise SystemExit(
        f"--primary-color: expected '#RRGGBB' or 'R,G,B' (0-255 or 0-1), got {spec!r}."
    )


def _with_lightness(rgb: RGB, lightness: float, saturation_scale: float = 1.0) -> RGB:
    """Same hue, prescribed lightness -- the basis of the sequential palette."""
    hue, _light, sat = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(hue, min(max(lightness, 0.0), 1.0), min(1.0, sat * saturation_scale))


def tint(rgb: RGB, amount: float) -> RGB:
    """Blend towards white; `amount` 0 leaves the colour, 1 gives white."""
    return (
        rgb[0] + (1.0 - rgb[0]) * amount,
        rgb[1] + (1.0 - rgb[1]) * amount,
        rgb[2] + (1.0 - rgb[2]) * amount,
    )


def sequential_palette(primary: RGB, n: int) -> List[RGB]:
    """
    Light-to-dark ramp of the primary hue for the *ordinal* factor (#views): an
    ordered quantity deserves an ordered colour scale.
    """
    if n <= 1:
        return [_with_lightness(primary, 0.45)]
    return [
        _with_lightness(primary, 0.68 - 0.42 * (i / (n - 1)), 0.85 + 0.30 * (i / (n - 1)))
        for i in range(n)
    ]


# Okabe & Ito's colour-blind-safe qualitative set, the usual choice for nominal
# factors in print. Used verbatim rather than as a hue rotation of the primary:
# evenly spaced hues are equally spaced in degrees, not in perceived difference.
OKABE_ITO = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


def _hue_distance(a: RGB, b: RGB) -> float:
    delta = abs(colorsys.rgb_to_hls(*a)[0] - colorsys.rgb_to_hls(*b)[0])
    return min(delta, 1.0 - delta)


def categorical_palette(primary: RGB, n: int) -> List[RGB]:
    """
    The Okabe-Ito set, for the *nominal* factor (view identity), which has no
    order to encode. The primary is deliberately not part of it -- it stays the
    colour of the pooled summary the points are drawn against -- and entries too
    close to it in hue are dropped so the two never read as the same series.
    Beyond the set, darkened repeats are appended.
    """
    colors: List[RGB] = [
        parse_color(code)
        for code in OKABE_ITO
        if _hue_distance(parse_color(code), primary) > 0.045
    ]
    base = list(colors)
    round_index = 0
    while len(colors) < n:
        round_index += 1
        colors.extend(_with_lightness(c, max(0.20, 0.60 - 0.15 * round_index)) for c in base)
    return colors[:n]


def resolve_font(spec: str) -> Optional[str]:
    """
    Resolve --font to a registered family name. Accepts a path to a font file or
    a comma-separated list of family names tried in order. Returns None, with a
    warning, when nothing matches, leaving matplotlib's default in place.
    """
    candidates = [c.strip() for c in spec.split(",") if c.strip()]
    installed = {f.name for f in font_manager.fontManager.ttflist}

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.suffix.lower() in {".ttf", ".otf", ".ttc"} and path.is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
        if candidate in installed:
            return candidate
        candidate_lower = candidate.lower()
        exact_casefold = next((name for name in installed if name.lower() == candidate_lower), None)
        if exact_casefold:
            return exact_casefold

    # Not in matplotlib's cache: either a font installed after the cache was
    # built, or a family whose packaged name carries a suffix, as Debian's
    # fonts-linuxlibertine does with 'Linux Biolinum O'. One scan of the system
    # fonts covers every candidate; it only runs when the cache has missed.
    system_fonts: List[Tuple[str, str]] = []
    for font_path in font_manager.findSystemFonts():
        try:
            system_fonts.append((font_manager.FontProperties(fname=font_path).get_name(), font_path))
        except (RuntimeError, OSError):
            continue
    for candidate in candidates:
        needle = candidate.lower()
        for name, font_path in system_fonts:
            if name.lower() == needle or name.lower().startswith(needle):
                font_manager.fontManager.addfont(font_path)
                return name

    warn(f"font {spec!r} not found; falling back to matplotlib's default family.")
    return None


@dataclass
class Style:
    """Everything the figures take from the CLI, resolved once."""

    primary: RGB
    font_family: Optional[str]
    dpi: int
    fmt: str
    max_points: int
    # How the raw distribution beside each summary box is rendered: one point
    # per frame value, or one box per colour group.
    distribution_style: str = DISTRIBUTION_POINTS
    # Draw values outside their group's 1.5 x IQR fence? Affects the figures
    # only; metrics_summary.json / .csv always cover every finite value.
    drop_fliers: bool = False
    # Keep the single 'all runs pooled' box of the GROUPING_OVERALL figures? When
    # False their colour groups are laid out on their own x axis instead, as in
    # the per-#views and per-view figures.
    overall_pooling: bool = True
    # Break the y axis of a box figure when one group sits far outside the rest,
    # instead of letting it compress every other box into a few pixels.
    dynamic_y_axis: bool = False
    value_labels: bool = True
    captions: bool = True
    monochrome: bool = False
    box_line_width: float = 1.2
    mean_marker: str = "x"
    format_pixels: Optional[Tuple[int, int]] = None
    scientific: bool = False
    metric_is_3d: bool = False
    _palettes: Dict[str, List[RGB]] = field(default_factory=dict)

    def apply(self) -> None:
        """
        Install a plain, print-oriented rcParams set: no top or right spine,
        outward ticks, a faint grid behind the data, unframed legends.
        """
        plt.rcParams.update(
            {
                "figure.dpi": self.dpi,
                "savefig.dpi": self.dpi,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
                "font.size": 9.0,
                "axes.titlesize": 9.5,
                "axes.labelsize": 9.0,
                "xtick.labelsize": 7.5,
                "ytick.labelsize": 7.5,
                "legend.fontsize": 7.0,
                "legend.title_fontsize": 7.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": "#777777",
                "axes.labelcolor": "black",
                "font.style": "normal",
                "font.weight": "normal",
                "axes.linewidth": 0.8,
                "axes.axisbelow": True,
                "axes.titlelocation": "left",
                "axes.titlepad": 7.0,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.color": "#777777",
                "ytick.color": "#777777",
                "xtick.labelcolor": "black",
                "ytick.labelcolor": "black",
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "xtick.major.size": 3.0,
                "ytick.major.size": 3.0,
                "ytick.minor.size": 1.8,
                "grid.color": "#9A9A9A",
                "grid.linewidth": 0.5,
                "grid.alpha": 0.30,
                "legend.frameon": False,
                "legend.handlelength": 1.1,
                "legend.borderaxespad": 0.0,
                "lines.linewidth": 1.2,
                "lines.markersize": 4.0,
            }
        )
        if self.scientific:
            plt.rcParams.update(
                {
                    "font.size": 7.5,
                    "axes.titlesize": 8.0,
                    "axes.labelsize": 7.5,
                    "xtick.labelsize": 6.5,
                    "ytick.labelsize": 6.5,
                    "legend.fontsize": 6.5,
                    "legend.title_fontsize": 7.0,
                    "axes.linewidth": 0.7,
                    "xtick.major.width": 0.7,
                    "ytick.major.width": 0.7,
                    "xtick.major.size": 2.5,
                    "ytick.major.size": 2.5,
                    "grid.linewidth": 0.35,
                    "grid.alpha": 0.18,
                    "figure.facecolor": "white",
                    "axes.facecolor": "white",
                    "savefig.facecolor": "white",
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                    "svg.fonttype": "none",
                }
            )
        if self.font_family:
            plt.rcParams["font.family"] = [self.font_family]

    def palette(self, kind: str, n: int) -> List[RGB]:
        """Cached palette; `kind` is 'n_views' (ordinal) or 'view' (nominal)."""
        key = f"{kind}:{n}"
        if key not in self._palettes:
            if self.monochrome:
                self._palettes[key] = [self.primary] * n
            else:
                builder = sequential_palette if kind == "n_views" else categorical_palette
                self._palettes[key] = builder(self.primary, n)
        return self._palettes[key]

    def figure_size(self, default: Tuple[float, float]) -> Tuple[float, float]:
        """Return figure size in inches; --format, when set, is width/height in pixels."""
        if self.format_pixels is None:
            return default
        width_px, height_px = self.format_pixels
        return width_px / self.dpi, height_px / self.dpi


# --------------------------------------------------------------------------
# Figure primitives
# --------------------------------------------------------------------------

# Point strips sit beside their box rather than on top of it, so that neither
# hides the other; within a strip each colour group gets its own sub-band.
BOX_WIDTH = 0.30
STRIP_OFFSET = 0.28
STRIP_WIDTH = 0.34


def slugify(name: str) -> str:
    """Filename-safe token, matching sweep_view_combinations.sanitize()."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return token or "unnamed"


VIEW_LABEL_WIDTH = 17  # characters per line of a wrapped view name


def _view_label(view: str, n_samples: Optional[int] = None) -> str:
    """
    View name wrapped to a fixed width so the ticks can stay horizontal, which
    reads better than rotated labels, optionally carrying the group's support.
    """
    label = textwrap.fill(view, VIEW_LABEL_WIDTH)
    return label if n_samples is None else f"{label}\nn = {n_samples}"


def _axis_cosmetics(ax: AxesLike, ylabel: str, xlabel: str = "") -> None:
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    for panel in _panels_of(ax):
        panel.yaxis.set_minor_locator(AutoMinorLocator(2))
        panel.tick_params(axis="both", which="both", colors="#777777", labelcolor="black")
        panel.grid(axis="y", which="major")


def _figure_width(n_positions: int, per_position: float, minimum: float) -> float:
    return max(minimum, per_position * n_positions)


def _label_headroom(ax: AxesLike, style: Style, fraction: float = 0.14, below: float = 0.0) -> None:
    """
    Room around the drawn data for the mean/median labels and the title. Each
    panel of a broken axis gets its own, since each carries labels of its own;
    the lower panel's headroom stops at the upper panel's floor, so widening it
    can narrow the break but can never make the two panels show the same values.
    """
    if not style.value_labels:
        return
    panels = _panels_of(ax)  # upper panel first
    ceiling: Optional[float] = None
    for index, panel in enumerate(panels):
        low, high = panel.get_ylim()
        span = high - low
        new_high = high + fraction * span
        if ceiling is not None:
            new_high = min(new_high, ceiling)
        # `below` is room under the figure's data, so only the lowest panel gets it.
        new_low = low - (below * span if index == len(panels) - 1 else 0.0)
        panel.set_ylim(new_low, new_high)
        ceiling = new_low


CAPTION_FONT_SIZE = 6.5
CAPTION_CHARS_PER_INCH = 21  # at CAPTION_FONT_SIZE, close enough for wrapping


def _save(
    fig: Figure, ax: AxesLike, out_path: Path, title: str, caption: str, style: Style
) -> None:
    """
    Title above the axes, caption beneath the figure, as in a paper. The caption
    is anchored to the drawn extent of the figure rather than to the axes box,
    so it clears tick labels and legends whatever their size, and is hard-wrapped
    to that extent instead of relying on matplotlib's word wrapping.
    """
    ax.set_title(title)
    if not style.captions:
        fig.savefig(
            out_path,
            format=style.fmt,
            bbox_inches=None if style.format_pixels is not None else "tight",
        )
        plt.close(fig)
        return
    fig.canvas.draw()
    extent = fig.get_tightbbox(fig.canvas.get_renderer())
    wrapped = "\n".join(
        textwrap.fill(line, max(int(extent.width * CAPTION_CHARS_PER_INCH), 40))
        for line in caption.split("\n")
    )
    fig.text(
        extent.x0 / fig.get_figwidth(),
        extent.y0 / fig.get_figheight() - 0.035,
        wrapped,
        ha="left",
        va="top",
        fontsize=CAPTION_FONT_SIZE,
        color="#3A3A3A",
        linespacing=1.35,
    )
    fig.savefig(
        out_path,
        format=style.fmt,
        bbox_inches=None if style.format_pixels is not None else "tight",
    )
    plt.close(fig)


def iqr_fence(values: Sequence[float]) -> Tuple[float, float]:
    """
    Tukey's 1.5 x IQR fence, the same rule matplotlib's whiskers use. Returns an
    infinite fence for fewer than two values, where no fence is defined.
    """
    if len(values) < 2:
        return (float("-inf"), float("inf"))
    q1, _q2, q3 = quantiles(values, n=4, method="inclusive")
    reach = 1.5 * (q3 - q1)
    return (q1 - reach, q3 + reach)


def _for_display(values: Sequence[float], style: Style) -> List[float]:
    """
    The values a figure draws for one group: all of them, or only those inside
    the group's own fence when --drop-fliers is set. Statistics are never taken
    from this, so trimming the figure cannot silently trim the reported numbers.
    """
    if not style.drop_fliers:
        return list(values)
    low, high = iqr_fence(values)
    return [v for v in values if low <= v <= high]


def _whisker_span(values: Sequence[float]) -> Optional[Tuple[float, float]]:
    """
    Lower and upper whisker cap of one group, i.e. the vertical extent _draw_boxes
    actually draws for it. None for an empty group, which draws nothing.
    """
    if not values:
        return None
    low, high = iqr_fence(values)
    inside = [v for v in values if low <= v <= high] or list(values)
    return (min(inside), max(inside))


def _whisker_top(values: Sequence[float]) -> float:
    """Upper whisker cap: the largest value inside the fence, as drawn."""
    span = _whisker_span(values)
    return float("nan") if span is None else span[1]


def _format_value(value: float) -> str:
    """Compact fixed-significance number for an on-figure label."""
    return "n/a" if value is None or math.isnan(value) else f"{value:.3g}"


def _annotate_stats(
    ax: AxesLike,
    position: float,
    anchor: float,
    stats: Dict[str, float],
    style: Style,
    fontsize: float = 6.0,
    rotation: float = 0.0,
) -> None:
    """
    Print the group's mean and median above its box. The numbers come from the
    summary, i.e. from every finite value, so they agree with metrics_summary.*
    even when --drop-fliers has narrowed what the figure shows.
    """
    if not style.value_labels or math.isnan(anchor):
        return
    separator = "  " if rotation else "\n"
    ax.annotate(
        f"mean {_format_value(stats['mean'])}{separator}med {_format_value(stats['median'])}",
        (position, anchor),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        rotation=rotation,
        color="#222222",
        zorder=5,
    )


def _thin(values: Sequence[float], limit: int, rng: random.Random) -> List[float]:
    """Deterministic thinning of an over-full point strip."""
    if limit <= 0 or len(values) <= limit:
        return list(values)
    return rng.sample(list(values), limit)


def _draw_boxes(
    ax: AxesLike,
    positions: Sequence[float],
    datasets: Sequence[Sequence[float]],
    color: RGB,
    style: Style,
    width: float = BOX_WIDTH,
) -> None:
    """
    Box = interquartile range, line = median, whiskers = 1.5 x IQR, x = mean.
    Fliers are suppressed because every raw point is drawn beside the box.
    """
    keep = [(p, list(d)) for p, d in zip(positions, datasets) if len(d) > 0]
    if not keep:
        return
    fill_color = tint(color, 0.22) if style.metric_is_3d else color
    ax.boxplot(
        [d for _p, d in keep],
        positions=[p for p, _d in keep],
        widths=width,
        showfliers=False,
        whis=1.5,
        patch_artist=True,
        showmeans=True,
        manage_ticks=False,
        boxprops={"facecolor": fill_color, "edgecolor": "black", "linewidth": style.box_line_width},
        whiskerprops={"color": "black", "linewidth": style.box_line_width},
        capprops={"color": "black", "linewidth": style.box_line_width},
        medianprops={"color": "black", "linewidth": style.box_line_width},
        meanprops={
            "marker": style.mean_marker,
            "markersize": 4.5,
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markeredgewidth": style.box_line_width,
        },
        zorder=3,
    )


def _draw_strip(
    ax: AxesLike,
    position: float,
    groups: Sequence[Tuple[Any, List[float]]],
    colors: Sequence[RGB],
    rng: random.Random,
    style: Style,
    offset: float = STRIP_OFFSET,
    width: float = STRIP_WIDTH,
) -> Tuple[int, int]:
    """
    Every raw value behind one box, in a band beside it, one sub-band per colour
    group. Returns (drawn, total) so the caption can flag any thinning.
    """
    drawn = total = 0
    n_groups = max(len(groups), 1)
    sub_width = width / n_groups
    for index, ((_key, values), color) in enumerate(zip(groups, colors)):
        total += len(values)
        drawable = _for_display(values, style)
        if not drawable:
            continue
        shown = _thin(drawable, style.max_points, rng)
        drawn += len(shown)
        centre = position + offset + (index - (n_groups - 1) / 2.0) * sub_width
        jitter = [centre + rng.uniform(-0.38, 0.38) * sub_width for _ in shown]
        ax.scatter(jitter, shown, s=3.0, color=color, alpha=0.45, linewidths=0.0,
                   zorder=2, rasterized=True)
    return drawn, total


def _draw_group_boxes(
    ax: AxesLike,
    position: float,
    groups: Sequence[Tuple[Any, List[float]]],
    colors: Sequence[RGB],
    style: Style,
    offset: float = STRIP_OFFSET,
    width: float = STRIP_WIDTH,
) -> Tuple[int, int]:
    """
    One box per colour group, side by side in the band beside the summary box:
    the --distribution-style=boxes counterpart of the point strip. Returns
    (represented, total) values, equal because a box represents all of them.
    """
    total = 0
    n_groups = max(len(groups), 1)
    sub_width = width / n_groups
    for index, ((_key, values), color) in enumerate(zip(groups, colors)):
        total += len(values)
        shown = _for_display(values, style)
        if not shown:
            continue
        centre = position + offset + (index - (n_groups - 1) / 2.0) * sub_width
        _draw_boxes(ax, [centre], [shown], color, style, width=sub_width * 0.68)
    return total, total


def _draw_distribution(
    ax: AxesLike,
    position: float,
    groups: Sequence[Tuple[Any, List[float]]],
    colors: Sequence[RGB],
    rng: random.Random,
    style: Style,
    offset: float = STRIP_OFFSET,
    width: float = STRIP_WIDTH,
) -> Tuple[int, int]:
    """Render the per-group distribution in whichever style was requested."""
    if style.distribution_style == DISTRIBUTION_BOXES:
        return _draw_group_boxes(ax, position, groups, colors, style, offset, width)
    return _draw_strip(ax, position, groups, colors, rng, style, offset, width)


# --------------------------------------------------------------------------
# Broken y axis (--dynamic-y-axis)
# --------------------------------------------------------------------------
#
# One group whose values sit far above the rest -- the two-view combinations of an
# MPJPE, typically -- stretches a linear axis until every other box collapses into
# a band a few pixels high, and the figure then answers nothing about the groups it
# was drawn to compare. The answer here is a broken axis rather than a log scale:
# these metrics are read as body-length ratios, a log axis distorts exactly that
# reading, and an error of 0 is a legitimate value it cannot place at all.
#
# The break is proposed from the extent as DRAWN (whisker caps, --drop-fliers
# already applied), for the same reason the annotations are anchored there: what
# is crushed is what is on the page, not what is in the summary.
#
# There are two complementary split heuristics:
#   1. A genuine empty gap between the whisker spans of groups.
#   2. A dominant group whose upper quartile is substantially above all other
#      groups. This catches the common MPJPE case where one box has a much wider
#      distribution and therefore overlaps the rest with its whisker, so the old
#      gap-only test could never fire.

DYNAMIC_Y_MIN_GAP_SHARE = 0.30  # empty band, as a share of the full drawn range
DYNAMIC_Y_MAX_BULK_SHARE = 0.45  # what the crushed cluster may occupy, same unit
DYNAMIC_Y_DOMINANT_Q3_RATIO = 2.0  # dominant Q3 must be >= this x the next-highest Q3
DYNAMIC_Y_DOMINANT_Q1_RATIO = 2.0  # dominant low Q1 must be <= 1/this x the next-lowest Q1
DYNAMIC_Y_PAD_SHARE = 0.10  # slack around a panel, as a share of that panel's range
DYNAMIC_Y_MIN_PAD_SHARE = 0.005  # floor under that slack, as a share of the whole range
DYNAMIC_Y_HEIGHT_RATIOS = (1.0, 2.0)  # upper (the outlier) : lower (the rest)
DYNAMIC_Y_HSPACE = 0.06
DYNAMIC_Y_FIGURE_SCALE = 1.15  # the second panel needs a little more paper
BREAK_MARK_SIZE = 7.0  # length of the axis cut marks, in points
BREAK_WAVE_HEIGHT = 0.012  # amplitude of a box's cut mark, in axes fractions


class YSplit(NamedTuple):
    """The y limits of the lower and of the upper panel of a broken y axis."""

    bottom: Tuple[float, float]
    top: Tuple[float, float]


# A drawing surface: a bare Axes for the figures that never break, or the
# SplitAxes below, which stands in for one.
AxesLike = Any


def _panels_of(ax: AxesLike) -> List[Axes]:
    """The concrete Axes behind a drawing surface, upper panel first."""
    return list(ax.panels) if isinstance(ax, SplitAxes) else [ax]


def _padded(low: float, high: float, total: float) -> Tuple[float, float]:
    """
    One panel's limits: its own range plus a slack proportional to that range, not
    to the whole. The point of the break is that the two panels have their own
    scales, and a slack taken from the full range would reintroduce the outlier's
    magnitude into the panel that was split off from it. The floor keeps a panel
    whose group is a single value from collapsing to zero height.
    """
    pad = max(DYNAMIC_Y_PAD_SHARE * (high - low), DYNAMIC_Y_MIN_PAD_SHARE * total)
    return (low - pad, high + pad)


def detect_y_split(
    datasets: Sequence[Sequence[float]],
    style: Style,
    spanning: Sequence[Sequence[float]] = (),
) -> Optional[YSplit]:
    """
    The two panels a broken y axis would need, or None to keep one linear axis.

    `datasets` are the groups whose separation decides the break -- the per-group
    boxes, which are what a far-out group crushes -- while `spanning` only widens
    the panels. A summary box pooling every group straddles the break by
    construction, so letting it vote would close the very gap it is meant to
    bridge; it is drawn across the break instead, with a cut mark on it.
    """
    if not style.dynamic_y_axis:
        return None
    spans = [s for s in (_whisker_span(d) for d in datasets) if s is not None]
    if len(spans) < 2:
        return None
    outer = spans + [s for s in (_whisker_span(d) for d in spanning) if s is not None]
    low = min(s[0] for s in outer)
    high = max(s[1] for s in outer)
    total = high - low
    if not math.isfinite(total) or total <= 0.0:
        return None

    best: Optional[Tuple[float, float, float]] = None  # (gap, bulk top, outlier floor)
    for cut in sorted({s[0] for s in spans})[1:]:
        bulk_top = max(s[1] for s in spans if s[0] < cut)
        gap = cut - bulk_top
        if gap <= 0.0 or gap / total < DYNAMIC_Y_MIN_GAP_SHARE:
            continue
        if (bulk_top - low) / total > DYNAMIC_Y_MAX_BULK_SHARE:
            continue
        if best is None or gap > best[0]:
            best = (gap, bulk_top, cut)
    if best is not None:
        _gap, bulk_top, outlier_floor = best
        return YSplit(_padded(low, bulk_top, total), _padded(outlier_floor, high, total))

    # Fallback: one dominant/wide group can be visually isolated even when its
    # whisker overlaps the rest. Detect that from the boxes' Q3 values instead of
    # demanding a literal empty whisker-to-whisker interval.
    quartiles = []
    for values in datasets:
        finite = _finite(values)
        if len(finite) < 2:
            continue
        q1, _median, q3 = quantiles(finite, n=4, method="inclusive")
        span = _whisker_span(finite)
        if span is not None and math.isfinite(q3):
            quartiles.append((q1, q3, span[0], span[1]))

    if len(quartiles) < 2:
        return None

    # Upper-side fallback: one group is much higher/wider than the others.
    dominant_index = max(range(len(quartiles)), key=lambda i: quartiles[i][1])
    dominant_q3 = quartiles[dominant_index][1]
    other = [q for i, q in enumerate(quartiles) if i != dominant_index]
    other_max_q3 = max(q[1] for q in other)
    other_max_whisker = max(q[3] for q in other)

    if other_max_q3 <= 0.0:
        dominant_ratio = float("inf") if dominant_q3 > 0.0 else 1.0
    else:
        dominant_ratio = dominant_q3 / other_max_q3

    # The break is placed above the entire non-dominant cluster and below the
    # dominant box's Q3. The dominant box can therefore span the break, which is
    # already handled by SplitAxes._mark_spanning().
    if (
        dominant_q3 > other_max_whisker
        and dominant_ratio >= DYNAMIC_Y_DOMINANT_Q3_RATIO
    ):
        bulk_top = other_max_whisker
        outlier_floor = dominant_q3
        if outlier_floor > bulk_top:
            return YSplit(
                _padded(low, bulk_top, total),
                _padded(outlier_floor, high, total),
            )

    # Lower-side mirror: one group is much lower/wider than the others.
    # For non-negative metrics (the normal case here), this is the direct mirror
    # of the Q3 test above: its Q1 must be at most 1/R times the smallest Q1 of
    # the other groups, and its lower whisker must extend below the whole cluster.
    dominant_low_index = min(range(len(quartiles)), key=lambda i: quartiles[i][0])
    dominant_q1 = quartiles[dominant_low_index][0]
    other_low = [q for i, q in enumerate(quartiles) if i != dominant_low_index]
    other_min_q1 = min(q[0] for q in other_low)
    other_min_whisker = min(q[2] for q in other_low)

    if other_min_q1 > 0.0:
        low_ratio = other_min_q1 / dominant_q1 if dominant_q1 > 0.0 else float("inf")
    else:
        low_ratio = float("inf") if dominant_q1 < other_min_q1 else 1.0

    # The lower panel contains the exceptional group's lower tail. The upper
    # panel contains the rest; the exceptional box may span the break.
    if (
        dominant_q1 < other_min_whisker
        and low_ratio >= DYNAMIC_Y_DOMINANT_Q1_RATIO
    ):
        outlier_ceiling = other_min_whisker
        bulk_bottom = dominant_q1
        if outlier_ceiling > bulk_bottom:
            return YSplit(
                _padded(low, bulk_bottom, total),
                _padded(outlier_ceiling, high, total),
            )

    return None


def _draw_axis_break(top: Axes, bottom: Axes) -> None:
    """The two diagonal cuts that say the y axis is not continuous."""
    marks = {
        "marker": [(-1.0, -0.6), (1.0, 0.6)],
        "markersize": BREAK_MARK_SIZE,
        "linestyle": "none",
        "color": "black",
        "markeredgecolor": "black",
        "markeredgewidth": 0.9,
        "clip_on": False,
    }
    top.plot([0.0], [0.0], transform=top.transAxes, **marks)
    bottom.plot([0.0], [1.0], transform=bottom.transAxes, **marks)


def _mark_spanning_box(ax: Axes, centre: float, width: float, at_top: bool) -> None:
    """
    A wave across one box where the break cuts it, so a whisker that continues in
    the other panel is never read as a whisker that ended there. x is in data
    coordinates and y in axes fractions, which puts the wave on the panel edge
    whatever the panel's limits are.
    """
    steps = 9
    edge = 1.0 if at_top else 0.0
    xs = [centre - width / 2.0 + width * i / (steps - 1) for i in range(steps)]
    ys = [edge + (BREAK_WAVE_HEIGHT if i % 2 else -BREAK_WAVE_HEIGHT) for i in range(steps)]
    transform = ax.get_xaxis_transform()
    # White underlay first: the cut has to read as a cut, not as another whisker.
    ax.plot(xs, ys, transform=transform, color="white", linewidth=2.6,
            solid_capstyle="butt", clip_on=False, zorder=6)
    ax.plot(xs, ys, transform=transform, color="black", linewidth=0.8,
            solid_capstyle="butt", clip_on=False, zorder=7)


class SplitAxes:
    """
    The drawing surface of a box figure: one Axes, or the two stacked, x-sharing
    Axes of a broken y axis, upper panel first.

    It carries the part of the Axes interface the box primitives use, so
    `_draw_boxes`, `_draw_strip` and `_annotate_stats` draw through it unchanged:
    every artist goes to both panels and each panel's y limits clip it, which is
    what makes a box that spans the break appear in each, while the ticks, the
    legend, the title and each annotation are routed to the one panel they belong
    on. With no split it is a thin pass-through, so there is one code path.
    """

    def __init__(self, fig: Figure, panels: Sequence[Axes], split: Optional[YSplit]) -> None:
        self.fig = fig
        self.panels: List[Axes] = list(panels)
        self.split = split

    @property
    def top(self) -> Axes:
        return self.panels[0]

    @property
    def bottom(self) -> Axes:
        return self.panels[-1]

    def panel_at(self, value: float) -> Axes:
        """The panel a value belongs on; the lower one for anything in the gap."""
        if self.split is None or value is None or math.isnan(value):
            return self.bottom
        threshold = 0.5 * (self.split.bottom[1] + self.split.top[0])
        return self.top if value >= threshold else self.bottom

    # -- the Axes interface the primitives call --------------------------------

    def boxplot(self, datasets: Sequence[Sequence[float]], **kwargs: Any) -> Any:
        result = None
        for panel in self.panels:
            result = panel.boxplot(datasets, **kwargs)
        self._mark_spanning(datasets, kwargs.get("positions") or [], kwargs.get("widths"))
        return result

    def scatter(self, *args: Any, **kwargs: Any) -> None:
        for panel in self.panels:
            panel.scatter(*args, **kwargs)

    def annotate(self, text: str, xy: Tuple[float, float], **kwargs: Any) -> None:
        self.panel_at(xy[1]).annotate(text, xy, **kwargs)

    def legend(self, *args: Any, **kwargs: Any) -> None:
        self.top.legend(*args, **kwargs)

    def set_title(self, title: str, **kwargs: Any) -> None:
        self.top.set_title(title, **kwargs)

    def set_xticks(self, ticks: Sequence[float]) -> None:
        self.bottom.set_xticks(ticks)

    def set_xticklabels(self, labels: Sequence[str]) -> None:
        self.bottom.set_xticklabels(labels)

    def set_xlim(self, *args: Any, **kwargs: Any) -> None:
        self.bottom.set_xlim(*args, **kwargs)

    def set_xlabel(self, label: str) -> None:
        self.bottom.set_xlabel(label)

    def set_ylabel(self, label: str) -> None:
        # One label for the pair, on the lower and larger panel: a figure-level
        # label would be centred over both, but it is placed in figure coordinates
        # and so has to be kept clear of the tick labels by hand, which cannot be
        # done reliably before the ticks are known. An axes label is positioned
        # against the rendered ticks by matplotlib itself.
        self.bottom.set_ylabel(label)

    # -- break marks -----------------------------------------------------------

    def _mark_spanning(
        self, datasets: Sequence[Sequence[float]], positions: Sequence[float], widths: Any
    ) -> None:
        """Cut every box whose whiskers cross the break, in both panels."""
        if self.split is None:
            return
        if not isinstance(widths, (list, tuple)):
            widths = [widths if widths else BOX_WIDTH] * len(datasets)
        for values, position, width in zip(datasets, positions, widths):
            span = _whisker_span(values)
            if span is None or span[0] >= self.split.bottom[1] or span[1] <= self.split.top[0]:
                continue
            _mark_spanning_box(self.top, position, width, at_top=False)
            _mark_spanning_box(self.bottom, position, width, at_top=True)


def _make_axes(
    figsize: Tuple[float, float], split: Optional[YSplit], exact_size: bool = False
) -> Tuple[Figure, SplitAxes]:
    """
    The figure and its drawing surface: one Axes, or the two panels of a broken y
    axis with the outlier range on top, their limits set and the cuts drawn.
    """
    if split is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, SplitAxes(fig, [ax], None)

    fig, (top, bottom) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(figsize[0], figsize[1] if exact_size else figsize[1] * DYNAMIC_Y_FIGURE_SCALE),
        gridspec_kw={
            "height_ratios": list(DYNAMIC_Y_HEIGHT_RATIOS),
            "hspace": DYNAMIC_Y_HSPACE,
        },
    )
    top.set_ylim(*split.top)
    bottom.set_ylim(*split.bottom)
    # The break replaces the spine between the panels; the x axis stays on the
    # lower one only, so the two never look like two independent figures.
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    _draw_axis_break(top, bottom)
    return fig, SplitAxes(fig, [top, bottom], split)


def _distribution_legend(
    ax: AxesLike, title: str, labels: Sequence[str], colors: Sequence[RGB], style: Style
) -> None:
    """Colour key for the distribution, plus the box's median and mean symbols."""
    boxes = style.distribution_style == DISTRIBUTION_BOXES
    handles = [
        Line2D(
            [], [],
            marker="s" if boxes else "o",
            linestyle="none",
            markersize=4.0 if boxes else 3.5,
            markerfacecolor=color,
            markeredgecolor=color,
            color=color,
            label=label,
        )
        for label, color in zip(labels, colors)
    ]
    handles += [
        Line2D([], [], color="black", linewidth=style.box_line_width, label="median"),
        Line2D([], [], marker=style.mean_marker, linestyle="none", markersize=4.5,
               markeredgewidth=style.box_line_width, color="black", label="mean"),
    ]
    ax.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=(1.01, 1.0))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

# Builders return (figure, axes, n_groups, n_points_drawn, n_points_total).
FigureResult = Tuple[Figure, Axes, int, int, int]


def fig_bar_per_view(table, spec, block, style, rng) -> FigureResult:
    """Bars = mean per view over all runs; diamonds = median."""
    views = views_of(table)
    stats = block[GROUPING_PER_VIEW]
    positions = list(range(len(views)))

    fig, ax = plt.subplots(figsize=style.figure_size((_figure_width(len(views), 1.10, 4.8), 3.5)))
    ax.bar(
        positions,
        [stats[v]["mean"] for v in views],
        width=0.62,
        facecolor=style.primary,
        edgecolor=style.primary,
        linewidth=0.9,
        label="mean",
        zorder=2,
    )
    ax.scatter(
        positions,
        [stats[v]["median"] for v in views],
        marker="D", s=15, facecolor="white", edgecolor="black", linewidth=0.7,
        zorder=3, label="median",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([_view_label(v, stats[v]["n_samples"]) for v in views])
    _axis_cosmetics(ax, spec.label, "view")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax, len(views), 0, 0


def fig_bar_per_view_by_n_views(table, spec, block, style, rng) -> FigureResult:
    """Bars = mean per view, one bar per #views; diamonds = median."""
    views = views_of(table)
    n_values = n_views_of(table)
    stats = block[GROUPING_PER_VIEW_PER_N]
    colors = style.palette("n_views", len(n_values))
    width = 0.78 / max(len(n_values), 1)

    fig, ax = plt.subplots(figsize=style.figure_size((_figure_width(len(views), 1.45, 5.6), 3.6)))
    for index, (k, color) in enumerate(zip(n_values, colors)):
        offset = -0.39 + width * (index + 0.5)
        positions = [x + offset for x in range(len(views))]
        cell_stats = [stats[v].get(str(k)) for v in views]
        ax.bar(
            positions,
            [s["mean"] if s else float("nan") for s in cell_stats],
            width=width, facecolor=color, edgecolor=color, linewidth=0.7,
            label=str(k), zorder=2,
        )
        ax.scatter(
            positions,
            [s["median"] if s else float("nan") for s in cell_stats],
            marker="D", s=8, facecolor="white", edgecolor="black", linewidth=0.5, zorder=3,
        )
    ax.set_xticks(list(range(len(views))))
    ax.set_xticklabels([_view_label(v) for v in views])
    _axis_cosmetics(ax, spec.label, "view")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], marker="D", linestyle="none", markersize=3.0, markerfacecolor="white",
               markeredgecolor="black")
    )
    labels.append("median")
    ax.legend(handles, labels, title="#views", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax, len(views) * len(n_values), 0, 0


def fig_line_vs_n_views(table, spec, block, style, rng) -> FigureResult:
    """Mean and median against the number of views, pooled over all views."""
    n_values = n_views_of(table)
    stats = block[GROUPING_PER_N]

    means = [stats[str(k)]["mean"] for k in n_values]
    medians = [stats[str(k)]["median"] for k in n_values]

    fig, ax = plt.subplots(figsize=style.figure_size((4.8, 3.5)))
    ax.plot(n_values, means, marker="o", color=style.primary, label="mean")
    median_color = style.primary if style.monochrome else _with_lightness(style.primary, 0.28)
    ax.plot(n_values, medians, marker="s", markerfacecolor="white", linestyle="--",
            color=median_color, label="median")
    if style.value_labels:
        # Means above their marker, medians below, so the two never collide.
        for series, offset, valign in ((means, 5, "bottom"), (medians, -6, "top")):
            for k, value in zip(n_values, series):
                ax.annotate(
                    _format_value(value), (k, value), textcoords="offset points",
                    xytext=(0, offset), ha="center", va=valign, fontsize=6.0,
                    color="#222222",
                )
    ax.set_xticks(n_values)
    _axis_cosmetics(ax, spec.label, "number of views in the reconstruction")
    _label_headroom(ax, style, 0.10, below=0.08)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax, len(n_values), 0, 0


def fig_box_per_view(table, spec, block, style, rng) -> FigureResult:
    """One box per view over all runs; points beside it, coloured by #views."""
    views = views_of(table)
    n_values = n_views_of(table)
    colors = style.palette("n_views", len(n_values))
    stats = block[GROUPING_PER_VIEW]
    cells = {(view, k): _finite(table.get((view, k), [])) for view in views for k in n_values}
    summaries = {view: _for_display(pooled(table, view=view), style) for view in views}
    split = detect_y_split(
        [_for_display(values, style) for values in cells.values()],
        style,
        spanning=list(summaries.values()),
    )

    fig, ax = _make_axes(
        style.figure_size((_figure_width(len(views), 1.30, 5.2), 3.8)), split,
        exact_size=style.format_pixels is not None,
    )
    drawn = total = 0
    for position, view in enumerate(views):
        shown = summaries[view]
        _draw_boxes(ax, [position], [shown], style.primary, style)
        groups = [(k, cells[(view, k)]) for k in n_values]
        d, t = _draw_distribution(ax, position, groups, colors, rng, style)
        drawn, total = drawn + d, total + t
        _annotate_stats(ax, position, _whisker_top(shown), stats[view], style)
    ax.set_xticks(list(range(len(views))))
    ax.set_xticklabels([_view_label(v, stats[v]["n_samples"]) for v in views])
    _axis_cosmetics(ax, spec.label, "view")
    _distribution_legend(ax, "#views", [str(k) for k in n_values], colors, style)
    _label_headroom(ax, style)
    return fig, ax, len(views), drawn, total


def fig_box_per_view_and_n_views(table, spec, block, style, rng) -> FigureResult:
    """One box per (view, #views) cell; points beside it, coloured by #views."""
    views = views_of(table)
    n_values = n_views_of(table)
    colors = style.palette("n_views", len(n_values))
    span = 0.84
    step = span / max(len(n_values), 1)

    split = detect_y_split(
        [_for_display(_finite(table.get((view, k), [])), style)
         for view in views for k in n_values],
        style,
    )

    width = _figure_width(len(views) * max(len(n_values), 1), 0.48, 5.6)
    fig, ax = _make_axes(
        style.figure_size((width, max(3.8, min(width * 0.36, 5.4)))), split,
        exact_size=style.format_pixels is not None,
    )
    # In box mode the cell box already is the group box, so the band beside it
    # would only duplicate it: the cell boxes are then simply centred.
    as_boxes = style.distribution_style == DISTRIBUTION_BOXES
    drawn = total = boxes = 0
    for view_index, view in enumerate(views):
        for k_index, (k, color) in enumerate(zip(n_values, colors)):
            values = _finite(table.get((view, k), []))
            shown = _for_display(values, style)
            if not shown:
                continue
            position = view_index - span / 2 + step * (k_index + 0.5)
            box_position = position if as_boxes else position - step * 0.20
            _draw_boxes(ax, [box_position], [shown], color, style, width=step * 0.34)
            if not as_boxes:
                d, t = _draw_strip(
                    ax, position, [(k, values)], [color], rng, style,
                    offset=step * 0.22, width=step * 0.34,
                )
                drawn, total = drawn + d, total + t
            else:
                drawn, total = drawn + len(values), total + len(values)
            boxes += 1
            _annotate_stats(
                ax, box_position, _whisker_top(shown),
                block[GROUPING_PER_VIEW_PER_N][view][str(k)], style,
                fontsize=5.0, rotation=90.0,
            )
    ax.set_xticks(list(range(len(views))))
    ax.set_xticklabels([_view_label(v) for v in views])
    _axis_cosmetics(ax, spec.label, "view")
    _distribution_legend(ax, "#views", [str(k) for k in n_values], colors, style)
    _label_headroom(ax, style, 0.20)
    return fig, ax, boxes, drawn, total


def fig_box_vs_n_views(table, spec, block, style, rng) -> FigureResult:
    """One box per #views over all views; points beside it, coloured by view."""
    views = views_of(table)
    n_values = n_views_of(table)
    colors = style.palette("view", len(views))
    stats = block[GROUPING_PER_N]
    cells = {(view, k): _finite(table.get((view, k), [])) for view in views for k in n_values}
    summaries = {k: _for_display(pooled(table, n_views=k), style) for k in n_values}
    split = detect_y_split(
        [_for_display(values, style) for values in cells.values()],
        style,
        spanning=list(summaries.values()),
    )

    fig, ax = _make_axes(
        style.figure_size((_figure_width(len(n_values), 1.2, 4.6), 3.8)), split,
        exact_size=style.format_pixels is not None,
    )
    drawn = total = 0
    for position, k in enumerate(n_values):
        shown = summaries[k]
        _draw_boxes(ax, [position], [shown], style.primary, style)
        groups = [(view, cells[(view, k)]) for view in views]
        d, t = _draw_distribution(ax, position, groups, colors, rng, style)
        drawn, total = drawn + d, total + t
        _annotate_stats(ax, position, _whisker_top(shown), stats[str(k)], style)
    ax.set_xticks(list(range(len(n_values))))
    ax.set_xticklabels([f"{k}\nn = {stats[str(k)]['n_samples']}" for k in n_values])
    _axis_cosmetics(ax, spec.label, "number of views in the reconstruction")
    _distribution_legend(ax, "view", list(views), colors, style)
    _label_headroom(ax, style)
    return fig, ax, len(n_values), drawn, total


def _fig_box_overall(table, spec, block, style, rng, colour_by: str) -> FigureResult:
    """A single box over every run; points beside it, coloured by one factor."""
    if colour_by == "n_views":
        keys: List[Any] = n_views_of(table)
        labels = [str(k) for k in keys]
        groups = [(k, pooled(table, n_views=k)) for k in keys]
        legend_title = "#views"
    else:
        keys = views_of(table)
        labels = [str(v) for v in keys]
        groups = [(view, pooled(table, view=view)) for view in keys]
        legend_title = "view"
    colors = style.palette(colour_by, len(keys))

    if not style.overall_pooling and style.distribution_style == DISTRIBUTION_BOXES:
        return _fig_box_per_group(spec, block, style, colour_by, keys, groups, colors, legend_title)

    shown = _for_display(pooled(table), style)
    split = detect_y_split(
        [_for_display(values, style) for _key, values in groups], style, spanning=[shown]
    )

    fig, ax = _make_axes(
        style.figure_size((3.6, 3.6)), split, exact_size=style.format_pixels is not None
    )
    _draw_boxes(ax, [0.0], [shown], style.primary, style, width=0.26)
    drawn, total = _draw_distribution(ax, 0.0, groups, colors, rng, style)
    _annotate_stats(ax, 0.0, _whisker_top(shown), block[GROUPING_OVERALL], style)
    ax.set_xticks([0.14])
    ax.set_xticklabels([f"all runs pooled\nn = {block[GROUPING_OVERALL]['n_samples']}"])
    ax.set_xlim(-0.28, 0.58)
    _axis_cosmetics(ax, spec.label)
    _distribution_legend(ax, legend_title, labels, colors, style)
    _label_headroom(ax, style)
    return fig, ax, 1, drawn, total


def _fig_box_per_group(
    spec: MetricSpec,
    block: Dict[str, Any],
    style: Style,
    colour_by: str,
    keys: Sequence[Any],
    groups: Sequence[Tuple[Any, List[float]]],
    colors: Sequence[RGB],
    legend_title: str,
) -> FigureResult:
    """
    The --no-overall-pooling form of _fig_box_overall: the box over all runs is
    dropped and the groups that were narrow boxes beside it take the x axis, one
    box at its own position, laid out and labelled as fig_box_vs_n_views and
    fig_box_per_view lay theirs out. Only the pooled box goes -- the groups are
    the same values, split the same way, so the figure stays comparable to the
    pooled one it replaces.
    """
    if colour_by == "n_views":
        stats = block[GROUPING_PER_N]
        group_stats = [stats[str(k)] for k in keys]
        tick_labels = [f"{k}\nn = {stats[str(k)]['n_samples']}" for k in keys]
        xlabel = "number of views in the reconstruction"
        figsize = (_figure_width(len(keys), 0.58, 3.8), 3.8)
    else:
        stats = block[GROUPING_PER_VIEW]
        group_stats = [stats[view] for view in keys]
        tick_labels = [_view_label(view, stats[view]["n_samples"]) for view in keys]
        xlabel = "view"
        figsize = (_figure_width(len(keys), 0.68, 4.2), 3.8)

    drawable = [_for_display(values, style) for _key, values in groups]
    split = detect_y_split(drawable, style)

    fig, ax = _make_axes(
        style.figure_size(figsize), split, exact_size=style.format_pixels is not None
    )
    for position, (shown, color, cell_stats) in enumerate(zip(drawable, colors, group_stats)):
        if not shown:
            continue
        box_width = 0.68 if colour_by == "n_views" else 0.72
        _draw_boxes(ax, [position], [shown], color, style, width=box_width)
        _annotate_stats(ax, position, _whisker_top(shown), cell_stats, style)
    ax.set_xticks(list(range(len(keys))))
    ax.set_xticklabels(tick_labels)
    _axis_cosmetics(ax, spec.label, xlabel)
    _distribution_legend(ax, legend_title, [str(k) for k in keys], colors, style)
    _label_headroom(ax, style)
    # A box represents every value behind it, as in _draw_group_boxes.
    total = sum(len(values) for _key, values in groups)
    return fig, ax, len(keys), total, total


def fig_box_overall_by_n_views(table, spec, block, style, rng) -> FigureResult:
    return _fig_box_overall(table, spec, block, style, rng, colour_by="n_views")


def fig_box_overall_by_view(table, spec, block, style, rng) -> FigureResult:
    return _fig_box_overall(table, spec, block, style, rng, colour_by="view")


# --------------------------------------------------------------------------
# Figure orchestration
# --------------------------------------------------------------------------


# Wording and file-name tokens of the two distribution styles, so that a figure
# is never described as showing points when it shows boxes.
DISTRIBUTION_TOKENS: Dict[str, Dict[str, str]] = {
    DISTRIBUTION_POINTS: {"file": "all_points", "noun": "points"},
    DISTRIBUTION_BOXES: {"file": "per_group_boxes", "noun": "boxes"},
}


@dataclass(frozen=True)
class FigureKind:
    """
    One figure recipe. `filename_kind` is the self-describing file-name stem and
    may carry the {file}/{noun} distribution tokens; `what` becomes the caption's
    first sentence, stating exactly what is pooled. `boxes_overrides` replaces
    any of those texts when --distribution-style=boxes changes what is drawn.
    """

    filename_kind: str
    builder: Callable[..., FigureResult]
    title: str
    what: str
    plot_kind: str
    grouping: str
    point_colouring: str
    boxes_overrides: Optional[Dict[str, str]] = None
    # Replaces the texts again when --no-overall-pooling drops the pooled box of a
    # GROUPING_OVERALL figure and lays its groups out on their own x axis. Applied
    # after boxes_overrides, since that layout only exists in the boxes style.
    unpooled_overrides: Optional[Dict[str, str]] = None
    # Only these react to --distribution-style and --drop-fliers; the mean/median
    # summaries do not, and must not claim in their caption that they do.
    draws_distribution: bool = False
    # True when the figure puts views on the x axis or splits by view. Skipped for
    # a metric with MetricSpec.view_axis=False, where it would draw one column
    # labelled with the sentinel view and imply a comparison that does not exist.
    # Figures that only vary over #views are kept: that is the sweep's question,
    # and the 3D metrics answer it.
    requires_view_axis: bool = False

    def resolve(self, style: Style) -> Dict[str, str]:
        """The file-name stem and index/caption texts for the active style."""
        tokens = DISTRIBUTION_TOKENS[style.distribution_style]
        fields = {
            "filename_kind": self.filename_kind,
            "what": self.what,
            "plot_kind": self.plot_kind,
            "point_colouring": self.point_colouring,
        }
        if style.distribution_style == DISTRIBUTION_BOXES and self.boxes_overrides:
            fields.update(self.boxes_overrides)
        if (
            style.distribution_style == DISTRIBUTION_BOXES
            and not style.overall_pooling
            and self.unpooled_overrides
        ):
            fields.update(self.unpooled_overrides)
        return {key: value.format(**tokens) for key, value in fields.items()}


FIGURE_KINDS: Tuple[FigureKind, ...] = (
    FigureKind(
        "bar_mean_and_median_per_view",
        fig_bar_per_view,
        "mean and median per view",
        "Bar height is the mean and the diamond the median of every frame value pooled over all "
        "runs that included the view, irrespective of that run's number of views.",
        "mean/median summary",
        GROUPING_PER_VIEW,
        "not applicable",
        requires_view_axis=True,
    ),
    FigureKind(
        "bar_mean_and_median_per_view_grouped_by_number_of_views",
        fig_bar_per_view_by_n_views,
        "mean and median per view, grouped by #views",
        "Bar height is the mean and the diamond the median of every frame value from the runs of "
        "that combination size which included the view.",
        "mean/median summary",
        GROUPING_PER_VIEW_PER_N,
        "bars coloured by #views",
        requires_view_axis=True,
    ),
    FigureKind(
        "line_mean_and_median_versus_number_of_views",
        fig_line_vs_n_views,
        "mean and median versus #views",
        "Each point pools every frame value of every view of every run of that combination size.",
        "mean/median summary",
        GROUPING_PER_N,
        "not applicable",
    ),
    FigureKind(
        "box_iqr_with_{file}_per_view__{noun}_coloured_by_number_of_views",
        fig_box_per_view,
        "distribution per view",
        "The box spans the interquartile range with the median as a line, whiskers at 1.5 x IQR "
        "and the mean as a diamond; beside it every raw frame value of that view is drawn, in one "
        "sub-band per number of views.",
        "box plot (IQR) with all raw points",
        GROUPING_PER_VIEW,
        "{noun} coloured by #views",
        draws_distribution=True,
        requires_view_axis=True,
        boxes_overrides={
            "what": "The wide box spans the interquartile range of the view with the median as a "
                    "line, whiskers at 1.5 x IQR and the mean as a diamond; beside it one narrow "
                    "box per number of views splits the same values by combination size.",
            "plot_kind": "box plot (IQR) with a box per group",
        },
    ),
    FigureKind(
        "box_iqr_with_{file}_per_view_and_number_of_views__{noun}_coloured_by_number_of_views",
        fig_box_per_view_and_n_views,
        "distribution per view and #views",
        "One box per (view, number of views) cell, with that cell's raw frame values drawn beside "
        "it; box and points share the colour of their combination size.",
        "box plot (IQR) with all raw points",
        GROUPING_PER_VIEW_PER_N,
        "boxes and points coloured by #views",
        draws_distribution=True,
        requires_view_axis=True,
        boxes_overrides={
            "what": "One box per (view, number of views) cell, coloured by combination size. This "
                    "grouping is already one box per group, so no second band is drawn beside it.",
            "plot_kind": "box plot (IQR) with a box per group",
            "point_colouring": "boxes coloured by #views",
        },
    ),
    FigureKind(
        "box_iqr_with_{file}_versus_number_of_views__{noun}_coloured_by_view",
        fig_box_vs_n_views,
        "distribution versus #views",
        "One box per combination size, pooling all of its views, with every raw frame value drawn "
        "beside it in one sub-band per view.",
        "box plot (IQR) with all raw points",
        GROUPING_PER_N,
        "{noun} coloured by view",
        draws_distribution=True,
        boxes_overrides={
            "what": "One wide box per combination size, pooling all of its views, with one narrow "
                    "box per view beside it splitting the same values by view.",
            "plot_kind": "box plot (IQR) with a box per group",
        },
    ),
    FigureKind(
        "box_iqr_with_{file}_all_runs_pooled__{noun}_coloured_by_number_of_views",
        fig_box_overall_by_n_views,
        "distribution over all runs",
        "A single box over every run, view and frame; the raw values beside it are grouped and "
        "coloured by the number of views of the run they come from.",
        "box plot (IQR) with all raw points",
        GROUPING_OVERALL,
        "{noun} coloured by #views",
        draws_distribution=True,
        boxes_overrides={
            "what": "A single box over every run, view and frame, with one box per combination "
                    "size beside it splitting the same values.",
            "plot_kind": "box plot (IQR) with a box per group",
        },
        unpooled_overrides={
            "filename_kind": "box_iqr_with_{file}_per_number_of_views__"
                             "{noun}_coloured_by_number_of_views",
            "what": "One box per combination size, each pooling every run, view and frame of "
                    "that size, on an axis of combination sizes; the box over all runs "
                    "together is not drawn.",
            "point_colouring": "{noun} coloured by #views",
        },
    ),
    FigureKind(
        "box_iqr_with_{file}_all_runs_pooled__{noun}_coloured_by_view",
        fig_box_overall_by_view,
        "distribution over all runs",
        "The same pooled box, with the raw values grouped and coloured by the view they come from.",
        "box plot (IQR) with all raw points",
        GROUPING_OVERALL,
        "{noun} coloured by view",
        draws_distribution=True,
        requires_view_axis=True,
        boxes_overrides={
            "what": "The same pooled box, with one box per view beside it splitting the same "
                    "values by view.",
            "plot_kind": "box plot (IQR) with a box per group",
        },
        unpooled_overrides={
            "filename_kind": "box_iqr_with_{file}_per_view__{noun}_coloured_by_view",
            "what": "One box per view, each pooling every run and frame that included it, on an "
                    "axis of views; the box over all runs together is not drawn.",
            "point_colouring": "{noun} coloured by view",
        },
    ),
)

PLOT_INDEX_COLUMNS = [
    "filename",
    "metric",
    "keypoint",
    "plot_kind",
    "grouping",
    "point_colouring",
    "n_groups",
    "n_points_drawn",
    "n_points_total",
    "orientation",
    "description",
]


def _plot_path(plots_dir: Path, measure_stem: str, kind: str, style: Style) -> Path:
    """Self-describing file name, abbreviated only if it would grow over-long."""
    name = f"{measure_stem}__{kind}.{style.fmt}"
    if len(name) > MAX_PLOT_NAME_LEN:
        # Same guard as the sweep's leaf names; plots_index.csv always carries
        # the unabbreviated description of every figure.
        budget = MAX_PLOT_NAME_LEN - len(kind) - len(style.fmt) - 12
        digest = f"{abs(hash(measure_stem)) % 10**6:06d}"
        name = f"{measure_stem[:max(budget, 8)]}_{digest}__{kind}.{style.fmt}"
    return plots_dir / name


def _caption(
    spec: MetricSpec, what: str, drawn: int, total: int, style: Style, draws_distribution: bool
) -> str:
    """Figure caption: what is pooled, what the metric is, how to read it."""
    parts = [
        what,
        spec.description,
        f"Orientation: {spec.orientation}. Missing or undefined frames (NaN, null) are excluded, "
        "not zero-filled.",
    ]
    if not spec.view_axis:
        parts.append(
            "Measured on the fused 3D reconstruction, so this quantity has one value per (run, "
            f"frame) and no per-view breakdown; where a view appears it is the placeholder "
            f"'{VIEW_3D}', not a camera."
        )
    if style.drop_fliers and draws_distribution:
        parts.append(
            "Values outside their group's 1.5 x IQR fence are omitted from the figure; the "
            "mean and median labels, and metrics_summary.json / .csv, still cover every value."
        )
    if spec.combination_invariant:
        parts.append(
            "Note: per (frame, view) this quantity does not depend on the view combination, so "
            "differences across combination sizes reflect group composition only."
        )
    if total and drawn < total:
        parts.append(f"Point strips thinned: {drawn} of {total} raw values drawn.")
    return "\n".join(parts)


def _metric_is_3d(metric: str) -> bool:
    """Whether a metric belongs to the world-space 3D metric block."""
    return metric.startswith((
        "IoU_3d_",
        "keypoint_distance_3d",
        "body_length_3d",
        "volume_ratio_3d",
        "MPVE_3d_",
        "MPJPE_3d_",
    ))


def plot_measure(
    cells: CellTable,
    metric: str,
    keypoint: Optional[str],
    block: Dict[str, Any],
    plots_dir: Path,
    style: Style,
) -> List[Dict[str, Any]]:
    """Render every FIGURE_KIND for one measure; returns its index rows."""
    if block[GROUPING_OVERALL]["n_samples"] == 0:
        warn(f"{metric}{'' if keypoint is None else f' / {keypoint}'}: no finite values; no plots.")
        return []

    table = cells[(metric, keypoint)]
    spec = spec_for(metric)
    # A keypoint metric's second axis is a keypoint, the per-bone-group MPJPE's is a
    # bone group; both travel on the same field, so the noun comes from the spec.
    member = spec.member_noun
    stem = (
        slugify(metric) if keypoint is None
        else f"{slugify(metric)}__{slugify(member)}_{slugify(keypoint)}"
    )
    subject = spec.label if keypoint is None else f"{spec.label}, {member} '{keypoint}'"

    rows: List[Dict[str, Any]] = []
    for kind in FIGURE_KINDS:
        if kind.requires_view_axis and not spec.view_axis:
            # Drawing it would produce a single column labelled with the sentinel
            # view and invite the reader to compare cameras that were never
            # separable for this metric. Skipped silently: it is a property of the
            # metric, not a fault, and warning once per measure per figure would
            # bury the real warnings.
            continue
        # One generator per figure: identical thinning for identical inputs,
        # independent of the order the figures happen to be rendered in.
        rng = random.Random(STRIP_RNG_SEED)
        resolved = kind.resolve(style)
        path = _plot_path(plots_dir, stem, resolved["filename_kind"], style)
        title = f"{subject}: {kind.title}"
        style.metric_is_3d = _metric_is_3d(metric)
        fig, ax, n_groups, drawn, total = kind.builder(table, spec, block, style, rng)
        caption = _caption(
            spec, resolved["what"], drawn, total, style, kind.draws_distribution
        )
        _save(fig, ax, path, title, caption, style)
        rows.append(
            {
                "filename": path.name,
                "metric": metric,
                "keypoint": keypoint or "",
                "plot_kind": resolved["plot_kind"],
                "grouping": kind.grouping,
                "point_colouring": resolved["point_colouring"],
                "n_groups": n_groups,
                "n_points_drawn": drawn,
                "n_points_total": total,
                "orientation": spec.orientation,
                "description": f"{title}. {resolved['what']}",
            }
        )
    return rows


def generate_plots(cells: CellTable, summary: Dict[str, Any], plots_dir: Path, style: Style) -> int:
    """Render every measure and write plots_index.csv describing each figure."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for metric, block in summary.items():
        measures: List[Tuple[Optional[str], Dict[str, Any]]] = [(None, block)]
        measures += list(block.get("by_keypoint", {}).items())
        for keypoint, measure_block in measures:
            rows.extend(plot_measure(cells, metric, keypoint, measure_block, plots_dir, style))
        log(f"  {metric}: {len(measures)} measure(s) plotted")

    index_path = plots_dir / "plots_index.csv"
    with index_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=PLOT_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Wrote {len(rows)} figure(s) to {plots_dir}, indexed in {index_path}")
    return len(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def load_collected_metrics(path: Path) -> Dict[str, Any]:
    """Load collected_metrics.json (run key -> raw metrics payload)."""
    if not path.is_file():
        raise SystemExit(f"{path}: collected metrics file not found.")
    with path.open() as fp:
        collected = json.load(fp)  # json accepts the NaN literals DSKv2 writes
    if not isinstance(collected, dict):
        raise SystemExit(f"{path}: expected a JSON object mapping run key -> metrics.")
    if not collected:
        raise SystemExit(f"{path}: no runs to analyse.")
    return collected


def load_collected_3d_metrics(
    path: Optional[Path], explicit: bool
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Load collected_3d_metrics.json, or ({}, None) when there is none to load.

    A missing file is fatal only when the user named it: the 3D metrics come from a
    separate Blender pass that a sweep may simply not have run yet, so the default
    location is probed and the analysis proceeds 2D-only if it is not there.
    """
    if path is None:
        return {}, None
    resolved = path.expanduser()
    if not resolved.is_file():
        if explicit:
            raise SystemExit(f"{resolved}: 3D metrics file not found.")
        log(f"3D metrics       : none at {resolved} (2D only)")
        return {}, None
    with resolved.open() as fp:
        collected = json.load(fp)
    if not isinstance(collected, dict):
        raise SystemExit(f"{resolved}: expected a JSON object mapping run key -> 3D metrics.")
    if not collected:
        warn(f"{resolved}: no 3D runs in the file; ignoring it.")
        return {}, None
    log(f"3D metrics        : {resolved}")
    return collected, resolved


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the per-frame metrics of a view-combination sweep by view and by number of "
            "views, and plot location and distribution for every metric."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "collected_metrics_path",
        type=Path,
        nargs="?",
        default=DEFAULT_COLLECTED_PATH,
        help=f"Path to the sweep's collected_metrics.json. Default: {DEFAULT_COLLECTED_PATH}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for the summary and the plots. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--collected-3d-metrics",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to the batch operator's collected_3d_metrics.json. Default: "
            f"'{DEFAULT_3D_NAME}' resolved one level above the collected metrics file's "
            "directory, which is where 'Batch 3D Metrics from PTS2 Dir' writes it; silently "
            "skipped when absent. Its runs are matched to the 2D ones by run key and its "
            "metrics ("
            + ", ".join(list(THREE_D_SCALARS) + list(THREE_D_KEYPOINTS) + list(THREE_D_NESTED_SCALARS))
            + ") are summarised and plotted alongside them."
        ),
    )
    parser.add_argument(
        "--no-3d-metrics",
        action="store_true",
        help="Ignore collected_3d_metrics.json even if it is present.",
    )
    parser.add_argument(
        "--exclude-blocked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exclude blocked frames -- those the reconstruction emitted but did not obtain by "
            "fitting the optimizer to that frame, currently poses gap-filled by interpolation "
            "across a detection gap. On by default: scoring them measures the interpolation, "
            "and because the runs with the most gap-filled frames are the view combinations "
            "whose detections failed most, including them flatters exactly the weakest "
            "combinations. Excluded frames are still counted as available in the coverage "
            "ratio rho. Use --no-exclude-blocked to score every frame."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help=(
            "Restrict the analysis to these metric keys. Default: every per-view metric present in "
            "the file. Known keys: " + ", ".join(METRIC_REGISTRY)
        ),
    )
    parser.add_argument(
        "--primary-color",
        "--primary-colour",
        dest="primary_color",
        default=DEFAULT_PRIMARY_COLOR,
        metavar="COLOR",
        help=(
            "Primary figure colour, '#RRGGBB' or 'R,G,B' (0-255 or 0-1). The ordinal palette "
            "(number of views) is its lightness ramp, the nominal palette (view) its hue "
            f"rotation. Default: {DEFAULT_PRIMARY_COLOR}, turquoise."
        ),
    )
    parser.add_argument(
        "--monochrome",
        action="store_true",
        help="Use one colour for all colour-coded groups, using --primary-color for that colour.",
    )
    parser.add_argument(
        "--scientific",
        action="store_true",
        help=(
            "Use a compact publication-oriented figure style: smaller typography, reduced visual "
            "clutter, subtle grids, white backgrounds, and editable TrueType/vector text."
        ),
    )
    parser.add_argument(
        "--box-line-width",
        type=float,
        default=1.2,
        metavar="POINTS",
        help="Line width of box, whisker, cap, median and mean-cross strokes, in points. Default: 1.2.",
    )
    parser.add_argument(
        "--mean-marker",
        default="x",
        metavar="MARKER",
        help="Matplotlib marker used for the mean indicator. Default: 'x'.",
    )
    parser.add_argument(
        "--format",
        dest="format_pixels",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Exact output figure width and height in pixels. When set, every plot uses this size; --dpi determines the corresponding physical size.",
    )
    parser.add_argument(
        "--font",
        default=DEFAULT_FONT,
        metavar="FONT",
        help=(
            "Path to a .ttf/.otf/.ttc file, or a comma-separated list of installed family names "
            f"tried in order. Default: '{DEFAULT_FONT}'."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of the saved figures. Default: 300.",
    )
    parser.add_argument(
        "--plot-format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure file format. Default: png.",
    )
    parser.add_argument(
        "--distribution-style",
        default=DISTRIBUTION_POINTS,
        choices=[DISTRIBUTION_POINTS, DISTRIBUTION_BOXES],
        help=(
            "How the distribution beside each summary box is drawn: 'points' plots every raw "
            f"frame value, 'boxes' replaces them with one box per colour group, giving several "
            f"boxes side by side. Default: {DISTRIBUTION_POINTS}."
        ),
    )
    parser.add_argument(
        "--drop-fliers",
        action="store_true",
        help=(
            "Omit values outside their group's 1.5 x IQR fence from the figures, in both "
            "distribution styles. The summary files and the mean/median labels are unaffected."
        ),
    )
    parser.add_argument(
        "--no-overall-pooling",
        action="store_true",
        help=(
            "In the two 'distribution over all runs' figures, drop the single pooled box and put "
            "the groups that flanked it on their own x axis -- one box per number of views, or "
            "one per view -- laid out and labelled like the per-#views and per-view figures. "
            f"Applies to --distribution-style={DISTRIBUTION_BOXES}, the style that draws those "
            "groups as boxes; the figure names and captions follow the change."
        ),
    )
    parser.add_argument(
        "--dynamic-y-axis",
        action="store_true",
        help=(
            "Break the y axis of a box figure when one group's box and whiskers sit far outside "
            "the rest -- two views against three or more in an MPJPE, say -- instead of letting "
            "it compress every other box into a band a few pixels high. The outlier range is "
            "drawn in a small upper panel and the rest in a larger lower one, both linear, with "
            "a cut mark on the axis and on every box that spans the break. The break is only "
            "taken when the groups really do fall into two well separated clusters."
        ),
    )
    parser.add_argument(
        "--no-value-labels",
        action="store_true",
        help="Do not print the mean and median next to each box and line marker.",
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help=(
            "Do not print the descriptive caption under each figure. Titles, axis labels and "
            "plots_index.csv still describe every figure."
        ),
    )
    parser.add_argument(
        "--max-points-per-group",
        type=int,
        default=1500,
        help=(
            "Cap on the raw points drawn per colour sub-band. Thinning is deterministic and is "
            "reported in the caption and in plots_index.csv; 0 disables it. Default: 1500."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and write the numeric summary, but generate no plots.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    collected_path = args.collected_metrics_path.expanduser()
    out_dir = args.out_dir.expanduser()
    # Validated before any work is done, so a typo fails immediately rather than
    # after the aggregation has already been written.
    primary = parse_color(args.primary_color)
    if not math.isfinite(args.box_line_width) or args.box_line_width <= 0.0:
        raise SystemExit("--box-line-width must be a finite value greater than 0.")
    try:
        MarkerStyle(args.mean_marker)
    except (ValueError, TypeError):
        raise SystemExit(f"--mean-marker: invalid matplotlib marker {args.mean_marker!r}.")
    if args.format_pixels is not None and any(v <= 0 for v in args.format_pixels):
        raise SystemExit("--format WIDTH HEIGHT must use positive pixel dimensions.")

    log(f"Collected metrics : {collected_path}")
    log(f"Output directory  : {out_dir}")

    EXCLUDE_BLOCKED[0] = bool(args.exclude_blocked)

    collected = load_collected_metrics(collected_path)

    explicit_3d = args.collected_3d_metrics is not None
    if args.no_3d_metrics:
        if explicit_3d:
            raise SystemExit("--collected-3d-metrics and --no-3d-metrics are contradictory.")
        path_3d = None
    elif explicit_3d:
        path_3d = args.collected_3d_metrics
    else:
        path_3d = collected_path.parent.parent / DEFAULT_3D_NAME
    collected_3d, resolved_3d = load_collected_3d_metrics(path_3d, explicit_3d)

    if collected_3d:
        only_2d = sorted(set(collected) - set(collected_3d))
        only_3d = sorted(set(collected_3d) - set(collected))
        # Not fatal in either direction: a sweep may have been re-run partially, and
        # the groupings pool per metric, so a run missing on one side simply does not
        # contribute there. It has to be visible, though, or a metric silently
        # summarising a different subset of runs than its neighbour looks comparable.
        if only_2d:
            warn(f"{len(only_2d)} run(s) have 2D metrics but no 3D ones, e.g. {only_2d[0]}")
        if only_3d:
            warn(f"{len(only_3d)} run(s) have 3D metrics but no 2D ones, e.g. {only_3d[0]}")

    # Filled by the sample iterators as a side channel: rho needs the frames that were
    # EXCLUDED as well as the ones that were kept, and those never reach the cell table.
    availability: Dict[str, Dict[str, int]] = {}

    # One table over both sources: same Sample shape, same run keys, so every
    # grouping, summary and figure downstream treats them identically.
    cells = build_cell_table(
        itertools.chain(
            iter_samples(collected, args.metrics,
                         exclude_blocked=args.exclude_blocked,
                         availability=availability),
            # No availability side channel: the 3D runs share the 2D run keys, so they would
            # collide in one map, and rho is a statement about a run's per-frame ARRAYS. The
            # 3D files carry their own coverage block, computed at the source in Blender.
            iter_3d_samples(collected_3d, args.metrics,
                            exclude_blocked=args.exclude_blocked),
        )
    )
    if not cells:
        raise SystemExit("No usable per-view metrics found; nothing to summarize.")

    summary = summarize(cells)
    report = build_report(summary, cells, collected_path)
    if resolved_3d is not None:
        report["meta"]["source_3d"] = str(resolved_3d)
    report["meta"]["frame_availability"] = coverage_report(availability)
    log("Metrics           : " + ", ".join(summary))
    write_summary(report, out_dir)

    if args.dry_run:
        log("[dry-run] skipping plot generation.")
        return

    style = Style(
        primary=primary,
        font_family=resolve_font(args.font),
        dpi=args.dpi,
        fmt=args.plot_format,
        max_points=args.max_points_per_group,
        distribution_style=args.distribution_style,
        drop_fliers=args.drop_fliers,
        overall_pooling=not args.no_overall_pooling,
        dynamic_y_axis=args.dynamic_y_axis,
        value_labels=not args.no_value_labels,
        captions=not args.no_captions,
        monochrome=args.monochrome,
        box_line_width=args.box_line_width,
        mean_marker=args.mean_marker,
        format_pixels=tuple(args.format_pixels) if args.format_pixels is not None else None,
    )
    style.apply()
    log(f"Figure font       : {style.font_family or plt.rcParams['font.family']}")
    generate_plots(cells, summary, out_dir / "plots", style)


if __name__ == "__main__":
    main()