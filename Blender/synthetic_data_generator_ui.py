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
        "timing",
        "total_duration_min",
        "seconds_per_frame",
    }
)

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
}


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

# Mirrors sweep_view_combinations.MAX_LEAF_NAME_LEN, for the same reason: stay
# clear of the 255-byte ext4 filename limit.
MAX_PLOT_NAME_LEN = 200

DEFAULT_COLLECTED_PATH = Path("metrics_collected/collected_metrics.json")
DEFAULT_OUT_DIR = Path("analysis_output")

GROUPING_OVERALL = "overall"
GROUPING_PER_VIEW = "per_view_overall"
GROUPING_PER_VIEW_PER_N = "per_view_per_n_views"
GROUPING_PER_N = "per_n_views"

DEFAULT_PRIMARY_COLOR = "#40E0D0"  # turquoise
# Debian's fonts-linuxlibertine installs the family as 'Linux Biolinum O'; the
# bare name is tried first so a differently packaged install also resolves.
DEFAULT_FONT = "Linux Biolinum,Linux Biolinum O"
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


def _iter_scalar_samples(
    metric: str, metric_data: Dict[str, Any], n_views: int, run_key: str
) -> Iterator[Sample]:
    for view, values in metric_data.items():
        if not isinstance(values, list):
            warn(f"{run_key}: '{metric}' / '{view}' is not a frame list; skipping it.")
            continue
        view_name = sys.intern(str(view))
        for value in values:
            yield Sample(metric, None, view_name, n_views, _to_float(value))


def _iter_keypoint_samples(
    metric: str, metric_data: Dict[str, Any], n_views: int, run_key: str
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
            for value in values:
                yield Sample(metric, keypoint_name, view_name, n_views, _to_float(value))


def iter_samples(
    collected: Dict[str, Any], wanted_metrics: Optional[Sequence[str]] = None
) -> Iterator[Sample]:
    """
    Stream every run's per-frame values as tidy samples. A run with an
    unparseable key or a malformed payload is warned about and skipped rather
    than aborting the analysis, as is an individual malformed metric entry.
    """
    n_runs = 0
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
        for metric in sorted(set(run_metrics) - NON_METRIC_KEYS, key=metric_sort_key):
            if wanted_metrics is not None and metric not in wanted_metrics:
                continue
            kind = classify_metric(run_metrics[metric], view_names)
            if kind is None:
                warn(f"{run_key}: '{metric}' is not a per-view frame series; skipping it.")
                continue
            metric_name = sys.intern(metric)
            if kind == SCALAR:
                yield from _iter_scalar_samples(metric_name, run_metrics[metric], n_views, run_key)
            else:
                yield from _iter_keypoint_samples(metric_name, run_metrics[metric], n_views, run_key)

    log(f"Read {n_runs} run(s).")


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


def build_report(summary: Dict[str, Any], cells: CellTable, source: Path) -> Dict[str, Any]:
    """Summary plus the provenance needed to read it without the source file."""
    reference_table = cells[(next(iter(summary)), None)]
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
                "axes.linewidth": 0.8,
                "axes.axisbelow": True,
                "axes.titlelocation": "left",
                "axes.titlepad": 7.0,
                "xtick.direction": "out",
                "ytick.direction": "out",
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
        if self.font_family:
            plt.rcParams["font.family"] = [self.font_family]

    def palette(self, kind: str, n: int) -> List[RGB]:
        """Cached palette; `kind` is 'n_views' (ordinal) or 'view' (nominal)."""
        key = f"{kind}:{n}"
        if key not in self._palettes:
            builder = sequential_palette if kind == "n_views" else categorical_palette
            self._palettes[key] = builder(self.primary, n)
        return self._palettes[key]


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


def _axis_cosmetics(ax: Axes, ylabel: str, xlabel: str = "") -> None:
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis="y", which="major")


def _figure_width(n_positions: int, per_position: float, minimum: float) -> float:
    return max(minimum, per_position * n_positions)


CAPTION_FONT_SIZE = 6.5
CAPTION_CHARS_PER_INCH = 21  # at CAPTION_FONT_SIZE, close enough for wrapping


def _save(fig: Figure, ax: Axes, out_path: Path, title: str, caption: str, style: Style) -> None:
    """
    Title above the axes, caption beneath the figure, as in a paper. The caption
    is anchored to the drawn extent of the figure rather than to the axes box,
    so it clears tick labels and legends whatever their size, and is hard-wrapped
    to that extent instead of relying on matplotlib's word wrapping.
    """
    ax.set_title(title)
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
    fig.savefig(out_path, format=style.fmt)
    plt.close(fig)


def _thin(values: Sequence[float], limit: int, rng: random.Random) -> List[float]:
    """Deterministic thinning of an over-full point strip."""
    if limit <= 0 or len(values) <= limit:
        return list(values)
    return rng.sample(list(values), limit)


def _draw_boxes(
    ax: Axes,
    positions: Sequence[float],
    datasets: Sequence[Sequence[float]],
    color: RGB,
    width: float = BOX_WIDTH,
) -> None:
    """
    Box = interquartile range, line = median, whiskers = 1.5 x IQR, diamond =
    mean. Fliers are suppressed because every raw point is drawn beside the box.
    """
    keep = [(p, list(d)) for p, d in zip(positions, datasets) if len(d) > 0]
    if not keep:
        return
    ax.boxplot(
        [d for _p, d in keep],
        positions=[p for p, _d in keep],
        widths=width,
        showfliers=False,
        whis=1.5,
        patch_artist=True,
        showmeans=True,
        manage_ticks=False,
        boxprops={"facecolor": tint(color, 0.80), "edgecolor": color, "linewidth": 0.9},
        whiskerprops={"color": color, "linewidth": 0.9},
        capprops={"color": color, "linewidth": 0.9},
        medianprops={"color": "black", "linewidth": 1.3},
        meanprops={
            "marker": "D",
            "markersize": 3.0,
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 0.6,
        },
        zorder=3,
    )


def _draw_strip(
    ax: Axes,
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
        if not values:
            continue
        shown = _thin(values, style.max_points, rng)
        drawn += len(shown)
        centre = position + offset + (index - (n_groups - 1) / 2.0) * sub_width
        jitter = [centre + rng.uniform(-0.38, 0.38) * sub_width for _ in shown]
        ax.scatter(jitter, shown, s=3.0, color=color, alpha=0.45, linewidths=0.0,
                   zorder=2, rasterized=True)
    return drawn, total


def _distribution_legend(ax: Axes, title: str, labels: Sequence[str], colors: Sequence[RGB]) -> None:
    """Colour key for the points, plus the box's median and mean symbols."""
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.5, color=color, label=label)
        for label, color in zip(labels, colors)
    ]
    handles += [
        Line2D([], [], color="black", linewidth=1.3, label="median"),
        Line2D([], [], marker="D", linestyle="none", markersize=3.0, markerfacecolor="white",
               markeredgecolor="black", label="mean"),
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

    fig, ax = plt.subplots(figsize=(_figure_width(len(views), 1.10, 4.8), 3.5))
    ax.bar(
        positions,
        [stats[v]["mean"] for v in views],
        width=0.62,
        facecolor=tint(style.primary, 0.55),
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

    fig, ax = plt.subplots(figsize=(_figure_width(len(views), 1.45, 5.6), 3.6))
    for index, (k, color) in enumerate(zip(n_values, colors)):
        offset = -0.39 + width * (index + 0.5)
        positions = [x + offset for x in range(len(views))]
        cell_stats = [stats[v].get(str(k)) for v in views]
        ax.bar(
            positions,
            [s["mean"] if s else float("nan") for s in cell_stats],
            width=width, facecolor=tint(color, 0.35), edgecolor=color, linewidth=0.7,
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

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.plot(n_values, [stats[str(k)]["mean"] for k in n_values], marker="o",
            color=style.primary, label="mean")
    ax.plot(n_values, [stats[str(k)]["median"] for k in n_values], marker="s",
            markerfacecolor="white", linestyle="--",
            color=_with_lightness(style.primary, 0.28), label="median")
    ax.set_xticks(n_values)
    _axis_cosmetics(ax, spec.label, "number of views in the reconstruction")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax, len(n_values), 0, 0


def fig_box_per_view(table, spec, block, style, rng) -> FigureResult:
    """One box per view over all runs; points beside it, coloured by #views."""
    views = views_of(table)
    n_values = n_views_of(table)
    colors = style.palette("n_views", len(n_values))
    stats = block[GROUPING_PER_VIEW]

    fig, ax = plt.subplots(figsize=(_figure_width(len(views), 1.30, 5.2), 3.8))
    drawn = total = 0
    for position, view in enumerate(views):
        _draw_boxes(ax, [position], [pooled(table, view=view)], style.primary)
        groups = [(k, _finite(table.get((view, k), []))) for k in n_values]
        d, t = _draw_strip(ax, position, groups, colors, rng, style)
        drawn, total = drawn + d, total + t
    ax.set_xticks(list(range(len(views))))
    ax.set_xticklabels([_view_label(v, stats[v]["n_samples"]) for v in views])
    _axis_cosmetics(ax, spec.label, "view")
    _distribution_legend(ax, "#views", [str(k) for k in n_values], colors)
    return fig, ax, len(views), drawn, total


def fig_box_per_view_and_n_views(table, spec, block, style, rng) -> FigureResult:
    """One box per (view, #views) cell; points beside it, coloured by #views."""
    views = views_of(table)
    n_values = n_views_of(table)
    colors = style.palette("n_views", len(n_values))
    span = 0.84
    step = span / max(len(n_values), 1)

    width = _figure_width(len(views) * max(len(n_values), 1), 0.48, 5.6)
    fig, ax = plt.subplots(figsize=(width, max(3.8, min(width * 0.36, 5.4))))
    drawn = total = boxes = 0
    for view_index, view in enumerate(views):
        for k_index, (k, color) in enumerate(zip(n_values, colors)):
            values = _finite(table.get((view, k), []))
            if not values:
                continue
            position = view_index - span / 2 + step * (k_index + 0.5)
            _draw_boxes(ax, [position - step * 0.20], [values], color, width=step * 0.34)
            d, t = _draw_strip(
                ax, position, [(k, values)], [color], rng, style,
                offset=step * 0.22, width=step * 0.34,
            )
            drawn, total, boxes = drawn + d, total + t, boxes + 1
    ax.set_xticks(list(range(len(views))))
    ax.set_xticklabels([_view_label(v) for v in views])
    _axis_cosmetics(ax, spec.label, "view")
    _distribution_legend(ax, "#views", [str(k) for k in n_values], colors)
    return fig, ax, boxes, drawn, total


def fig_box_vs_n_views(table, spec, block, style, rng) -> FigureResult:
    """One box per #views over all views; points beside it, coloured by view."""
    views = views_of(table)
    n_values = n_views_of(table)
    colors = style.palette("view", len(views))
    stats = block[GROUPING_PER_N]

    fig, ax = plt.subplots(figsize=(_figure_width(len(n_values), 1.2, 4.6), 3.8))
    drawn = total = 0
    for position, k in enumerate(n_values):
        _draw_boxes(ax, [position], [pooled(table, n_views=k)], style.primary)
        groups = [(view, _finite(table.get((view, k), []))) for view in views]
        d, t = _draw_strip(ax, position, groups, colors, rng, style)
        drawn, total = drawn + d, total + t
    ax.set_xticks(list(range(len(n_values))))
    ax.set_xticklabels([f"{k}\nn = {stats[str(k)]['n_samples']}" for k in n_values])
    _axis_cosmetics(ax, spec.label, "number of views in the reconstruction")
    _distribution_legend(ax, "view", list(views), colors)
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

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    _draw_boxes(ax, [0.0], [pooled(table)], style.primary, width=0.26)
    drawn, total = _draw_strip(ax, 0.0, groups, colors, rng, style)
    ax.set_xticks([0.14])
    ax.set_xticklabels([f"all runs pooled\nn = {block[GROUPING_OVERALL]['n_samples']}"])
    ax.set_xlim(-0.28, 0.58)
    _axis_cosmetics(ax, spec.label)
    _distribution_legend(ax, legend_title, labels, colors)
    return fig, ax, 1, drawn, total


def fig_box_overall_by_n_views(table, spec, block, style, rng) -> FigureResult:
    return _fig_box_overall(table, spec, block, style, rng, colour_by="n_views")


def fig_box_overall_by_view(table, spec, block, style, rng) -> FigureResult:
    return _fig_box_overall(table, spec, block, style, rng, colour_by="view")


# --------------------------------------------------------------------------
# Figure orchestration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureKind:
    """
    One figure recipe. `filename_kind` is the self-describing file-name stem;
    `what` becomes the caption's first sentence, stating exactly what is pooled.
    """

    filename_kind: str
    builder: Callable[..., FigureResult]
    title: str
    what: str
    plot_kind: str
    grouping: str
    point_colouring: str


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
        "box_iqr_with_all_points_per_view__points_coloured_by_number_of_views",
        fig_box_per_view,
        "distribution per view",
        "The box spans the interquartile range with the median as a line, whiskers at 1.5 x IQR "
        "and the mean as a diamond; beside it every raw frame value of that view is drawn, in one "
        "sub-band per number of views.",
        "box plot (IQR) with all raw points",
        GROUPING_PER_VIEW,
        "points coloured by #views",
    ),
    FigureKind(
        "box_iqr_with_all_points_per_view_and_number_of_views__points_coloured_by_number_of_views",
        fig_box_per_view_and_n_views,
        "distribution per view and #views",
        "One box per (view, number of views) cell, with that cell's raw frame values drawn beside "
        "it; box and points share the colour of their combination size.",
        "box plot (IQR) with all raw points",
        GROUPING_PER_VIEW_PER_N,
        "boxes and points coloured by #views",
    ),
    FigureKind(
        "box_iqr_with_all_points_versus_number_of_views__points_coloured_by_view",
        fig_box_vs_n_views,
        "distribution versus #views",
        "One box per combination size, pooling all of its views, with every raw frame value drawn "
        "beside it in one sub-band per view.",
        "box plot (IQR) with all raw points",
        GROUPING_PER_N,
        "points coloured by view",
    ),
    FigureKind(
        "box_iqr_with_all_points_all_runs_pooled__points_coloured_by_number_of_views",
        fig_box_overall_by_n_views,
        "distribution over all runs",
        "A single box over every run, view and frame; the raw values beside it are grouped and "
        "coloured by the number of views of the run they come from.",
        "box plot (IQR) with all raw points",
        GROUPING_OVERALL,
        "points coloured by #views",
    ),
    FigureKind(
        "box_iqr_with_all_points_all_runs_pooled__points_coloured_by_view",
        fig_box_overall_by_view,
        "distribution over all runs",
        "The same pooled box, with the raw values grouped and coloured by the view they come from.",
        "box plot (IQR) with all raw points",
        GROUPING_OVERALL,
        "points coloured by view",
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


def _caption(spec: MetricSpec, what: str, drawn: int, total: int) -> str:
    """Figure caption: what is pooled, what the metric is, how to read it."""
    parts = [
        what,
        spec.description,
        f"Orientation: {spec.orientation}. Missing or undefined frames (NaN, null) are excluded, "
        "not zero-filled.",
    ]
    if spec.combination_invariant:
        parts.append(
            "Note: per (frame, view) this quantity does not depend on the view combination, so "
            "differences across combination sizes reflect group composition only."
        )
    if total and drawn < total:
        parts.append(f"Point strips thinned: {drawn} of {total} raw values drawn.")
    return "\n".join(parts)


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
    stem = slugify(metric) if keypoint is None else f"{slugify(metric)}__keypoint_{slugify(keypoint)}"
    subject = spec.label if keypoint is None else f"{spec.label}, keypoint '{keypoint}'"

    rows: List[Dict[str, Any]] = []
    for kind in FIGURE_KINDS:
        # One generator per figure: identical thinning for identical inputs,
        # independent of the order the figures happen to be rendered in.
        rng = random.Random(STRIP_RNG_SEED)
        path = _plot_path(plots_dir, stem, kind.filename_kind, style)
        title = f"{subject}: {kind.title}"
        fig, ax, n_groups, drawn, total = kind.builder(table, spec, block, style, rng)
        _save(fig, ax, path, title, _caption(spec, kind.what, drawn, total), style)
        rows.append(
            {
                "filename": path.name,
                "metric": metric,
                "keypoint": keypoint or "",
                "plot_kind": kind.plot_kind,
                "grouping": kind.grouping,
                "point_colouring": kind.point_colouring,
                "n_groups": n_groups,
                "n_points_drawn": drawn,
                "n_points_total": total,
                "orientation": spec.orientation,
                "description": f"{title}. {kind.what}",
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

    log(f"Collected metrics : {collected_path}")
    log(f"Output directory  : {out_dir}")

    collected = load_collected_metrics(collected_path)
    cells = build_cell_table(iter_samples(collected, args.metrics))
    if not cells:
        raise SystemExit("No usable per-view metrics found; nothing to summarize.")

    summary = summarize(cells)
    report = build_report(summary, cells, collected_path)
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
    )
    style.apply()
    log(f"Figure font       : {style.font_family or plt.rcParams['font.family']}")
    generate_plots(cells, summary, out_dir / "plots", style)


if __name__ == "__main__":
    main()