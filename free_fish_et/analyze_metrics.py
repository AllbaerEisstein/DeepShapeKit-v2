#!/usr/bin/env python3
"""
Analyse the metrics collected by `sweep_view_combinations.py --collect-results`.

The sweep reconstructs the same sequence from every camera-view combination of
size >= 2 and writes one metrics file per run; the collector merges them into
`<out_path>/metrics_collected/collected_metrics.json`, a mapping

    run key (see combo_folder_name) -> raw metrics_instance_N.json content

This script pools the RAW per-frame values (never per-run pre-averages) of

    IoU_reconstruction_and_gt
    IoU_reconstruction_and_mask_detection
    keypoint_L2_distance

over three groupings -- per view, per (view, #views), per #views -- and, for the
keypoint metric, repeats those groupings for each individual keypoint as well as
for all keypoints pooled. Missing/undetected frames are encoded as NaN (float
metrics) or null (keypoint metric) and are excluded from every statistic rather
than coerced to zero, following DSKv2_demo._mean_median.

Usage:
    python analyze_metrics.py out/metrics_collected/collected_metrics.json
    python analyze_metrics.py --out-dir analysis_output --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, DefaultDict, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402

# Metrics shaped {view: [value per frame]} and {view: {keypoint: [value per frame]}}.
SCALAR_METRICS: Tuple[str, ...] = (
    "IoU_reconstruction_and_gt",
    "IoU_reconstruction_and_mask_detection",
)
KEYPOINT_METRIC = "keypoint_L2_distance"
METRIC_ORDER: Tuple[str, ...] = SCALAR_METRICS + (KEYPOINT_METRIC,)

# combo_folder_name() emits 'k{n}__v{i0-i1-...}' with an optional descriptive
# suffix that is dropped when the leaf name would exceed 200 chars, so the view
# count is taken from the prefix only.
RUN_KEY_PATTERN = re.compile(r"^k(\d+)__")

DEFAULT_COLLECTED_PATH = Path("metrics_collected/collected_metrics.json")
DEFAULT_OUT_DIR = Path("analysis_output")

GROUPING_PER_VIEW = "per_view_overall"
GROUPING_PER_VIEW_PER_N = "per_view_per_n_views"
GROUPING_PER_N = "per_n_views"


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


def parse_n_views(run_key: str) -> Optional[int]:
    """Number of views in a run, from the run key prefix. None if unparseable."""
    match = RUN_KEY_PATTERN.match(run_key)
    return int(match.group(1)) if match else None


def _to_float(value: Any) -> float:
    """Frame value as float; None, bools and junk become NaN (i.e. 'missing')."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def _finite(values: Sequence[Any]) -> List[float]:
    """Drop None and NaN (missing/undetected frames) and cast the rest to float."""
    return [
        float(v)
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]


def _iter_scalar_samples(
    metric: str, metric_data: Any, n_views: int, run_key: str
) -> Iterator[Sample]:
    if not isinstance(metric_data, dict):
        warn(f"{run_key}: metric '{metric}' is not a view mapping; skipping it.")
        return
    for view, values in metric_data.items():
        if not isinstance(values, list):
            warn(f"{run_key}: metric '{metric}', view '{view}' is not a list; skipping it.")
            continue
        for value in values:
            yield Sample(metric, None, str(view), n_views, _to_float(value))


def _iter_keypoint_samples(
    metric: str, metric_data: Any, n_views: int, run_key: str
) -> Iterator[Sample]:
    if not isinstance(metric_data, dict):
        warn(f"{run_key}: metric '{metric}' is not a view mapping; skipping it.")
        return
    for view, keypoints in metric_data.items():
        if not isinstance(keypoints, dict):
            warn(f"{run_key}: metric '{metric}', view '{view}' is not a keypoint mapping; skipping it.")
            continue
        for keypoint, values in keypoints.items():
            if not isinstance(values, list):
                warn(
                    f"{run_key}: metric '{metric}', view '{view}', keypoint "
                    f"'{keypoint}' is not a list; skipping it."
                )
                continue
            for value in values:
                yield Sample(metric, str(keypoint), str(view), n_views, _to_float(value))


def build_samples(collected: Dict[str, Any]) -> List[Sample]:
    """
    Flatten the collected runs into a tidy list of samples. Runs with an
    unparseable key or a malformed metrics payload are warned about and skipped
    rather than aborting the analysis.
    """
    samples: List[Sample] = []
    n_runs = 0

    for run_key, run_metrics in collected.items():
        n_views = parse_n_views(run_key)
        if n_views is None:
            warn(f"{run_key}: run key does not start with 'k<N>__'; skipping run.")
            continue
        if not isinstance(run_metrics, dict):
            warn(f"{run_key}: metrics payload is not an object; skipping run.")
            continue

        n_runs += 1
        for metric in SCALAR_METRICS:
            if metric not in run_metrics:
                warn(f"{run_key}: metric '{metric}' missing; skipping it.")
                continue
            samples.extend(_iter_scalar_samples(metric, run_metrics[metric], n_views, run_key))

        if KEYPOINT_METRIC not in run_metrics:
            warn(f"{run_key}: metric '{KEYPOINT_METRIC}' missing; skipping it.")
        else:
            samples.extend(
                _iter_keypoint_samples(KEYPOINT_METRIC, run_metrics[KEYPOINT_METRIC], n_views, run_key)
            )

    n_finite = sum(1 for s in samples if not math.isnan(s.value))
    log(
        f"Parsed {n_runs} run(s) into {len(samples)} frame value(s), "
        f"{len(samples) - n_finite} of them missing (NaN/null)."
    )
    return samples


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
    """Group summary: finite-sample count plus the DSKv2 mean/median pair."""
    finite_values = _finite(values)
    return {"n_samples": len(finite_values), **_mean_median(finite_values, metric_name)}


# (metric, keypoint or None) identifies one "measure"; keypoint=None pools all
# keypoints for KEYPOINT_METRIC and is the only key for the scalar metrics.
MeasureKey = Tuple[str, Optional[str]]


def summarize(samples: Sequence[Sample]) -> Dict[str, Any]:
    """
    Pool the tidy samples into the three groupings for every measure and return
    the nested summary written to metrics_summary.json.
    """
    per_view: DefaultDict[Tuple[MeasureKey, str], List[float]] = defaultdict(list)
    per_view_per_n: DefaultDict[Tuple[MeasureKey, str, int], List[float]] = defaultdict(list)
    per_n: DefaultDict[Tuple[MeasureKey, int], List[float]] = defaultdict(list)
    views_seen: DefaultDict[MeasureKey, set] = defaultdict(set)
    n_views_seen: DefaultDict[MeasureKey, set] = defaultdict(set)
    keypoints_seen: DefaultDict[str, set] = defaultdict(set)

    for sample in samples:
        measures: List[MeasureKey] = [(sample.metric, None)]
        if sample.keypoint is not None:
            # A keypoint sample feeds both its own measure and the pooled one.
            measures.append((sample.metric, sample.keypoint))
            keypoints_seen[sample.metric].add(sample.keypoint)
        for measure in measures:
            per_view[(measure, sample.view)].append(sample.value)
            per_view_per_n[(measure, sample.view, sample.n_views)].append(sample.value)
            per_n[(measure, sample.n_views)].append(sample.value)
            views_seen[measure].add(sample.view)
            n_views_seen[measure].add(sample.n_views)

    def measure_block(measure: MeasureKey) -> Dict[str, Any]:
        """The three groupings of one (metric, keypoint-or-None) measure."""
        metric = measure[0]
        views = sorted(views_seen[measure])
        n_views_values = sorted(n_views_seen[measure])
        return {
            GROUPING_PER_VIEW: {
                view: _stats(per_view[(measure, view)], metric) for view in views
            },
            GROUPING_PER_VIEW_PER_N: {
                view: {
                    str(k): _stats(per_view_per_n[(measure, view, k)], metric)
                    for k in n_views_values
                    if (measure, view, k) in per_view_per_n
                }
                for view in views
            },
            GROUPING_PER_N: {
                str(k): _stats(per_n[(measure, k)], metric) for k in n_views_values
            },
        }

    summary: Dict[str, Any] = {}
    for metric in METRIC_ORDER:
        if (metric, None) not in views_seen:
            continue
        block = measure_block((metric, None))
        if metric == KEYPOINT_METRIC:
            block["by_keypoint"] = {
                keypoint: measure_block((metric, keypoint))
                for keypoint in sorted(keypoints_seen[metric])
            }
        summary[metric] = block

    return summary


# --------------------------------------------------------------------------
# Output: JSON + flat CSV
# --------------------------------------------------------------------------

CSV_COLUMNS = ["metric", "keypoint", "grouping", "view", "n_views", "n_samples", "mean", "median"]


def _csv_rows_for_block(metric: str, keypoint: str, block: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    for view, stats in block[GROUPING_PER_VIEW].items():
        yield {
            "metric": metric, "keypoint": keypoint, "grouping": GROUPING_PER_VIEW,
            "view": view, "n_views": "", **stats,
        }
    for view, by_n in block[GROUPING_PER_VIEW_PER_N].items():
        for n_views, stats in by_n.items():
            yield {
                "metric": metric, "keypoint": keypoint, "grouping": GROUPING_PER_VIEW_PER_N,
                "view": view, "n_views": n_views, **stats,
            }
    for n_views, stats in block[GROUPING_PER_N].items():
        yield {
            "metric": metric, "keypoint": keypoint, "grouping": GROUPING_PER_N,
            "view": "", "n_views": n_views, **stats,
        }


def summary_to_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the nested summary into one row per (measure, grouping, group)."""
    rows: List[Dict[str, Any]] = []
    for metric, block in summary.items():
        rows.extend(_csv_rows_for_block(metric, "", block))
        for keypoint, kp_block in block.get("by_keypoint", {}).items():
            rows.extend(_csv_rows_for_block(metric, keypoint, kp_block))
    return rows


def write_summary(summary: Dict[str, Any], out_dir: Path) -> Tuple[Path, Path]:
    """Write metrics_summary.json and its flat metrics_summary.csv counterpart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics_summary.json"
    csv_path = out_dir / "metrics_summary.csv"

    with json_path.open("w") as fp:
        json.dump(summary, fp, indent=2)

    rows = summary_to_rows(summary)
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Wrote {json_path}")
    log(f"Wrote {csv_path} ({len(rows)} rows)")
    return json_path, csv_path


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Filename-safe token, matching sweep_view_combinations.sanitize()."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return token or "unnamed"


def _sorted_n_views(block: Dict[str, Any]) -> List[int]:
    return sorted(int(k) for k in block[GROUPING_PER_N])


def _bar_per_view_overall(block: Dict[str, Any], label: str, out_path: Path) -> None:
    """Mean per view as bars, median overlaid as a marker."""
    stats_by_view = block[GROUPING_PER_VIEW]
    views = sorted(stats_by_view)
    if not views:
        return
    positions = range(len(views))
    means = [stats_by_view[v]["mean"] for v in views]
    medians = [stats_by_view[v]["median"] for v in views]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(views)), 5.0))
    ax.bar(positions, means, width=0.6, color="#4C72B0", label="mean")
    ax.scatter(positions, medians, marker="D", color="black", zorder=3, label="median")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(views, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(f"{label} per view (all runs pooled)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _bar_per_view_per_n_views(block: Dict[str, Any], label: str, out_path: Path) -> None:
    """Mean per view grouped by #views, median overlaid as a marker per bar."""
    stats_by_view = block[GROUPING_PER_VIEW_PER_N]
    views = sorted(stats_by_view)
    n_views_values = _sorted_n_views(block)
    if not views or not n_views_values:
        return

    width = 0.8 / len(n_views_values)
    colors = plt.get_cmap("viridis")([
        i / max(1, len(n_views_values) - 1) for i in range(len(n_views_values))
    ])

    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * len(views)), 5.0))
    for i, n_views in enumerate(n_views_values):
        offset = -0.4 + width * (i + 0.5)
        positions = [x + offset for x in range(len(views))]
        means = [stats_by_view[v].get(str(n_views), {}).get("mean", float("nan")) for v in views]
        medians = [stats_by_view[v].get(str(n_views), {}).get("median", float("nan")) for v in views]
        ax.bar(positions, means, width=width, color=colors[i], label=f"k={n_views}")
        ax.scatter(positions, medians, marker="D", s=14, color="black", zorder=3)

    ax.set_xticks(list(range(len(views))))
    ax.set_xticklabels(views, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(f"{label} per view, by number of views (bars = mean, diamonds = median)")
    ax.legend(title="#views", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _line_per_n_views(block: Dict[str, Any], label: str, out_path: Path) -> None:
    """Mean and median against the number of views, pooled across views."""
    stats_by_n = block[GROUPING_PER_N]
    n_views_values = _sorted_n_views(block)
    if not n_views_values:
        return
    means = [stats_by_n[str(k)]["mean"] for k in n_views_values]
    medians = [stats_by_n[str(k)]["median"] for k in n_views_values]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(n_views_values, means, marker="o", label="mean")
    ax.plot(n_views_values, medians, marker="s", linestyle="--", label="median")
    ax.set_xticks(n_views_values)
    ax.set_xlabel("number of views in the reconstruction")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs. number of views")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_plots(summary: Dict[str, Any], plots_dir: Path) -> int:
    """Render the three figure families for every measure. Returns file count."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    def render(block: Dict[str, Any], metric: str, keypoint: Optional[str]) -> int:
        stem = slugify(metric) if keypoint is None else f"{slugify(metric)}_{slugify(keypoint)}"
        label = metric if keypoint is None else f"{metric} [{keypoint}]"
        targets = [
            (_bar_per_view_overall, plots_dir / f"bar_perview_overall_{stem}.png"),
            (_bar_per_view_per_n_views, plots_dir / f"bar_perview_perviews_{stem}.png"),
            (_line_per_n_views, plots_dir / f"line_perviews_{stem}.png"),
        ]
        for plot_fn, path in targets:
            plot_fn(block, label, path)
        return len(targets)

    for metric, block in summary.items():
        written += render(block, metric, None)
        for keypoint, kp_block in block.get("by_keypoint", {}).items():
            written += render(kp_block, metric, keypoint)

    log(f"Wrote {written} plot(s) to {plots_dir}")
    return written


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
            "Aggregate the per-frame metrics of a view-combination sweep by view "
            "and by number of views, and plot the result."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "collected_metrics_path",
        type=Path,
        nargs="?",
        default=DEFAULT_COLLECTED_PATH,
        help=(
            "Path to the sweep's collected_metrics.json. Default: "
            f"{DEFAULT_COLLECTED_PATH}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for the summary and plots. Default: {DEFAULT_OUT_DIR}",
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

    log(f"Collected metrics : {collected_path}")
    log(f"Output directory  : {out_dir}")

    collected = load_collected_metrics(collected_path)
    samples = build_samples(collected)
    if not samples:
        raise SystemExit("No usable metric values found; nothing to summarize.")

    summary = summarize(samples)
    write_summary(summary, out_dir)

    if args.dry_run:
        log("[dry-run] skipping plot generation.")
    else:
        generate_plots(summary, out_dir / "plots")


if __name__ == "__main__":
    main()