import argparse
import json
import multiprocessing as mp
import re
import threading
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from queue import Empty
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError:  # pragma: no cover - tkinter unavailable in some envs
    tk = None

from src.extract_frames_edit import (
    detect_keypoints_yolo,
    extract_from_video,
    predict_masks_yolo,
)
from src.multiview_reconstruction_edit import (
    reconstruct,
    render_pose_time_series,
)


class ConfigError(Exception):
    """Raised when the pipeline configuration is invalid."""


DEFAULT_VIDEOS = [
    "bluegill_data/videos/bluegill_renders/006_Positive Z (Fish Ventral Side).mp4",
    "bluegill_data/videos/bluegill_renders/004a_Positive X (Fish Front).mp4",
    "bluegill_data/videos/bluegill_renders/003_Fish Top R.mp4",
]


@dataclass
class PipelineConfig:
    videos: List[str] = field(default_factory=lambda: DEFAULT_VIDEOS.copy())
    segmentation_model_path: str = "src/DSKv2/cygill_seg.pt"
    pose_model_path: str = "src/DSKv2/cygill_pose.pt"
    mesh_path: str = "src/DSKv2/Bluegill_Body_mesh.json"
    cam_matrices_path: str = "src/DSKv2/cam_matrices.json"
    out_path: str = "src/results/cygill"
    final_output_folder: str = "src/results/output/"
    frame_range: Optional[str] = "65-89"
    instance_number: int = 0
    pose_time_series_path: Optional[str] = "src/DSKv2/pose_time_series_Bluegill_Body.json"
    pose_time_series_deform: bool = False
    pose_time_series_offset_by_range_start: bool = False
    dataset_folder_name: str = "dataset"
    seed: int = 700
    save_models: bool = True
    conf_threshold: float = 0.8
    also_create_frame2video_csv: bool = True
    undistort: bool = True

    def dataset_folder(self) -> Path:
        return Path(self.out_path) / self.dataset_folder_name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_fields}
        return cls(**filtered)


def load_config(config_path: Path) -> PipelineConfig:
    with config_path.open() as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ConfigError("Configuration file must contain a JSON object.")
    return PipelineConfig.from_dict(data)


def save_config(config: PipelineConfig, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as fp:
        json.dump(config.to_dict(), fp, indent=2)


def parse_frame_selection(frame_selection: Optional[str]) -> List[int]:
    if frame_selection is None:
        raise ConfigError("Frame range is required for reconstruction.")

    tokens = re.split(r"[\s,]+", frame_selection.strip())
    indices: List[int] = []

    for token in tokens:
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:
                raise ConfigError(f"Invalid frame range token: '{token}'.") from exc
            if start > end:
                raise ConfigError(f"Frame range start must be <= end for '{token}'.")
            indices.extend(range(start, end + 1))
        else:
            try:
                indices.append(int(token))
            except ValueError as exc:
                raise ConfigError(f"Invalid frame index token: '{token}'.") from exc

    if not indices:
        raise ConfigError("No frame indices parsed from frame range input.")

    return indices


def read_keypoint_list(mesh_path: Path) -> List[str]:
    with mesh_path.open() as fp:
        mesh_data = json.load(fp)

    for key in ("keypoint_list", "kpt_list"):
        if key in mesh_data:
            keypoints = mesh_data[key]
            if not isinstance(keypoints, Iterable):
                raise ConfigError(f"Mesh field '{key}' must be an iterable.")
            return [str(name) for name in keypoints]

    raise ConfigError(
        "Mesh file does not contain a 'keypoint_list' or 'kpt_list' field."
    )


def run_pipeline(
    config: PipelineConfig,
    steps: List[str],
    pause_event: Optional[Any] = None,
) -> None:
    videos = [Path(video).absolute() for video in config.videos]
    if not videos:
        raise ConfigError("At least one video path must be provided.")

    reconstruct_pause_event = (
        pause_event if pause_event is not None and "reconstruct" in steps else None
    )

    out_dir = Path(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_folder_path = config.dataset_folder()
    dataset_folder_path.mkdir(parents=True, exist_ok=True)

    final_output_folder = Path(config.final_output_folder)
    final_output_folder.mkdir(parents=True, exist_ok=True)

    mesh_path: Optional[Path] = None
    keypoint_list: Optional[List[str]] = None
    kpt_name_dict: Dict[int, str] = {}

    if any(step in steps for step in ("keypoints", "reconstruct", "render_time_series")):
        mesh_path = Path(config.mesh_path)
        keypoint_list = read_keypoint_list(mesh_path)
        kpt_name_dict = {index: name for index, name in enumerate(keypoint_list)}

    if "extract" in steps:
        extract_from_video(
            videos,
            Path(config.cam_matrices_path),
            out_dir,
            dataset_folder_name=config.dataset_folder_name,
            also_create_frame2video_csv=config.also_create_frame2video_csv,
            undistort=config.undistort,
            frame_indices=parse_frame_selection(config.frame_range) if config.frame_range else None
        )

    if "masks" in steps:
        predict_masks_yolo(
            dataset_path=dataset_folder_path,
            model_path=Path(config.segmentation_model_path),
            conf_threshold=config.conf_threshold,
            frame_indices=parse_frame_selection(config.frame_range) if config.frame_range else None
        )

    if "keypoints" in steps:
        detect_keypoints_yolo(
            dataset_path=dataset_folder_path,
            model_path=Path(config.pose_model_path),
            kpt_names_dict=kpt_name_dict,
            frame_indices=parse_frame_selection(config.frame_range) if config.frame_range else None
        )

    if "render_time_series" in steps:
        if not config.pose_time_series_path:
            raise ConfigError(
                "pose_time_series_path must be provided to render the time series."
            )
        if mesh_path is None:
            raise ConfigError("mesh_path must be provided to render the time series.")
        render_pose_time_series(
            mesh_path=str(mesh_path),
            dataset_dir=str(dataset_folder_path),
            pose_time_series_file_path=config.pose_time_series_path,
            outdir=str(final_output_folder),
            deform=config.pose_time_series_deform,
            frame_range=parse_frame_selection(config.frame_range) if config.frame_range else None,
            offset_by_frame_range_start=config.pose_time_series_offset_by_range_start,
        )

    if "reconstruct" in steps:
        frame_indices = parse_frame_selection(config.frame_range)
        if mesh_path is None:
            raise ConfigError("mesh_path must be provided for reconstruction.")
        selected_views = resolve_dataset_views(dataset_folder_path, videos)
        reconstruct(
            mesh_path=str(mesh_path),
            dataset_dir=str(dataset_folder_path),
            outdir=str(final_output_folder),
            frame_indices=frame_indices,
            instance_number=config.instance_number,
            seed=config.seed,
            save_models=config.save_models,
            video_names=selected_views,
            pause_event=reconstruct_pause_event,
        )


def _mean_median(values: List[float], metric_name: str) -> Dict[str, float]:
    if not values:
        raise ValueError(f"No values provided for metric '{metric_name}'.")
    return {"mean": float(mean(values)), "median": float(median(values))}


def _summarize_scalar_metric(metric_name: str, metric_data: Dict[str, List[float]]) -> Dict[str, Any]:
    if not metric_data:
        raise ValueError(f"Metric '{metric_name}' is empty.")

    overall_values: List[float] = []
    per_view: Dict[str, Dict[str, float]] = {}
    per_frame_values: defaultdict[int, List[float]] = defaultdict(list)

    for view, values in metric_data.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Metric '{metric_name}' for view '{view}' is empty or invalid.")
        per_view[view] = _mean_median(values, metric_name)
        overall_values.extend(values)
        for idx, value in enumerate(values):
            per_frame_values[idx].append(value)

    summary: Dict[str, Any] = {
        "overall": _mean_median(overall_values, metric_name),
        "by_view": per_view,
        "by_frame": {
            str(frame_idx): _mean_median(frame_values, metric_name)
            for frame_idx, frame_values in sorted(per_frame_values.items())
        },
    }

    return summary


def _summarize_keypoint_metric(metric_name: str, metric_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    if not metric_data:
        raise ValueError(f"Metric '{metric_name}' is empty.")

    overall_values: List[float] = []
    per_view_values: defaultdict[str, List[float]] = defaultdict(list)
    per_keypoint_values: defaultdict[str, List[float]] = defaultdict(list)
    per_frame_values: defaultdict[int, List[float]] = defaultdict(list)
    per_view_keypoint_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for view, keypoints in metric_data.items():
        if not isinstance(keypoints, dict) or not keypoints:
            raise ValueError(f"Metric '{metric_name}' for view '{view}' is empty or invalid.")
        view_summary: Dict[str, Dict[str, float]] = {}
        for keypoint, values in keypoints.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"Metric '{metric_name}' for view '{view}', keypoint '{keypoint}' is empty or invalid."
                )
            stats = _mean_median(values, metric_name)
            view_summary[keypoint] = stats
            overall_values.extend(values)
            per_view_values[view].extend(values)
            per_keypoint_values[keypoint].extend(values)
            for idx, value in enumerate(values):
                per_frame_values[idx].append(value)
        per_view_keypoint_summary[view] = view_summary

    summary: Dict[str, Any] = {
        "overall": _mean_median(overall_values, metric_name),
        "by_view": {
            view: _mean_median(values, metric_name)
            for view, values in per_view_values.items()
        },
        "by_frame": {
            str(frame_idx): _mean_median(values, metric_name)
            for frame_idx, values in sorted(per_frame_values.items())
        },
        "by_keypoint": {
            keypoint: _mean_median(values, metric_name)
            for keypoint, values in per_keypoint_values.items()
        },
        "by_view_and_keypoint": per_view_keypoint_summary,
    }

    return summary


def compute_metrics_summary(metrics_data: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    scalar_metrics = ["orig_IoU", "mask_IoU"]

    for metric_name in scalar_metrics:
        metric_values = metrics_data.get(metric_name)
        if metric_values is not None:
            summary[metric_name] = _summarize_scalar_metric(metric_name, metric_values)

    keypoint_metric = metrics_data.get("keypoint_L2_distance")
    if keypoint_metric is not None:
        summary["keypoint_L2_distance"] = _summarize_keypoint_metric(
            "keypoint_L2_distance", keypoint_metric
        )

    if not summary:
        raise ValueError("No metrics found to summarize.")

    return summary


def format_metrics_summary_text(summary: Dict[str, Any]) -> str:
    def fmt(stats: Dict[str, Optional[float]]) -> str:
        mean_val = stats.get("mean")
        median_val = stats.get("median")
        if mean_val is None or median_val is None:
            return "mean=NA, median=NA"
        return f"mean={mean_val:.4f}, median={median_val:.4f}"

    sections: List[str] = []
    metric_order = ["orig_IoU", "mask_IoU", "keypoint_L2_distance"]

    sections.append("Overall metrics")
    for metric_name in metric_order:
        metric_summary = summary.get(metric_name)
        if metric_summary and "overall" in metric_summary:
            sections.append(f"  {metric_name}: {fmt(metric_summary['overall'])}")
    sections.append("")

    sections.append("Metrics by view")
    for metric_name in metric_order:
        metric_summary = summary.get(metric_name)
        if not metric_summary:
            continue
        by_view = metric_summary.get("by_view", {})
        if not by_view:
            continue
        sections.append(f"  {metric_name}:")
        for view, stats in sorted(by_view.items()):
            sections.append(f"    {view}: {fmt(stats)}")
    sections.append("")

    sections.append("Metrics by frame")
    for metric_name in metric_order:
        metric_summary = summary.get(metric_name)
        if not metric_summary:
            continue
        by_frame = metric_summary.get("by_frame", {})
        if not by_frame:
            continue
        sections.append(f"  {metric_name}:")
        for frame_idx, stats in sorted(by_frame.items(), key=lambda x: int(x[0])):
            sections.append(f"    frame {frame_idx}: {fmt(stats)}")
    sections.append("")

    keypoint_summary = summary.get("keypoint_L2_distance")
    if keypoint_summary:
        sections.append("Keypoint_L2_distance by keypoint")
        for keypoint, stats in sorted(keypoint_summary.get("by_keypoint", {}).items()):
            sections.append(f"  {keypoint}: {fmt(stats)}")
        sections.append("")

        sections.append("Keypoint_L2_distance by keypoint and view")
        for view, keypoint_stats in sorted(
            keypoint_summary.get("by_view_and_keypoint", {}).items()
        ):
            sections.append(f"  {view}:")
            for keypoint, stats in sorted(keypoint_stats.items()):
                sections.append(f"    {keypoint}: {fmt(stats)}")
        sections.append("")

        sections.append("Keypoint_L2_distance by frame")
        for frame_idx, stats in sorted(
            keypoint_summary.get("by_frame", {}).items(), key=lambda x: int(x[0])
        ):
            sections.append(f"  frame {frame_idx}: {fmt(stats)}")
        sections.append("")

        sections.append("Keypoint_L2_distance by view")
        for view, stats in sorted(keypoint_summary.get("by_view", {}).items()):
            sections.append(f"  {view}: {fmt(stats)}")
        sections.append("")

        sections.append("Keypoint_L2_distance overall")
        sections.append(f"  {fmt(keypoint_summary['overall'])}")

    return "\n".join(sections).strip()


def resolve_dataset_views(dataset_dir: Path, requested_videos: List[Path]) -> List[str]:
    index_path = dataset_dir / "index.json"
    if not index_path.exists():
        raise ConfigError(f"Dataset index file not found at {index_path}.")

    with index_path.open() as fp:
        index_data = json.load(fp)

    available_views = index_data.get("frame_folders")
    if not isinstance(available_views, list) or not available_views:
        raise ConfigError("Dataset index does not list any frame folders.")

    resolved: List[str] = []
    missing: List[str] = []

    for video_path in requested_videos:
        stem = video_path.stem

        candidate: Optional[str]
        if stem in available_views:
            candidate = stem
        else:
            undistorted = f"{stem}_undistorted"
            if undistorted in available_views:
                candidate = undistorted
            else:
                prefixed = next(
                    (name for name in available_views if name.startswith(f"{stem}_")),
                    None,
                )
                candidate = prefixed

        if candidate is None:
            missing.append(stem)
        elif candidate not in resolved:
            resolved.append(candidate)

    if missing:
        raise ConfigError(
            "Requested view(s) not present in dataset: " + ", ".join(missing)
        )

    if not resolved:
        raise ConfigError("No valid views selected for reconstruction.")

    return resolved


def _run_pipeline_subprocess(
    config_dict: Dict[str, Any],
    steps: List[str],
    queue: "mp.Queue",
    pause_event: Optional[Any] = None,
) -> None:
    try:
        config = PipelineConfig.from_dict(config_dict)
        run_pipeline(config, steps, pause_event=pause_event)
    except Exception as exc:  # pragma: no cover - propagated back to GUI
        queue.put(("error", repr(exc), traceback.format_exc()))
        raise
    else:
        queue.put(("success", None, None))


class PipelineGUI:
    def __init__(self, root: "tk.Tk", config: PipelineConfig, steps: List[str]):
        self.root = root
        self.config = PipelineConfig.from_dict(config.to_dict())
        self.steps = steps
        self.video_paths = list(self.config.videos)

        self.segmentation_model_var = tk.StringVar(value=self.config.segmentation_model_path)
        self.pose_model_var = tk.StringVar(value=self.config.pose_model_path)
        self.mesh_var = tk.StringVar(value=self.config.mesh_path)
        self.cam_var = tk.StringVar(value=self.config.cam_matrices_path)
        self.out_path_var = tk.StringVar(value=self.config.out_path)
        self.final_output_var = tk.StringVar(value=self.config.final_output_folder)
        self.frame_range_var = tk.StringVar(value=self.config.frame_range or "")
        self.instance_var = tk.StringVar(value=str(self.config.instance_number))
        self.undistort_var = tk.BooleanVar(value=self.config.undistort)
        self.pose_time_series_var = tk.StringVar(value=self.config.pose_time_series_path or "")
        self.pose_time_series_deform_var = tk.BooleanVar(value=self.config.pose_time_series_deform)
        self.pose_time_series_offset_var = tk.BooleanVar(
            value=self.config.pose_time_series_offset_by_range_start
        )
        self.advanced_visible = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Idle.")
        self.action_buttons: List[tk.Widget] = []
        self.worker_thread: Optional[threading.Thread] = None
        self.worker_process: Optional[mp.Process] = None
        self.worker_queue: Optional[mp.Queue] = None
        self.pause_requested: bool = False
        self.worker_pause_event: Optional[mp.Event] = None
        self.current_step: Optional[str] = None

        self._build_ui()
        self._refresh_video_listbox()

    def _build_ui(self) -> None:
        self.root.title("DeepShapeKit v2 Pipeline")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        for col in range(3):
            weight = 1 if col == 1 else 0
            main.columnconfigure(col, weight=weight)
        main.rowconfigure(0, weight=1)

        ttk.Label(main, text="Videos").grid(row=0, column=0, sticky="nw")

        video_frame = ttk.Frame(main)
        video_frame.grid(row=0, column=1, columnspan=2, sticky="nsew", pady=(0, 8))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self.video_listbox = tk.Listbox(video_frame, height=6, selectmode=tk.EXTENDED)
        self.video_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(video_frame, orient="vertical", command=self.video_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.video_listbox.configure(yscrollcommand=scrollbar.set)

        video_buttons = ttk.Frame(video_frame)
        video_buttons.grid(row=0, column=2, padx=(8, 0), sticky="ns")
        add_button = ttk.Button(video_buttons, text="Add…", command=self.add_videos)
        add_button.grid(row=0, column=0, sticky="ew")
        remove_button = ttk.Button(video_buttons, text="Remove", command=self.remove_selected_videos)
        remove_button.grid(row=1, column=0, sticky="ew", pady=4)
        clear_button = ttk.Button(video_buttons, text="Clear", command=self.clear_videos)
        clear_button.grid(row=2, column=0, sticky="ew")
        self.action_buttons.extend([add_button, remove_button, clear_button])

        row = 1
        self._add_path_row(main, row, "Segmentation model", self.segmentation_model_var, self.browse_segmentation_model)
        row += 1
        self._add_path_row(main, row, "Pose model", self.pose_model_var, self.browse_pose_model)
        row += 1
        self._add_path_row(main, row, "Mesh", self.mesh_var, self.browse_mesh)
        row += 1
        self._add_path_row(main, row, "Camera matrices", self.cam_var, self.browse_cam_matrices)
        row += 1
        self._add_directory_row(main, row, "Output path", self.out_path_var, self.browse_out_path)
        row += 1
        self._add_directory_row(
            main, row, "Final output folder", self.final_output_var, self.browse_final_output
        )
        row += 1

        undistort_check = ttk.Checkbutton(
            main,
            text="Undistort videos during extraction",
            variable=self.undistort_var,
        )
        undistort_check.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        ttk.Label(main, text="Frame range").grid(row=row, column=0, sticky="w", pady=2)
        frame_entry = ttk.Entry(main, textvariable=self.frame_range_var)
        frame_entry.grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Label(main, text="e.g. 10-23", foreground="#777").grid(row=row, column=2, sticky="w", pady=2)
        row += 1

        ttk.Label(main, text="Instance number").grid(row=row, column=0, sticky="w", pady=2)
        instance_entry = ttk.Entry(main, textvariable=self.instance_var)
        instance_entry.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        self.advanced_toggle = ttk.Button(main, text="Advanced ▸", command=self._toggle_advanced)
        self.advanced_toggle.grid(row=row, column=0, sticky="w", pady=(6, 2))
        row += 1

        self.advanced_frame = ttk.Frame(main)
        self.advanced_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.advanced_frame.columnconfigure(1, weight=1)

        self._add_path_row(
            self.advanced_frame,
            0,
            "Pose time series",
            self.pose_time_series_var,
            self.browse_pose_time_series,
        )
        deform_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Deform mesh when rendering time series",
            variable=self.pose_time_series_deform_var,
        )
        deform_check.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))
        offset_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Pose time series start is frame range start",
            variable=self.pose_time_series_offset_var,
        )
        offset_check.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))
        render_button = ttk.Button(
            self.advanced_frame,
            text="render_pose_time_series",
            command=lambda: self.run_step("render_time_series"),
        )
        render_button.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.action_buttons.append(render_button)
        self.advanced_frame.grid_remove()
        row += 1

        control_frame = ttk.Frame(main)
        control_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        for col in range(8):
            control_frame.columnconfigure(col, weight=1)

        self._add_action_button(control_frame, 0, "Save config…", self.save_config)
        self._add_action_button(control_frame, 1, "Load config…", self.load_config)
        self._add_action_button(control_frame, 2, "extract_from_video", lambda: self.run_step("extract"))
        self._add_action_button(control_frame, 3, "predict_masks_yolo", lambda: self.run_step("masks"))
        self._add_action_button(control_frame, 4, "detect_keypoints_yolo", lambda: self.run_step("keypoints"))
        self._add_action_button(control_frame, 5, "reconstruct", lambda: self.run_step("reconstruct"))
        self._add_action_button(control_frame, 6, "Show metrics", self.show_metrics)
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.pause_execution, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=7, padx=2, sticky="ew")

        status_label = ttk.Label(main, textvariable=self.status_var, relief="sunken", anchor="w")
        status_label.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    def _add_path_row(
        self,
        parent: "tk.Widget",
        row: int,
        label: str,
        variable: "tk.StringVar",
        browse_command,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        button = ttk.Button(parent, text="Browse", command=browse_command)
        button.grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=2)

    def _add_directory_row(
        self,
        parent: "tk.Widget",
        row: int,
        label: str,
        variable: "tk.StringVar",
        browse_command,
    ) -> None:
        self._add_path_row(parent, row, label, variable, browse_command)

    def _add_action_button(self, parent: "tk.Widget", column: int, text: str, command) -> None:
        button = ttk.Button(parent, text=text, command=command)
        button.grid(row=0, column=column, padx=2, sticky="ew")
        self.action_buttons.append(button)

    def _refresh_video_listbox(self) -> None:
        self.video_listbox.delete(0, tk.END)
        for path in self.video_paths:
            self.video_listbox.insert(tk.END, path)

    def add_videos(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Select video files",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not file_paths:
            return
        for path in file_paths:
            if path not in self.video_paths:
                self.video_paths.append(path)
        self._refresh_video_listbox()

    def remove_selected_videos(self) -> None:
        selection = list(self.video_listbox.curselection())
        if not selection:
            return
        for index in reversed(selection):
            del self.video_paths[index]
        self._refresh_video_listbox()

    def clear_videos(self) -> None:
        if messagebox.askyesno("Clear videos", "Remove all videos from the list?"):
            self.video_paths.clear()
            self._refresh_video_listbox()

    def browse_segmentation_model(self) -> None:
        self._browse_file(self.segmentation_model_var, "Select segmentation model")

    def browse_pose_model(self) -> None:
        self._browse_file(self.pose_model_var, "Select pose model")

    def browse_mesh(self) -> None:
        self._browse_file(self.mesh_var, "Select mesh file")

    def browse_cam_matrices(self) -> None:
        self._browse_file(self.cam_var, "Select camera matrices file")

    def browse_pose_time_series(self) -> None:
        self._browse_file(self.pose_time_series_var, "Select pose time series file")

    def browse_out_path(self) -> None:
        self._browse_directory(self.out_path_var, "Select output directory")

    def browse_final_output(self) -> None:
        self._browse_directory(self.final_output_var, "Select final output directory")

    def _browse_file(self, variable: "tk.StringVar", title: str) -> None:
        file_path = filedialog.askopenfilename(title=title)
        if file_path:
            variable.set(file_path)

    def _browse_directory(self, variable: "tk.StringVar", title: str) -> None:
        directory = filedialog.askdirectory(title=title)
        if directory:
            variable.set(directory)

    def gather_config(self) -> PipelineConfig:
        config = PipelineConfig.from_dict(self.config.to_dict())
        config.videos = list(self.video_paths)
        config.segmentation_model_path = self.segmentation_model_var.get().strip()
        config.pose_model_path = self.pose_model_var.get().strip()
        config.mesh_path = self.mesh_var.get().strip()
        config.cam_matrices_path = self.cam_var.get().strip()
        config.out_path = self.out_path_var.get().strip()
        config.final_output_folder = self.final_output_var.get().strip()
        frame_range = self.frame_range_var.get().strip()
        config.frame_range = frame_range or None

        instance_value = self.instance_var.get().strip()
        if instance_value:
            try:
                config.instance_number = int(instance_value)
            except ValueError as exc:
                raise ConfigError("Instance number must be an integer.") from exc
        else:
            config.instance_number = 0
        config.undistort = bool(self.undistort_var.get())
        pose_time_series = self.pose_time_series_var.get().strip()
        config.pose_time_series_path = pose_time_series or None
        config.pose_time_series_deform = bool(self.pose_time_series_deform_var.get())
        config.pose_time_series_offset_by_range_start = bool(self.pose_time_series_offset_var.get())

        return config

    def save_config(self) -> None:
        try:
            config = self.gather_config()
        except ConfigError as exc:
            messagebox.showerror("Configuration error", str(exc))
            return

        file_path = filedialog.asksaveasfilename(
            title="Save configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            save_config(config, Path(file_path))
            self.config = config
            self.set_status(f"Configuration saved to {file_path}")
        except Exception as exc:  # pragma: no cover - file system errors
            messagebox.showerror("Save failed", str(exc))

    def load_config(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Load configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            config = load_config(Path(file_path))
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        self.apply_config(config)
        self.set_status(f"Configuration loaded from {file_path}")

    def apply_config(self, config: PipelineConfig) -> None:
        self.config = config
        self.video_paths = list(config.videos)
        self.segmentation_model_var.set(config.segmentation_model_path)
        self.pose_model_var.set(config.pose_model_path)
        self.mesh_var.set(config.mesh_path)
        self.cam_var.set(config.cam_matrices_path)
        self.out_path_var.set(config.out_path)
        self.final_output_var.set(config.final_output_folder)
        self.frame_range_var.set(config.frame_range or "")
        self.instance_var.set(str(config.instance_number))
        self.undistort_var.set(bool(config.undistort))
        self.pose_time_series_var.set(config.pose_time_series_path or "")
        self.pose_time_series_deform_var.set(bool(config.pose_time_series_deform))
        self.pose_time_series_offset_var.set(bool(config.pose_time_series_offset_by_range_start))
        self._refresh_video_listbox()

    def _toggle_advanced(self) -> None:
        if self.advanced_visible.get():
            self.advanced_visible.set(False)
            self.advanced_frame.grid_remove()
            self.advanced_toggle.configure(text="Advanced ▸")
        else:
            self.advanced_visible.set(True)
            self.advanced_frame.grid()
            self.advanced_toggle.configure(text="Advanced ▾")

    def run_step(self, step: str) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A step is already running. Please wait.")
            return

        try:
            config = self.gather_config()
        except ConfigError as exc:
            messagebox.showerror("Configuration error", str(exc))
            return

        self.config = config
        self.set_status(f"Running {step}…")
        self._set_buttons_state(tk.DISABLED)

        self.pause_button.configure(state=tk.NORMAL)
        self.pause_requested = False
        self.current_step = step

        pause_event = mp.Event() if step == "reconstruct" else None
        self.worker_pause_event = pause_event

        def task() -> None:
            queue: mp.Queue = mp.Queue()
            process = mp.Process(
                target=_run_pipeline_subprocess,
                args=(config.to_dict(), [step], queue, pause_event),
            )
            self.worker_process = process
            self.worker_queue = queue
            process.start()
            process.join()
            exit_code = process.exitcode

            result = None
            try:
                result = queue.get_nowait()
            except Empty:
                result = None
            finally:
                queue.close()
                queue.join_thread()

            self.worker_process = None
            self.worker_queue = None
            self.worker_pause_event = None
            self.current_step = None

            if self.pause_requested:
                self.root.after(0, lambda: self._on_step_paused(step))
                return

            if exit_code == 0:
                self.root.after(0, lambda: self._on_step_finished(step))
                return

            if result and result[0] == "error":
                error_message = result[1] or "Pipeline step failed."
                trace = result[2] or ""
                exc = RuntimeError(error_message)
            else:
                exc = RuntimeError(f"Process exited with code {exit_code}")
                trace = ""

            self.root.after(0, lambda: self._on_step_failed(step, exc, trace))

        self.worker_thread = threading.Thread(target=task, daemon=True)
        self.worker_thread.start()

    def pause_execution(self) -> None:
        if not self.worker_process or not self.worker_process.is_alive():
            self.set_status("No running step to pause.")
            return

        self.pause_requested = True
        self.pause_button.configure(state=tk.DISABLED)

        if self.current_step == "reconstruct" and self.worker_pause_event is not None:
            self.set_status("Pause requested; finishing current frame before stopping.")
            self.worker_pause_event.set()
            return

        self.set_status("Pausing current step…")
        self.worker_process.terminate()

    def _on_step_finished(self, step: str) -> None:
        self._set_buttons_state(tk.NORMAL)
        self.set_status(f"Finished {step}.")
        self.pause_button.configure(state=tk.DISABLED)
        self.worker_thread = None
        self.pause_requested = False
        messagebox.showinfo("Success", f"Step '{step}' completed successfully.")

    def _on_step_failed(self, step: str, exc: Exception, trace: str) -> None:
        self._set_buttons_state(tk.NORMAL)
        self.set_status(f"{step} failed: {exc}")
        self.pause_button.configure(state=tk.DISABLED)
        self.worker_thread = None
        self.pause_requested = False
        details = f"An error occurred while running '{step}':\n{exc}"
        if trace:
            details += f"\n\nDetails:\n{trace}"
        messagebox.showerror("Step failed", details)

    def _on_step_paused(self, step: str) -> None:
        self._set_buttons_state(tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED)
        self.worker_thread = None
        self.pause_requested = False
        self.set_status(f"{step} paused.")
        messagebox.showinfo("Paused", f"Step '{step}' was paused.")

    def _set_buttons_state(self, state: str) -> None:
        for button in self.action_buttons:
            button.configure(state=state)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def show_metrics(self) -> None:
        try:
            config = self.gather_config()
        except ConfigError as exc:
            messagebox.showerror("Configuration error", str(exc))
            return

        metrics_path = Path(config.final_output_folder) / f"metrics_instance_{config.instance_number}.json"
        if not metrics_path.exists():
            messagebox.showerror("Metrics not found", f"No metrics file found at {metrics_path}.")
            return

        try:
            with metrics_path.open() as fp:
                metrics_data = json.load(fp)
        except Exception as exc:
            messagebox.showerror("Read failed", f"Could not read metrics file: {exc}")
            return

        metrics_data = dict(metrics_data)
        metrics_data.pop("metrics_summary", None)

        try:
            summary = compute_metrics_summary(metrics_data)
            display_text = format_metrics_summary_text(summary)
        except Exception as exc:
            messagebox.showerror("Metrics error", str(exc))
            return

        metrics_data["metrics_summary"] = summary

        try:
            with metrics_path.open("w") as fp:
                json.dump(metrics_data, fp, indent=2)
        except Exception as exc:
            messagebox.showerror("Write failed", f"Could not update metrics file: {exc}")
            return

        self.set_status(f"Metrics summary saved to {metrics_path}.")
        self._display_text_window("Metrics summary", display_text)

    def _display_text_window(self, title: str, content: str) -> None:
        window = tk.Toplevel(self.root)
        window.title(title)
        text_area = ScrolledText(window, wrap="word", width=100, height=40)
        text_area.pack(fill="both", expand=True)
        text_area.insert("1.0", content)
        text_area.configure(state="disabled")


def run_gui(initial_config: PipelineConfig, steps: List[str]) -> None:
    if tk is None:
        raise RuntimeError("tkinter is required for the GUI but is not available.")

    root = tk.Tk()
    PipelineGUI(root, initial_config, steps)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepShapeKit v2 pipeline.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["extract", "masks", "keypoints", "reconstruct"],
        help="Pipeline steps to execute.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI instead of running through the CLI.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a JSON configuration file to load.",
    )
    parser.add_argument("--videos", nargs="+", help="One or more input video paths.")
    parser.add_argument(
        "--segmentation-model",
        dest="segmentation_model_path",
        help="Path to the segmentation model.",
    )
    parser.add_argument(
        "--pose-model",
        dest="pose_model_path",
        help="Path to the pose model.",
    )
    parser.add_argument("--mesh", dest="mesh_path", help="Path to the mesh JSON file.")
    parser.add_argument(
        "--cam-matrices",
        dest="cam_matrices_path",
        help="Path to the camera matrices JSON file.",
    )
    parser.add_argument(
        "--out-path",
        dest="out_path",
        help="Output directory for intermediate results.",
    )
    parser.add_argument(
        "--final-output-folder",
        dest="final_output_folder",
        help="Directory for final reconstruction outputs.",
    )
    parser.add_argument(
        "--frame-range",
        dest="frame_range",
        help="Frame range for reconstruction (e.g. '10-23' or '0-5,8').",
    )
    parser.add_argument(
        "--instance-number",
        dest="instance_number",
        type=int,
        help="Instance number for reconstruction cache management.",
    )
    parser.add_argument(
        "--pose-time-series",
        dest="pose_time_series_path",
        help="Path to a pose time series JSON for rendering silhouettes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        help="Random seed for reconstruction.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        dest="conf_threshold",
        help="Confidence threshold for mask prediction.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_false",
        dest="save_models",
        help="Disable saving models during reconstruction.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        dest="save_models",
        help="Enable saving models during reconstruction.",
    )
    parser.set_defaults(save_models=True)
    return parser


def update_config_from_args(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    if args.videos:
        config.videos = args.videos
    if args.segmentation_model_path:
        config.segmentation_model_path = args.segmentation_model_path
    if args.pose_model_path:
        config.pose_model_path = args.pose_model_path
    if args.mesh_path:
        config.mesh_path = args.mesh_path
    if args.cam_matrices_path:
        config.cam_matrices_path = args.cam_matrices_path
    if args.out_path:
        config.out_path = args.out_path
    if args.final_output_folder:
        config.final_output_folder = args.final_output_folder
    if args.frame_range:
        config.frame_range = args.frame_range
    if args.instance_number is not None:
        config.instance_number = args.instance_number
    if args.pose_time_series_path:
        config.pose_time_series_path = args.pose_time_series_path
    if args.seed is not None:
        config.seed = args.seed
    if args.conf_threshold is not None:
        config.conf_threshold = args.conf_threshold
    if hasattr(args, "save_models") and args.save_models is not None:
        config.save_models = args.save_models
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.gui and tk is None:
        parser.error("tkinter is not available; GUI mode cannot be used.")

    config = PipelineConfig()

    if args.config:
        config = load_config(Path(args.config))

    config = update_config_from_args(config, args)

    if args.gui:
        run_gui(config, args.steps)
        return

    run_pipeline(config, args.steps)


if __name__ == "__main__":
    main()
