#!/usr/bin/env python3
"""
Sweep DSKv2 reconstructions over every view combination of size >= 2.

Given a base full_config in which every camera view is present as a video input,
this script generates one config per view combination and launches a DSKv2
reconstruction for each, with a bounded number of concurrent processes.

Across runs, everything is held static except:
  * final_output_folder -- a leaf folder encoding the included views
  * videos              -- the subset for this combination
  * view_weights        -- the base CSV subset by the SAME positional indices,
                           preserving each view's original weight value

Positional subsetting is required, not cosmetic: DSKv2_demo.resolve_dataset_views
preserves the order of config.videos, and run_pipeline zips view_weights against
that order. A weights list that is reordered independently of the videos silently
applies the wrong weight to the wrong camera -- it does not raise.

Preparation steps (extract / masks / keypoints) depend only on the videos, never
on the combination, and write into out_path/dataset_folder_name, which is shared
by every run. They are therefore executed ONCE, serially, over the full view set
before the parallel phase. Use --prep-per-run to override.

Usage:
    python sweep_view_combinations.py base_config.json \\
        --prep extract_from_video masks keypoints \\
        --jobs 4

    python sweep_view_combinations.py base_config.json --prep masks --dry-run
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Public prep-step names -> DSKv2_demo internal step names.
PREP_STEP_ALIASES = {
    "extract_from_video": "extract",
    "masks": "masks",
    "keypoints": "keypoints",
}
# Canonical pipeline order, mirroring DSKv2_demo.VALID_STEPS.
CANONICAL_STEP_ORDER = ["extract", "masks", "keypoints", "reconstruct"]

MAX_LEAF_NAME_LEN = 200  # stay clear of the 255-byte ext4 filename limit

_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


# --------------------------------------------------------------------------
# Config handling
# --------------------------------------------------------------------------


def load_base_config(path: Path) -> Dict[str, Any]:
    with path.open() as fp:
        config = json.load(fp)
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: config must be a JSON object.")

    for key in ("videos", "out_path"):
        if key not in config:
            raise SystemExit(f"{path}: missing required key '{key}'.")

    videos = config["videos"]
    if not isinstance(videos, list) or len(videos) < 2:
        raise SystemExit(f"{path}: 'videos' must be a list of at least 2 paths.")

    stems = [Path(v).stem for v in videos]
    duplicates = {s for s in stems if stems.count(s) > 1}
    if duplicates:
        # resolve_dataset_views maps videos to dataset folders by stem and
        # de-duplicates, so identical stems collapse into one view and desync
        # the weights from the videos.
        raise SystemExit(
            f"{path}: duplicate video stems would collide in the dataset index: "
            + ", ".join(sorted(duplicates))
        )

    return config


def parse_view_weights(raw: Any, n_views: int, config_path: Path) -> List[str]:
    """
    Return one weight token per view, as strings so the original formatting of
    each weight is preserved verbatim into the generated configs.
    """
    if raw is None or str(raw).strip() == "":
        return ["1"] * n_views

    tokens = [tok.strip() for tok in str(raw).split(",")]
    if any(tok == "" for tok in tokens):
        raise SystemExit(f"{config_path}: view_weights has an empty entry: {raw!r}")

    for tok in tokens:
        try:
            if float(tok) < 0:
                raise SystemExit(
                    f"{config_path}: view_weights entries must be non-negative: {tok!r}"
                )
        except ValueError:
            raise SystemExit(
                f"{config_path}: view_weights contains a non-numeric value: {tok!r}"
            )

    if len(tokens) == 1:
        # DSKv2 broadcasts a single weight across all views.
        return tokens * n_views
    if len(tokens) != n_views:
        raise SystemExit(
            f"{config_path}: view_weights has {len(tokens)} entries but there are "
            f"{n_views} videos. They must match (or be a single broadcast value)."
        )
    return tokens


def sanitize(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return token or "view"


def combo_folder_name(indices: Sequence[int], stems: Sequence[str]) -> str:
    """
    Leaf folder name describing the combination. Prefers readable view stems and
    falls back to a pure index encoding when the descriptive form would produce
    an over-long filename. The generated config is written into the folder either
    way, so the exact view set is always recoverable.
    """
    idx_part = "-".join(str(i) for i in indices)
    descriptive = "+".join(sanitize(stems[i]) for i in indices)
    name = f"k{len(indices)}__v{idx_part}__{descriptive}"
    if len(name) > MAX_LEAF_NAME_LEN:
        name = f"k{len(indices)}__v{idx_part}"
    return name


def build_combo_config(
    base: Dict[str, Any],
    indices: Sequence[int],
    weight_tokens: Sequence[str],
    leaf_name: str,
) -> Dict[str, Any]:
    config = copy.deepcopy(base)
    config["videos"] = [base["videos"][i] for i in indices]
    # Same positional indices as the videos -- this is what keeps each view
    # bound to its own original weight.
    config["view_weights"] = ",".join(weight_tokens[i] for i in indices)
    config["final_output_folder"] = str(Path(base["out_path"]) / leaf_name)
    return config


# --------------------------------------------------------------------------
# Process orchestration
# --------------------------------------------------------------------------


def build_command(
    python_exe: str, demo_path: Path, config_path: Path, steps: Sequence[str]
) -> List[str]:
    return [
        python_exe,
        str(demo_path),
        "--headless",
        "--config",
        str(config_path),
        "--steps",
        *steps,
    ]


def run_command(
    command: Sequence[str],
    log_path: Path,
    cwd: Path,
    label: str,
    cuda_device: Optional[str] = None,
) -> Tuple[str, int, float]:
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log(f"[start ] {label}")

    with log_path.open("w") as handle:
        handle.write(f"$ {' '.join(command)}\n\n")
        handle.flush()
        proc = subprocess.run(
            list(command), cwd=str(cwd), env=env, stdout=handle,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - started
    status = "ok    " if proc.returncode == 0 else "FAILED"
    log(f"[{status}] {label} ({elapsed:.1f}s, rc={proc.returncode}) -> {log_path}")
    return label, proc.returncode, elapsed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DSKv2 reconstructions over every view combination of size >= 2 "
            "derived from a base full_config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "base_config",
        type=Path,
        help="Path to the base full_config JSON, with all views present as videos.",
    )
    parser.add_argument(
        "--prep",
        nargs="*",
        default=[],
        choices=sorted(PREP_STEP_ALIASES),
        metavar="STEP",
        help=(
            "Preparation steps to run before optimization, any subset of: "
            + ", ".join(sorted(PREP_STEP_ALIASES))
            + ". Default: none (assumes the dataset is already prepared)."
        ),
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of DSKv2 reconstructions to run concurrently. Default: 1.",
    )
    parser.add_argument(
        "--demo",
        type=Path,
        default=Path("DSKv2_demo.py"),
        help="Path to DSKv2_demo.py. Default: ./DSKv2_demo.py",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch DSKv2. Default: this interpreter.",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=2,
        help="Minimum combination size. Default: 2.",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=None,
        help="Maximum combination size. Default: all views.",
    )
    parser.add_argument(
        "--prep-per-run",
        action="store_true",
        help=(
            "Run the prep steps inside every combination run instead of once up "
            "front. Redundant and races on the shared dataset folder unless each "
            "combination uses a distinct out_path."
        ),
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help=(
            "Comma-separated GPU ids assigned round-robin via CUDA_VISIBLE_DEVICES, "
            "e.g. '0,1'. Default: inherit the environment."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-run logs. Default: <out_path>/_sweep_logs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a combination whose output folder already holds a config.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write every config and print every command, but launch nothing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1.")

    demo_path = args.demo.expanduser().resolve()
    if not demo_path.is_file():
        raise SystemExit(f"DSKv2_demo.py not found at {demo_path} (use --demo).")

    base_path = args.base_config.expanduser().resolve()
    base = load_base_config(base_path)

    videos = base["videos"]
    n_views = len(videos)
    stems = [Path(v).stem for v in videos]
    weight_tokens = parse_view_weights(base.get("view_weights"), n_views, base_path)

    min_views = max(2, args.min_views)
    max_views = args.max_views if args.max_views is not None else n_views
    if max_views > n_views:
        max_views = n_views
    if min_views > max_views:
        raise SystemExit(
            f"--min-views ({min_views}) exceeds the available view count ({max_views})."
        )

    out_root = Path(base["out_path"]).expanduser()
    log_dir = (args.log_dir or out_root / "_sweep_logs").expanduser()

    prep_steps = [PREP_STEP_ALIASES[s] for s in args.prep]
    prep_steps = [s for s in CANONICAL_STEP_ORDER if s in set(prep_steps)]

    log(f"Base config      : {base_path}")
    log(f"Views ({n_views})        : " + ", ".join(f"[{i}] {s}" for i, s in enumerate(stems)))
    log(f"Base weights     : {','.join(weight_tokens)}")
    log(f"Output root      : {out_root}")
    log(f"Prep steps       : {', '.join(prep_steps) if prep_steps else '(none)'}")
    log(f"Parallel jobs    : {args.jobs}")

    # ---- Combination enumeration -----------------------------------------
    combos = [
        combo
        for size in range(min_views, max_views + 1)
        for combo in itertools.combinations(range(n_views), size)
    ]
    log(f"Combinations     : {len(combos)} (sizes {min_views}..{max_views})\n")

    # ---- Phase 1: shared preparation -------------------------------------
    if prep_steps and not args.prep_per_run:
        prep_config_path = out_root / "_prep_config.json"
        prep_config_path.parent.mkdir(parents=True, exist_ok=True)
        with prep_config_path.open("w") as fp:
            json.dump(base, fp, indent=2)

        command = build_command(args.python, demo_path, prep_config_path, prep_steps)
        if args.dry_run:
            log(f"[dry-run] prep: {' '.join(command)}")
        else:
            log("Running preparation once over the full view set...")
            _, rc, _ = run_command(
                command, log_dir / "_prep.log", demo_path.parent, "prep (all views)"
            )
            if rc != 0:
                raise SystemExit("Preparation failed; aborting before the sweep.")
            log("")

    # ---- Phase 2: generate configs ---------------------------------------
    run_steps = ["reconstruct"]
    if prep_steps and args.prep_per_run:
        run_steps = prep_steps + ["reconstruct"]

    jobs: List[Dict[str, Any]] = []
    skipped = 0
    for indices in combos:
        leaf = combo_folder_name(indices, stems)
        config = build_combo_config(base, indices, weight_tokens, leaf)
        run_dir = Path(config["final_output_folder"])
        config_path = run_dir / "config.json"

        if args.skip_existing and config_path.exists():
            skipped += 1
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as fp:
            json.dump(config, fp, indent=2)

        jobs.append(
            {
                "label": leaf,
                "config_path": config_path,
                "command": build_command(args.python, demo_path, config_path, run_steps),
            }
        )

    log(f"Wrote {len(jobs)} configs" + (f" ({skipped} skipped)." if skipped else "."))

    if args.dry_run:
        log("\n[dry-run] commands that would be launched:\n")
        for job in jobs:
            weights = json.loads(job["config_path"].read_text())["view_weights"]
            log(f"  {job['label']}  weights={weights}")
            log(f"    {' '.join(job['command'])}")
        return 0

    if not jobs:
        log("Nothing to run.")
        return 0

    # ---- Phase 3: bounded parallel execution ------------------------------
    devices = (
        [d.strip() for d in args.cuda_devices.split(",") if d.strip()]
        if args.cuda_devices
        else None
    )

    log(f"\nLaunching {len(jobs)} reconstructions, {args.jobs} at a time...\n")
    results: List[Tuple[str, int, float]] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_command,
                job["command"],
                log_dir / f"{job['label']}.log",
                demo_path.parent,
                job["label"],
                devices[i % len(devices)] if devices else None,
            ): job["label"]
            for i, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            results.append(future.result())

    # ---- Summary ----------------------------------------------------------
    failures = [(label, rc) for label, rc, _ in results if rc != 0]
    log(f"\n{'-' * 60}")
    log(f"Completed {len(results)} runs in {time.time() - started:.1f}s")
    log(f"  succeeded : {len(results) - len(failures)}")
    log(f"  failed    : {len(failures)}")
    for label, rc in sorted(failures):
        log(f"    - {label} (rc={rc}) -> {log_dir / (label + '.log')}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())