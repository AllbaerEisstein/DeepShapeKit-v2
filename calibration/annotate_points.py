import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2 as cv


Point = Tuple[float, float]


def _load_points(json_path: Path, list_index: int) -> List[Point]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("JSON root must be a list.")
    if list_index < 0 or list_index >= len(payload):
        raise IndexError(f"Index {list_index} out of range for {len(payload)} point sets.")

    points = payload[list_index]
    if not isinstance(points, Iterable):
        raise ValueError(f"Entry at index {list_index} is not a list of points.")

    normalized: List[Point] = []
    for idx, pair in enumerate(points):
        if (
            not isinstance(pair, Sequence)
            or len(pair) != 2
            or not isinstance(pair[0], (int, float))
            or not isinstance(pair[1], (int, float))
        ):
            raise ValueError(f"Point #{idx} is not a numeric x,y pair.")
        normalized.append((float(pair[0]), float(pair[1])))

    return normalized


def draw_points(
    image_path: Path,
    points: List[Point],
    *,
    start_label: int = 0,
    index: int = 0,
    circle_radius: int = 6,
    circle_color=(0, 0, 255),
    text_color=(0, 255, 0),
) -> Path:
    image = cv.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for label, (x, y) in enumerate(points, start=start_label):
        center = (int(round(x)), int(round(y)))
        cv.circle(image, center, circle_radius, circle_color, thickness=-1)
        text_position = (center[0] + circle_radius, center[1] - circle_radius)
        cv.putText(
            image,
            str(label),
            text_position,
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            thickness=2,
            lineType=cv.LINE_AA,
        )

    output_path = Path.cwd() / f"{image_path.stem}_points_{index}.png"
    if not cv.imwrite(str(output_path), image):
        raise IOError(f"Failed to write annotated image to {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a selected set of checkerboard points from a JSON file onto an image. "
            "The original image is left untouched; an annotated copy is saved to the current directory."
        )
    )
    parser.add_argument("image", help="Path to the image file to annotate.")
    parser.add_argument("points_json", help="Path to the JSON file with point detections (list of lists).")
    parser.add_argument(
        "index",
        type=int,
        help="Index of the list of points in the JSON file to draw.",
    )
    parser.add_argument(
        "--start-label",
        type=int,
        default=0,
        help="Starting number for point labels (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    json_path = Path(args.points_json)

    points = _load_points(json_path, args.index)
    output_path = draw_points(
        image_path=image_path,
        points=points,
        start_label=args.start_label,
        index=args.index,
    )
    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
