"""Download a small MOT17 subset, run the C++ pipeline, and score its output."""

from __future__ import annotations

import argparse
import csv
import subprocess
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

SEQUENCE = "MOT17-05-FRCNN"
FRAME_COUNT = 100
IOU_THRESHOLD = 0.50
MIRROR_COMMIT = "f93908467986c77765667925a526ad07be8ad630"
MIRROR_ROOT = (
    "https://huggingface.co/datasets/Lekim89/MOT17/resolve/"
    f"{MIRROR_COMMIT}/train/{SEQUENCE}"
)
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
PEDESTRIAN_CLASS = 1
IGNORED_CLASSES = frozenset({2, 7, 8, 12})


@dataclass(frozen=True)
class Row:
    frame: int
    identity: int
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int = -1
    visibility: float = -1.0


def intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def match_boxes(
    ground_truth: list[Row],
    predictions: list[Row],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    if not ground_truth or not predictions:
        return []
    costs = np.array(
        [
            [
                1.0 - intersection_over_union(gt.box, prediction.box)
                for prediction in predictions
            ]
            for gt in ground_truth
        ]
    )
    gt_indices, prediction_indices = linear_sum_assignment(costs)
    return [
        (int(gt_index), int(prediction_index))
        for gt_index, prediction_index in zip(
            gt_indices, prediction_indices, strict=True
        )
        if costs[gt_index, prediction_index] <= 1.0 - iou_threshold
    ]


def suppress_ignored_predictions(
    targets: list[Row],
    ignored: list[Row],
    predictions: list[Row],
    iou_threshold: float,
) -> list[Row]:
    protected = {
        prediction_index
        for _, prediction_index in match_boxes(targets, predictions, iou_threshold)
    }
    return [
        prediction
        for index, prediction in enumerate(predictions)
        if index in protected
        or not any(
            intersection_over_union(prediction.box, ignored_row.box) >= iou_threshold
            for ignored_row in ignored
        )
    ]


def parse_mot(path: Path, ground_truth: bool = False) -> dict[int, list[Row]]:
    rows: dict[int, list[Row]] = defaultdict(list)
    with path.open(newline="") as stream:
        for line_number, values in enumerate(csv.reader(stream), start=1):
            if not values:
                continue
            if len(values) < 7:
                raise ValueError(f"{path}:{line_number}: expected at least 7 columns")
            try:
                frame = int(float(values[0]))
                identity = int(float(values[1]))
                box = tuple(float(value) for value in values[2:6])
                confidence = float(values[6])
                class_id = int(float(values[7])) if ground_truth else -1
                visibility = float(values[8]) if ground_truth else -1.0
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid numeric value"
                ) from error
            if frame <= 0 or box[2] <= 0.0 or box[3] <= 0.0:
                raise ValueError(
                    f"{path}:{line_number}: invalid frame or box dimensions"
                )
            rows[frame].append(
                Row(frame, identity, box, confidence, class_id, visibility)
            )
    return rows


def calculate_metrics(
    gt_by_frame: dict[int, list[Row]],
    detections_by_frame: dict[int, list[Row]],
    tracks_by_frame: dict[int, list[Row]],
    frame_count: int,
    iou_threshold: float,
) -> dict[str, float | int]:
    true_positives = false_positives = false_negatives = 0
    accumulator = mm.MOTAccumulator(auto_id=False)

    for frame in range(1, frame_count + 1):
        gt_rows = [row for row in gt_by_frame.get(frame, []) if row.confidence == 1.0]
        targets = [row for row in gt_rows if row.class_id == PEDESTRIAN_CLASS]
        ignored = [row for row in gt_rows if row.class_id in IGNORED_CLASSES]

        detections = suppress_ignored_predictions(
            targets, ignored, detections_by_frame.get(frame, []), iou_threshold
        )
        matches = match_boxes(targets, detections, iou_threshold)
        true_positives += len(matches)
        false_positives += len(detections) - len(matches)
        false_negatives += len(targets) - len(matches)

        tracks = suppress_ignored_predictions(
            targets, ignored, tracks_by_frame.get(frame, []), iou_threshold
        )
        distances = np.array(
            [
                [
                    1.0 - intersection_over_union(target.box, track.box)
                    if intersection_over_union(target.box, track.box) >= iou_threshold
                    else np.nan
                    for track in tracks
                ]
                for target in targets
            ]
        )
        if not targets:
            distances = np.empty((0, len(tracks)))
        elif not tracks:
            distances = np.empty((len(targets), 0))
        accumulator.update(
            [row.identity for row in targets],
            [row.identity for row in tracks],
            distances,
            frameid=frame,
        )

    metric_names = [
        "mota",
        "idf1",
        "num_false_positives",
        "num_misses",
        "num_switches",
    ]
    summary = mm.metrics.create().compute(
        accumulator, metrics=metric_names, name=SEQUENCE
    )
    denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    return {
        "precision": true_positives / denominator if denominator else 0.0,
        "recall": true_positives / recall_denominator if recall_denominator else 0.0,
        "mota": float(summary.at[SEQUENCE, "mota"]),
        "idf1": float(summary.at[SEQUENCE, "idf1"]),
        "fp": int(summary.at[SEQUENCE, "num_false_positives"]),
        "fn": int(summary.at[SEQUENCE, "num_misses"]),
        "id_switches": int(summary.at[SEQUENCE, "num_switches"]),
    }


def download_file(url: str, destination: Path, downloaded_bytes: int) -> int:
    if destination.exists():
        return downloaded_bytes
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "mot-evaluation/1.0"})
    with urllib.request.urlopen(request) as response:
        content_length = int(response.headers.get("Content-Length", "0"))
        if downloaded_bytes + content_length > MAX_DOWNLOAD_BYTES:
            raise RuntimeError("MOT17 download would exceed the 500 MB safety limit")
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                downloaded_bytes += len(chunk)
                if downloaded_bytes > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(
                        "MOT17 download exceeded the 500 MB safety limit"
                    )
                output.write(chunk)
        temporary.replace(destination)
    return downloaded_bytes


def acquire_subset(root: Path, frame_count: int) -> int:
    sequence_root = root / SEQUENCE
    downloaded_bytes = download_file(
        f"{MIRROR_ROOT}/gt/gt.txt", sequence_root / "gt" / "gt.txt", 0
    )
    for frame in range(1, frame_count + 1):
        filename = f"{frame:06d}.jpg"
        downloaded_bytes = download_file(
            f"{MIRROR_ROOT}/img1/{filename}",
            sequence_root / "img1" / filename,
            downloaded_bytes,
        )
    return downloaded_bytes


def run(command: Iterable[str], root: Path) -> None:
    subprocess.run(list(command), cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    args = parser.parse_args()
    if args.frames <= 0 or args.frames > FRAME_COUNT:
        parser.error(f"--frames must be between 1 and {FRAME_COUNT}")
    if not 0.0 < args.iou <= 1.0:
        parser.error("--iou must be in (0, 1]")

    root = Path(__file__).resolve().parents[1]
    model = root / "data" / "yolo11n.onnx"
    if not model.is_file():
        raise FileNotFoundError(f"required model does not exist: {model}")

    data_root = root / "data" / "mot17"
    downloaded_bytes = acquire_subset(data_root, args.frames)
    print(f"Downloaded {downloaded_bytes / (1024 * 1024):.2f} MiB")

    run(["cmake", "--preset", "debug"], root)
    run(["cmake", "--build", "--preset", "debug", "--target", "evaluate_mot"], root)
    output_root = data_root / "results"
    detection_path = output_root / "detections.txt"
    track_path = output_root / "tracks.txt"
    run(
        [
            str(root / "build" / "debug" / "evaluate_mot"),
            str(model),
            str(data_root / SEQUENCE / "img1"),
            str(detection_path),
            str(track_path),
            str(args.frames),
        ],
        root,
    )

    metrics = calculate_metrics(
        parse_mot(data_root / SEQUENCE / "gt" / "gt.txt", ground_truth=True),
        parse_mot(detection_path),
        parse_mot(track_path),
        args.frames,
        args.iou,
    )
    print(f"MOT17-05-FRCNN frames 1-{args.frames}, IoU {args.iou:.2f}")
    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}"
        )


if __name__ == "__main__":
    main()
