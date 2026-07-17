#!/usr/bin/env python3
"""Generate annotated crops from videos or still images located under real_data/."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "real_data"
DEFAULT_IMAGE_DIR = ROOT / "data" / "images"
DEFAULT_ANNOTATION_FILE = ROOT / "data" / "annotation.txt"
DEFAULT_BACKUP_SUFFIX = ".realdata.bak"
DEFAULT_IMAGES_PER_FOLDER = 1000
DEFAULT_FOLDER_START = 1
DEFAULT_FRAME_STEP = 1
MIN_DIMENSION = 6
MAX_DIMENSION = 750
DEFAULT_DETECTOR_MODEL = ROOT / "deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx"
DETECTOR_INPUT_SIZE = 640
DETECTOR_BODY_LABEL = 0
DETECTOR_BODY_THRESHOLD = 0.35
DEFAULT_CLASS_PIE_FILE = ROOT / "data" / "class_distribution.png"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CLASS_PREFIX_TO_ID = {
    "no_possession": 0,
    "possession": 1,
}
ORDERED_CLASS_PREFIXES = sorted(CLASS_PREFIX_TO_ID.items(), key=lambda item: len(item[0]), reverse=True)
CLASS_ID_TO_PREFIX = {class_id: prefix for prefix, class_id in CLASS_PREFIX_TO_ID.items()}


@dataclass
class FrameAnnotation:
    filename: str
    video_id: str
    timestamp: str
    person_id: int
    class_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mp4 videos or labeled image folders into annotated PNG crops."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Directory containing real_data mp4 files.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory where cropped PNGs are stored.")
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help="Annotation text file to append to.",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=DEFAULT_BACKUP_SUFFIX,
        help="Suffix appended to annotation file for backups (default: %(default)s).",
    )
    parser.add_argument("--images-per-folder", type=int, default=DEFAULT_IMAGES_PER_FOLDER, help="Maximum PNGs per folder (default: 1000).")
    parser.add_argument(
        "--start-folder",
        type=int,
        default=DEFAULT_FOLDER_START,
        help="Numeric folder index to start from when saving images (default: 1).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help="Take every Nth frame from each video (default: 1).",
    )
    parser.add_argument("--person-id", type=int, default=1, help="Person identifier stored in annotations (default: 1).")
    parser.add_argument("--min-dimension", type=int, default=MIN_DIMENSION, help="Minimum width/height required (default: 6).")
    parser.add_argument("--max-dimension", type=int, default=MAX_DIMENSION, help="Maximum width/height allowed (default: 750).")
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL,
        help="ONNX detector used for cropping and filtering (default: deimv2...640.onnx).",
    )
    parser.add_argument(
        "--allow-multi-body",
        action="store_true",
        help="When set, keep crops for all detected bodies instead of just the best scoring one.",
    )
    parser.add_argument(
        "--input-image-dir",
        type=Path,
        default=None,
        help="Directory containing still images grouped by class folder names.",
    )
    parser.add_argument(
        "--class-pie-file",
        type=Path,
        default=DEFAULT_CLASS_PIE_FILE,
        help="PNG path for a pie chart summarizing generated class counts.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs if duplicates occur.")
    parser.add_argument("--dry-run", action="store_true", help="Plan operations without writing files.")
    args = parser.parse_args()
    if args.images_per_folder < 1:
        parser.error("--images-per-folder must be at least 1")
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1")
    if args.max_dimension < args.min_dimension:
        parser.error("--max-dimension must be greater than or equal to --min-dimension")
    return args


def load_detector_session(model_path: Path) -> tuple[ort.InferenceSession, str]:
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model not found: {model_path}")
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": ".",
                "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def _prepare_detector_blob(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        image,
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    blob = resized.transpose(2, 0, 1).astype(np.float32, copy=False)
    blob = np.expand_dims(blob, axis=0)
    return blob


def _run_detector(session: ort.InferenceSession, input_name: str, image: np.ndarray) -> np.ndarray:
    blob = _prepare_detector_blob(image)
    return session.run(None, {input_name: blob})[0][0]


def detect_person_boxes(
    session: ort.InferenceSession,
    input_name: str,
    frame: np.ndarray,
) -> list[tuple[tuple[float, float, float, float], float]]:
    detections = _run_detector(session, input_name, frame)
    valid: list[tuple[tuple[float, float, float, float], float]] = []
    for det in detections:
        label = int(round(det[0]))
        score = float(det[5])
        if label != DETECTOR_BODY_LABEL or score < DETECTOR_BODY_THRESHOLD:
            continue
        det_array = det
        box = (float(det_array[1]), float(det_array[2]), float(det_array[3]), float(det_array[4]))
        valid.append((box, score))
    valid.sort(key=lambda item: item[1], reverse=True)
    return valid


def crop_frame_using_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
) -> tuple[np.ndarray, int, int] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    x1_px = max(int(round(x1 * width)), 0)
    y1_px = max(int(round(y1 * height)), 0)
    x2_px = min(int(round(x2 * width)), width)
    y2_px = min(int(round(y2 * height)), height)
    if x2_px <= x1_px or y2_px <= y1_px:
        return None
    crop = frame[y1_px:y2_px, x1_px:x2_px].copy()
    return crop, crop.shape[1], crop.shape[0]


def classify_name(name: str) -> int | None:
    normalized = name.lower().replace("-", "_")
    for prefix, class_id in ORDERED_CLASS_PREFIXES:
        if normalized.startswith(prefix):
            return class_id
        compact = prefix.replace("_", "")
        if compact and normalized.startswith(compact):
            return class_id
    return None


def classify_video(path: Path) -> int | None:
    return classify_name(path.stem)


def iter_video_files(input_dir: Path) -> list[tuple[Path, int]]:
    videos: list[tuple[Path, int]] = []
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for video_path in sorted(input_dir.glob("*.mp4")):
        class_id = classify_video(video_path)
        if class_id is None:
            print(f"[skip] {video_path.name} does not match expected class prefixes.", file=sys.stderr)
            continue
        videos.append((video_path, class_id))
    if not videos:
        print("[info] No matching mp4 files found.", file=sys.stderr)
    return videos


def iter_image_files(image_root: Path) -> list[tuple[Path, int]]:
    images: list[tuple[Path, int]] = []
    if not image_root.exists():
        raise FileNotFoundError(f"Input image directory not found: {image_root}")

    def gather(directory: Path, class_id: int) -> None:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append((path, class_id))

    root_class_id = classify_name(image_root.name)
    if root_class_id is not None:
        gather(image_root, root_class_id)
    else:
        found_dir = False
        for entry in sorted(image_root.iterdir()):
            if not entry.is_dir():
                continue
            found_dir = True
            class_id = classify_name(entry.name)
            if class_id is None:
                print(f"[skip] {entry.name} does not match expected class prefixes.", file=sys.stderr)
                continue
            gather(entry, class_id)
        if not found_dir:
            print(f"[info] No class folders found under {image_root}", file=sys.stderr)

    if not images:
        print(f"[info] No image files found under {image_root}", file=sys.stderr)
    return images


def ensure_backup(annotation_file: Path, suffix: str) -> Path | None:
    if not annotation_file.exists():
        return None
    backup_path = annotation_file.with_suffix(annotation_file.suffix + suffix)
    shutil.copy2(annotation_file, backup_path)
    print(f"[backup] Copied {annotation_file} -> {backup_path}")
    return backup_path


def next_image_path(
    image_dir: Path,
    start_folder: int,
    images_per_folder: int,
    generated_count: int,
    filename: str,
) -> Path:
    folder_index = start_folder + generated_count // images_per_folder
    folder_name = f"{folder_index:04d}"
    folder_path = image_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path / filename


def save_frame(frame, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.save(output_path)


def process_video(
    video_path: Path,
    class_id: int,
    args: argparse.Namespace,
    generated_count: int,
    detector_session: ort.InferenceSession,
    detector_input_name: str,
) -> tuple[int, list[FrameAnnotation]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    fps = fps if fps > 0 else 30.0
    frame_index = 0
    kept = 0
    annotations: list[FrameAnnotation] = []
    video_id = video_path.stem

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if frame_index % args.frame_step != 0:
            frame_index += 1
            continue

        detections = detect_person_boxes(detector_session, detector_input_name, frame)
        if not detections:
            frame_index += 1
            continue
        selected = detections if args.allow_multi_body else detections[:1]
        timestamp_seconds = frame_index / fps
        timestamp = f"{timestamp_seconds:.3f}"
        for det_index, (box, _) in enumerate(selected):
            crop_result = crop_frame_using_box(frame, box)
            if crop_result is None:
                continue
            crop, width_px, height_px = crop_result
            if (
                width_px < args.min_dimension
                or height_px < args.min_dimension
                or width_px > args.max_dimension
                or height_px > args.max_dimension
            ):
                continue

            filename_suffix = ""
            if args.allow_multi_body and len(selected) > 1:
                filename_suffix = f"_{det_index + 1}"
            filename = f"{video_id}_{frame_index:06d}_{class_id}{filename_suffix}.png"

            if not args.dry_run:
                output_path = next_image_path(
                    args.image_dir, args.start_folder, args.images_per_folder, generated_count + kept, filename
                )
                try:
                    save_frame(crop, output_path, overwrite=args.overwrite)
                except FileExistsError:
                    continue

            annotations.append(
                FrameAnnotation(
                    filename=filename,
                    video_id=video_id,
                    timestamp=timestamp,
                    person_id=args.person_id,
                    class_id=class_id,
                )
            )
            kept += 1
        frame_index += 1

    capture.release()
    print(f"[info] Processed {video_path.name}: kept {kept} frames.")
    return kept, annotations


def process_image_file(
    image_path: Path,
    class_id: int,
    args: argparse.Namespace,
    generated_count: int,
    detector_session: ort.InferenceSession,
    detector_input_name: str,
) -> tuple[int, list[FrameAnnotation]]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[warn] Skipping unreadable image {image_path}", file=sys.stderr)
        return 0, []

    detections = detect_person_boxes(detector_session, detector_input_name, frame)
    if not detections:
        return 0, []

    selected = detections if args.allow_multi_body else detections[:1]
    annotations: list[FrameAnnotation] = []
    timestamp = "0.000"
    kept = 0

    for det_index, (box, _) in enumerate(selected):
        crop_result = crop_frame_using_box(frame, box)
        if crop_result is None:
            continue
        crop, width_px, height_px = crop_result
        if (
            width_px < args.min_dimension
            or height_px < args.min_dimension
            or width_px > args.max_dimension
            or height_px > args.max_dimension
        ):
            continue

        sequence_index = generated_count + kept
        filename_suffix = ""
        if args.allow_multi_body and len(selected) > 1:
            filename_suffix = f"_{det_index + 1}"
        filename = f"{image_path.stem}_{sequence_index:06d}_{class_id}{filename_suffix}.png"

        if not args.dry_run:
            output_path = next_image_path(
                args.image_dir, args.start_folder, args.images_per_folder, sequence_index, filename
            )
            try:
                save_frame(crop, output_path, overwrite=args.overwrite)
            except FileExistsError:
                continue

        annotations.append(
            FrameAnnotation(
                filename=filename,
                video_id=image_path.stem,
                timestamp=timestamp,
                person_id=args.person_id,
                class_id=class_id,
            )
        )
        kept += 1

    return kept, annotations


def append_annotations(annotation_file: Path, rows: list[FrameAnnotation]) -> None:
    if not rows:
        print("[annotation] No rows to append.")
        return
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    with annotation_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            (row.filename, row.video_id, row.timestamp, row.person_id, row.class_id) for row in rows
        )
    print(f"[annotation] Appended {len(rows)} rows to {annotation_file}.")


def load_annotation_class_counts(annotation_file: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not annotation_file.exists():
        return counts
    with annotation_file.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if len(row) < 5:
                continue
            try:
                class_id = int(row[4])
            except ValueError:
                continue
            if class_id not in CLASS_ID_TO_PREFIX:
                raise ValueError(
                    f"Unsupported class_id={class_id} in {annotation_file} at line {line_number}; "
                    "PPC annotations only allow class_id 0 and 1."
                )
            counts[class_id] += 1
    return counts


def save_class_distribution_pie(counts: Counter[int], output_path: Path) -> None:
    if not counts:
        print("[pie] No annotations available; skipping pie chart.")
        return
    labels = []
    sizes = []
    for class_id, count in sorted(counts.items()):
        prefix = CLASS_ID_TO_PREFIX.get(class_id, str(class_id))
        labels.append(f"{prefix} ({count})")
        sizes.append(count)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
    )
    ax.axis("equal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[pie] Saved class distribution chart to {output_path}")


def log_class_split_counts(counts: Counter[int]) -> None:
    total = sum(counts.values())
    if total == 0:
        print("[counts] No annotations available; skipping split counts output.")
        return
    print("[counts] Split counts (existing + new):")
    print(f"[counts] Total annotations: {total}")
    for class_id, count in sorted(counts.items()):
        label = CLASS_ID_TO_PREFIX.get(class_id, str(class_id))
        percentage = (count / total) * 100.0
        print(f"[counts] {label}: {count} ({percentage:.1f}%)")


def main() -> None:
    args = parse_args()
    videos: list[tuple[Path, int]] = []
    if args.input_image_dir is None:
        videos = iter_video_files(args.input_dir)
    else:
        print("[info] --input-image-dir specified; skipping video processing.")
    image_files: list[tuple[Path, int]] = []
    if args.input_image_dir is not None:
        image_files = iter_image_files(args.input_image_dir)

    if not videos and not image_files:
        return

    # Reject legacy PUC annotations before any new files or rows are written.
    load_annotation_class_counts(args.annotation_file)

    detector_session, detector_input_name = load_detector_session(args.detector_model)

    total_kept = 0
    all_rows: list[FrameAnnotation] = []
    for video_path, class_id in videos:
        kept, rows = process_video(
            video_path,
            class_id,
            args,
            total_kept,
            detector_session,
            detector_input_name,
        )
        total_kept += kept
        all_rows.extend(rows)

    for image_path, class_id in image_files:
        kept, rows = process_image_file(
            image_path,
            class_id,
            args,
            total_kept,
            detector_session,
            detector_input_name,
        )
        total_kept += kept
        all_rows.extend(rows)

    if total_kept == 0:
        print("[info] No frames met the filtering criteria; nothing to write.")
        return

    if args.dry_run:
        print("[dry-run] Skipping file writes and annotation updates.")
        return

    ensure_backup(args.annotation_file, args.backup_suffix)
    append_annotations(args.annotation_file, all_rows)
    class_counts = load_annotation_class_counts(args.annotation_file)
    save_class_distribution_pie(class_counts, args.class_pie_file)
    log_class_split_counts(class_counts)
    print(f"[done] Generated {total_kept} frames from {len(videos)} videos and {len(image_files)} images.")


if __name__ == "__main__":
    main()
