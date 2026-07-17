#!/usr/bin/env python3
"""Convert labeled image folders under data/ into a parquet dataset."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_DIR / "dataset.parquet"
DEFAULT_CLASS_PIE_FILE = DEFAULT_DATA_DIR / "class_distribution.png"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LABEL_MAP = {
    0: "no_possession",
    1: "possession",
}
CLASS_NAME_TO_ID = {label: class_id for class_id, label in LABEL_MAP.items()}


def validate_ppc_dataframe(df: pd.DataFrame, *, require_both_classes: bool = True) -> None:
    required_columns = {"class_id", "label"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"PPC dataset is missing required columns: {missing_columns}")
    if df[["class_id", "label"]].isna().any().any():
        raise ValueError("PPC dataset class_id and label values must not be null.")

    class_ids = {int(value) for value in df["class_id"]}
    invalid_ids = sorted(class_ids - set(LABEL_MAP))
    if invalid_ids:
        raise ValueError(
            f"PPC dataset contains unsupported class_id values {invalid_ids}; only 0 and 1 are allowed."
        )
    for row_index, row in df[["class_id", "label"]].iterrows():
        class_id = int(row["class_id"])
        expected_label = LABEL_MAP[class_id]
        if str(row["label"]).strip() != expected_label:
            raise ValueError(
                f"PPC label/class_id mismatch at row {row_index}: class_id={class_id} requires "
                f"label={expected_label!r}, got {row['label']!r}."
            )
    if require_both_classes and class_ids != set(LABEL_MAP):
        missing_ids = sorted(set(LABEL_MAP) - class_ids)
        raise ValueError(f"PPC dataset must contain both classes; missing class_id values {missing_ids}.")
    if require_both_classes and "split" in df.columns:
        train_ids = {int(value) for value in df.loc[df["split"] == "train", "class_id"]}
        if train_ids != set(LABEL_MAP):
            missing_ids = sorted(set(LABEL_MAP) - train_ids)
            raise ValueError(
                f"PPC training split must contain both classes; missing class_id values {missing_ids}."
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset.parquet from data/<label> image folders.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing label folders (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Destination parquet file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Target ratio for the training split.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Target ratio for the validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Seed controlling deterministic split assignment.")
    parser.add_argument(
        "--class-pie-file",
        type=Path,
        default=DEFAULT_CLASS_PIE_FILE,
        help=f"PNG path for a class distribution pie chart generated from the parquet (default: {DEFAULT_CLASS_PIE_FILE})",
    )
    parser.add_argument(
        "--pie-only",
        action="store_true",
        help="Only read the parquet specified by --output and write the class distribution pie chart.",
    )
    parser.add_argument(
        "--no-embed-images",
        action="store_true",
        help="Store image paths only instead of embedding raw image bytes in the parquet.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing parquet file.",
    )
    return parser.parse_args()


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0:
        raise ValueError("train-ratio must be greater than zero.")
    if val_ratio < 0:
        raise ValueError("val-ratio must be non-negative.")
    if train_ratio + val_ratio <= 0:
        raise ValueError("At least one split ratio must be positive.")


def _half_up_count(total: int, numerator: float, denominator: float) -> int:
    value = Decimal(total) * Decimal(str(numerator)) / Decimal(str(denominator))
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _iter_images(label_dir: Path) -> Iterable[Path]:
    for path in sorted(label_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _format_image_path(path: Path, dataset_root: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_image_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image file missing while embedding: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def _warn_unknown_label_dirs(data_dir: Path) -> None:
    known_labels = set(CLASS_NAME_TO_ID)
    unknown_dirs = sorted(path.name for path in data_dir.iterdir() if path.is_dir() and path.name not in known_labels)
    if unknown_dirs:
        print(
            f"[warn] Ignoring unknown label directories under {data_dir}: {', '.join(unknown_dirs)}",
            file=sys.stderr,
        )


def _build_rows_for_label(
    *,
    label_name: str,
    class_id: int,
    data_dir: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    embed_images: bool,
) -> list[dict[str, object]]:
    label_dir = data_dir / label_name
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    image_paths = list(_iter_images(label_dir))
    if not image_paths:
        raise RuntimeError(f"No supported image files found under {label_dir}")

    rng = random.Random(seed + class_id)
    rng.shuffle(image_paths)

    ratio_total = train_ratio + val_ratio
    val_count = _half_up_count(len(image_paths), val_ratio, ratio_total)
    split_by_path = {
        path: "val" if index < val_count else "train"
        for index, path in enumerate(image_paths)
    }

    rows: list[dict[str, object]] = []
    for path in sorted(image_paths):
        try:
            source = path.parent.resolve().relative_to(data_dir.resolve()).as_posix()
        except ValueError:
            source = path.parent.name

        row: dict[str, object] = {
            "split": split_by_path[path],
            "image_path": _format_image_path(path, output_root),
            "class_id": class_id,
            "label": label_name,
            "source": source,
            "filename": path.name,
        }
        if embed_images:
            row["image_bytes"] = _read_image_bytes(path)
        rows.append(row)

    return rows


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    _validate_ratios(args.train_ratio, args.val_ratio)
    output_root = args.output.resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)
    _warn_unknown_label_dirs(data_dir)

    embed_images = not args.no_embed_images
    rows: list[dict[str, object]] = []
    for class_id, label_name in LABEL_MAP.items():
        rows.extend(
            _build_rows_for_label(
                label_name=label_name,
                class_id=class_id,
                data_dir=data_dir,
                output_root=output_root,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed,
                embed_images=embed_images,
            )
        )

    if not rows:
        raise RuntimeError(f"No dataset rows were produced from {data_dir}")

    ordered_columns = ["split", "image_path", "class_id", "label", "source", "filename"]
    if embed_images:
        ordered_columns.insert(2, "image_bytes")

    dataset_df = pd.DataFrame(rows)
    dataset_df = dataset_df[ordered_columns]
    dataset_df = dataset_df.sort_values(["split", "class_id", "source", "filename"]).reset_index(drop=True)
    validate_ppc_dataframe(dataset_df)
    return dataset_df


def _summarize(df: pd.DataFrame) -> None:
    split_counts = Counter(df["split"])
    label_counts = Counter(df["label"])
    split_label_counts = Counter(zip(df["split"], df["label"]))

    print("Split counts:")
    for split in ("train", "val"):
        print(f"  {split:>5}: {split_counts.get(split, 0)}")

    print("Label counts:")
    for label in LABEL_MAP.values():
        print(f"  {label:>16}: {label_counts.get(label, 0)}")

    print("Split/label counts:")
    for split in ("train", "val"):
        for label in LABEL_MAP.values():
            print(f"  {split:>5} {label:>16}: {split_label_counts.get((split, label), 0)}")


def _load_parquet_class_counts(parquet_path: Path) -> Counter[int]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["class_id", "label"])
    if df.empty:
        return Counter()
    validate_ppc_dataframe(df)
    return Counter(int(value) for value in df["class_id"].dropna())


def save_class_distribution_pie_from_parquet(parquet_path: Path, output_path: Path) -> None:
    counts = _load_parquet_class_counts(parquet_path)
    if not counts:
        print("[pie] No parquet rows available; skipping pie chart.")
        return

    labels = []
    sizes = []
    for class_id, count in sorted(counts.items()):
        label = LABEL_MAP.get(class_id, str(class_id))
        labels.append(f"{label} ({count})")
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


def main() -> None:
    args = _parse_args()
    output = args.output
    if args.pie_only:
        save_class_distribution_pie_from_parquet(output, args.class_pie_file)
        return

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; use --overwrite to replace it.")

    dataset_df = build_dataset(args)
    dataset_df.to_parquet(output, index=False)
    print(f"Wrote {output} ({len(dataset_df)} rows).")
    _summarize(dataset_df)
    save_class_distribution_pie_from_parquet(output, args.class_pie_file)


if __name__ == "__main__":
    main()
