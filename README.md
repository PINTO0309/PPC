# PPC

A binary classification pipeline that determines whether a subject is holding an object from a 48x48 RGB image.

| class_id | label | Model output index |
| ---: | --- | ---: |
| 0 | `no_possession` | 0 |
| 1 | `possession` | 1 |

The PyTorch model and exported ONNX model always return two probabilities in the following order:
`[no_possession_probability, possession_probability]`.

## Setup

The Python version and dependencies are pinned in `pyproject.toml`.

```bash
uv sync
```

Run the commands below in the managed environment by prefixing them with `uv run`.

## Dataset

Place images under `data/` using the following structure. Image filenames are not used to determine labels;
only the top-level label directory is used.

```text
data/
├── no_possession/
│   └── .../*.png
└── possession/
    └── .../*.png
```

Generate the Parquet dataset. By default, raw image bytes are embedded, and each class is split into
train and validation sets using a seeded 90/10 split.

```bash
uv run python 02_make_parquet.py
```

To store image paths without embedding image bytes, run:

```bash
uv run python 02_make_parquet.py --no-embed-images --overwrite
```

To merge multiple PPC Parquet files, specify each input path relative to `data/`.

```bash
uv run python 03_merge_parquet.py dataset_a.parquet dataset_b.parquet --overwrite
```

An optional preprocessing command can generate crops and annotations from videos or labeled image directories.
Use `--detector-model` to specify the detector ONNX model.

```bash
uv run python 01_data_prep_realdata.py \
  --input-image-dir /path/to/labeled-images \
  --detector-model /path/to/detector.onnx
```

## Training and inference

```bash
uv run python -m ppc train \
  --data_root data/dataset.parquet \
  --output_dir runs/ppc_baseline \
  --epochs 30
```

The best model is selected using validation `possession_f1` and saved as
`ppc_best_*_possession_f1_*.pt`. Training history includes accuracy, possession precision/recall/F1,
and macro precision/recall/F1. When training finishes, `runs/ppc_baseline/ppc_best.onnx` is
automatically exported from the best checkpoint and simplified with onnxsim. The generated ONNX path
is recorded as `onnx_model` in `summary.json`, while the simplification result is recorded as
`onnx_simplified`.

```bash
uv run python -m ppc predict \
  --checkpoint runs/ppc_baseline/ppc_best_epoch0001_possession_f1_0.9000.pt \
  --inputs data/possession/000000 \
  --output runs/ppc_baseline/predictions.csv
```

The prediction CSV contains `pred_label`, `pred_class`, `prob_no_possession`, and `prob_possession`.

## ONNX

```bash
uv run python -m ppc exportonnx \
  --checkpoint runs/ppc_baseline/ppc_best_epoch0001_possession_f1_0.9000.pt \
  --output ppc_s_48x48.onnx
```

Visualize feature maps:

```bash
uv run python 10_visualize_ppc_heatmaps.py \
  --model ppc_s_48x48.onnx \
  --image data/possession/000000/point_000000.png
```

Run feature ablation against the `possession` probability:

```bash
uv run python 11_ablate_ppc_features.py \
  --model ppc_s_48x48.onnx \
  --image-dir data/possession/000000 \
  --target-class possession
```

Legacy three-class PUC Parquet files, checkpoints, and ONNX models are not compatible with PPC.
`demo_puc.py` and `demo_phone_gaze_classification.py` are independent legacy demos and are outside
the scope of this PPC pipeline.
