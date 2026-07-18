# PPC

A model that performs binary classification to determine whether the subject is holding a smartphone. 48x48 RGB image.

| class_id | label | Model output index |
| ---: | --- | ---: |
| 0 | `no_possession` | 0 |
| 1 | `possession` | 1 |

The PyTorch model and exported ONNX model always return two probabilities in the following order:
`[no_possession_probability, possession_probability]`.

https://github.com/user-attachments/assets/715f87c7-e1ed-4849-b838-377bb010a99f

## Data sample

|no<br>possession|no<br>possession|possession|possession|possession|possession|
|:-:|:-:|:-:|:-:|:-:|:-:|
<img width="48" height="48" alt="no_action_008364" src="https://github.com/user-attachments/assets/327c71a0-c636-4ea9-8700-ed5a28a0050e" />|<img width="48" height="48" alt="no_action_008001" src="https://github.com/user-attachments/assets/9d080aa1-fc47-4c83-85a6-9e1f1fa087b8" />|<img width="48" height="48" alt="point_somewhere_002145" src="https://github.com/user-attachments/assets/2d816974-3ddd-4d2c-ae61-0df6cbe5c14f" />|<img width="48" height="48" alt="point_somewhere_002068" src="https://github.com/user-attachments/assets/108d9652-0c07-47f0-8b34-428ee5f23dfa" />|<img width="48" height="48" alt="point_003496" src="https://github.com/user-attachments/assets/0bc4d7bd-e85e-43f9-a893-dbbb070f46da" />|<img width="48" height="48" alt="point_003008" src="https://github.com/user-attachments/assets/cbf23eed-5ded-4709-8e7a-0cc8eab920eb" />|

## Setup

The Python version and dependencies are pinned in `pyproject.toml`.

```bash
git clone https://github.com/PINTO0309/PPC.git && cd PPC
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
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

## Inference
```bash
uv run python demo_phone_gaze_classification.py \
-v 0 \
-pm puc_l_48x48.onnx \
-dlr -dnm -dgm -dhm \
-ep cuda \
-gm gazelle_dinov3_vit_tiny_inout_1x3x640x640_1xNx4.onnx \
--enable-heatmap

uv run python demo_phone_gaze_classification.py \
-v 0 \
-pm puc_l_48x48.onnx \
-dlr -dnm -dgm -dhm \
-ep tensorrt \
-gm gazelle_dinov3_vit_tiny_inout_1x3x640x640_1xNx4.onnx \
--enable-heatmap
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

## PUC demos with the PPC possession gate

`demo_ppc.py` retains the existing three-class PUC action inference and adds PPC inference on the
same center-expanded RGB hand crop. Both classifiers use `INTER_LINEAR` resizing, 0-to-1
normalization, and CHW conversion. The PPC output order must be `[no_possession, possession]`.

```bash
uv run python demo_ppc.py \
  --model /path/to/wholebody_detector.onnx \
  --puc_model /path/to/puc_model.onnx \
  --ppc_model /path/to/ppc_best.onnx \
  --video /path/to/input.mp4
```

The gaze-enabled demo uses the same PPC gate while preserving Gazelle processing:

```bash
uv run python demo_phone_gaze_classification.py \
  --model /path/to/wholebody_detector.onnx \
  --puc_model /path/to/puc_model.onnx \
  --enable-puc \
  --ppc_model /path/to/ppc_best.onnx \
  --gazelle-model /path/to/gazelle.onnx \
  --video /path/to/input.mp4
```

Gaze-to-hand overlap is evaluated in a region obtained by expanding the original detected Hand box
by 1.5x around its center. This is independent of the 2.5x crop used as the PUC/PPC model input.

`--ppc_model` and `--ppc-model` are equivalent. Both demos use `ppc_l_48x48.onnx` by default when
the option is omitted. PPC is still inferred for each detected hand, then both hands belonging to a
body are combined into one PPC label and confidence. A hand is labeled `possession` when its
`possession` probability is 0.5 or greater. `possession` takes priority over
`no_possession`, and the highest-confidence result within that class becomes the body result.

The combined PPC result is tracked per body and smoothed with the same `state_verdict` algorithm
previously used by PUC. The default long and short histories are 10 and 6 frames. A new body remains
`no_possession` until all history buffers are populated; with continuous `possession`, the default
gate opens on frame 10. It then requires at least 5 positive samples in the 10-frame history and at
least 5 in the latest 6 frames. Configure the buffers with `--ppc-long-history-size` and
`--ppc-short-history-size`; the existing `--hand-*` and `--body-*` names remain aliases.

PUC action inference, UI tracking, and gaze activation proceed only while the history-confirmed PPC
label is `possession`. PUC itself has no temporal history, so its current-frame result is reflected
immediately after the PPC gate. PPC errors and frames without a valid PPC result append a negative
history sample and still force immediate `no_action` for that frame.

The body overlay shows the history-confirmed PPC label but always displays the representative hand's
current-frame `possession` probability. It may therefore show a value below 0.5 while a historical
`possession` state is being maintained. In the gaze demo, a suppressed body is excluded from
gaze-based activation. The PUC action gate is disabled by default; pass `--enable-puc` to restrict
gaze candidates to Hands classified by PUC as `point_somewhere` or `point`. PUC inference and the
existing display values remain available when the gate is disabled.
