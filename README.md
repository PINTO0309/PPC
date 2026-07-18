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
-pm ppc_l_48x48.onnx \
-dlr -dnm -dgm -dhm \
-ep cuda \
-gm gazelle_dinov3_vit_tiny_inout_1x3x640x640_1xNx4.onnx \
--enable-heatmap

uv run python demo_phone_gaze_classification.py \
-v 0 \
-pm ppc_l_48x48.onnx \
-dlr -dnm -dgm -dhm \
-ep tensorrt \
-gm gazelle_dinov3_vit_tiny_inout_1x3x640x640_1xNx4.onnx \
--enable-heatmap
```

## Training Pipeline

- Use the labeled image folders under `data/no_action`, `data/point_somewhere`, and `data/point`.
- `02_make_parquet.py` writes pre-defined train/val splits into `data/dataset.parquet` using an image-level 9:1 split per class.
- The training loop relies on `BCEWithLogitsLoss` plus class-balanced `pos_weight` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities. Use `--train_resampling weighted` to switch on the previous `WeightedRandomSampler` behaviour, or `--train_resampling balanced` to physically duplicate minority classes before shuffling.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `ppc_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `ppc_best_epoch0004_f1_0.9321.pt` (also latest 10).
- The backbone can be switched with `--arch_variant`. Supported combinations with `--head_variant` are:

  | `--arch_variant` | Default (`--head_variant auto`) | Explicitly selectable heads | Remarks |
  |------------------|-----------------------------|---------------------------|------|
  | `baseline`       | `avg`                       | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, you need to adjust the height and width of the feature map so that they are divisible by `--token_mixer_grid` (if left as is, an exception will occur during ONNX conversion or inference). |
  | `inverted_se`    | `avgmax_mlp`                | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, it is necessary to adjust `--token_mixer_grid` as above. |
  | `convnext`       | `transformer`               | `avg`, `avgmax_mlp`, `transformer`, `mlp_mixer` | For token mixer heads, the feature map dimensions must be divisible by `--token_mixer_grid` (default `2x3`). |
- The classification head is selected with `--head_variant` (`avg`, `avgmax_mlp`, `transformer`, `mlp_mixer`, or `auto` which derives a sensible default from the backbone).
- Pass `--rgb_to_yuv_to_y` to convert RGB crops to YUV, keep only the Y (luma) channel inside the network, and train a single-channel stem without modifying the dataloader.
- Alternatively, use `--rgb_to_lab` or `--rgb_to_luv` to convert inputs to CIE Lab/Luv (3-channel) before the stem; these options are mutually exclusive with each other and with `--rgb_to_yuv_to_y`.
- Mixed precision can be enabled with `--use_amp` when CUDA is available.
- Resume training with `--resume path/to/ppc_epoch_XXXX.pt`; all optimiser/scheduler/AMP states and history are restored.
- Loss/accuracy/F1 metrics are logged to TensorBoard under `output_dir`, and `tqdm` progress bars expose per-epoch progress for train/val/test loops.

Baseline depthwise-separable CNN:

```bash
SIZE=48x48
uv run python -m ppc train \
--data_root data/dataset.parquet \
--output_dir runs/ppc_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant baseline \
--seed 42 \
--device auto \
--use_amp
```

Inverted residual + SE variant (recommended for higher capacity):

```bash
SIZE=48x48
VAR=s
uv run python -m ppc train \
--data_root data/dataset.parquet \
--output_dir runs/ppc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--seed 42 \
--device auto \
--use_amp

```

ConvNeXt-style backbone with transformer head over pooled tokens:

```bash
SIZE=48x48
uv run python -m ppc train \
--data_root data/dataset.parquet \
--output_dir runs/ppc_convnext_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant convnext \
--head_variant transformer \
--token_mixer_grid 2x2 \
--seed 42 \
--device auto \
--use_amp
```

- Outputs include the latest 10 `ppc_epoch_*.pt`, the latest 10 `ppc_best_epochXXXX_f1_YYYY.pt` (highest validation F1, or training F1 when no validation split), `history.json`, `summary.json`, optional `test_predictions.csv`, and `train.log`.
- After every epoch a confusion matrix and ROC curve are saved under `runs/ppc/diagnostics/<split>/confusion_<split>_epochXXXX.png` and `roc_<split>_epochXXXX.png`.
- `--image_size` accepts either a single integer for square crops (e.g. `--image_size 48`) or `HEIGHTxWIDTH` to resize non-square frames (e.g. `--image_size 64x48`).
- Add `--resume <checkpoint>` to continue from an earlier epoch. Remember that `--epochs` indicates the desired total epoch count (e.g. resuming `--epochs 40` after training to epoch 30 will run 10 additional epochs).
- Launch TensorBoard with:
  ```bash
  tensorboard --logdir runs/ppc
  ```

### ONNX Export

```bash
uv run python -m ppc exportonnx \
--checkpoint runs/ppc_is_s_48x48/ppc_best_epoch0049_f1_0.9939.pt \
--output ppc_s_48x48.onnx \
--opset 17
```

- The saved graph exposes `images` as input and `prob_pointing` as output (batch dimension is dynamic); probabilities can be consumed directly.
- After exporting, the tool runs `onnxsim` for simplification and rewrites any remaining BatchNormalization nodes into affine `Mul`/`Add` primitives. If simplification fails, a warning is emitted and the unsimplified model is preserved.

## Arch

<img width="350" alt="puc_p_48x48" src="https://github.com/user-attachments/assets/4f4bacfd-5ac1-4af6-a68b-379377f3dc49" />

## Ultra-lightweight classification model series
1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [HSC: Happy smile classifier](https://github.com/PINTO0309/HSC) - MIT License
7. [WHC: Waving Hand Classification](https://github.com/PINTO0309/WHC) - MIT License
8. [UHD: Ultra-lightweight human detection](https://github.com/PINTO0309/UHD) - MIT License
9. [MWC: Mask wearing classifier](https://github.com/PINTO0309/MWC) - MIT License
10. [SGC: Classification of wearing vs. not wearing sunglasses. 48x48.](https://github.com/PINTO0309/SGC) - MIT License
11. [HHC: Head Hat Classification. HHC is a binary classifier for cropped head images. 48x48.](https://github.com/PINTO0309/HHC) - MIT License
12. [BPC: Background Plain classification. 48x48.](https://github.com/PINTO0309/BPC) - MIT License
