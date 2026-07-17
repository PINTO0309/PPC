from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image

from ppc.pipeline import TrainConfig, predict_images, train_pipeline


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_tiny_dataset(path: Path) -> tuple[Path, Path]:
    rows = []
    inference_images = []
    for class_id, (label, color) in enumerate(
        (("no_possession", (0, 0, 0)), ("possession", (255, 255, 255)))
    ):
        for split, count in (("train", 2), ("val", 1)):
            for index in range(count):
                image_path = path.parent / f"{label}_{split}_{index}.png"
                Image.new("RGB", (48, 48), color=color).save(image_path)
                inference_images.append(image_path)
                rows.append(
                    {
                        "split": split,
                        "image_path": image_path.name,
                        "image_bytes": _png_bytes(color),
                        "class_id": class_id,
                        "label": label,
                        "source": f"{label}/{split}",
                        "filename": image_path.name,
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return inference_images[0], inference_images[-1]


def test_train_predict_export_and_onnx_inference(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    negative_image, positive_image = _write_tiny_dataset(dataset_path)
    output_dir = tmp_path / "run"

    summary = train_pipeline(
        TrainConfig(
            data_root=dataset_path,
            output_dir=output_dir,
            epochs=1,
            batch_size=4,
            num_workers=0,
            base_channels=4,
            num_blocks=1,
            dropout=0.0,
            device="cpu",
        )
    )
    checkpoint = Path(summary["checkpoint"])
    assert checkpoint.name.startswith("ppc_best_epoch0001_possession_f1_")
    onnx_path = Path(summary["onnx_model"])
    assert onnx_path == output_dir / "ppc_best.onnx"
    assert summary["onnx_opset"] == 17
    assert summary["onnx_simplified"] is True
    assert onnx_path.is_file()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["model_family"] == "ppc"
    assert payload["model_config"]["num_classes"] == 2
    assert payload["label_map"] == {0: "no_possession", 1: "possession"}

    predictions = predict_images(checkpoint, [str(negative_image), str(positive_image)], device_spec="cpu")
    assert list(predictions.columns) == [
        "path",
        "pred_label",
        "pred_class",
        "logits",
        "prob_no_possession",
        "prob_possession",
    ]
    np.testing.assert_allclose(
        predictions[["prob_no_possession", "prob_possession"]].sum(axis=1).to_numpy(),
        np.ones(2),
        atol=1e-6,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: np.zeros((1, 3, 48, 48), dtype=np.float32)})[0]
    assert output.shape == (1, 2)
    np.testing.assert_allclose(output.sum(axis=1), np.ones(1), atol=1e-5)
