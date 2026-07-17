from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import pytest
import torch
from PIL import Image
from onnx import TensorProto, helper
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ppc.data import CLASS_ID_TO_NAME, NUM_CLASSES, _to_label
from ppc.model import ModelConfig, PPC
from ppc.pipeline import (
    PPC_CHECKPOINT_VERSION,
    PPC_MODEL_FAMILY,
    _run_epoch,
    _validate_checkpoint_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("arch_variant", "head_variant"),
    [
        ("baseline", "auto"),
        ("inverted_se", "auto"),
        ("convnext", "auto"),
        ("baseline", "avg"),
        ("baseline", "avgmax_mlp"),
        ("baseline", "transformer"),
        ("baseline", "mlp_mixer"),
    ],
)
def test_all_model_variants_return_two_probabilities(arch_variant: str, head_variant: str) -> None:
    config = ModelConfig(
        base_channels=8,
        num_blocks=2,
        dropout=0.0,
        arch_variant=arch_variant,
        head_variant=head_variant,
    )
    model = PPC(config).eval()
    inputs = torch.rand(2, 3, 48, 48)
    with torch.no_grad():
        logits = model(inputs)
        probabilities = model.predict_proba(inputs)
    assert logits.shape == (2, NUM_CLASSES)
    assert probabilities.shape == (2, NUM_CLASSES)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))


def test_ppc_rejects_non_binary_model_config() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        ModelConfig(num_classes=3)


def test_label_contract_rejects_legacy_and_mismatched_values() -> None:
    assert CLASS_ID_TO_NAME == {0: "no_possession", 1: "possession"}
    assert _to_label(pd.Series({"class_id": 1, "label": "possession"})) == 1
    with pytest.raises(ValueError, match="Unsupported class_id"):
        _to_label(pd.Series({"class_id": 2, "label": "point"}))
    with pytest.raises(ValueError, match="mismatch"):
        _to_label(pd.Series({"class_id": 1, "label": "no_possession"}))


def test_make_parquet_uses_folder_labels_and_reproducible_split(tmp_path: Path) -> None:
    script = _load_script("02_make_parquet.py")
    data_dir = tmp_path / "data"
    for label, color in (("no_possession", (0, 0, 0)), ("possession", (255, 255, 255))):
        label_dir = data_dir / label / "000000"
        label_dir.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (48, 48), color=color).save(label_dir / f"legacy_name_{index:03d}.png")

    args = argparse.Namespace(
        data_dir=data_dir,
        output=tmp_path / "dataset.parquet",
        train_ratio=0.9,
        val_ratio=0.1,
        seed=42,
        no_embed_images=False,
    )
    first = script.build_dataset(args)
    second = script.build_dataset(args)

    assert len(first) == 20
    assert first.groupby(["label", "split"]).size().to_dict() == {
        ("no_possession", "train"): 9,
        ("no_possession", "val"): 1,
        ("possession", "train"): 9,
        ("possession", "val"): 1,
    }
    assert all(isinstance(value, bytes) for value in first["image_bytes"])
    pd.testing.assert_frame_equal(first, second)

    legacy = first.copy()
    legacy.loc[0, ["class_id", "label"]] = [2, "point"]
    with pytest.raises(ValueError, match="unsupported class_id"):
        script.validate_ppc_dataframe(legacy)


class _MetricDataset(Dataset):
    def __init__(self) -> None:
        self.labels = torch.tensor([0] * 10 + [1, 1], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {"image": torch.tensor([index], dtype=torch.float32), "label": self.labels[index]}


class _FixedPredictionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        predictions = torch.tensor([0] * 8 + [1, 1, 0, 1], dtype=torch.long)
        logits = torch.full((len(predictions), 2), -4.0)
        logits[torch.arange(len(predictions)), predictions] = 4.0
        self.register_buffer("fixed_logits", logits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        indices = inputs[:, 0].long()
        return self.fixed_logits[indices]


def test_metrics_report_possession_and_macro_values() -> None:
    loader = DataLoader(_MetricDataset(), batch_size=12, shuffle=False)
    metrics, _ = _run_epoch(
        _FixedPredictionModel(),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        optimizer=None,
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["possession_precision"] == pytest.approx(1 / 3)
    assert metrics["possession_recall"] == pytest.approx(1 / 2)
    assert metrics["possession_f1"] == pytest.approx(0.4)
    assert metrics["macro_f1"] == pytest.approx((16 / 19 + 0.4) / 2)


def test_checkpoint_contract_rejects_legacy_puc(tmp_path: Path) -> None:
    valid = {
        "model_family": PPC_MODEL_FAMILY,
        "checkpoint_version": PPC_CHECKPOINT_VERSION,
        "model_config": {"num_classes": 2},
        "label_map": {0: "no_possession", 1: "possession"},
        "model_state": {},
        "normalization": {},
    }
    _validate_checkpoint_payload(valid, tmp_path / "valid.pt")

    legacy = dict(valid)
    legacy.update(
        model_family="puc",
        model_config={"num_classes": 3},
        label_map={0: "no_action", 1: "point_somewhere", 2: "point"},
    )
    with pytest.raises(ValueError, match="Legacy PUC checkpoints"):
        _validate_checkpoint_payload(legacy, tmp_path / "legacy.pt")


def test_ablation_target_parser_accepts_only_ppc_classes() -> None:
    script = _load_script("11_ablate_ppc_features.py")
    assert script._parse_target_score("no_possession").indices == (0,)
    assert script._parse_target_score("possession").indices == (1,)
    assert script._parse_target_score("0").label == "no_possession"
    assert script._parse_target_score("1").label == "possession"
    for invalid in ("point", "phone_usage", "2", "0,1"):
        with pytest.raises(argparse.ArgumentTypeError):
            script._parse_target_score(invalid)


def test_analysis_scripts_reject_three_class_onnx(tmp_path: Path) -> None:
    model_path = tmp_path / "legacy_puc.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["probabilities"],
                value=helper.make_tensor(
                    "legacy_probabilities",
                    TensorProto.FLOAT,
                    [1, 3],
                    [0.2, 0.3, 0.5],
                ),
            )
        ],
        "legacy_puc",
        [],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 3])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)

    for filename in ("10_visualize_ppc_heatmaps.py", "11_ablate_ppc_features.py"):
        script = _load_script(filename)
        with pytest.raises(ValueError, match="must contain 2 probabilities"):
            script._load_model(model_path)
