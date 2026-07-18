from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
DEMO_FILENAMES = ("demo_ppc.py", "demo_phone_gaze_classification.py")


def _load_script(filename: str):
    path = ROOT / filename
    module_name = f"test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _StaticClassifier:
    def __init__(self, probabilities=None, error: Exception | None = None) -> None:
        self.probabilities = probabilities
        self.error = error
        self.calls = 0

    def __call__(self, *, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return np.asarray(self.probabilities, dtype=np.float32)


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
@pytest.mark.parametrize(
    ("possession_score", "expected_ppc_class", "expected_forced"),
    [(0.499, 0, True), (0.5, 1, False), (0.501, 1, False)],
)
def test_ppc_gate_boundary(
    filename: str,
    possession_score: float,
    expected_ppc_class: int,
    expected_forced: bool,
) -> None:
    demo = _load_script(filename)
    puc = _StaticClassifier([0.1, 0.2, 0.7])
    ppc = _StaticClassifier([1.0 - possession_score, possession_score])

    result = demo.classify_hand_crop(
        np.zeros((12, 10, 3), dtype=np.uint8),
        puc_classifier=puc,
        ppc_classifier=ppc,
    )

    assert puc.calls == 1
    assert ppc.calls == 1
    assert result.phone_class == 2
    assert result.ppc_class == expected_ppc_class
    assert result.puc_forced_no_action is False
    assert result.phone_probs == pytest.approx([0.1, 0.2, 0.7])

    hand = demo.Box(23, 1.0, 0, 0, 10, 10, 5, 5)
    body = demo.Box(0, 1.0, 0, 0, 20, 20, 10, 10)
    demo.apply_hand_classification_result(hand, result)
    demo.assign_phone_usage_to_bodies([hand], [body])
    demo.assign_ppc_to_bodies([hand], [body])
    assert body.puc_forced_no_action is expected_forced
    if expected_forced:
        demo.finalize_phone_usage_current_frame(body)
        assert body.phone_class == 0
        assert body.phone_confidence == pytest.approx(0.1)
        assert body.phone_state == 0
    else:
        assert body.phone_class == 2
        assert body.phone_confidence == pytest.approx(0.7)


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
def test_ppc_failure_is_fail_safe_and_puc_failure_remains_unknown(filename: str) -> None:
    demo = _load_script(filename)
    image = np.zeros((12, 10, 3), dtype=np.uint8)

    ppc_failure = demo.classify_hand_crop(
        image,
        puc_classifier=_StaticClassifier([0.1, 0.2, 0.7]),
        ppc_classifier=_StaticClassifier(error=RuntimeError("PPC failed")),
    )
    assert ppc_failure.phone_class == 2
    assert ppc_failure.phone_confidence == pytest.approx(0.7)
    assert ppc_failure.ppc_inference_failed is True
    assert ppc_failure.puc_forced_no_action is False

    failed_hand = demo.Box(23, 1.0, 0, 0, 10, 10, 5, 5)
    failed_body = demo.Box(0, 1.0, 0, 0, 20, 20, 10, 10)
    demo.apply_hand_classification_result(failed_hand, ppc_failure)
    demo.assign_phone_usage_to_bodies([failed_hand], [failed_body])
    demo.assign_ppc_to_bodies([failed_hand], [failed_body])
    assert failed_body.ppc_inference_failed is True
    assert failed_body.puc_forced_no_action is True
    demo.finalize_phone_usage_current_frame(failed_body)
    assert failed_body.phone_class == 0
    assert failed_body.phone_confidence == pytest.approx(0.1)

    puc_failure = demo.classify_hand_crop(
        image,
        puc_classifier=_StaticClassifier(error=RuntimeError("PUC failed")),
        ppc_classifier=_StaticClassifier([0.499, 0.501]),
    )
    assert puc_failure.phone_class == -1
    assert puc_failure.phone_confidence == -1.0
    assert puc_failure.phone_state == -1
    assert puc_failure.ppc_inference_failed is False
    assert puc_failure.puc_forced_no_action is False

    invalid_ppc = demo.classify_hand_crop(
        image,
        puc_classifier=_StaticClassifier([0.1, 0.2, 0.7]),
        ppc_classifier=_StaticClassifier([0.2, 0.3, 0.5]),
    )
    assert invalid_ppc.phone_class == 2
    assert invalid_ppc.ppc_inference_failed is True
    assert invalid_ppc.puc_forced_no_action is False


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
def test_current_frame_suppression_and_overlay_contract(filename: str) -> None:
    demo = _load_script(filename)
    assert not hasattr(demo, "PhoneUsageStateHistory")
    body = demo.Box(0, 1.0, 0, 0, 20, 20, 10, 10)
    body.phone_class = 2
    body.phone_confidence = 0.7
    body.phone_probs = [0.1, 0.2, 0.7]
    body.phone_state = 1
    body.phone_label = "point"
    body.puc_forced_no_action = True
    demo.finalize_phone_usage_current_frame(body)
    assert body.phone_class == 0
    assert body.phone_confidence == pytest.approx(0.1)
    assert body.phone_state == 0
    assert body.phone_label == ""

    box = demo.Box(0, 1.0, 0, 0, 10, 10, 5, 5)
    box.ppc_class = 0
    box.ppc_confidence = 0.501
    box.ppc_label = "no_possession"
    box.ppc_possession_score = 0.499
    box.puc_forced_no_action = True
    assert demo.format_ppc_gate_overlay(box) == "PPC no_possession: 0.499"
    box.ppc_class = 1
    box.ppc_confidence = 0.5
    box.ppc_label = "possession"
    box.ppc_possession_score = 0.5
    box.puc_forced_no_action = False
    assert demo.format_ppc_gate_overlay(box) == "PPC possession: 0.500"
    box.ppc_inference_failed = True
    assert demo.format_ppc_gate_overlay(box) == "PPC error"


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
def test_puc_state_changes_immediately_without_temporal_history(filename: str) -> None:
    demo = _load_script(filename)
    body = demo.Box(0, 1.0, 0, 0, 20, 20, 10, 10)
    body.puc_forced_no_action = False
    body.phone_class = 2
    body.phone_state = 1
    body.gaze_inout_pass = True

    demo.finalize_phone_usage_current_frame(body)
    expected_label = "Check the phone" if "gaze" in filename else "point"
    assert body.phone_state == 1
    assert body.phone_label == expected_label

    body.phone_class = 0
    body.phone_state = 0
    demo.finalize_phone_usage_current_frame(body)
    assert body.phone_state == 0
    assert body.phone_label == ""


def _body_with_ppc_frame(
    demo,
    *,
    x_offset: int = 0,
    frame_class: int,
    possession_score: float,
    inference_failed: bool = False,
):
    body = demo.Box(0, 1.0, x_offset, 0, x_offset + 100, 100, x_offset + 50, 50)
    body.ppc_frame_class = frame_class
    body.ppc_class = frame_class
    body.ppc_possession_score = possession_score
    body.ppc_inference_failed = inference_failed
    if frame_class in (0, 1):
        body.ppc_probs = [1.0 - possession_score, possession_score]
        body.ppc_confidence = body.ppc_probs[frame_class]
        body.ppc_label = demo.PPC_LABELS[frame_class]
    return body


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
def test_ppc_history_strict_warmup_and_single_frame_smoothing(filename: str) -> None:
    demo = _load_script(filename)
    manager = demo.PPCBodyHistoryManager(long_size=10, short_size=6)

    for frame_index in range(10):
        body = _body_with_ppc_frame(
            demo,
            frame_class=1,
            possession_score=0.8,
        )
        manager.update([body])
        assert body.ppc_frame_class == 1
        assert body.ppc_possession_score == pytest.approx(0.8)
        if frame_index < 9:
            assert body.ppc_class == 0
            assert body.ppc_label == "no_possession"
            assert body.puc_forced_no_action is True
        else:
            assert body.ppc_class == 1
            assert body.ppc_label == "possession"
            assert body.puc_forced_no_action is False

    one_negative = _body_with_ppc_frame(
        demo,
        frame_class=0,
        possession_score=0.4,
    )
    manager.update([one_negative])
    assert one_negative.ppc_frame_class == 0
    assert one_negative.ppc_class == 1
    assert one_negative.puc_forced_no_action is False
    assert demo.format_ppc_gate_overlay(one_negative) == "PPC possession: 0.400"

    second_negative = _body_with_ppc_frame(
        demo,
        frame_class=0,
        possession_score=0.3,
    )
    manager.update([second_negative])
    assert second_negative.ppc_class == 0
    assert second_negative.puc_forced_no_action is True


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
@pytest.mark.parametrize("inference_failed", [False, True])
def test_ppc_missing_or_error_is_immediately_suppressed_and_history_recovers(
    filename: str,
    inference_failed: bool,
) -> None:
    demo = _load_script(filename)
    manager = demo.PPCBodyHistoryManager(long_size=10, short_size=6)
    for _ in range(10):
        body = _body_with_ppc_frame(demo, frame_class=1, possession_score=0.8)
        manager.update([body])
    assert body.puc_forced_no_action is False

    invalid = _body_with_ppc_frame(
        demo,
        frame_class=-1,
        possession_score=-1.0,
        inference_failed=inference_failed,
    )
    manager.update([invalid])
    assert invalid.ppc_class == -1
    assert invalid.puc_forced_no_action is True
    assert demo.format_ppc_gate_overlay(invalid) == ("PPC error" if inference_failed else "")

    recovered = _body_with_ppc_frame(demo, frame_class=1, possession_score=0.75)
    manager.update([recovered])
    assert recovered.ppc_class == 1
    assert recovered.puc_forced_no_action is False


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
def test_ppc_histories_are_isolated_and_removed_with_tracks(filename: str) -> None:
    demo = _load_script(filename)
    manager = demo.PPCBodyHistoryManager(long_size=10, short_size=6)

    for _ in range(10):
        possession = _body_with_ppc_frame(
            demo,
            x_offset=0,
            frame_class=1,
            possession_score=0.8,
        )
        no_possession = _body_with_ppc_frame(
            demo,
            x_offset=200,
            frame_class=0,
            possession_score=0.2,
        )
        manager.update([possession, no_possession])

    assert possession.ppc_state_track_id != no_possession.ppc_state_track_id
    assert possession.puc_forced_no_action is False
    assert no_possession.puc_forced_no_action is True
    assert len(manager.histories) == 2

    for _ in range(31):
        manager.update([])
    assert manager.histories == {}
    assert manager.tracker.tracks == []


def test_gaze_candidates_use_history_confirmed_ppc_gate() -> None:
    demo = _load_script("demo_phone_gaze_classification.py")
    manager = demo.PPCBodyHistoryManager(long_size=10, short_size=6)
    action_hand = demo.Box(23, 1.0, 10, 10, 30, 30, 20, 20)
    action_hand.phone_class = 2

    for frame_index in range(10):
        body = _body_with_ppc_frame(demo, frame_class=1, possession_score=0.8)
        manager.update([body])
        candidates = demo.positive_puc_hands_by_body({0: [action_hand]}, [body])
        assert bool(candidates) is (frame_index == 9)


@pytest.mark.parametrize(
    "filename",
    ("demo_phone_gaze_classification.py", "demo_phone_gaze_classification_puc.py"),
)
def test_gaze_overlap_uses_1_5x_original_hand_box(filename: str) -> None:
    demo = _load_script(filename)
    hand = demo.Box(23, 1.0, 40, 40, 60, 60, 50, 50)
    assert demo.GAZE_HAND_BOX_EXPANSION == pytest.approx(1.5)
    assert demo.calculate_expanded_box_bounds(
        hand,
        frame_height=100,
        frame_width=100,
        expansion=demo.GAZE_HAND_BOX_EXPANSION,
    ) == (35, 35, 65, 65)

    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[50, 75] = 1.0
    assert demo.gaze_heatmap_overlaps_box(
        heatmap,
        hand,
        frame_height=100,
        frame_width=100,
        overlap_ratio=0.8,
        expansion=demo.GAZE_HAND_BOX_EXPANSION,
    ) is False
    assert demo.gaze_heatmap_overlaps_box(
        heatmap,
        hand,
        frame_height=100,
        frame_width=100,
        overlap_ratio=0.8,
        expansion=3.75,
    ) is True


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
def test_ppc_history_size_validation(filename: str) -> None:
    demo = _load_script(filename)
    with pytest.raises(ValueError, match="must not exceed"):
        demo.PPCBodyHistoryManager(long_size=4, short_size=6)


def test_gaze_candidates_exclude_suppressed_body_but_keep_independent_action() -> None:
    demo = _load_script("demo_phone_gaze_classification.py")
    suppressed_body = demo.Box(0, 1.0, 0, 0, 40, 40, 20, 20)
    suppressed_body.puc_forced_no_action = True
    active_body = demo.Box(0, 1.0, 50, 0, 90, 40, 70, 20)
    active_body.puc_forced_no_action = False
    suppressed_action = demo.Box(23, 1.0, 0, 0, 10, 10, 5, 5)
    suppressed_action.phone_class = 2
    active_action = demo.Box(23, 1.0, 60, 0, 70, 10, 65, 5)
    active_action.phone_class = 1

    candidates = demo.positive_puc_hands_by_body(
        {0: [suppressed_action], 1: [active_action]},
        [suppressed_body, active_body],
    )
    assert candidates == {1: [active_action]}


def test_gaze_puc_gate_is_enabled_only_when_requested() -> None:
    demo = _load_script("demo_phone_gaze_classification.py")
    body = demo.Box(0, 1.0, 0, 0, 40, 40, 20, 20)
    body.puc_forced_no_action = False
    no_action_hand = demo.Box(23, 1.0, 5, 5, 15, 15, 10, 10)
    no_action_hand.phone_class = 0

    assert demo.positive_puc_hands_by_body(
        {0: [no_action_hand]},
        [body],
        enable_puc_gate=True,
    ) == {}
    assert demo.positive_puc_hands_by_body(
        {0: [no_action_hand]},
        [body],
        enable_puc_gate=False,
    ) == {0: [no_action_hand]}

    body.phone_class = 0
    body.phone_confidence = 0.8
    body.phone_state = 1
    body.gaze_inout_pass = True
    demo.finalize_phone_usage_current_frame(body, enable_puc_gate=True)
    assert body.phone_label == ""

    body.phone_state = 1
    demo.finalize_phone_usage_current_frame(body, enable_puc_gate=False)
    assert body.phone_state == 1
    assert body.phone_label == demo.PHONE_CHECK_LABEL

    body.puc_forced_no_action = True
    body.phone_state = 1
    demo.finalize_phone_usage_current_frame(body, enable_puc_gate=False)
    assert body.phone_state == 0
    assert body.phone_label == ""


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
def test_both_hands_are_aggregated_into_one_body_ppc_label(filename: str) -> None:
    demo = _load_script(filename)
    body = demo.Box(0, 1.0, 0, 0, 100, 100, 50, 50)
    no_possession = demo.Box(23, 1.0, 10, 10, 30, 30, 20, 20)
    no_possession.phone_class = 2
    no_possession.phone_confidence = 0.7
    no_possession.phone_probs = [0.1, 0.2, 0.7]
    no_possession.ppc_class = 0
    no_possession.ppc_confidence = 0.6
    no_possession.ppc_label = "no_possession"
    no_possession.ppc_probs = [0.6, 0.4]
    no_possession.ppc_possession_score = 0.4

    demo.assign_phone_usage_to_bodies([no_possession], [body])
    demo.assign_ppc_to_bodies([no_possession], [body])
    assert body.phone_class == 2
    assert body.ppc_class == 0
    assert body.ppc_possession_score == pytest.approx(0.4)
    assert body.puc_forced_no_action is True
    demo.finalize_phone_usage_current_frame(body)
    assert body.phone_class == 0

    possession = demo.Box(23, 1.0, 60, 10, 80, 30, 70, 20)
    possession.phone_class = 1
    possession.phone_confidence = 0.6
    possession.phone_probs = [0.2, 0.6, 0.2]
    possession.ppc_class = 1
    possession.ppc_confidence = 0.8
    possession.ppc_label = "possession"
    possession.ppc_probs = [0.2, 0.8]
    possession.ppc_possession_score = 0.8

    demo.assign_phone_usage_to_bodies([no_possession, possession], [body])
    demo.assign_ppc_to_bodies([no_possession, possession], [body])
    assert body.phone_class == 2
    assert body.ppc_class == 1
    assert body.ppc_label == "possession"
    assert body.ppc_confidence == pytest.approx(0.8)
    assert body.ppc_possession_score == pytest.approx(0.8)
    assert body.puc_forced_no_action is False


def _write_constant_classifier(
    path: Path,
    *,
    channels: int,
    probabilities: list[float],
) -> None:
    output_name = "probabilities"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[output_name],
                value=helper.make_tensor(
                    "constant_probabilities",
                    TensorProto.FLOAT,
                    [1, len(probabilities)],
                    probabilities,
                ),
            )
        ],
        "constant_classifier",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, channels, 8, 8])],
        [
            helper.make_tensor_value_info(
                output_name,
                TensorProto.FLOAT,
                [1, len(probabilities)],
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)


@pytest.mark.parametrize("filename", DEMO_FILENAMES)
def test_synthetic_onnx_contract_and_shared_preprocessing(filename: str, tmp_path: Path) -> None:
    demo = _load_script(filename)
    puc_path = tmp_path / "puc.onnx"
    ppc_path = tmp_path / "ppc.onnx"
    legacy_ppc_path = tmp_path / "legacy_ppc.onnx"
    invalid_input_path = tmp_path / "invalid_input.onnx"
    _write_constant_classifier(puc_path, channels=3, probabilities=[0.1, 0.2, 0.7])
    _write_constant_classifier(ppc_path, channels=3, probabilities=[0.75, 0.25])
    _write_constant_classifier(legacy_ppc_path, channels=3, probabilities=[0.2, 0.3, 0.5])
    _write_constant_classifier(invalid_input_path, channels=1, probabilities=[0.75, 0.25])

    puc = demo.PUCClassifier(model_path=str(puc_path), providers=["CPUExecutionProvider"])
    ppc = demo.PPCClassifier(model_path=str(ppc_path), providers=["CPUExecutionProvider"])
    rgb_crop = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)

    np.testing.assert_array_equal(puc._preprocess(rgb_crop), ppc._preprocess(rgb_crop))
    assert puc(rgb_crop).shape == (3,)
    assert ppc(rgb_crop).shape == (2,)

    with pytest.raises(ValueError, match=r"\[batch, 2\]"):
        demo.PPCClassifier(model_path=str(legacy_ppc_path), providers=["CPUExecutionProvider"])
    with pytest.raises(ValueError, match="three channels"):
        demo.PPCClassifier(model_path=str(invalid_input_path), providers=["CPUExecutionProvider"])
