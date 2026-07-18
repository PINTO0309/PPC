#!/usr/bin/env python

from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import copy
import cv2
import math
import time
from pprint import pprint
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser, ArgumentTypeError
from typing import Tuple, Optional, List, Dict, Any, Deque
import importlib.util
from collections import Counter, deque
from abc import ABC, abstractmethod

from bbalg.main import state_verdict

AVERAGE_HEAD_WIDTH: float = 0.16 + 0.10 # 16cm + Margin Compensation

BOX_COLORS = [
    [(216, 67, 21),"Front"],
    [(255, 87, 34),"Right-Front"],
    [(123, 31, 162),"Right-Side"],
    [(255, 193, 7),"Right-Back"],
    [(76, 175, 80),"Back"],
    [(33, 150, 243),"Left-Back"],
    [(156, 39, 176),"Left-Side"],
    [(0, 188, 212),"Left-Front"],
]

PHONE_USAGE_LABELS = {
    0: "no_action",
    1: "point_somewhere",
    2: "point",
}
PHONE_USAGE_COLORS = {
    1: (0, 220, 60),        # vivid green: point_somewhere
    2: (255, 144, 30),      # bold orange: point
}
PPC_LABELS = {
    0: "no_possession",
    1: "possession",
}
PPC_COLORS = {
    0: (0, 0, 255),         # red: no_possession
    1: (0, 220, 60),        # vivid green: possession
}
PHONE_USAGE_TARGET_CLASS_ID = 23
PHONE_USAGE_CROP_EXPANSION = 2.5
PPC_POSSESSION_THRESHOLD = 0.5
DEFAULT_PUC_MODEL = 'puc_s_48x48.onnx'
DEFAULT_PUC_INPUT_SIZE = (48, 48)
DEFAULT_PPC_MODEL = 'ppc_l_48x48.onnx'
DEFAULT_PPC_INPUT_SIZE = (48, 48)

HAND_CLASS_ID = 23
HAND_LEFT_CLASS_ID = 24
HAND_RIGHT_CLASS_ID = 25
OBJECT_CLASS_IDS = {0, 5, 6, 7, 16, 17, 18, 19, 20, HAND_CLASS_ID, HAND_LEFT_CLASS_ID, HAND_RIGHT_CLASS_ID, 27}
ATTRIBUTE_CLASS_IDS = {1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15}
KEYPOINT_CLASS_IDS = {21, 22, 26}
YOLO_NMS_IOU_THRESHOLD = 0.45

# YOLOMIT wholebody28 does not provide the previous full keypoint chain.
EDGES: List[Tuple[int, int]] = []

PPC_LONG_HISTORY_SIZE = 10
PPC_SHORT_HISTORY_SIZE = 6

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    generation: int = -1 # -1: Unknown, 0: Adult, 1: Child
    gender: int = -1 # -1: Unknown, 0: Male, 1: Female
    handedness: int = -1 # -1: Unknown, 0: Left, 1: Right
    head_pose: int = -1 # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
    is_used: bool = False
    person_id: int = -1
    track_id: int = -1
    phone_confidence: float = -1.0
    phone_state: int = -1  # -1: Unknown, 0: no_action, 1: action
    phone_label: str = ''
    phone_class: int = -1
    phone_probs: Optional[List[float]] = None
    ppc_class: int = -1
    ppc_frame_class: int = -1
    ppc_confidence: float = -1.0
    ppc_label: str = ''
    ppc_probs: Optional[List[float]] = None
    ppc_possession_score: float = -1.0
    ppc_inference_failed: bool = False
    ppc_history_active: bool = False
    ppc_state_track_id: int = -1
    puc_forced_no_action: bool = False


class PPCStateHistory:
    def __init__(self, long_size: int, short_size: int) -> None:
        self.long_history: Deque[bool] = deque(maxlen=long_size)
        self.short_history: Deque[bool] = deque(maxlen=short_size)
        self.interval_active = False

    def append(self, is_possession: bool) -> bool:
        self.long_history.append(is_possession)
        self.short_history.append(is_possession)
        self.interval_active, _, _ = state_verdict(
            long_tracking_history=self.long_history,
            short_tracking_history=self.short_history,
        )
        return self.interval_active


def apply_ppc_suppression_to_body(body_box: Box) -> bool:
    """Force the current body result to no_action when PPC suppresses PUC."""
    if not body_box.puc_forced_no_action:
        return False
    body_box.phone_class = 0
    body_box.phone_confidence = (
        float(body_box.phone_probs[0])
        if body_box.phone_probs is not None and len(body_box.phone_probs) >= 1
        else 0.0
    )
    body_box.phone_state = 0
    body_box.phone_label = ''
    return True


def finalize_phone_usage_current_frame(body_box: Box) -> None:
    """Apply the current-frame PUC result without temporal smoothing."""
    if apply_ppc_suppression_to_body(body_box):
        return
    predicted_class = body_box.phone_class if body_box.phone_class is not None else -1
    if predicted_class > 0:
        body_box.phone_state = 1
        body_box.phone_label = PHONE_USAGE_LABELS.get(
            predicted_class,
            f'class_{predicted_class}',
        )
    else:
        body_box.phone_state = 0
        body_box.phone_label = ''


class SimpleSortTracker:
    """Minimal SORT-style tracker based on IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: List[Dict[str, Any]] = []
        self.frame_index = 0

    @staticmethod
    def _iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        if inter_w == 0 or inter_h == 0:
            return 0.0

        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    def update(self, boxes: List[Box]) -> None:
        self.frame_index += 1

        for box in boxes:
            box.track_id = -1

        if not boxes and not self.tracks:
            return

        iou_matrix = None
        if self.tracks and boxes:
            iou_matrix = np.zeros((len(self.tracks), len(boxes)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                track_bbox = track['bbox']
                for d_idx, box in enumerate(boxes):
                    det_bbox = (box.x1, box.y1, box.x2, box.y2)
                    iou_matrix[t_idx, d_idx] = self._iou(track_bbox, det_bbox)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: List[Tuple[int, int]] = []

        if iou_matrix is not None and iou_matrix.size > 0:
            while True:
                best_track = -1
                best_det = -1
                best_iou = self.iou_threshold
                for t_idx in range(len(self.tracks)):
                    if t_idx in matched_tracks:
                        continue
                    for d_idx in range(len(boxes)):
                        if d_idx in matched_detections:
                            continue
                        iou = float(iou_matrix[t_idx, d_idx])
                        if iou > best_iou:
                            best_iou = iou
                            best_track = t_idx
                            best_det = d_idx
                if best_track == -1:
                    break
                matched_tracks.add(best_track)
                matched_detections.add(best_det)
                matches.append((best_track, best_det))

        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            det_box = boxes[d_idx]
            track['bbox'] = (det_box.x1, det_box.y1, det_box.x2, det_box.y2)
            track['missed'] = 0
            track['last_seen'] = self.frame_index
            det_box.track_id = track['id']

        surviving_tracks: List[Dict[str, Any]] = []
        for idx, track in enumerate(self.tracks):
            if idx in matched_tracks:
                surviving_tracks.append(track)
                continue
            track['missed'] += 1
            if track['missed'] <= self.max_age:
                surviving_tracks.append(track)
        self.tracks = surviving_tracks

        for d_idx, det_box in enumerate(boxes):
            if d_idx in matched_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            det_box.track_id = track_id
            self.tracks.append(
                {
                    'id': track_id,
                    'bbox': (det_box.x1, det_box.y1, det_box.x2, det_box.y2),
                    'missed': 0,
                    'last_seen': self.frame_index,
                }
            )

        if not boxes:
            return


def apply_ppc_history_to_body(body_box: Box, history: PPCStateHistory) -> None:
    """Apply the Body-level PPC history while preserving the current-frame score."""
    frame_class = body_box.ppc_frame_class
    has_valid_result = frame_class in PPC_LABELS
    history_active = history.append(has_valid_result and frame_class == 1)
    body_box.ppc_history_active = history_active

    if body_box.ppc_inference_failed or not has_valid_result:
        body_box.ppc_class = -1
        body_box.ppc_confidence = -1.0
        body_box.ppc_label = ''
        body_box.puc_forced_no_action = True
        return

    effective_class = 1 if history_active else 0
    body_box.ppc_class = effective_class
    body_box.ppc_label = PPC_LABELS[effective_class]
    if body_box.ppc_probs is not None and len(body_box.ppc_probs) == 2:
        body_box.ppc_confidence = float(body_box.ppc_probs[effective_class])
    elif body_box.ppc_possession_score >= 0.0:
        body_box.ppc_confidence = (
            body_box.ppc_possession_score
            if effective_class == 1
            else 1.0 - body_box.ppc_possession_score
        )
    body_box.puc_forced_no_action = not history_active


class PPCBodyHistoryManager:
    """Track bodies independently of UI tracking and maintain their PPC histories."""

    def __init__(self, long_size: int, short_size: int) -> None:
        if long_size < 2 or short_size < 2:
            raise ValueError('PPC history sizes must be at least 2.')
        if short_size > long_size:
            raise ValueError('PPC short history size must not exceed long history size.')
        self.long_size = long_size
        self.short_size = short_size
        self.tracker = SimpleSortTracker()
        self.histories: Dict[int, PPCStateHistory] = {}

    def _get_history(self, track_id: int) -> PPCStateHistory:
        history = self.histories.get(track_id)
        if history is None:
            history = PPCStateHistory(self.long_size, self.short_size)
            self.histories[track_id] = history
        return history

    def update(self, body_boxes: List[Box]) -> None:
        self.tracker.update(body_boxes)
        matched_track_ids: set[int] = set()

        for body_box in body_boxes:
            track_id = body_box.track_id
            body_box.ppc_state_track_id = track_id
            body_box.track_id = -1
            if track_id <= 0:
                body_box.puc_forced_no_action = True
                continue
            matched_track_ids.add(track_id)
            apply_ppc_history_to_body(body_box, self._get_history(track_id))

        current_track_ids = {track['id'] for track in self.tracker.tracks}
        for track_id in current_track_ids - matched_track_ids:
            self._get_history(track_id).append(False)

        for track_id in list(self.histories):
            if track_id not in current_track_ids:
                self.histories.pop(track_id, None)

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.25,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                    # onnxruntime>=1.21.0 breaking changes
                    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
                    # https://github.com/microsoft/onnxruntime/pull/22681/files
                    # https://github.com/microsoft/onnxruntime/pull/23893/files
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._keypoint_th = keypoint_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            onnxruntime.set_default_logger_severity(3) # ERROR
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            print(f'{Color.GREEN("Enabled ONNX ExecutionProviders:")}')
            pprint(f'{self._providers}')

            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3

        elif self._runtime in ['ai_edge_litert', 'tensorflow']:
            if self._runtime == 'ai_edge_litert':
                from ai_edge_litert.interpreter import Interpreter
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['ai_edge_litert', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()

class WholeBodyDetector(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolomit_t_wholebody28_1x3x480x640.onnx',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for WholeBodyDetector. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for the YOLOMIT whole-body object detector.

        obj_class_score_th: Optional[float]
            Object score threshold. Default: 0.35

        attr_class_score_th: Optional[float]
            Attributes score threshold. Default: 0.70

        keypoint_th: Optional[float]
            Keypoints score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            keypoint_th=keypoint_th,
            providers=providers,
        )
        self.mean: np.ndarray = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape([3,1,1]) # Not used in WholeBodyDetector
        self.std: np.ndarray = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape([3,1,1]) # Not used in WholeBodyDetector
        self._input_height, self._input_width = self._resolve_input_size()

    def _resolve_input_size(self) -> Tuple[int, int]:
        default_height, default_width = 480, 640
        input_shape: Optional[List[int]] = None
        if self._runtime == 'onnx':
            input_shape = list(self._interpreter.get_inputs()[0].shape)
        elif self._input_details:
            input_shape = self._input_details[0].get('shape')
            if input_shape is not None:
                input_shape = list(input_shape)

        def _safe_dim(value: Any, default: int) -> int:
            try:
                if value is None:
                    return default
                return int(value)
            except (TypeError, ValueError):
                return default

        if not input_shape or len(input_shape) <= max(self._h_index, self._w_index):
            return default_height, default_width

        height = _safe_dim(input_shape[self._h_index], default_height)
        width = _safe_dim(input_shape[self._w_index], default_width)
        return height, width

    def __call__(
        self,
        image: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, attributes, is_used=False]
        """
        temp_image = copy.deepcopy(image)
        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )
        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]
        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            RGB CHW float32 image normalized to 0.0-1.0.
        """
        image = cv2.resize(image, (self._input_width, self._input_height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(self._swap)
        image = \
            np.ascontiguousarray(
                image,
                dtype=np.float32,
            )
        return image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            YOLOMIT raw output. Expected shape is [32, 6300] or [1, 32, 6300],
            containing [cx, cy, w, h, class0..class27].

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, attributes, is_used=False]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        box_score_threshold: float = min([self._obj_class_score_th, self._attr_class_score_th, self._keypoint_th])

        if len(boxes) > 0:
            result_boxes = self._decode_yolomit28_output(
                raw_output=boxes,
                image_width=image_width,
                image_height=image_height,
                score_threshold=box_score_threshold,
            )

            if len(result_boxes) > 0:
                result_boxes = self._classwise_nms(
                    boxes=result_boxes,
                    iou_threshold=YOLO_NMS_IOU_THRESHOLD,
                )
                # Object filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in OBJECT_CLASS_IDS and box.score >= self._obj_class_score_th) or box.classid not in OBJECT_CLASS_IDS
                ]
                # Attribute filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in ATTRIBUTE_CLASS_IDS and box.score >= self._attr_class_score_th) or box.classid not in ATTRIBUTE_CLASS_IDS
                ]
                # Keypoint filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in KEYPOINT_CLASS_IDS and box.score >= self._keypoint_th) or box.classid not in KEYPOINT_CLASS_IDS
                ]

                # Adult, Child merge
                # classid: 0 -> Body
                #   classid: 1 -> Adult
                #   classid: 2 -> Child
                # 1. Calculate Adult and Child IoUs for Body detection results
                # 2. Connect either the Adult or the Child with the highest score and the highest IoU with the Body.
                # 3. Exclude Adult and Child from detection results
                if not disable_generation_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    generation_boxes = [box for box in result_boxes if box.classid in [1, 2]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=generation_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [1, 2]]
                # Male, Female merge
                # classid: 0 -> Body
                #   classid: 3 -> Male
                #   classid: 4 -> Female
                # 1. Calculate Male and Female IoUs for Body detection results
                # 2. Connect either the Male or the Female with the highest score and the highest IoU with the Body.
                # 3. Exclude Male and Female from detection results
                if not disable_gender_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    gender_boxes = [box for box in result_boxes if box.classid in [3, 4]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=gender_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [3, 4]]
                # HeadPose merge
                # classid: 7 -> Head
                #   classid:  8 -> Front
                #   classid:  9 -> Right-Front
                #   classid: 10 -> Right-Side
                #   classid: 11 -> Right-Back
                #   classid: 12 -> Back
                #   classid: 13 -> Left-Back
                #   classid: 14 -> Left-Side
                #   classid: 15 -> Left-Front
                # 1. Calculate HeadPose IoUs for Head detection results
                # 2. Connect either the HeadPose with the highest score and the highest IoU with the Head.
                # 3. Exclude HeadPose from detection results
                if not disable_headpose_identification_mode:
                    head_boxes = [box for box in result_boxes if box.classid == 7]
                    headpose_boxes = [box for box in result_boxes if box.classid in [8,9,10,11,12,13,14,15]]
                    self._find_most_relevant_obj(base_objs=head_boxes, target_objs=headpose_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [8,9,10,11,12,13,14,15]]
                # Left and right hand merge
                # classid: 23 -> Hand
                #   classid: 24 -> Left-Hand
                #   classid: 25 -> Right-Hand
                # 1. Calculate Left-Hand and Right-Hand IoUs for Hand detection results
                # 2. Connect either the Left-Hand or the Right-Hand with the highest score and the highest IoU with the Hand.
                # 3. Exclude Left-Hand and Right-Hand from detection results
                if not disable_left_and_right_hand_identification_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == HAND_CLASS_ID]
                    left_right_hand_boxes = [box for box in result_boxes if box.classid in [HAND_LEFT_CLASS_ID, HAND_RIGHT_CLASS_ID]]
                    self._find_most_relevant_obj(base_objs=hand_boxes, target_objs=left_right_hand_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [HAND_LEFT_CLASS_ID, HAND_RIGHT_CLASS_ID]]

                # Keypoints NMS
                # Suppression of overdetection
                # classid: 21 -> shoulder
                # classid: 22 -> elbow
                # classid: 26 -> knee
                for target_classid in KEYPOINT_CLASS_IDS:
                    keypoints_boxes = [box for box in result_boxes if box.classid == target_classid]
                    filtered_keypoints_boxes = self._nms(target_objs=keypoints_boxes, iou_threshold=0.20)
                    result_boxes = [box for box in result_boxes if box.classid != target_classid]
                    result_boxes = result_boxes + filtered_keypoints_boxes
        return result_boxes

    def _decode_yolomit28_output(
        self,
        *,
        raw_output: np.ndarray,
        image_width: int,
        image_height: int,
        score_threshold: float,
    ) -> List[Box]:
        predictions = np.asarray(raw_output)
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise ValueError(f"Unsupported YOLOMIT output shape: {predictions.shape}")
        if predictions.shape[0] == 32:
            predictions = predictions.T
        elif predictions.shape[1] != 32:
            raise ValueError(f"Unsupported YOLOMIT output shape: {predictions.shape}")

        boxes: List[Box] = []
        scale_x = image_width / float(self._input_width)
        scale_y = image_height / float(self._input_height)

        xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        best_class_ids = np.argmax(class_scores, axis=1)
        best_scores = np.max(class_scores, axis=1)
        keep_indices = np.where(best_scores > score_threshold)[0]

        for idx in keep_indices:
            classid = int(best_class_ids[idx])
            if classid < 0 or classid > 27:
                continue

            cx_in, cy_in, w_in, h_in = xywh[idx]
            x_min = int(round((float(cx_in) - float(w_in) / 2.0) * scale_x))
            y_min = int(round((float(cy_in) - float(h_in) / 2.0) * scale_y))
            x_max = int(round((float(cx_in) + float(w_in) / 2.0) * scale_x))
            y_max = int(round((float(cy_in) + float(h_in) / 2.0) * scale_y))

            x_min = max(0, min(x_min, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            x_max = max(0, min(x_max, image_width))
            y_max = max(0, min(y_max, image_height))
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append(
                Box(
                    classid=classid,
                    score=float(best_scores[idx]),
                    x1=x_min,
                    y1=y_min,
                    x2=x_max,
                    y2=y_max,
                    cx=(x_min + x_max) // 2,
                    cy=(y_min + y_max) // 2,
                    generation=-1,
                    gender=-1,
                    handedness=-1,
                    head_pose=-1,
                )
            )
        return boxes

    def _classwise_nms(
        self,
        *,
        boxes: List[Box],
        iou_threshold: float,
    ) -> List[Box]:
        selected: List[Box] = []
        class_ids = sorted({box.classid for box in boxes})
        for classid in class_ids:
            candidates = sorted(
                [box for box in boxes if box.classid == classid],
                key=lambda box: box.score,
                reverse=True,
            )
            while candidates:
                current = candidates.pop(0)
                selected.append(current)
                candidates = [
                    box for box in candidates
                    if self._calculate_iou(base_obj=current, target_obj=box) < iou_threshold
                ]
        return selected

    def _find_most_relevant_obj(
        self,
        *,
        base_objs: List[Box],
        target_objs: List[Box],
    ):
        for base_obj in base_objs:
            most_relevant_obj: Box = None
            best_score = 0.0
            best_iou = 0.0
            best_distance = float('inf')

            for target_obj in target_objs:
                distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                # Process only unused objects with center Euclidean distance less than or equal to 10.0
                if not target_obj.is_used and distance <= 10.0:
                    # Prioritize high-score objects
                    if target_obj.score >= best_score:
                        # IoU Calculation
                        iou: float = \
                            self._calculate_iou(
                                base_obj=base_obj,
                                target_obj=target_obj,
                            )
                        # Adopt object with highest IoU
                        if iou > best_iou:
                            most_relevant_obj = target_obj
                            best_iou = iou
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            best_distance = distance
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score
            if most_relevant_obj:
                if most_relevant_obj.classid == 1:
                    base_obj.generation = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 2:
                    base_obj.generation = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 3:
                    base_obj.gender = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 4:
                    base_obj.gender = 1
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 8:
                    base_obj.head_pose = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 9:
                    base_obj.head_pose = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 10:
                    base_obj.head_pose = 2
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 11:
                    base_obj.head_pose = 3
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 12:
                    base_obj.head_pose = 4
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 13:
                    base_obj.head_pose = 5
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 14:
                    base_obj.head_pose = 6
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 15:
                    base_obj.head_pose = 7
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == HAND_LEFT_CLASS_ID:
                    base_obj.handedness = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == HAND_RIGHT_CLASS_ID:
                    base_obj.handedness = 1
                    most_relevant_obj.is_used = True

    def _nms(
        self,
        *,
        target_objs: List[Box],
        iou_threshold: float,
    ):
        filtered_objs: List[Box] = []

        # 1. Sorted in order of highest score
        #    key=lambda box: box.score to get the score, and reverse=True to sort in descending order
        sorted_objs = sorted(target_objs, key=lambda box: box.score, reverse=True)

        # 2. Scan the box list after sorting
        while sorted_objs:
            # Extract the first (highest score)
            current_box = sorted_objs.pop(0)

            # If you have already used it, skip it
            if current_box.is_used:
                continue

            # Add to filtered_objs and set the use flag
            filtered_objs.append(current_box)
            current_box.is_used = True

            # 3. Mark the boxes where the current_box and IOU are above the threshold as used or exclude them
            remaining_boxes = []
            for box in sorted_objs:
                if not box.is_used:
                    # Calculating IoU
                    iou_value = self._calculate_iou(base_obj=current_box, target_obj=box)

                    # If the IOU threshold is exceeded, it is considered to be the same object and is removed as a duplicate
                    if iou_value >= iou_threshold:
                        # Leave as used (exclude later)
                        box.is_used = True
                    else:
                        # If the IOU threshold is not met, the candidate is still retained
                        remaining_boxes.append(box)

            # Only the remaining_boxes will be handled in the next loop
            sorted_objs = remaining_boxes

        # 4. Return the box that is left over in the end
        return filtered_objs

    def _calculate_iou(
        self,
        *,
        base_obj: Box,
        target_obj: Box,
    ) -> float:
        # Calculate areas of overlap
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)
        # If there is no overlap
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        # Calculate area of overlap and area of each bounding box
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
        # Calculate IoU
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou

def preprocess_hand_classifier_image(
    image: np.ndarray,
    *,
    height: int,
    width: int,
    classifier_name: str,
) -> np.ndarray:
    """Apply the shared RGB hand-crop preprocessing used by PUC and PPC."""
    if image is None or image.size == 0:
        raise ValueError(f'Input image for {classifier_name} is empty.')
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f'Input image for {classifier_name} must be RGB HWC with three channels; '
            f'got shape {image.shape}.'
        )
    if height <= 0 or width <= 0:
        raise ValueError(f'Invalid target size for {classifier_name} preprocessing.')
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32) / 255.0
    resized = resized.transpose(2, 0, 1)
    return np.ascontiguousarray(resized, dtype=np.float32)


def validate_probability_vector(
    probabilities: Any,
    *,
    expected_classes: int,
    classifier_name: str,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 1 or probs.shape[0] != expected_classes:
        raise ValueError(
            f'{classifier_name} must return exactly {expected_classes} probabilities; '
            f'got shape {probs.shape}.'
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError(f'{classifier_name} returned non-finite probabilities.')
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError(f'{classifier_name} probabilities must be in the range [0, 1].')
    if not np.isclose(float(np.sum(probs)), 1.0, atol=1e-3):
        raise ValueError(f'{classifier_name} probabilities must sum to 1.0.')
    return probs


class PUCClassifier(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = DEFAULT_PUC_MODEL,
        providers: Optional[List] = None,
    ):
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            obj_class_score_th=0.0,
            attr_class_score_th=0.0,
            keypoint_th=0.0,
            providers=providers,
        )
        self._input_height, self._input_width = self._resolve_input_size()
        inputs = self._interpreter.get_inputs()
        outputs = self._interpreter.get_outputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError(
                f'PUC must expose one rank-4 image input; got {[input.shape for input in inputs]}.'
            )
        if inputs[0].shape[1] != 3:
            raise ValueError(f'PUC image input must have three channels; got {inputs[0].shape}.')
        if len(outputs) != 1 or len(outputs[0].shape) != 2 or outputs[0].shape[-1] != 3:
            raise ValueError(
                'PUC must expose one [batch, 3] probability output; '
                f'got {[output.shape for output in outputs]}.'
            )

    def _resolve_input_size(self) -> Tuple[int, int]:
        default_height, default_width = DEFAULT_PUC_INPUT_SIZE
        input_shape: Optional[List[int]] = None
        if self._runtime == 'onnx':
            input_shape = list(self._interpreter.get_inputs()[0].shape)
        elif self._input_details:
            input_shape = self._input_details[0].get('shape')
            if input_shape is not None:
                input_shape = list(input_shape)

        def _safe_dim(value: Any, default: int) -> int:
            try:
                if value is None:
                    return default
                return int(value)
            except (TypeError, ValueError):
                return default

        if not input_shape or len(input_shape) <= max(self._h_index, self._w_index):
            return default_height, default_width

        height = _safe_dim(input_shape[self._h_index], default_height)
        width = _safe_dim(input_shape[self._w_index], default_width)
        return height, width

    def __call__(self, image: np.ndarray) -> np.ndarray:
        resized_image = self._preprocess(image=image)
        inference_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inference_image])
        probs = np.squeeze(outputs[0])
        if probs.ndim == 0:
            probs = np.array([float(probs)], dtype=np.float32)
        return np.asarray(probs, dtype=np.float32)

    def _preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        return preprocess_hand_classifier_image(
            image,
            height=self._input_height,
            width=self._input_width,
            classifier_name='PUC',
        )

    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        return []


class PPCClassifier(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = None,
        providers: Optional[List] = None,
    ):
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            obj_class_score_th=0.0,
            attr_class_score_th=0.0,
            keypoint_th=0.0,
            providers=providers,
        )
        self._input_height, self._input_width = self._resolve_input_size()
        inputs = self._interpreter.get_inputs()
        outputs = self._interpreter.get_outputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError(
                f'PPC must expose one rank-4 image input; got {[input.shape for input in inputs]}.'
            )
        if inputs[0].shape[1] != 3:
            raise ValueError(f'PPC image input must have three channels; got {inputs[0].shape}.')
        if len(outputs) != 1 or len(outputs[0].shape) != 2 or outputs[0].shape[-1] != 2:
            raise ValueError(
                'PPC must expose one [batch, 2] probability output ordered as '
                '[no_possession, possession]; '
                f'got {[output.shape for output in outputs]}.'
            )

    def _resolve_input_size(self) -> Tuple[int, int]:
        default_height, default_width = DEFAULT_PPC_INPUT_SIZE
        input_shape = list(self._interpreter.get_inputs()[0].shape)

        def _safe_dim(value: Any, default: int) -> int:
            try:
                if value is None:
                    return default
                return int(value)
            except (TypeError, ValueError):
                return default

        if len(input_shape) <= max(self._h_index, self._w_index):
            return default_height, default_width
        return (
            _safe_dim(input_shape[self._h_index], default_height),
            _safe_dim(input_shape[self._w_index], default_width),
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        resized_image = self._preprocess(image=image)
        inference_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inference_image])
        probs = np.squeeze(outputs[0])
        if probs.ndim == 0:
            probs = np.array([float(probs)], dtype=np.float32)
        return np.asarray(probs, dtype=np.float32)

    def _preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        return preprocess_hand_classifier_image(
            image,
            height=self._input_height,
            width=self._input_width,
            classifier_name='PPC',
        )

    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        return []


@dataclass(frozen=True)
class HandClassificationResult:
    phone_class: int
    phone_confidence: float
    phone_state: int
    phone_probs: Optional[List[float]]
    ppc_class: int
    ppc_confidence: float
    ppc_label: str
    ppc_probs: Optional[List[float]]
    ppc_possession_score: float
    ppc_inference_failed: bool
    puc_forced_no_action: bool


def classify_hand_crop(
    rgb_hand_crop: np.ndarray,
    *,
    puc_classifier: Any,
    ppc_classifier: Any,
) -> HandClassificationResult:
    """Run both classifiers and apply the fail-safe PPC possession gate."""
    puc_probs: Optional[np.ndarray] = None
    ppc_probs: Optional[np.ndarray] = None
    try:
        puc_probs = validate_probability_vector(
            puc_classifier(image=rgb_hand_crop),
            expected_classes=3,
            classifier_name='PUC',
        )
    except Exception:
        puc_probs = None

    ppc_inference_failed = False
    try:
        ppc_probs = validate_probability_vector(
            ppc_classifier(image=rgb_hand_crop),
            expected_classes=2,
            classifier_name='PPC',
        )
    except Exception:
        ppc_probs = None
        ppc_inference_failed = True

    possession_score = float(ppc_probs[1]) if ppc_probs is not None else -1.0
    ppc_class = (
        1
        if ppc_probs is not None and possession_score >= PPC_POSSESSION_THRESHOLD
        else (0 if ppc_probs is not None else -1)
    )
    ppc_confidence = float(ppc_probs[ppc_class]) if ppc_probs is not None else -1.0
    ppc_label = PPC_LABELS.get(ppc_class, '')
    puc_probs_list = puc_probs.tolist() if puc_probs is not None else None
    ppc_probs_list = ppc_probs.tolist() if ppc_probs is not None else None

    if puc_probs is None:
        return HandClassificationResult(
            phone_class=-1,
            phone_confidence=-1.0,
            phone_state=-1,
            phone_probs=None,
            ppc_class=ppc_class,
            ppc_confidence=ppc_confidence,
            ppc_label=ppc_label,
            ppc_probs=ppc_probs_list,
            ppc_possession_score=possession_score,
            ppc_inference_failed=ppc_inference_failed,
            puc_forced_no_action=False,
        )

    top_class = int(np.argmax(puc_probs))
    return HandClassificationResult(
        phone_class=top_class,
        phone_confidence=float(puc_probs[top_class]),
        phone_state=1 if top_class > 0 else 0,
        phone_probs=puc_probs_list,
        ppc_class=ppc_class,
        ppc_confidence=ppc_confidence,
        ppc_label=ppc_label,
        ppc_probs=ppc_probs_list,
        ppc_possession_score=possession_score,
        ppc_inference_failed=ppc_inference_failed,
        puc_forced_no_action=False,
    )


def apply_hand_classification_result(box: Box, result: HandClassificationResult) -> None:
    box.phone_class = result.phone_class
    box.phone_confidence = result.phone_confidence
    box.phone_state = result.phone_state
    box.phone_probs = result.phone_probs
    box.ppc_class = result.ppc_class
    box.ppc_frame_class = result.ppc_class
    box.ppc_confidence = result.ppc_confidence
    box.ppc_label = result.ppc_label
    box.ppc_probs = result.ppc_probs
    box.ppc_possession_score = result.ppc_possession_score
    box.ppc_inference_failed = result.ppc_inference_failed
    box.puc_forced_no_action = result.puc_forced_no_action


def format_ppc_gate_overlay(box: Box) -> str:
    if box.ppc_inference_failed:
        return 'PPC error'
    if box.ppc_class not in PPC_LABELS or box.ppc_possession_score < 0.0:
        return ''
    return f'PPC {PPC_LABELS[box.ppc_class]}: {box.ppc_possession_score:.3f}'


def draw_ppc_gate_overlay(image: np.ndarray, box: Box) -> None:
    text = format_ppc_gate_overlay(box)
    if not text:
        return
    x = min(max(box.x1 + 5, 0), max(image.shape[1] - 5, 0))
    y = max(box.y1 - 50, 20)
    if y >= box.y1:
        y = min(max(box.y2 + 20, 20), image.shape[0] - 5)
    color = PPC_COLORS.get(box.ppc_class, (255, 255, 255))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def list_image_files(dir_path: str) -> List[str]:
    path = Path(dir_path)
    image_files = []
    for extension in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(path.rglob(extension))
    return sorted([str(file) for file in image_files])

def crop_image_with_margin(
    image: np.ndarray,
    box: Box,
    *,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
) -> Optional[np.ndarray]:
    """Extracts a region with the specified pixel margins."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    x1 = max(int(box.x1) - margin_left, 0)
    y1 = max(int(box.y1) - margin_top, 0)
    x2 = min(int(box.x2) + margin_right, w)
    y2 = min(int(box.y2) + margin_bottom, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()

def crop_image_with_expansion(
    image: np.ndarray,
    box: Box,
    *,
    expansion: float,
) -> Optional[np.ndarray]:
    """Extract a center-expanded detection region, matching data/extract_hand_crops.py."""
    if image is None or image.size == 0:
        return None
    if expansion <= 0:
        raise ValueError('Expansion must be positive.')

    frame_h, frame_w = image.shape[:2]
    box_width = float(box.x2 - box.x1)
    box_height = float(box.y2 - box.y1)
    if box_width <= 0 or box_height <= 0:
        return None

    center_x = (float(box.x1) + float(box.x2)) / 2.0
    center_y = (float(box.y1) + float(box.y2)) / 2.0
    expanded_width = box_width * expansion
    expanded_height = box_height * expansion

    x1 = int(np.floor(center_x - expanded_width / 2.0))
    y1 = int(np.floor(center_y - expanded_height / 2.0))
    x2 = int(np.ceil(center_x + expanded_width / 2.0))
    y2 = int(np.ceil(center_y + expanded_height / 2.0))

    x1 = max(0, min(frame_w, x1))
    x2 = max(0, min(frame_w, x2))
    y1 = max(0, min(frame_h, y1))
    y2 = max(0, min(frame_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()

def find_body_for_phone_usage_target(target_box: Box, body_boxes: List[Box]) -> Optional[Box]:
    candidates = [
        body_box
        for body_box in body_boxes
        if body_box.x1 <= target_box.cx <= body_box.x2 and body_box.y1 <= target_box.cy <= body_box.y2
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda body_box: (body_box.cx - target_box.cx) ** 2 + (body_box.cy - target_box.cy) ** 2,
    )

def phone_usage_target_priority(target_box: Box) -> Tuple[int, float]:
    is_action = int(target_box.phone_class is not None and target_box.phone_class > 0)
    return is_action, float(target_box.phone_confidence)

def assign_phone_usage_to_bodies(target_boxes: List[Box], body_boxes: List[Box]) -> None:
    best_target_by_body_index: Dict[int, Box] = {}

    for body_box in body_boxes:
        body_box.phone_confidence = -1.0
        body_box.phone_state = -1
        body_box.phone_label = ''
        body_box.phone_class = -1
        body_box.phone_probs = None
        body_box.puc_forced_no_action = False

    for target_box in target_boxes:
        if target_box.phone_class is None or target_box.phone_class < 0 or target_box.phone_confidence < 0:
            continue
        body_box = find_body_for_phone_usage_target(target_box=target_box, body_boxes=body_boxes)
        if body_box is None:
            continue
        body_index = next(
            (index for index, candidate_body in enumerate(body_boxes) if candidate_body is body_box),
            None,
        )
        if body_index is None:
            continue
        current_best = best_target_by_body_index.get(body_index)
        if current_best is None or phone_usage_target_priority(target_box) > phone_usage_target_priority(current_best):
            best_target_by_body_index[body_index] = target_box

    for body_index, target_box in best_target_by_body_index.items():
        body_box = body_boxes[body_index]
        body_box.phone_class = target_box.phone_class
        body_box.phone_confidence = target_box.phone_confidence
        body_box.phone_state = 1 if target_box.phone_class > 0 else 0
        body_box.phone_label = PHONE_USAGE_LABELS.get(target_box.phone_class, f"class_{target_box.phone_class}")
        body_box.phone_probs = list(target_box.phone_probs) if target_box.phone_probs is not None else None


def ppc_target_priority(target_box: Box) -> Tuple[int, float]:
    is_possession = int(target_box.ppc_class == 1)
    return is_possession, float(target_box.ppc_confidence)


def assign_ppc_to_bodies(target_boxes: List[Box], body_boxes: List[Box]) -> None:
    """Aggregate both hands into one PPC label and gate per body."""
    valid_targets_by_body_index: Dict[int, List[Box]] = {}
    failed_body_indices: set[int] = set()

    for body_box in body_boxes:
        body_box.ppc_class = -1
        body_box.ppc_frame_class = -1
        body_box.ppc_confidence = -1.0
        body_box.ppc_label = ''
        body_box.ppc_probs = None
        body_box.ppc_possession_score = -1.0
        body_box.ppc_inference_failed = False
        body_box.ppc_history_active = False
        body_box.ppc_state_track_id = -1
        body_box.puc_forced_no_action = True

    for target_box in target_boxes:
        body_box = find_body_for_phone_usage_target(target_box=target_box, body_boxes=body_boxes)
        if body_box is None:
            continue
        body_index = next(
            (index for index, candidate_body in enumerate(body_boxes) if candidate_body is body_box),
            None,
        )
        if body_index is None:
            continue
        if target_box.ppc_inference_failed:
            failed_body_indices.add(body_index)
        if target_box.ppc_class in PPC_LABELS and target_box.ppc_confidence >= 0.0:
            valid_targets_by_body_index.setdefault(body_index, []).append(target_box)

    for body_index, candidates in valid_targets_by_body_index.items():
        selected_target = max(candidates, key=ppc_target_priority)
        body_box = body_boxes[body_index]
        body_box.ppc_class = selected_target.ppc_class
        body_box.ppc_frame_class = selected_target.ppc_class
        body_box.ppc_confidence = selected_target.ppc_confidence
        body_box.ppc_label = PPC_LABELS[selected_target.ppc_class]
        body_box.ppc_probs = (
            list(selected_target.ppc_probs) if selected_target.ppc_probs is not None else None
        )
        body_box.ppc_possession_score = selected_target.ppc_possession_score
        body_box.puc_forced_no_action = selected_target.ppc_class != 1

    for body_index in failed_body_indices:
        if body_index not in valid_targets_by_body_index:
            body_boxes[body_index].ppc_inference_failed = True

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)

def distance_euclid(p1: Tuple[int,int], p2: Tuple[int,int]) -> float:
    """2点 (x1, y1), (x2, y2) のユークリッド距離を返す"""
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def draw_skeleton(
    image: np.ndarray,
    boxes: List[Box],
    color=(0,255,255),
    max_dist_threshold=500.0
):
    """
    与えられた boxes (各クラスIDの関節候補) を基に、EDGESで定義された親子を
    「もっとも近い距離のペアから順番に」接合していく。ただし、
    classid=0 (人物) のバウンディングボックス内にあるキーポイント同士のみを
    接続対象とする。
    """
    # -------------------------
    # 1) 人物ボックスに ID を付与する
    # -------------------------
    person_boxes = [b for b in boxes if b.classid == 0]
    for i, pbox in enumerate(person_boxes):
        # 便宜上、Boxクラスに person_id 属性がないので動的に付与する例
        pbox.person_id = i

    # -------------------------------------------------
    # 2) キーポイントがどの人物ボックスに属するか判断して person_id を記録
    #    （複数人のバウンディングボックスが重なっている場合は、
    #      先に見つかったものを採用、など適宜ルールを決める）
    # -------------------------------------------------
    keypoint_ids = KEYPOINT_CLASS_IDS
    for box in boxes:
        if box.classid in keypoint_ids:
            box.person_id = -1
            for pbox in person_boxes:
                if (pbox.x1 <= box.cx <= pbox.x2) and (pbox.y1 <= box.cy <= pbox.y2):
                    box.person_id = pbox.person_id
                    break

    # -------------------------
    # 3) クラスIDごとに仕分け
    # -------------------------
    classid_to_boxes: Dict[int, List[Box]] = {}
    for b in boxes:
        classid_to_boxes.setdefault(b.classid, []).append(b)

    edge_counts = Counter(EDGES)

    # 結果のラインを入れる
    lines_to_draw = []

    # ユークリッド距離計算の簡易関数
    def distance_euclid(p1, p2):
        import math
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    # 各 (pid, cid) ペアに対してグルーピング
    for (pid, cid), repeat_count in edge_counts.items():
        parent_list = classid_to_boxes.get(pid, [])
        child_list  = classid_to_boxes.get(cid, [])

        if not parent_list or not child_list:
            continue

        # 親クラスIDが21 or 29の時はEDGESに書かれている回数(=repeat_count)だけマッチ可
        # それ以外は1回だけ
        for_parent = repeat_count if (pid in [21, 29]) else 1

        parent_capacity = [for_parent]*len(parent_list)  # 親ごとに繋げる上限

        # 子は常に1回のみ
        child_used = [False]*len(child_list)

        # 距離が小さいペアから順に確定していくために、全ペアの距離を計算
        pair_candidates = []
        for i, pbox in enumerate(parent_list):
            for j, cbox in enumerate(child_list):
                # ここで "同じ person_id 同士であること" をチェック
                if (pbox.person_id is not None) and (cbox.person_id is not None) and (pbox.person_id == cbox.person_id):

                    dist = distance_euclid((pbox.cx, pbox.cy), (cbox.cx, cbox.cy))
                    if dist <= max_dist_threshold:
                        pair_candidates.append((dist, i, j))

        # 距離の小さい順に並べ替え
        pair_candidates.sort(key=lambda x: x[0])

        # 貪欲に割り当て
        for dist, i, j in pair_candidates:
            if parent_capacity[i] > 0 and (not child_used[j]):
                # 親iがまだマッチ可能 & 子jが未使用ならマッチ確定
                pbox = parent_list[i]
                cbox = child_list[j]

                lines_to_draw.append(((pbox.cx, pbox.cy), (cbox.cx, cbox.cy)))
                parent_capacity[i] -= 1
                child_used[j] = True

    # -------------------------
    # 4) ラインを描画
    # -------------------------
    for (pt1, pt2) in lines_to_draw:
        cv2.line(image, pt1, pt2, color, thickness=2)

def main():
    parser = ArgumentParser()

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 2:
            raise ArgumentTypeError(f"Invalid Value: {ivalue}. Please specify an integer of 2 or greater.")
        return ivalue

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='yolomit_t_wholebody28_1x3x480x640.onnx',
        help='ONNX/TFLite file path for the YOLOMIT whole-body object detector.',
    )
    parser.add_argument(
        '-pm',
        '--puc_model',
        type=str,
        default=DEFAULT_PUC_MODEL,
        help='ONNX file path for the PUC phone usage classifier.',
    )
    parser.add_argument(
        '--ppc_model',
        '--ppc-model',
        dest='ppc_model',
        type=str,
        default=DEFAULT_PPC_MODEL,
        help=f'ONNX file path for the PPC possession classifier. Default: {DEFAULT_PPC_MODEL}',
    )
    group_v_or_i = parser.add_mutually_exclusive_group(required=True)
    group_v_or_i.add_argument(
        '-v',
        '--video',
        type=str,
        help='Video file path or camera index.',
    )
    group_v_or_i.add_argument(
        '-i',
        '--images_dir',
        type=str,
        help='jpg, png images folder path.',
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='cuda',
        help='Execution provider for ONNXRuntime.',
    )
    parser.add_argument(
        '-it',
        '--inference_type',
        type=str,
        choices=['fp16', 'int8'],
        default='fp16',
        help='Inference type. Default: fp16',
    )
    parser.add_argument(
        '-dvw',
        '--disable_video_writer',
        action='store_true',
        help=\
            'Disable video writer. '+
            'Eliminates the file I/O load associated with automatic recording to MP4. '+
            'Devices that use a MicroSD card or similar for main storage can speed up overall processing.',
    )
    parser.add_argument(
        '-dwk',
        '--disable_waitKey',
        action='store_true',
        help=\
            'Disable cv2.waitKey(). '+
            'When you want to process a batch of still images, '+
            ' disable key-input wait and process them continuously.',
    )
    parser.add_argument(
        '-ost',
        '--object_socre_threshold',
        type=float,
        default=0.35,
        help=\
            'The detection score threshold for object detection. Default: 0.35',
    )
    parser.add_argument(
        '-ast',
        '--attribute_socre_threshold',
        type=float,
        default=0.70,
        help=\
            'The attribute score threshold for object detection. Default: 0.70',
    )
    parser.add_argument(
        '-kst',
        '--keypoint_threshold',
        type=float,
        default=0.30,
        help=\
            'The keypoint score threshold for object detection. Default: 0.30',
    )
    parser.add_argument(
        '--phone-usage-target-class-id',
        type=int,
        default=PHONE_USAGE_TARGET_CLASS_ID,
        help=f'Detector class ID used as the PUC phone usage target. Default: {PHONE_USAGE_TARGET_CLASS_ID}',
    )
    parser.add_argument(
        '--phone-usage-crop-expansion',
        type=float,
        default=PHONE_USAGE_CROP_EXPANSION,
        help=f'Center-based expansion factor applied before cropping the PUC target. Default: {PHONE_USAGE_CROP_EXPANSION}',
    )
    parser.add_argument(
        '--ppc-long-history-size',
        '--hand-long-history-size',
        '--body-long-history-size',
        dest='ppc_long_history_size',
        type=check_positive,
        default=PPC_LONG_HISTORY_SIZE,
        help=f'Long Body-level PPC history length. Default: {PPC_LONG_HISTORY_SIZE}',
    )
    parser.add_argument(
        '--ppc-short-history-size',
        '--hand-short-history-size',
        '--body-short-history-size',
        dest='ppc_short_history_size',
        type=check_positive,
        default=PPC_SHORT_HISTORY_SIZE,
        help=f'Short Body-level PPC history length. Default: {PPC_SHORT_HISTORY_SIZE}',
    )
    parser.add_argument(
        '-kdm',
        '--keypoint_drawing_mode',
        type=str,
        choices=['dot', 'box', 'both'],
        default='dot',
        help='Key Point Drawing Mode. Default: dot',
    )
    parser.add_argument(
        '-ebm',
        '--enable_bone_drawing_mode',
        action='store_true',
        help=\
            'Enable bone drawing mode. (Press B on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dnm',
        '--disable_generation_identification_mode',
        action='store_true',
        help=\
            'Disable generation identification mode. (Press N on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dgm',
        '--disable_gender_identification_mode',
        action='store_true',
        help=\
            'Disable gender identification mode. (Press G on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dlr',
        '--disable_left_and_right_hand_identification_mode',
        action='store_true',
        help=\
            'Disable left and right hand identification mode. (Press H on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dhm',
        '--disable_headpose_identification_mode',
        action='store_true',
        help=\
            'Disable HeadPose identification mode. (Press P on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-drc',
        '--disable_render_classids',
        type=int,
        nargs="*",
        default=[],
        help=\
            'Class ID to disable bounding box drawing. List[int]. e.g. -drc 17 18 19',
    )
    parser.add_argument(
        '-efm',
        '--enable_face_mosaic',
        action='store_true',
        help=\
            'Enable face mosaic.',
    )
    parser.add_argument(
        '-dtk',
        '--disable_tracking',
        action='store_true',
        help=\
            'Disable instance tracking. (Press R on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dti',
        '--disable_trackid_overlay',
        action='store_true',
        help=\
            'Disable TrackID overlay. (Press T on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dhd',
        '--disable_head_distance_measurement',
        action='store_true',
        help=\
            'Disable Head distance measurement. (Press M on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-oyt',
        '--output_yolo_format_text',
        action='store_true',
        help=\
            'Output YOLO format texts and images.',
    )
    parser.add_argument(
        '-bblw',
        '--bounding_box_line_width',
        type=check_positive,
        default=2,
        help=\
            'Bounding box line width. Default: 2',
    )
    parser.add_argument(
        '-chf',
        '--camera_horizontal_fov',
        type=int,
        default=90,
        help=\
            'Camera horizontal FOV. Default: 90',
    )
    args = parser.parse_args()
    ppc_long_history_size = args.ppc_long_history_size
    ppc_short_history_size = args.ppc_short_history_size
    if ppc_short_history_size > ppc_long_history_size:
        parser.error('--ppc-short-history-size must not exceed --ppc-long-history-size.')
    phone_usage_target_class_id = args.phone_usage_target_class_id
    phone_usage_crop_expansion = args.phone_usage_crop_expansion
    if phone_usage_crop_expansion <= 0:
        raise ValueError('--phone-usage-crop-expansion must be positive.')

    # runtime check
    model_file: str = args.model
    model_dir_path = os.path.dirname(os.path.abspath(model_file))
    model_ext: str = os.path.splitext(model_file)[1][1:].lower()
    runtime: str = None
    if model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif model_ext == 'tflite':
        if is_package_installed('ai_edge_litert'):
            runtime = 'ai_edge_litert'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: ai_edge_litert or tensorflow is not installed.'))
            sys.exit(0)
    video: str = args.video
    images_dir: str = args.images_dir
    disable_waitKey: bool = args.disable_waitKey
    object_socre_threshold: float = args.object_socre_threshold
    attribute_socre_threshold: float = args.attribute_socre_threshold
    keypoint_threshold: float = args.keypoint_threshold
    keypoint_drawing_mode: str = args.keypoint_drawing_mode
    enable_bone_drawing_mode: bool = args.enable_bone_drawing_mode
    disable_generation_identification_mode: bool = args.disable_generation_identification_mode
    disable_gender_identification_mode: bool = args.disable_gender_identification_mode
    disable_left_and_right_hand_identification_mode: bool = args.disable_left_and_right_hand_identification_mode
    disable_headpose_identification_mode: bool = args.disable_headpose_identification_mode
    disable_render_classids: List[int] = args.disable_render_classids
    enable_face_mosaic: bool = args.enable_face_mosaic
    enable_tracking: bool = not args.disable_tracking
    enable_trackid_overlay: bool = not args.disable_trackid_overlay
    enable_head_distance_measurement: bool = not args.disable_head_distance_measurement
    output_yolo_format_text: bool = args.output_yolo_format_text
    execution_provider: str = args.execution_provider
    inference_type: str = args.inference_type
    inference_type = inference_type.lower()
    bounding_box_line_width: int = args.bounding_box_line_width
    camera_horizontal_fov: int = args.camera_horizontal_fov
    puc_model_file: str = args.puc_model
    ppc_model_file: str = args.ppc_model
    for classifier_name, classifier_path in {
        'PUC classifier': puc_model_file,
        'PPC classifier': ppc_model_file,
    }.items():
        if not os.path.isfile(classifier_path):
            parser.error(f'{classifier_name} model file does not exist: {classifier_path}')
        if os.path.splitext(classifier_path)[1].lower() != '.onnx':
            parser.error(f'{classifier_name} must be an ONNX model: {classifier_path}')
    providers: List[Tuple[str, Dict] | str] = None

    if execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'tensorrt':
        ep_type_params = {}
        if inference_type == 'fp16':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        elif inference_type == 'int8':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                    "trt_int8_enable": True,
                    "trt_int8_calibration_table_name": "calibration.flatbuffers",
                }
        else:
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    'trt_engine_cache_enable': True, # .engine, .profile export
                    'trt_engine_cache_path': f'{model_dir_path}',
                    # 'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
                    # onnxruntime>=1.21.0 breaking changes
                    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
                    # https://github.com/microsoft/onnxruntime/pull/22681/files
                    # https://github.com/microsoft/onnxruntime/pull/23893/files
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                } | ep_type_params,
            ),
            "CUDAExecutionProvider",
            'CPUExecutionProvider',
        ]

    print(Color.GREEN('Provider parameters:'))
    pprint(providers)

    # Model initialization
    model = WholeBodyDetector(
        runtime=runtime,
        model_path=model_file,
        obj_class_score_th=object_socre_threshold,
        attr_class_score_th=attribute_socre_threshold,
        keypoint_th=keypoint_threshold,
        providers=providers,
    )
    phone_usage_classifier = PUCClassifier(
        runtime='onnx',
        model_path=puc_model_file,
        providers=providers,
    )
    possession_classifier = PPCClassifier(
        runtime='onnx',
        model_path=ppc_model_file,
        providers=providers,
    )

    file_paths: List[str] = None
    cap = None
    video_writer = None
    if images_dir is not None:
        file_paths = list_image_files(dir_path=images_dir)
    else:
        cap = cv2.VideoCapture(
            int(video) if is_parsable_to_int(video) else video
        )
        disable_video_writer: bool = args.disable_video_writer
        if not disable_video_writer:
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                filename='output.mp4',
                fourcc=fourcc,
                fps=cap_fps,
                frameSize=(w, h),
            )

    file_paths_count = -1
    movie_frame_count = 0
    white_line_width = bounding_box_line_width
    colored_line_width = white_line_width - 1
    tracker = SimpleSortTracker()
    ppc_history_manager = PPCBodyHistoryManager(
        long_size=ppc_long_history_size,
        short_size=ppc_short_history_size,
    )
    track_color_cache: Dict[int, np.ndarray] = {}
    tracking_enabled_prev = enable_tracking
    while True:
        image: np.ndarray = None
        if file_paths is not None:
            file_paths_count += 1
            if file_paths_count <= len(file_paths) - 1:
                image = cv2.imread(file_paths[file_paths_count])
            else:
                break
        else:
            res, image = cap.read()
            if not res:
                break
            movie_frame_count += 1

        debug_image = copy.deepcopy(image)
        debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        start_time = time.perf_counter()
        boxes = model(
            image=debug_image,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        elapsed_time = time.perf_counter() - start_time
        for box in boxes:
            if box.classid != phone_usage_target_class_id:
                continue
            hand_crop = crop_image_with_expansion(
                image=image,
                box=box,
                expansion=phone_usage_crop_expansion,
            )
            if hand_crop is None or hand_crop.size == 0:
                rgb_hand_crop = np.empty((0, 0, 3), dtype=np.uint8)
            else:
                rgb_hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            result = classify_hand_crop(
                rgb_hand_crop,
                puc_classifier=phone_usage_classifier,
                ppc_classifier=possession_classifier,
            )
            apply_hand_classification_result(box, result)

        phone_usage_target_boxes = [box for box in boxes if box.classid == phone_usage_target_class_id]
        body_boxes = [box for box in boxes if box.classid == 0]
        assign_phone_usage_to_bodies(
            target_boxes=phone_usage_target_boxes,
            body_boxes=body_boxes,
        )
        assign_ppc_to_bodies(
            target_boxes=phone_usage_target_boxes,
            body_boxes=body_boxes,
        )
        ppc_history_manager.update(body_boxes)

        for body_box in body_boxes:
            finalize_phone_usage_current_frame(body_box)

        if file_paths is None:
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        current_tracking_enabled = enable_tracking
        if current_tracking_enabled:
            if not tracking_enabled_prev:
                tracker = SimpleSortTracker()
                track_color_cache.clear()
            tracker.update(body_boxes)
            active_track_ids = {track['id'] for track in tracker.tracks}
            stale_ids = [tid for tid in track_color_cache.keys() if tid not in active_track_ids]
            for tid in stale_ids:
                track_color_cache.pop(tid, None)
        else:
            if tracking_enabled_prev:
                tracker = SimpleSortTracker()
                track_color_cache.clear()
            for box in boxes:
                box.track_id = -1
        tracking_enabled_prev = current_tracking_enabled

        # Draw bounding boxes
        for box in boxes:
            classid: int = box.classid
            is_phone_usage_target = classid == phone_usage_target_class_id
            phone_label_active = classid == 0 and bool(box.phone_label)
            color = (255,255,255)

            if classid in disable_render_classids:
                continue

            if classid == 0:
                phone_color = PHONE_USAGE_COLORS.get(getattr(box, "phone_class", -1))
                # Body
                if phone_label_active and phone_color is not None:
                    color = phone_color
                elif not disable_gender_identification_mode:
                    # Body
                    if box.gender == 0:
                        # Male
                        color = (255,0,0)
                    elif box.gender == 1:
                        # Female
                        color = (139,116,225)
                    else:
                        # Unknown
                        color = (0,0,255)
                else:
                    # Body
                    color = (0,0,255)
            elif is_phone_usage_target:
                # PUC phone usage target, normally Hand
                color = (0,255,0)
            elif classid == 5:
                # Body-With-Wheelchair
                color = (0,200,255)
            elif classid == 6:
                # Body-With-Crutches
                color = (83,36,179)
            elif classid == 7:
                # Head
                if not disable_headpose_identification_mode:
                    color = BOX_COLORS[box.head_pose][0] if box.head_pose != -1 else (216,67,21)
                else:
                    color = (0,0,255)
            elif classid == 16:
                # Face
                color = (0,200,255)
            elif classid == 17:
                # Eye
                color = (255,0,0)
            elif classid == 18:
                # Nose
                color = (0,255,0)
            elif classid == 19:
                # Mouth
                color = (255,0,0)
            elif classid == 20:
                # Ear
                color = (203,192,255)

            elif classid == 21:
                # Shoulder
                color = (255,0,0)
            elif classid == 22:
                # Elbow
                color = (0,255,0)
            elif classid == 26:
                # Knee
                color = (0,0,255)
            elif classid == 27:
                # Foot
                color = (250,0,136)

            if (classid == 0 and not disable_gender_identification_mode) \
                or (classid == 7 and not disable_headpose_identification_mode) \
                or ((classid == HAND_CLASS_ID or is_phone_usage_target) and not disable_left_and_right_hand_identification_mode) \
                or classid == 16 \
                or classid in KEYPOINT_CLASS_IDS:

                # Body
                if classid == 0:
                    if box.gender == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Head
                elif classid == 7:
                    if box.head_pose == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Face
                elif classid == 16:
                    if enable_face_mosaic:
                        w = int(abs(box.x2 - box.x1))
                        h = int(abs(box.y2 - box.y1))
                        small_box = cv2.resize(debug_image[box.y1:box.y2, box.x1:box.x2, :], (3,3))
                        normal_box = cv2.resize(small_box, (w,h))
                        if normal_box.shape[0] != abs(box.y2 - box.y1) \
                            or normal_box.shape[1] != abs(box.x2 - box.x1):
                                normal_box = cv2.resize(small_box, (abs(box.x2 - box.x1), abs(box.y2 - box.y1)))
                        debug_image[box.y1:box.y2, box.x1:box.x2, :] = normal_box
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Hands
                elif classid == HAND_CLASS_ID or is_phone_usage_target:
                    if box.handedness == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Shoulder, Elbow, Knee
                elif classid in KEYPOINT_CLASS_IDS:
                    if keypoint_drawing_mode in ['dot', 'both']:
                        cv2.circle(debug_image, (box.cx, box.cy), 4, (255,255,255), -1)
                        cv2.circle(debug_image, (box.cx, box.cy), 3, color, -1)
                    if keypoint_drawing_mode in ['box', 'both']:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)

            else:
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

            if classid == 0:
                draw_ppc_gate_overlay(debug_image, box)

            # TrackID text
            track_text_right_x: Optional[int] = None
            track_text_y: Optional[int] = None
            if enable_trackid_overlay and classid == 0 and box.track_id > 0:
                track_text = f'ID: {box.track_id}'
                text_x = max(box.x1 - 5, 0)
                text_y = box.y1 - 30
                if text_y < 20:
                    text_y = min(box.y2 + 25, debug_image_h - 10)
                track_text_size, _ = cv2.getTextSize(
                    track_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    1,
                )
                track_text_right_x = text_x + track_text_size[0]
                track_text_y = text_y
                cached_color = track_color_cache.get(box.track_id)
                if isinstance(cached_color, np.ndarray):
                    text_color = tuple(int(np.clip(v, 0, 255)) for v in cached_color.tolist())
                else:
                    text_color = color if isinstance(color, tuple) else (0, 200, 255)
                cv2.putText(
                    debug_image,
                    track_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (10, 10, 10),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    track_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            # Attributes text
            generation_txt = ''
            if box.generation == -1:
                generation_txt = ''
            elif box.generation == 0:
                generation_txt = 'Adult'
            elif box.generation == 1:
                generation_txt = 'Child'

            gender_txt = ''
            if box.gender == -1:
                gender_txt = ''
            elif box.gender == 0:
                gender_txt = 'M'
            elif box.gender == 1:
                gender_txt = 'F'

            attr_txt = f'{generation_txt}({gender_txt})' if gender_txt != '' else f'{generation_txt}'

            headpose_txt = BOX_COLORS[box.head_pose][1] if box.head_pose != -1 else ''
            attr_txt = f'{attr_txt} {headpose_txt}' if headpose_txt != '' else f'{attr_txt}'
            if classid == 0:
                if phone_label_active:
                    attr_txt = f'{box.phone_label} {box.phone_confidence:.3f}'

            attr_color = (
                PHONE_USAGE_COLORS.get(getattr(box, "phone_class", -1), color)
                if phone_label_active
                else color
            )
            if attr_txt:
                attr_x = box.x1 if box.x1+50 < debug_image_w else debug_image_w-50
                attr_y = box.y1-10 if box.y1-25 > 0 else 20
                if phone_label_active and track_text_right_x is not None and track_text_y is not None:
                    attr_text_size, _ = cv2.getTextSize(
                        attr_txt,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        1,
                    )
                    preferred_attr_x = track_text_right_x + 10
                    if preferred_attr_x + attr_text_size[0] <= debug_image_w - 2:
                        attr_x = preferred_attr_x
                    else:
                        attr_x = max(0, debug_image_w - attr_text_size[0] - 2)
                    attr_y = track_text_y
                cv2.putText(
                    debug_image,
                    f'{attr_txt}',
                    (attr_x, attr_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f'{attr_txt}',
                    (attr_x, attr_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    attr_color,
                    1,
                    cv2.LINE_AA,
                )

            handedness_txt = ''
            if box.handedness == -1:
                handedness_txt = ''
            elif box.handedness == 0:
                handedness_txt = 'L'
            elif box.handedness == 1:
                handedness_txt = 'R'
            handedness_y = box.y1 - 10 if box.y1 - 25 > 0 else 20
            cv2.putText(
                debug_image,
                f'{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    handedness_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    handedness_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

            # Head distance
            if enable_head_distance_measurement and classid == 7:
                focalLength: float = 0.0
                if (camera_horizontal_fov > 90):
                    # Fisheye Camera (Equidistant Model)
                    focalLength = debug_image_w / (camera_horizontal_fov * (math.pi / 180))
                else:
                    # Normal camera (Pinhole Model)
                    focalLength = debug_image_w / (2 * math.tan((camera_horizontal_fov / 2) * (math.pi / 180)))
                # Meters
                distance = (AVERAGE_HEAD_WIDTH * focalLength) / abs(box.x2 - box.x1)

                cv2.putText(
                    debug_image,
                    f'{distance:.3f} m',
                    (
                        box.x1+5 if box.x1 < debug_image_w else debug_image_w-50,
                        box.y1+20 if box.y1-5 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f'{distance:.3f} m',
                    (
                        box.x1+5 if box.x1 < debug_image_w else debug_image_w-50,
                        box.y1+20 if box.y1-15 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (10, 10, 10),
                    1,
                    cv2.LINE_AA,
                )

            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (255, 255, 255),
            #     2,
            #     cv2.LINE_AA,
            # )
            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     color,
            #     1,
            #     cv2.LINE_AA,
            # )

        # Draw skeleton
        if enable_bone_drawing_mode:
            draw_skeleton(image=debug_image, boxes=boxes, color=(0, 255, 255), max_dist_threshold=300)

        if file_paths is not None:
            basename = os.path.basename(file_paths[file_paths_count])
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{basename}', debug_image)

        if file_paths is not None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_i.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_o.png', debug_image)
            with open(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        elif file_paths is None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{movie_frame_count:08d}.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_i.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_o.png', debug_image)
            with open(f'output/{movie_frame_count:08d}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

        if video_writer is not None:
            video_writer.write(debug_image)
            # video_writer.write(image)

        cv2.imshow("test", debug_image)

        key = cv2.waitKey(1) & 0xFF if file_paths is None or disable_waitKey else cv2.waitKey(0) & 0xFF
        if key == ord('\x1b'): # 27, ESC
            break
        elif key == ord('b'): # 98, B, Bone drawing mode switch
            enable_bone_drawing_mode = not enable_bone_drawing_mode
        elif key == ord('n'): # 110, N, Generation mode switch
            disable_generation_identification_mode = not disable_generation_identification_mode
        elif key == ord('g'): # 103, G, Gender mode switch
            disable_gender_identification_mode = not disable_gender_identification_mode
        elif key == ord('p'): # 112, P, HeadPose mode switch
            disable_headpose_identification_mode = not disable_headpose_identification_mode
        elif key == ord('h'): # 104, H, HandsLR mode switch
            disable_left_and_right_hand_identification_mode = not disable_left_and_right_hand_identification_mode
        elif key == ord('k'): # 107, K, Keypoints mode switch
            if keypoint_drawing_mode == 'dot':
                keypoint_drawing_mode = 'box'
            elif keypoint_drawing_mode == 'box':
                keypoint_drawing_mode = 'both'
            elif keypoint_drawing_mode == 'both':
                keypoint_drawing_mode = 'dot'
        elif key == ord('r'): # 114, R, Tracking mode switch
            enable_tracking = not enable_tracking
            if enable_tracking and not enable_trackid_overlay:
                enable_trackid_overlay = True
        elif key == ord('t'): # 116, T, TrackID overlay mode switch
            enable_trackid_overlay = not enable_trackid_overlay
            if not enable_tracking:
                enable_trackid_overlay = False
        elif key == ord('m'): # 109, M, Head distance measurement mode switch
            enable_head_distance_measurement = not enable_head_distance_measurement

    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()

    try:
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    main()
