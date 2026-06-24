import math
import re
import subprocess

import cv2
import numpy as np

from examples.realtime_yolo.frame_source import validate_bgr_frame
from examples.realtime_yolo.triton_http import RequestedOutput, TensorSpec


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


def parse_model_info_scales(text: str):
    scales = {}
    current_name = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith('name: '):
            current_name = line.split(':', 1)[1].strip()
        elif line.startswith('scale data:') and current_name:
            values = line.split(':', 1)[1].strip()
            values = values.strip('()')
            scales[current_name] = [float(item) for item in re.split(r'\s*,\s*', values) if item]
    return scales


def load_model_info_scales(model_path: str):
    result = subprocess.run(
        ['hrt_model_exec', 'model_info', '--model_file', model_path],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_model_info_scales(result.stdout)


class Yolov5xAdapter:
    model_name = 'yolov5x_bpu'
    model_path = '/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm'
    input_width = 672
    input_height = 672
    output_sizes = {'output': 7197120, '1310': 1799280, '1312': 449820}
    output_shapes = {
        'output': [1, 84, 84, 255],
        '1310': [1, 42, 42, 255],
        '1312': [1, 21, 21, 255],
    }
    output_order = ['output', '1310', '1312']
    strides = {'output': 8, '1310': 16, '1312': 32}
    anchors = {
        'output': [(10, 13), (16, 30), (33, 23)],
        '1310': [(30, 61), (62, 45), (59, 119)],
        '1312': [(116, 90), (156, 198), (373, 326)],
    }

    def __init__(self, score_threshold=0.3, nms_threshold=0.45, nms_top_k=20, scales=None):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self._scales = scales

    def inputs_from_bgr(self, frame):
        validate_bgr_frame(frame)
        data_y, data_uv = self._bgr_to_nv12_split(frame)
        return [
            TensorSpec('data_y', [1, 672, 672, 1], 'UINT8', data_y),
            TensorSpec('data_uv', [1, 336, 336, 2], 'UINT8', data_uv),
        ]

    def requested_outputs(self):
        return [RequestedOutput(name) for name in self.output_order]

    def validate_outputs(self, outputs):
        for name, expected_size in self.output_sizes.items():
            actual_size = len(outputs.get(name, b''))
            if actual_size != expected_size:
                raise RuntimeError(f'unexpected output size for {name}: got {actual_size}, expected {expected_size}')

    def postprocess(self, outputs, frame_shape):
        self.validate_outputs(outputs)
        scales = self._get_scales()
        candidates = []
        original_height, original_width = frame_shape[:2]
        h_ratio = self.input_height / original_height
        w_ratio = self.input_width / original_width

        for name in self.output_order:
            tensor = np.frombuffer(outputs[name], dtype=np.int32).reshape(self.output_shapes[name])
            scale = np.asarray(scales[name], dtype=np.float32).reshape((1, 1, 1, 255))
            decoded = tensor.astype(np.float32) * scale
            candidates.extend(self._decode_output(name, decoded[0], original_width, original_height, w_ratio, h_ratio))
        return self._nms(candidates)

    def annotate(self, frame, detections, status_text: str):
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            name = det.get('name', 'object')
            score = float(det.get('score', 0.0))
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'{name} {score:.2f}',
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        cv2.putText(frame, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def _get_scales(self):
        if self._scales is None:
            self._scales = load_model_info_scales(self.model_path)
        missing = [name for name in self.output_order if name not in self._scales]
        if missing:
            raise RuntimeError(f'missing output scale metadata: {missing}')
        return self._scales

    def _decode_output(self, name, data, original_width, original_height, w_ratio, h_ratio):
        height, width = data.shape[:2]
        stride = self.strides[name]
        anchors = self.anchors[name]
        results = []
        for h in range(height):
            for w in range(width):
                cell = data[h, w].reshape((3, 85))
                for anchor_index, (anchor_x, anchor_y) in enumerate(anchors):
                    pred = cell[anchor_index]
                    class_id = int(np.argmax(pred[5:]))
                    confidence = self._sigmoid(float(pred[4])) * self._sigmoid(float(pred[class_id + 5]))
                    if confidence < self.score_threshold:
                        continue
                    center_x = (self._sigmoid(float(pred[0])) * 2 - 0.5 + w) * stride
                    center_y = (self._sigmoid(float(pred[1])) * 2 - 0.5 + h) * stride
                    box_w = (self._sigmoid(float(pred[2])) * 2) ** 2 * anchor_x
                    box_h = (self._sigmoid(float(pred[3])) * 2) ** 2 * anchor_y
                    xmin = max((center_x - box_w / 2.0) / w_ratio, 0.0)
                    ymin = max((center_y - box_h / 2.0) / h_ratio, 0.0)
                    xmax = min((center_x + box_w / 2.0) / w_ratio, original_width - 1.0)
                    ymax = min((center_y + box_h / 2.0) / h_ratio, original_height - 1.0)
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    results.append({
                        'id': class_id,
                        'name': COCO_CLASSES[class_id],
                        'score': confidence,
                        'bbox': [xmin, ymin, xmax, ymax],
                    })
        return results

    def _nms(self, detections):
        detections = sorted(detections, key=lambda item: item['score'], reverse=True)
        kept = []
        suppressed = [False] * len(detections)
        for i, det in enumerate(detections):
            if suppressed[i]:
                continue
            kept.append(det)
            if len(kept) >= self.nms_top_k:
                break
            for j in range(i + 1, len(detections)):
                if suppressed[j] or det['id'] != detections[j]['id']:
                    continue
                if self._iou(det['bbox'], detections[j]['bbox']) > self.nms_threshold:
                    suppressed[j] = True
        return kept

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        denom = area_a + area_b - inter
        return 0.0 if denom <= 0 else inter / denom

    @staticmethod
    def _sigmoid(value):
        return 1.0 / (1.0 + math.exp(-value))

    def _bgr_to_nv12_split(self, frame):
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        area = self.input_width * self.input_height
        yuv420p = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area].reshape((1, self.input_height, self.input_width, 1)).copy()
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv = uv_planar.transpose((1, 0)).reshape((1, self.input_height // 2, self.input_width // 2, 2)).copy()
        return y.tobytes(), uv.tobytes()
