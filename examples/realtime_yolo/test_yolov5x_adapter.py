import numpy as np

from examples.realtime_yolo.yolov5x_adapter import Yolov5xAdapter, parse_model_info_scales


def test_yolov5x_adapter_builds_expected_triton_inputs():
    adapter = Yolov5xAdapter()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    inputs = adapter.inputs_from_bgr(frame)

    assert [item.name for item in inputs] == ['data_y', 'data_uv']
    assert inputs[0].shape == [1, 672, 672, 1]
    assert inputs[1].shape == [1, 336, 336, 2]
    assert inputs[0].datatype == 'UINT8'
    assert inputs[1].datatype == 'UINT8'
    assert len(inputs[0].data) == 451584
    assert len(inputs[1].data) == 225792


def test_yolov5x_adapter_requested_outputs_are_model_specific():
    adapter = Yolov5xAdapter()
    assert [item.name for item in adapter.requested_outputs()] == ['output', '1310', '1312']


def test_yolov5x_adapter_validate_outputs_rejects_wrong_size():
    adapter = Yolov5xAdapter()
    outputs = {'output': b'1', '1310': b'2', '1312': b'3'}

    try:
        adapter.validate_outputs(outputs)
    except RuntimeError as exc:
        assert 'unexpected output size for output' in str(exc)
    else:
        raise AssertionError('expected RuntimeError')


def test_yolov5x_adapter_annotate_changes_frame_pixels():
    adapter = Yolov5xAdapter()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{'bbox': [10, 10, 50, 50], 'name': 'person', 'score': 0.9}]

    annotated = adapter.annotate(frame.copy(), detections, 'status')

    assert annotated.shape == frame.shape
    assert int(annotated.sum()) > 0


def test_parse_model_info_scales_extracts_output_scales():
    text = '''
output[0]:
name: output
valid shape: (1,84,84,255)
quanti type: SCALE
scale data: (0.1,2e-05,0.3)
quantizeAxis: 3
output[1]:
name: 1310
valid shape: (1,42,42,255)
quanti type: SCALE
scale data: (4.0,5.0)
quantizeAxis: 3
'''

    scales = parse_model_info_scales(text)

    assert scales['output'] == [0.1, 2e-05, 0.3]
    assert scales['1310'] == [4.0, 5.0]
