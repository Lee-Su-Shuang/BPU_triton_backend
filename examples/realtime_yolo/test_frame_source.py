import numpy as np

from examples.realtime_yolo.frame_source import validate_bgr_frame


def test_validate_bgr_frame_accepts_hwc_uint8_three_channel():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    validate_bgr_frame(frame)


def test_validate_bgr_frame_rejects_gray_frame():
    frame = np.zeros((10, 20), dtype=np.uint8)
    try:
        validate_bgr_frame(frame)
    except ValueError as exc:
        assert 'HxWx3' in str(exc)
    else:
        raise AssertionError('expected ValueError')
