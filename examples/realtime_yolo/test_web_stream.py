import numpy as np

from examples.realtime_yolo.web_stream import ValidationFramePipeline, encode_jpeg, render_index_html


def test_render_index_html_includes_stream_and_health():
    html = render_index_html()
    assert 'Realtime YOLO Validation' in html
    assert '<img id="stream" src="/stream.mjpg"' in html
    assert '/health' in html


def test_render_index_html_reconnects_stream_on_error():
    html = render_index_html()
    assert 'stream.onerror' in html
    assert 'setTimeout' in html
    assert '/stream.mjpg?ts=' in html


def test_encode_jpeg_returns_jpeg_bytes():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    encoded = encode_jpeg(frame)
    assert encoded.startswith(b'\xff\xd8')
    assert encoded.endswith(b'\xff\xd9')


def test_next_jpeg_does_not_block_on_inference_path():
    class Source:
        def read(self):
            return np.zeros((32, 32, 3), dtype=np.uint8)

    class Adapter:
        def inputs_from_bgr(self, frame):
            raise AssertionError('next_jpeg should not preprocess synchronously')

        def requested_outputs(self):
            raise AssertionError('next_jpeg should not request outputs synchronously')

        def postprocess(self, outputs, frame_shape):
            raise AssertionError('next_jpeg should not postprocess synchronously')

        def annotate(self, frame, detections, status_text):
            return frame

    class Client:
        def infer(self, inputs, outputs):
            raise AssertionError('next_jpeg should not call Triton synchronously')

    pipeline = ValidationFramePipeline(Source(), Adapter(), Client())

    encoded = pipeline.next_jpeg()

    assert encoded.startswith(b'\xff\xd8')
    assert encoded.endswith(b'\xff\xd9')
    assert pipeline.health()['last_error'] is None
