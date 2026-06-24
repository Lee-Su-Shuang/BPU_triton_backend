#!/usr/bin/env python3
import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.realtime_yolo.frame_source import UsbCameraSource
from examples.realtime_yolo.triton_http import TritonHttpClient
from examples.realtime_yolo.web_stream import ValidationFramePipeline, serve
from examples.realtime_yolo.yolov5x_adapter import Yolov5xAdapter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Realtime YOLOv5x validation harness for Triton BPU backend')
    parser.add_argument('--video-device', default='/dev/video0', help='USB camera device path')
    parser.add_argument('--triton-url', default='http://127.0.0.1:8000', help='Triton base HTTP URL')
    parser.add_argument('--host', default='0.0.0.0', help='HTTP server bind host')
    parser.add_argument('--port', type=int, default=8080, help='HTTP server port')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source = UsbCameraSource(args.video_device)
    try:
        adapter = Yolov5xAdapter()
        client = TritonHttpClient(args.triton_url, adapter.model_name)
        pipeline = ValidationFramePipeline(source, adapter, client)
        serve(pipeline, args.host, args.port)
    finally:
        source.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
