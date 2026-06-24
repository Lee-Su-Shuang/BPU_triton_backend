import subprocess
import sys

from examples.realtime_yolo.run_usb_yolov5x_web import parse_args


def test_parse_args_defaults_are_backend_validation_friendly():
    args = parse_args([])
    assert args.video_device == '/dev/video0'
    assert args.triton_url == 'http://127.0.0.1:8000'
    assert args.host == '0.0.0.0'
    assert args.port == 8080


def test_script_help_works_when_run_by_path():
    result = subprocess.run(
        [sys.executable, 'examples/realtime_yolo/run_usb_yolov5x_web.py', '--help'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert 'Realtime YOLOv5x validation harness for Triton BPU backend' in result.stdout
