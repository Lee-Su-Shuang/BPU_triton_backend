from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import traceback

import cv2
import numpy as np


def render_index_html(title='Realtime YOLO Validation'):
    return f'''<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
      body {{ font-family: sans-serif; background: #111; color: #eee; }}
      img {{ max-width: 100%; border: 1px solid #555; }}
      code {{ color: #9cf; }}
      #status {{ margin: 8px 0; color: #9cf; }}
    </style>
  </head>
  <body>
    <h1>{title}</h1>
    <p>Stream: <code>/stream.mjpg</code> Health: <a href="/health">/health</a></p>
    <div id="status">connecting...</div>
    <img id="stream" src="/stream.mjpg" alt="annotated stream">
    <script>
      const stream = document.getElementById('stream');
      const status = document.getElementById('status');
      stream.onerror = () => {{
        status.textContent = 'stream disconnected; reconnecting...';
        setTimeout(() => {{ stream.src = '/stream.mjpg?ts=' + Date.now(); }}, 1000);
      }};
      setInterval(async () => {{
        try {{
          const response = await fetch('/health?ts=' + Date.now());
          status.textContent = await response.text();
        }} catch (err) {{
          status.textContent = 'health error: ' + err;
        }}
      }}, 1000);
    </script>
  </body>
</html>'''


def encode_jpeg(frame):
    ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError('failed to encode JPEG frame')
    return encoded.tobytes()


class ValidationFramePipeline:
    def __init__(self, source, adapter, client, target_stream_fps: float = 10.0):
        self.source = source
        self.adapter = adapter
        self.client = client
        self.target_stream_fps = target_stream_fps
        self.last_error = None
        self.frame_count = 0
        self.inference_count = 0
        self.last_latency_ms = 0.0
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_detections = []
        self._stop = threading.Event()
        self._capture_thread = None
        self._inference_thread = None

    def start(self):
        if self._capture_thread is not None:
            return
        self._capture_thread = threading.Thread(target=self._capture_loop, name='capture-loop', daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, name='inference-loop', daemon=True)
        self._capture_thread.start()
        self._inference_thread.start()

    def stop(self):
        self._stop.set()
        for thread in (self._capture_thread, self._inference_thread):
            if thread is not None:
                thread.join(timeout=2)

    def health(self):
        with self._lock:
            return {
                'ok': self.last_error is None,
                'frames': self.frame_count,
                'inferences': self.inference_count,
                'last_latency_ms': round(self.last_latency_ms, 2),
                'last_error': self.last_error,
            }

    def next_jpeg(self):
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            detections = list(self._latest_detections)
            latency_ms = self.last_latency_ms
            error = self.last_error
            frame_count = self.frame_count
            inference_count = self.inference_count
        if frame is None:
            frame = self.source.read()
        status = f'frames={frame_count} infer={latency_ms:.1f}ms infer_count={inference_count} error={error or "none"}'
        return encode_jpeg(self.adapter.annotate(frame, detections, status))

    def stream_interval_s(self):
        if self.target_stream_fps <= 0:
            return 0.0
        return 1.0 / self.target_stream_fps

    def _capture_loop(self):
        while not self._stop.is_set():
            try:
                frame = self.source.read()
                with self._lock:
                    self._latest_frame = frame
                    self.frame_count += 1
                    if self.last_error and self.last_error.startswith('capture:'):
                        self.last_error = None
            except Exception as exc:
                with self._lock:
                    self.last_error = 'capture: ' + ''.join(traceback.format_exception_only(type(exc), exc)).strip()
                time.sleep(0.2)

    def _inference_loop(self):
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue
            start = time.time()
            try:
                inputs = self.adapter.inputs_from_bgr(frame)
                outputs = self.client.infer(inputs, self.adapter.requested_outputs())
                detections = self.adapter.postprocess(outputs, frame.shape)
                error = None
            except Exception as exc:
                detections = []
                error = 'infer: ' + ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            latency_ms = (time.time() - start) * 1000.0
            with self._lock:
                self._latest_detections = detections
                self.last_latency_ms = latency_ms
                self.inference_count += 1
                self.last_error = error


def make_handler(pipeline):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                body = render_index_html().encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == '/health':
                body = json.dumps(pipeline.health(), separators=(',', ':')).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == '/stream.mjpg':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                interval_s = pipeline.stream_interval_s()
                while True:
                    try:
                        jpeg = pipeline.next_jpeg()
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode('ascii'))
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                        if interval_s > 0:
                            time.sleep(interval_s)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def serve(pipeline, host: str, port: int):
    pipeline.start()
    httpd = ThreadingHTTPServer((host, port), make_handler(pipeline))
    print(f'Realtime YOLO validation harness listening on http://{host}:{port}/')
    try:
        httpd.serve_forever()
    finally:
        pipeline.stop()
