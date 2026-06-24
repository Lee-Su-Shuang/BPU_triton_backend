#!/usr/bin/env python3
import json
import sys
import time
import urllib.request

READY_URL = 'http://127.0.0.1:8000/v2/health/ready'
INFER_URL = 'http://127.0.0.1:8000/v2/models/yolov5x_bpu/infer'
EXPECTED_SIZES = {'output': 7197120, '1310': 1799280, '1312': 449820}


def wait_ready(timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(READY_URL, timeout=1) as response:
                if response.status == 200:
                    print('triton ready')
                    return
        except Exception as exc:  # readiness endpoint may not be up yet
            last = exc
        time.sleep(1)
    raise RuntimeError(f'Triton did not become ready: {last}')


def infer() -> None:
    data_y = bytes(1 * 672 * 672 * 1)
    data_uv = bytes(1 * 336 * 336 * 2)
    request = {
        'inputs': [
            {
                'name': 'data_y',
                'shape': [1, 672, 672, 1],
                'datatype': 'UINT8',
                'parameters': {'binary_data_size': len(data_y)},
            },
            {
                'name': 'data_uv',
                'shape': [1, 336, 336, 2],
                'datatype': 'UINT8',
                'parameters': {'binary_data_size': len(data_uv)},
            },
        ],
        'outputs': [
            {'name': 'output', 'parameters': {'binary_data': True}},
            {'name': '1310', 'parameters': {'binary_data': True}},
            {'name': '1312', 'parameters': {'binary_data': True}},
        ],
    }
    header = json.dumps(request, separators=(',', ':')).encode('utf-8')
    http_request = urllib.request.Request(
        INFER_URL,
        data=header + data_y + data_uv,
        method='POST',
        headers={
            'Content-Type': 'application/octet-stream',
            'Inference-Header-Content-Length': str(len(header)),
        },
    )
    with urllib.request.urlopen(http_request, timeout=60) as response:
        header_len = int(response.headers['Inference-Header-Content-Length'])
        payload = response.read()
    metadata = json.loads(payload[:header_len])
    raw = payload[header_len:]
    sizes = {out['name']: out['parameters']['binary_data_size'] for out in metadata['outputs']}
    parsed = sum(sizes.values())
    print(json.dumps(metadata, indent=2))
    print('raw_bytes', len(raw), 'parsed_bytes', parsed, 'sizes', sizes)
    if sizes != EXPECTED_SIZES:
        raise RuntimeError(f'unexpected output sizes: {sizes}')
    if len(raw) != parsed:
        raise RuntimeError(f'raw length mismatch: raw={len(raw)} parsed={parsed}')


if __name__ == '__main__':
    wait_ready()
    infer()
