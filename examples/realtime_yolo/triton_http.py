from dataclasses import dataclass
import json
import urllib.request


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: list
    datatype: str
    data: bytes


@dataclass(frozen=True)
class RequestedOutput:
    name: str


def build_binary_request(inputs, outputs):
    request = {
        'inputs': [
            {
                'name': item.name,
                'shape': list(item.shape),
                'datatype': item.datatype,
                'parameters': {'binary_data_size': len(item.data)},
            }
            for item in inputs
        ],
        'outputs': [{'name': item.name, 'parameters': {'binary_data': True}} for item in outputs],
    }
    header = json.dumps(request, separators=(',', ':')).encode('utf-8')
    payload = b''.join(item.data for item in inputs)
    return header, payload


def parse_binary_response(headers, payload: bytes):
    header_len = int(headers['Inference-Header-Content-Length'])
    metadata = json.loads(payload[:header_len])
    raw = payload[header_len:]
    offset = 0
    outputs = {}
    for output in metadata['outputs']:
        name = output['name']
        size = int(output['parameters']['binary_data_size'])
        outputs[name] = raw[offset:offset + size]
        offset += size
    if offset != len(raw):
        raise RuntimeError(f'Triton response size mismatch: parsed={offset} raw={len(raw)}')
    return outputs


class TritonHttpClient:
    def __init__(self, base_url: str, model_name: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout_s = timeout_s

    def infer(self, inputs, outputs):
        header, payload = build_binary_request(inputs, outputs)
        request = urllib.request.Request(
            f'{self.base_url}/v2/models/{self.model_name}/infer',
            data=header + payload,
            method='POST',
            headers={
                'Content-Type': 'application/octet-stream',
                'Inference-Header-Content-Length': str(len(header)),
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return parse_binary_response(response.headers, response.read())
