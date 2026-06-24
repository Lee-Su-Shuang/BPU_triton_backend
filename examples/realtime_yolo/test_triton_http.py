import json

from examples.realtime_yolo.triton_http import (
    RequestedOutput,
    TensorSpec,
    build_binary_request,
    parse_binary_response,
)


def test_build_binary_request_is_model_agnostic_and_preserves_payload_order():
    header, payload = build_binary_request(
        [
            TensorSpec('input_a', [1, 2], 'UINT8', b'ab'),
            TensorSpec('input_b', [1, 3], 'INT32', b'cdef'),
        ],
        [RequestedOutput('out0'), RequestedOutput('out1')],
    )

    request = json.loads(header.decode('utf-8'))
    assert [item['name'] for item in request['inputs']] == ['input_a', 'input_b']
    assert request['inputs'][0]['shape'] == [1, 2]
    assert request['inputs'][1]['datatype'] == 'INT32'
    assert request['inputs'][0]['parameters']['binary_data_size'] == 2
    assert request['inputs'][1]['parameters']['binary_data_size'] == 4
    assert [item['name'] for item in request['outputs']] == ['out0', 'out1']
    assert payload == b'abcdef'


def test_parse_binary_response_splits_outputs_by_response_metadata_order():
    metadata = {
        'outputs': [
            {'name': 'out0', 'parameters': {'binary_data_size': 3}},
            {'name': 'out1', 'parameters': {'binary_data_size': 2}},
        ]
    }
    header = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
    payload = header + b'abcde'
    headers = {'Inference-Header-Content-Length': str(len(header))}

    outputs = parse_binary_response(headers, payload)

    assert outputs == {'out0': b'abc', 'out1': b'de'}
