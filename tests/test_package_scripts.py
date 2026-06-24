from pathlib import Path


def test_verify_all_cleans_services_before_starting_triton():
    text = Path('verify_all.sh').read_text()
    assert './stop_services.sh' in text
    assert text.index('./stop_services.sh') < text.index('scripts/triton_e2e/start_bpu_triton.sh')


def test_run_realtime_yolo_cleans_services_before_starting_triton():
    text = Path('run_realtime_yolo.sh').read_text()
    assert './stop_services.sh' in text
    assert text.index('./stop_services.sh') < text.index('scripts/triton_e2e/start_bpu_triton.sh')
