import cv2
import numpy as np


def validate_bgr_frame(frame):
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('expected BGR frame with shape HxWx3')
    if frame.dtype != np.uint8:
        raise ValueError('expected BGR frame dtype uint8')


class UsbCameraSource:
    def __init__(self, device: str):
        self.device = device
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f'failed to open camera: {device}')

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f'failed to read frame from camera: {self.device}')
        validate_bgr_frame(frame)
        return frame

    def close(self):
        self.cap.release()
