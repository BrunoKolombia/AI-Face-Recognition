import cv2
import numpy as np
from typing import Optional


class CameraHandler:

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

        self.capture = None

        self.is_running = False

    def start(self) -> bool:
        if self.capture is not None:
            return True

        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.capture = None
            return False

        self.is_running = True
        return True

    def stop(self):
        self.is_running = False

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def read_frame(self) -> Optional[np.ndarray]:
        if self.capture is None or not self.is_running:
            return None

        ret, frame = self.capture.read()

        if ret:
            return frame
        return None

    def get_frame_size(self) -> tuple:
        if self.capture is None:
            return (640, 480)

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return (width, height)