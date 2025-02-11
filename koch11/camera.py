import cv2 as cv


class Camera:
    def __init__(
        self,
        device_id: str,
        fps: int = 120,
        width: int = 640,
        height: int = 480,
        fourcc: str = "MJPG",
    ):
        self.device_id_ = device_id
        self.cap = cv.VideoCapture(int(device_id))
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv.CAP_PROP_FPS, fps)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 4.0)

    def read(self, flip: bool = True):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read frame from camera.")
        if flip:
            frame = cv.flip(frame, 0)
            frame = cv.flip(frame, 1)
        return frame

    @property
    def device_id(self):
        return self._device_id_

    @property
    def width(self):
        return self.cap.get(cv.CAP_PROP_FRAME_WIDTH)

    @property
    def height(self):
        return self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
