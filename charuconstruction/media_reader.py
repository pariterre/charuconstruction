from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

import cv2
import numpy as np

from .frame import Frame

if TYPE_CHECKING:
    from .charuco import Charuco


class MediaReader(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def destroy(self):
        """
        Release any resources held by the reader.
        """

    # Reader as a list iterator
    def __iter__(self) -> "MediaReader":
        return self

    def __next__(self) -> Frame:
        frame = self._read_frame()
        if frame is None:
            raise StopIteration()

        return frame

    @abstractmethod
    def _read_frame(self) -> Frame | None:
        """
        Read a frame from the source.
        """


class ImageReader(MediaReader):
    def __init__(self, image_path: str):
        self._image_path = image_path
        self._image = cv2.imread(self._image_path)
        self._first = True
        super().__init__()

    def destroy(self):
        pass

    def _read_frame(self):
        if not self._first:
            return None
        self._first = False
        return Frame(self._image.copy())


class VideoReader(MediaReader):
    def __init__(self, video_path: str = 0):
        self._video_path = video_path
        self._cap = cv2.VideoCapture(self._video_path)
        super().__init__()

    def destroy(self):
        self._cap.release()

    def _read_frame(self):
        ret, frame = self._cap.read()
        if not ret:
            return None
        return Frame(frame)


class CharucoMockReader(MediaReader):
    def __init__(self, board1: Charuco, board2: Charuco, angles: Iterable[float] = (0,)):
        self._board1 = board1
        self._board2 = board2

        self._current_angle_index = 0
        self._angles = angles

        super().__init__()

    def destroy(self):
        pass

    def __iter__(self) -> "CharucoMockReader":
        self._current_angle_index = 0
        return self

    def _read_frame(self):
        if self._current_angle_index >= len(self._angles):
            return None
        angle = self._angles[self._current_angle_index]
        self._current_angle_index += 1

        # Create a composite image from two charuco boards, one still next to a rotating one
        board1_img = self._board1.cv2_board_image
        board2_img = CharucoMockReader._rotate_about_y_axis(img=self._board2.cv2_board_image, angle_deg=angle * 3)

        board1_img: np.ndarray = cv2.cvtColor(src=board1_img, code=cv2.COLOR_GRAY2BGR)
        board2_img: np.ndarray = cv2.cvtColor(src=board2_img, code=cv2.COLOR_GRAY2BGR)
        height = max(board1_img.shape[0], board2_img.shape[0])

        left = CharucoMockReader._pad_center_vertical(img=board1_img, target_height=height)
        right = CharucoMockReader._pad_center_vertical(img=board2_img, target_height=height)
        combined = np.hstack([left, right])

        success, buf = cv2.imencode(".png", combined)
        if not success:
            return None
        return Frame(cv2.imdecode(buf, cv2.IMREAD_COLOR))

    @staticmethod
    def _rotate_about_y_axis(
        img: np.ndarray, angle_deg: float, focal_length: int = 1000, background_color: int = 255
    ) -> np.ndarray:
        height, width = img.shape[:2]
        angle = np.deg2rad(angle_deg)

        cx, cy = width / 2, height / 2
        K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
        R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        H = K @ R @ np.linalg.inv(K)

        # Project original image corners
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        corners_h: np.ndarray = cv2.perspectiveTransform(corners[None, :, :], H)[0]

        min_x, min_y = corners_h.min(axis=0)
        max_x, max_y = corners_h.max(axis=0)

        new_width = int(np.ceil(max_x - min_x))
        new_height = int(np.ceil(max_y - min_y))

        # Translation to keep image fully visible
        T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        H_total = T @ H
        warped = cv2.warpPerspective(
            img, H_total, (new_width, new_height), flags=cv2.INTER_LINEAR, borderValue=background_color
        )
        return warped

    @staticmethod
    def _pad_center_vertical(
        img: np.ndarray, target_height: float, bg: tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        height, _ = img.shape[:2]
        if height >= target_height:
            return img
        top = (target_height - height) // 2
        bottom = target_height - height - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=bg)
