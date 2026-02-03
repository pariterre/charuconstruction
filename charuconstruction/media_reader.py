from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

import cv2
import numpy as np

from .camera import Camera
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
    def __init__(
        self,
        boards: Iterable[Charuco],
        angles: Iterable[Iterable[float]] = None,
        camera: Camera = Camera.default(),
    ):
        self._boards = boards
        if angles is None:
            angles = [(0.0,) for _ in boards]

        # Sanity checks for angles
        if len(angles) != len(self._boards):
            raise ValueError(
                "Number of angle sequences must match number of boards."
            )
        for i in range(len(boards)):
            if len(angles[i]) != len(angles[0]):
                raise ValueError(
                    "All angle sequences must have the same length."
                )
        self._angles = angles
        self._frame_count = len(angles[0])

        self._camera = camera
        self._current_angle_index = 0

        super().__init__()

    def destroy(self):
        pass

    def __iter__(self) -> "CharucoMockReader":
        self._current_angle_index = 0
        return self

    def _read_frame(self):
        if self._current_angle_index >= self._frame_count:
            return None

        # Move the image further away and rotate the boards and get their images
        cv2_imgs: list[np.ndarray] = []
        for board, angles in zip(self._boards, self._angles):
            angle = angles[self._current_angle_index]
            img = self._move_image(
                board.cv2_board_image, distance=10, angle_deg=angle
            )
            cv2_imgs.append(cv2.cvtColor(src=img, code=cv2.COLOR_GRAY2BGR))
        self._current_angle_index += 1

        # Create a composite image
        padded_imgs: list[np.ndarray] = []
        max_img_height = max(cv2_imgs, key=lambda im: im.shape[0]).shape[0]
        for img in cv2_imgs:
            padded_imgs.append(
                self._pad_center_vertical(img=img, target_height=max_img_height)
            )
        combined = np.hstack(padded_imgs)

        # Encode and decode to get a proper Frame object
        success, buf = cv2.imencode(".png", combined)
        if not success:
            return None
        return Frame(cv2.imdecode(buf, cv2.IMREAD_COLOR))

    def _move_image(
        self,
        img: np.ndarray,
        distance: float,
        angle_deg: float,
        background_color: int = 255,
    ) -> np.ndarray:
        angle = np.deg2rad(angle_deg)
        scale = self._camera.focal_length / (
            self._camera.focal_length + distance
        )

        rotation_matrix = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
        translation_matrix = np.array(
            [[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        transformation_matrix = (
            self._camera.matrix
            @ (translation_matrix @ rotation_matrix)
            @ np.linalg.inv(self._camera.matrix)
        )

        # Project original image corners
        height, width = img.shape[:2]
        corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )
        transformed_corners: np.ndarray = cv2.perspectiveTransform(
            corners[None, :, :], transformation_matrix
        )[0]

        min_x, min_y = transformed_corners.min(axis=0)
        max_x, max_y = transformed_corners.max(axis=0)
        new_width = int(np.ceil(max_x - min_x))
        new_height = int(np.ceil(max_y - min_y))

        # Translation to keep image fully visible
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        transformation_total = translation @ transformation_matrix
        return cv2.warpPerspective(
            img,
            transformation_total,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderValue=background_color,
        )

    @staticmethod
    def _pad_center_vertical(
        img: np.ndarray,
        target_height: float,
        background_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        height, _ = img.shape[:2]
        if height >= target_height:
            return img
        top = (target_height - height) // 2
        bottom = target_height - height - top
        return cv2.copyMakeBorder(
            img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=background_color
        )
