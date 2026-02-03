from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

import cv2
import numpy as np

from .camera import Camera
from .frame import Frame
from .math import Transformation

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
        camera: Camera,
        transformations: Iterable[Iterable[Transformation]] = None,
    ):
        self._boards = boards
        if transformations is None:
            transformations = [[Transformation()] * len(boards)]

        # Sanity checks for angles
        for transformation in transformations:
            if len(transformation) != len(self._boards):
                raise ValueError(
                    "Number of transformations must match number of boards in all frames."
                )
        self._transformations = transformations
        self._frame_count = len(transformations)

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

        # First create a cv2 image with a white background which corresponds to a
        # distant wall
        frame = np.full(
            (
                int(self._camera.sensor_height),
                int(self._camera.sensor_width),
                3,
            ),
            255,
            dtype=np.uint8,
        )

        # Move the image further away and rotate the boards and get their images
        for board, transformations in zip(
            self._boards, self._transformations[self._current_angle_index]
        ):
            img = self._project_board(
                board.cv2_board_image,
                transformation=transformations,
            )
            projected_img = cv2.cvtColor(src=img, code=cv2.COLOR_GRAY2BGR)

            masked = projected_img < 255
            frame[masked] = projected_img[masked]

        self._current_angle_index += 1

        # Encode and decode to get a proper Frame object
        success, buf = cv2.imencode(".png", frame)
        if not success:
            return None
        return Frame(cv2.imdecode(buf, cv2.IMREAD_COLOR))

    def _project_board(
        self,
        img: np.ndarray,
        transformation: Transformation,
    ) -> np.ndarray:
        """
        Project the Charuco board image with a given rotation and translation.

        Parameters:
            img (np.ndarray): The Charuco board image to project.
            transformation (Transformation): The transformation containing translation and rotation.
        """

        # For a planar homography: H = K @ (R - t*n^T/d) @ K^(-1)
        # where n is the normal to the plane (0,0,1) and d is distance
        # Since the board is in the z=0 plane initially, we use:
        # H = K @ (R + t @ [0,0,1]) @ K^(-1)
        normal = np.array([[0.0], [0.0], [1.0]])
        translation = transformation.translation.vector
        rotation = transformation.rotation.matrix
        transformation_matrix = (
            self._camera.matrix
            @ (rotation + translation @ normal.T)
            @ np.linalg.inv(self._camera.matrix)
        )

        # Project original image corners
        return cv2.warpPerspective(
            img,
            transformation_matrix,
            (int(self._camera.sensor_width), int(self._camera.sensor_height)),
            flags=cv2.INTER_LINEAR,
            borderValue=255,
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
