from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

import cv2
import numpy as np

from .camera import Camera
from .frame import Frame
from .math import Transformation, TranslationVector, RotationMatrix, Vector3

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
    def __init__(self, image_path: str | Iterable[str]):
        self._image_path = (
            (image_path,) if isinstance(image_path, str) else image_path
        )
        self._index = 0
        super().__init__()

    def destroy(self):
        pass

    def _read_frame(self):
        if self._index >= len(self._image_path):
            return None
        image = cv2.imread(self._image_path[self._index])
        self._index += 1
        return Frame(image)


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


class LiveVideoReader(MediaReader):
    # TODO
    pass


class CharucoMockReader(MediaReader):
    def __init__(
        self,
        boards: Iterable[Charuco],
        camera: Camera,
        transformations: dict[Charuco, Iterable[Transformation]] = None,
    ):
        """
        A mock reader that simulates a video feed of Charuco boards with given transformations.
        If no transformations are provided, a controller window is shown to manually adjust the
        pose of the boards.
        """
        self._boards = boards

        # Sanity checks for angles
        if transformations is None:
            self._frame_count = None
            self._current_index = None
        else:
            self._frame_count = (
                len(transformations[next(iter(transformations))])
                if transformations
                else 0
            )
            self._current_index = 0

            for charuco in transformations.keys():
                if len(transformations[charuco]) != self._frame_count:
                    raise ValueError(
                        "Number of transformations must match number of boards in all frames."
                    )
        self._transformations = transformations

        self._camera = camera
        self._dynamic_frame_ready = False

        super().__init__()

    def destroy(self):
        if self.with_gui:
            for board in self._boards:
                ctrl = f"Controle {board}"
                try:
                    cv2.destroyWindow(ctrl)
                except:
                    pass

    @property
    def with_gui(self) -> bool:
        return self._frame_count is None

    @property
    def frame_count(self) -> int | None:
        return self._frame_count

    def transformations(self, charuco: Charuco) -> Transformation:
        """
        Return the current transformation for a given Charuco board.
        """
        if self.with_gui:
            ctrl = f"Controle {charuco}"
            return Transformation(
                translation=TranslationVector(
                    x=cv2.getTrackbarPos("Trans X", ctrl) / 100,
                    y=cv2.getTrackbarPos("Trans Y", ctrl) / 100,
                    z=cv2.getTrackbarPos("Trans Z", ctrl) / 100,
                ),
                rotation=RotationMatrix.from_euler(
                    Vector3(
                        x=cv2.getTrackbarPos("Rot X", ctrl),
                        y=cv2.getTrackbarPos("Rot Y", ctrl),
                        z=cv2.getTrackbarPos("Rot Z", ctrl),
                    ),
                    sequence=RotationMatrix.Sequence.ZYX,
                    degrees=True,
                ),
            )
        else:
            return self._transformations[charuco][self._current_index - 1]

    def __iter__(self) -> "CharucoMockReader":
        if self.with_gui:
            for board in self._boards:
                ctrl = f"Controle {board}"
                cv2.namedWindow(ctrl, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(ctrl, 400, 300)

                cv2.createTrackbar("Trans X", ctrl, 0, 1000, self._has_moved)
                cv2.setTrackbarMin("Trans X", ctrl, -500)
                cv2.setTrackbarMax("Trans X", ctrl, 500)
                cv2.createTrackbar("Trans Y", ctrl, 0, 1000, self._has_moved)
                cv2.setTrackbarMin("Trans Y", ctrl, -500)
                cv2.setTrackbarMax("Trans Y", ctrl, 500)
                cv2.createTrackbar("Trans Z", ctrl, 150, 500, self._has_moved)
                cv2.setTrackbarMin("Trans Z", ctrl, 0)
                cv2.setTrackbarMax("Trans Z", ctrl, 500)
                cv2.createTrackbar("Rot X", ctrl, 0, 360, self._has_moved)
                cv2.setTrackbarMin("Rot X", ctrl, -360)
                cv2.setTrackbarMax("Rot X", ctrl, 360)
                cv2.createTrackbar("Rot Y", ctrl, 0, 360, self._has_moved)
                cv2.setTrackbarMin("Rot Y", ctrl, -360)
                cv2.setTrackbarMax("Rot Y", ctrl, 360)
                cv2.createTrackbar("Rot Z", ctrl, 0, 360, self._has_moved)
                cv2.setTrackbarMin("Rot Z", ctrl, -360)
                cv2.setTrackbarMax("Rot Z", ctrl, 360)

            self._dynamic_frame_ready = True
        else:
            self._current_index = 0

        return self

    def _has_moved(self, _) -> bool:
        self._dynamic_frame_ready = True

    def _read_frame(self):
        if self.with_gui:
            while not self._dynamic_frame_ready:
                cv2.waitKey(10)
                if cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
                    return None
            self._dynamic_frame_ready = False
        else:
            if self._current_index >= self._frame_count:
                return None
            # We increment now since the self.transformations uses the current_index - 1
            self._current_index += 1

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
        for board in self._boards:
            img = self._project_board(
                board.cv2_board_image,
                transformation=self.transformations(charuco=board),
            )
            projected_img = cv2.cvtColor(src=img, code=cv2.COLOR_GRAY2BGR)

            masked = projected_img < 255
            frame[masked] = projected_img[masked]

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

        # Corners of the img
        h, w = img.shape[:2]
        s_matrix = np.array(
            [[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], dtype=np.float32
        )

        translation = (
            transformation.translation.vector * self._camera.focal_length
        )
        rotation = transformation.rotation.matrix
        rt_matrix = np.hstack((rotation, translation))
        h_matrix = self._camera.matrix @ np.delete(rt_matrix, 2, 1)
        h_final = h_matrix @ s_matrix

        return cv2.warpPerspective(
            img,
            h_final,
            (int(self._camera.sensor_width), int(self._camera.sensor_height)),
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )
