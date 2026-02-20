import json
from pathlib import Path
import random

import cv2
import numpy as np


from .camera import Camera
from .media_reader import Frame
from .math import TranslationVector, RotationMatrix
from .resolution import Resolution


class Charuco:
    def __init__(
        self,
        vertical_squares_count: int,
        horizontal_squares_count: int,
        square_len: float,
        marker_len: float,
        resolution: Resolution | str,
        aruco_dict: int = cv2.aruco.DICT_7X7_1000,
        seed: int = None,
    ):
        """
        Initialize a ChArUco board with the given parameters.

        Parameters:
            vertical_squares_count (int): Number of squares in the vertical direction.
            horizontal_squares_count (int): Number of squares in the horizontal direction.
            square_len (float): Length of the squares in meters.
            marker_len (float): Length of the ArUco markers in meters (cannot be greater than square_len).
            resolution (Resolution | str): The number of pixels in one inch on the printed page.
                If a string is provided, it should be the name of a Resolution enum member (e.g., "DPI_100").
            aruco_dict (int): Predefined ArUco dictionary to use.
            seed (int): Seed for random number generator (will determine the order of aruco markers).
        """

        if marker_len > square_len:
            raise ValueError(
                "The marker length cannot be greater than the square length."
            )

        self._vert_count = vertical_squares_count
        self._horz_count = horizontal_squares_count
        self._square_len = square_len
        self._marker_len = marker_len
        self._resolution = (
            resolution
            if isinstance(resolution, Resolution)
            else Resolution.from_serialized(resolution)
        )

        self._predefined_aruco_dict = aruco_dict
        self._dictionary = cv2.aruco.getPredefinedDictionary(
            self._predefined_aruco_dict
        )

        aruco_indices = list(
            range(0, horizontal_squares_count * vertical_squares_count)
        )
        self._random_seed = seed
        rng = random.Random(seed)
        rng.shuffle(aruco_indices)
        self._dictionary.bytesList = self._dictionary.bytesList[aruco_indices]

        self._board = cv2.aruco.CharucoBoard(
            (self._horz_count, self._vert_count),
            self._square_len,
            self._marker_len,
            self._dictionary,
        )

        if self._horz_count >= self._vert_count:
            board_len = int(
                self._resolution.value * self._square_len * self._horz_count
            )
            img_len = (board_len, int(board_len * self.square_ratio))
        else:
            board_len = int(
                self._resolution.value * self._square_len * self._vert_count
            )
            img_len = (int(board_len / self.square_ratio), board_len)
        self._board_image: np.ndarray = cv2.aruco.CharucoBoard.generateImage(
            self._board, outSize=img_len, marginSize=0
        )

    @property
    def squares_count(self) -> int:
        """
        Get the total number of squares in the Charuco board.
        """
        return self._horz_count * self._vert_count

    @property
    def horizontal_squares_count(self) -> int:
        """
        Get the number of squares in the horizontal direction.
        """
        return self._horz_count

    @property
    def vertical_squares_count(self) -> int:
        """
        Get the number of squares in the vertical direction.
        """
        return self._vert_count

    @property
    def width_len(self) -> float:
        """
        Get the total width of the Charuco board in meters.
        """
        return self._horz_count * self._square_len

    @property
    def height_len(self) -> float:
        """
        Get the total height of the Charuco board in meters.
        """
        return self._vert_count * self._square_len

    @property
    def square_ratio(self) -> float:
        """
        Get the ratio of vertical to horizontal squares.
        """
        return self._vert_count / self._horz_count

    @property
    def aruco_ids(self) -> list[int]:
        """
        Get the list of ArUco marker IDs used in the Charuco board.
        """
        return self._board.getIds().flatten().tolist()

    @property
    def random_seed(self) -> int | None:
        """
        Get the random seed used for shuffling the ArUco markers.
        """
        return self._random_seed

    @property
    def cv2_board_image(self) -> np.ndarray:
        """
        Get the generated Charuco board image as a cv2 image.
        """
        return self._board_image

    def show(self) -> None:
        """
        Show the generated Charuco board image in a window.
        """
        cv2.imshow("img", self._board_image)
        cv2.waitKey()

    @classmethod
    def load(cls, load_folder: Path) -> "Charuco":
        """
        Load a Charuco board from a folder containing the board parameters and image.

        Parameters:
            load_folder (Path): Path to the folder containing the board parameters and image.

        Returns:
            Charuco: Loaded Charuco board object.
        """
        param_path = load_folder / "board.json"
        params = json.load(open(param_path, "r"))
        return cls(**params)

    def save(self, save_folder: Path, override: bool = False) -> None:
        """
        Save the Charuco board parameters and image to a folder.

        Parameters:
            save_folder (Path): Path to the folder where the board parameters and image will be saved
            override (bool): Whether to override the folder if it already exists.
        """
        if not override and save_folder.exists():
            raise FileExistsError(
                f"The folder {save_folder} already exists. Use override=True to overwrite."
            )
        save_folder.mkdir(parents=True, exist_ok=True)

        json.dump(
            self.serialize(), open(save_folder / "board.json", "w"), indent=2
        )
        cv2.imwrite(save_folder / "board.png", self._board_image)

    def serialize(self) -> dict:
        """
        Serialize the Charuco board parameters to a dictionary.

        Returns:
            dict: Dictionary containing the Charuco board parameters.
        """
        return {
            "vertical_squares_count": self._vert_count,
            "horizontal_squares_count": self._horz_count,
            "square_len": self._square_len,
            "marker_len": self._marker_len,
            "resolution": self._resolution.serialized,
            "aruco_dict": self._predefined_aruco_dict,
            "seed": self._random_seed,
        }

    def detect(
        self,
        frame: Frame,
        camera: Camera,
        translation_initial_guess: TranslationVector = None,
        rotation_initial_guess: RotationMatrix = None,
    ) -> tuple[TranslationVector | None, RotationMatrix | None]:
        """
        Detect markers and ChArUco corners in the given image.

        Parameters:
            frame (Frame): Frame to detect markers and corners from.
            camera (Camera): Camera parameters for pose estimation.
            translation_initial_guess (TranslationVector): Initial guess
              for translation vector.
            rotation_initial_guess (RotationMatrix): Initial guess
              for rotation matrix.
        Returns:
            tuple[TranslationVector | None, RotationMatrix | None]: translation vector and rotation matrix.
        """
        result = _detect_marker_corners(frame, self)
        if result is None:
            return None, None
        _, _, charuco_corners, charuco_ids = result

        # Default values
        translation_initial_guess = (
            translation_initial_guess.as_array()
            if translation_initial_guess is not None
            else np.zeros((3, 1))
        )
        rotation_initial_guess = (
            rotation_initial_guess.to_rodrigues()
            if rotation_initial_guess is not None
            else np.zeros((3, 1))
        )

        ret, rotation, translation = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            self._board,
            camera.matrix,
            camera.distorsion_coefficients,
            rotation_initial_guess,
            translation_initial_guess,
        )
        return (
            TranslationVector.from_array(translation) if ret > 0 else None,
            RotationMatrix(cv2.Rodrigues(rotation)[0]) if ret > 0 else None,
        )

    def draw_aruco_markers(self, frame: Frame) -> None:
        result = _detect_marker_corners(frame, self)
        if result is None:
            return
        corners, ids, charuco_corners, charuco_ids = result

        # Draw the detected markers and corners of the corresponding Charuco board
        cv2.aruco.drawDetectedMarkers(frame.get(), corners, ids)
        _draw_detected_corners_charuco_own(
            frame.get(), charuco_corners, charuco_ids
        )

    def draw_estimated_pose_axes(
        self,
        frame: Frame,
        camera: Camera,
        translation: TranslationVector,
        rotation: RotationMatrix,
        axes_length: float = 1.0,
    ) -> None:
        """
        Draw the axes of reference corresponding to the given pose estimation.

        Parameters:
            frame (Frame): Frame to draw the axes on.
            camera (Camera): Camera parameters for pose estimation.
            translation (TranslationVector): Translation vector of the pose estimation.
            rotation (RotationMatrix): Rotation matrix of the pose estimation.
            axes_length (float): Length in meter of the axes to draw in the output frame.
        """
        cv2.drawFrameAxes(
            frame.get(),
            camera.matrix,
            camera.distorsion_coefficients,
            rotation.to_rodrigues(),
            translation.as_array(),
            axes_length,
        )


def _detect_marker_corners(
    frame: Frame, charuco: Charuco
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Detect markers and ChArUco corners in the given image.
    """
    grayscale_frame = frame.get(grayscale=True)
    corners, ids = _detect_markers(grayscale_frame, charuco)
    if not corners or ids is None:
        return None

    # Read chessboard corners between markers
    corner_counts, charuco_corners, charuco_ids = (
        cv2.aruco.interpolateCornersCharuco(
            corners, ids, grayscale_frame, charuco._board
        )
    )
    if corner_counts == 0:
        return None

    return corners, ids, charuco_corners, charuco_ids


def _detect_markers(
    frame: np.ndarray, charuco: Charuco
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """
    Detect ArUco markers in the given image, filtering to only include those belonging to the specified Charuco board.
    """
    corners, ids, _ = cv2.aruco.detectMarkers(frame, charuco._dictionary)
    if not corners or ids is None:
        return [], None

    # Filter out the markers to only include those that belong to the current Charuco board
    # Strategy :
    #   1 - Remove all the markers which are not in the current board
    #   2 - Keep only unique markers IDs and compute the mid point of that reduced board
    #   3 - Reinsert all the markers which appear multiple times (other boards on screen) and
    #       remove them based on nearest distance to the mid-board position
    aruco_ids = charuco.aruco_ids
    corners_tp = []
    ids_tp = []
    for corner, id in zip(corners, ids):
        if id in aruco_ids:
            corners_tp.append(corner)
            ids_tp.append(id)
    ids = np.array(ids_tp)
    corners = corners_tp

    if len(ids) == 0:
        return [], None
    non_unique_ids: list[int] = ids[:, 0].tolist()
    unique_ids: set[int] = set(ids[:, 0].tolist())
    ids_tp = []
    corners_tp = []
    for id in unique_ids:
        if non_unique_ids.count(id) == 1:
            index = non_unique_ids.index(id)
            ids_tp.append(ids[index])
            corners_tp.append(corners[index])
    mid_point = np.mean(
        [np.mean(corner[:, 0, :], axis=0) for corner in corners_tp], axis=0
    )

    for id in unique_ids:
        if non_unique_ids.count(id) == 1:
            # Already processed
            continue

        min_distance = float("inf")
        best_so_far = -1
        all_indices = [i for i, cid in enumerate(ids) if cid[0] == id]
        for i in all_indices:
            corner_center = np.mean(corners[i].squeeze(), axis=0)
            distance = np.linalg.norm(corner_center - mid_point)
            if distance < min_distance:
                min_distance = distance
                best_so_far = i
        ids_tp.append(ids[best_so_far])
        corners_tp.append(corners[best_so_far])
    ids = np.array(ids_tp)
    corners = corners_tp

    return corners, ids


def _draw_detected_corners_charuco_own(img, corners, ids):
    """
    Draw rectangles and IDs to the corners
    """

    rect_size = 5
    id_font = cv2.FONT_HERSHEY_SIMPLEX
    id_scale = 0.5
    id_color = (255, 255, 0)
    rect_thickness = 1

    # Draw rectangels and IDs
    for corner, id in zip(corners, ids):
        corner_x = int(corner[0][0])
        corner_y = int(corner[0][1])
        id_text = "Id: {}".format(str(id[0]))
        id_coord = (corner_x + 2 * rect_size, corner_y + 2 * rect_size)
        cv2.rectangle(
            img,
            (corner_x - rect_size, corner_y - rect_size),
            (corner_x + rect_size, corner_y + rect_size),
            id_color,
            thickness=rect_thickness,
        )
        cv2.putText(img, id_text, id_coord, id_font, id_scale, id_color)
