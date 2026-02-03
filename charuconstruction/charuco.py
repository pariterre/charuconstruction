import json
from pathlib import Path
import random

import cv2
import numpy as np


from .camera import Camera
from .media_reader import Frame


class Charuco:
    def __init__(
        self,
        vertical_squares_count: int,
        horizontal_squares_count: int,
        square_len: float,
        marker_len: float,
        page_len: int,
        page_margin: int,
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
            page_len (int): Total length of the page in pixels when generating the board image.
            page_margin (int): Margin size in pixels when generating the board image.
            aruco_dict (int): Predefined ArUco dictionary to use.
            seed (int): Seed for random number generator (will determine the order of aruco markers).
        """

        self._vert_count = vertical_squares_count
        self._horz_count = horizontal_squares_count
        self._square_len = square_len
        self._marker_len = marker_len
        self._page_margin = page_margin
        self._page_len = page_len

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
        self._board_image: np.ndarray = cv2.aruco.CharucoBoard.generateImage(
            self._board,
            (self._page_len, int(self._page_len * self.square_ratio)),
            marginSize=self._page_margin,
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
            "page_len": self._page_len,
            "page_margin": self._page_margin,
            "aruco_dict": self._predefined_aruco_dict,
            "seed": self._random_seed,
        }

    def detect(
        self,
        frame: Frame,
        camera: Camera,
        initial_guess: tuple[np.ndarray, np.ndarray] = None,
    ) -> tuple[Frame, tuple[np.ndarray, np.ndarray]]:
        """
        Detect markers and ChArUco corners in the given image.

        Parameters:
            frame (Frame): Frame to detect markers and corners from.
            camera (Camera): Camera parameters for pose estimation.
            initial_guess (tuple[np.ndarray, np.ndarray]): Initial guess for translation and rotations vectors.
        Returns:
            tuple[Frame, tuple[np.ndarray, np.ndarray]]: Updated frame with detected markers and corners drawn, translation vector, and rotation vector.
        """
        grayscale_frame = frame.get(grayscale=True)
        output_frame = frame.get().copy()

        result = _detect_marker_corners(grayscale_frame, self)
        if result is None:
            return frame, (None, None)
        corners, ids, charuco_corners, charuco_ids = result

        # Draw the detected markers and corners of the corresponding Charuco board
        cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)
        _draw_detected_corners_charuco_own(
            output_frame, charuco_corners, charuco_ids
        )

        # Show the axes of reference
        # Default values
        axis_length = 1.0  # Length of axes in meters
        translation_initial_guess, rotation_initial_guess = (
            (np.zeros((3, 1)), np.zeros((3, 1)))
            if initial_guess is None
            else initial_guess
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
        if ret > 0:
            cv2.drawFrameAxes(
                output_frame,
                camera.matrix,
                camera.distorsion_coefficients,
                rotation,
                translation,
                axis_length,
            )
        return Frame(output_frame), (translation, rotation)


def _detect_marker_corners(
    frame: np.ndarray, charuco: Charuco
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Detect markers and ChArUco corners in the given image.
    """
    corners, ids = _detect_markers(frame, charuco)
    if not corners or ids is None:
        return None

    # Read chessboard corners between markers
    corner_counts, charuco_corners, charuco_ids = (
        cv2.aruco.interpolateCornersCharuco(corners, ids, frame, charuco._board)
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
