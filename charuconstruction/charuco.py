import json
from pathlib import Path
import random

import cv2
import numpy as np

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
        self._dictionary = cv2.aruco.getPredefinedDictionary(self._predefined_aruco_dict)

        aruco_indices = list(range(0, vertical_squares_count * horizontal_squares_count))
        self._random_seed = seed
        rng = random.Random(seed)
        rng.shuffle(aruco_indices)
        self._dictionary.bytesList = self._dictionary.bytesList[aruco_indices]

        self._board = cv2.aruco.CharucoBoard(
            (self._vert_count, self._horz_count),
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
        return self._vert_count * self._horz_count

    @property
    def vertical_squares_count(self) -> int:
        return self._vert_count

    @property
    def horizontal_squares_count(self) -> int:
        return self._horz_count

    @property
    def square_ratio(self) -> float:
        return self._horz_count / self._vert_count

    @property
    def aruco_ids(self) -> list[int]:
        return self._board.getIds().flatten().tolist()

    @property
    def random_seed(self) -> int | None:
        return self._random_seed

    @property
    def cv2_board_image(self) -> np.ndarray:
        return self._board_image

    def show(self) -> None:
        cv2.imshow("img", self._board_image)
        cv2.waitKey()

    @classmethod
    def load(cls, load_folder: Path) -> "Charuco":
        param_path = load_folder / "board.json"
        params = json.load(open(param_path, "r"))
        return cls(**params)

    def save(self, save_folder: Path, override: bool = False) -> None:
        save_folder.mkdir(parents=True, exist_ok=False)

        json.dump(self.serialize(), open(save_folder / "board.json", "w"), indent=2)
        cv2.imwrite(save_folder / "board.png", self._board_image)

    def serialize(self) -> dict:
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

    def detect(self, frame: Frame) -> Frame:
        """
        Detect markers and ChArUco corners in the given image.

        Parameters:
            frame (Frame): Frame to detect markers and corners from.
        Returns:
            Frame: Updated frame with detected markers and corners drawn.
        """
        grayscale_frame = frame.get(grayscale=True)
        output_frame = frame.get().copy()

        result = _detect_marker_corners(grayscale_frame, self)
        if result is None:
            return frame
        corners, ids, charuco_corners, charuco_ids = result

        # Draw the detected markers and corners of the corresponding Charuco board
        cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)
        _draw_detected_corners_charuco_own(output_frame, charuco_corners, charuco_ids)

        # Show the axes of reference

        # Default values
        height, width = output_frame.shape[:2]
        focal_length = width  # focal length ~ image width in pixels
        camera_matrix = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]], dtype=float)
        dist_coeffs = np.zeros(5)
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)

        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self._board, camera_matrix, dist_coeffs, rvec, tvec
        )
        if ret > 0:
            axis_length = 0.05  # Length of axes in your units (e.g., meters)
            cv2.drawFrameAxes(output_frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

        return Frame(output_frame)


def _detect_marker_corners(
    frame: np.ndarray, charuco: Charuco
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray] | None:
    corners, ids = _detect_markers(frame, charuco)
    if not corners or ids is None:
        return None

    # Read chessboard corners between markers
    corner_counts, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, frame, charuco._board
    )
    if corner_counts == 0:
        return None

    return corners, ids, charuco_corners, charuco_ids


def _detect_markers(frame: np.ndarray, charuco: Charuco) -> tuple[list[np.ndarray], np.ndarray | None]:
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
    mid_point = np.mean([np.mean(corner[:, 0, :], axis=0) for corner in corners_tp], axis=0)

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
