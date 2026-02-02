import random

import cv2


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

        self._dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)

        aruco_indices = list(range(0, vertical_squares_count * horizontal_squares_count))
        rng = random.Random(seed)
        rng.shuffle(aruco_indices)
        self._dictionary.bytesList = self._dictionary.bytesList[aruco_indices]

        self._board = cv2.aruco.CharucoBoard(
            (self._vert_count, self._horz_count),
            self._square_len,
            self._marker_len,
            self._dictionary,
        )
        self._board_image = cv2.aruco.CharucoBoard.generateImage(
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

    def show(self, save_path: str = None):
        cv2.imshow("img", self._board_image)
        if save_path is not None:
            cv2.imwrite(save_path, self._board_image)
            cv2.waitKey(2000)
        else:
            cv2.waitKey()
