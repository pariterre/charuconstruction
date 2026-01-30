import cv2


class Charuco:
    def __init__(
        self, vertical_squares_count: int, horizontal_squares_count: int, square_length: float, marker_length: float
    ):
        """
        Initialize a ChArUco board with the given parameters.

        Parameters:
          vertical_squares_count (int): Number of squares in the vertical direction.
          horizontal_squares_count (int): Number of squares in the horizontal direction.
          square_length (float): Length of the squares in meters.
          marker_length (float): Length of the ArUco markers in meters.
        """

        # https://medium.com/@ed.twomey1/using-charuco-boards-in-opencv-237d8bc9e40d
        self._vertical_squares = vertical_squares_count
        self._horizontal_squares = horizontal_squares_count
        self.square_length = square_length
        self.marker_length = marker_length
        self._aruco_dict = cv2.aruco.DICT_6X6_250
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dict)
        self.board = cv2.aruco.CharucoBoard(
            (self.horizontal_squares, self.vertical_squares), self.square_length, self.marker_length, self.dictionary
        )
