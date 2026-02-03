from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    """
    Camera calibration parameters.

    Attributes:
        focal_length (float): Focal length of the camera in pixels.
        sensor_width (float): Width of the camera sensor in millimeters.
        sensor_height (float): Height of the camera sensor in millimeters.
        matrix (np.ndarray): Camera matrix (3x3).
        distorsion_coefficients (np.ndarray): Distortion coefficients (1x5).
    """

    focal_length: float
    sensor_width: float
    sensor_height: float
    matrix: np.ndarray
    distorsion_coefficients: np.ndarray

    @classmethod
    def default(
        cls,
        focal_length: float = 1000.0,
        sensor_width: float = 36.0,
        sensor_height: float = 24.0,
    ) -> "Camera":
        matrix = np.array(
            [
                [focal_length, 0.0, sensor_width / 2.0],
                [0.0, focal_length, sensor_height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        return cls(
            focal_length, sensor_width, sensor_height, matrix, dist_coeffs
        )
