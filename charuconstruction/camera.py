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
    def generic_iphone_camera(
        cls,
        focal_length: float = 2200,
        sensor_width: float = 3000.0,
        sensor_height: float = 1800.0,
    ) -> "Camera":
        return cls(
            focal_length=focal_length,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            matrix=default_camera_matrix(
                focal_length, sensor_width, sensor_height
            ),
            distorsion_coefficients=np.zeros(5, dtype=np.float32),
        )

    @classmethod
    def pixel2_camera(
        cls, use_video_parameters: bool = False, is_vertical: bool = False
    ) -> "Camera":
        focal_length_cm = 3.4  # Focal length in centimeters
        pixel_size_cm = 0.0014  # Pixel size in centimeters (1.4 micrometers)
        focal_length = focal_length_cm / pixel_size_cm  # Focal length in pixels

        sensor_width = 4032.0
        raw_sensor_height = 3024.0
        height_crop_factor = 0.75
        sensor_height = raw_sensor_height * height_crop_factor

        if use_video_parameters:
            video_width = 3840.0  # 4K video width in pixels (1920x1080 * 2)
            video_crop_factor = video_width / sensor_width
            # video_crop_factor *= 0.5  # 1080p video crop factor

            focal_length *= video_crop_factor
            sensor_width *= video_crop_factor
            sensor_height *= video_crop_factor

        if is_vertical:
            sensor_width, sensor_height = sensor_height, sensor_width

        return cls(
            focal_length=focal_length,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            matrix=default_camera_matrix(
                focal_length, sensor_width, sensor_height
            ),
            distorsion_coefficients=np.array((-0.05, 0.02, 0.0, 0.0, 0.0)),
        )


def default_camera_matrix(
    focal_length: float, sensor_width: float, sensor_height: float
) -> np.ndarray:
    return np.array(
        [
            [focal_length, 0.0, sensor_width / 2.0],
            [0.0, focal_length, sensor_height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
