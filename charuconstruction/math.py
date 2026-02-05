from dataclasses import dataclass
from enum import Enum

import numpy as np


class Vector3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self._vector = np.array([[x], [y], [z]])

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Vector3":
        if len(array.shape) == 1:
            array = array[:, None]

        if len(array.shape) != 2 or array.shape != (3, 1):
            raise ValueError("Array must be of shape (3, ) or (3, 1)")

        return cls(array[0, 0], array[1, 0], array[2, 0])

    @classmethod
    def zero(cls) -> "Vector3":
        return cls(0.0, 0.0, 0.0)

    @property
    def x(self) -> float:
        return self._vector[0, 0]

    @property
    def y(self) -> float:
        return self._vector[1, 0]

    @property
    def z(self) -> float:
        return self._vector[2, 0]

    def as_array(self) -> np.ndarray:
        return self._vector


class TranslationVector(Vector3):
    @property
    def vector(self) -> np.ndarray:
        return self._vector


class RotationMatrix:
    class Sequence(Enum):
        ZYX = "zyx"
        ZXY = "zxy"
        YZX = "yzx"
        YXZ = "yxz"
        XZY = "xzy"
        XYZ = "xyz"

    def __init__(self, matrix: np.ndarray) -> None:
        if matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be of shape (3, 3)")
        self._matrix = matrix

    @classmethod
    def identity(cls) -> "RotationMatrix":
        return cls(np.eye(3))

    @classmethod
    def from_euler(
        cls,
        angles: Vector3,
        sequence: "RotationMatrix.Sequence" = Sequence.XYZ,
        degrees: bool = False,
    ) -> "RotationMatrix":
        """
        Construct a RotationMatrix from Euler angles (in radians if degrees=False,
        otherwise in degrees).

        Parameters:
            angles (Vector3): The rotation angles around the X, Y, and Z axes in
            radians if degrees=False, otherwise in degrees.
            sequence (RotationSequence): The order of rotations to apply.
            degrees (bool): Whether the input angles are in degrees. Default is False (radians).
        """
        x = np.deg2rad(angles.x) if degrees else angles.x
        y = np.deg2rad(angles.y) if degrees else angles.y
        z = np.deg2rad(angles.z) if degrees else angles.z

        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(x), -np.sin(x)],
                [0, np.sin(x), np.cos(x)],
            ]
        )

        rotation_y = np.array(
            [
                [np.cos(y), 0, np.sin(y)],
                [0, 1, 0],
                [-np.sin(y), 0, np.cos(y)],
            ]
        )

        rotation_z = np.array(
            [
                [np.cos(z), -np.sin(z), 0],
                [np.sin(z), np.cos(z), 0],
                [0, 0, 1],
            ]
        )

        if sequence == RotationMatrix.Sequence.ZYX:
            return cls(rotation_z @ rotation_y @ rotation_x)
        elif sequence == RotationMatrix.Sequence.ZXY:
            return cls(rotation_z @ rotation_x @ rotation_y)
        elif sequence == RotationMatrix.Sequence.YZX:
            return cls(rotation_y @ rotation_z @ rotation_x)
        elif sequence == RotationMatrix.Sequence.YXZ:
            return cls(rotation_y @ rotation_x @ rotation_z)
        elif sequence == RotationMatrix.Sequence.XZY:
            return cls(rotation_x @ rotation_z @ rotation_y)
        elif sequence == RotationMatrix.Sequence.XYZ:
            return cls(rotation_x @ rotation_y @ rotation_z)
        else:
            raise ValueError(f"Unsupported rotation sequence: {sequence}")

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def to_euler(
        self,
        sequence: "RotationMatrix.Sequence" = Sequence.XYZ,
        degrees: bool = False,
    ) -> Vector3:
        """
        Convert the rotation matrix to Euler angles.

        Parameters:
            sequence (RotationSequence): The order of rotations to extract.
            degrees (bool): Whether to return the angles in degrees. Default is False (radians).
        Returns:
            Vector3: The rotation angles around the X, Y, and Z axes.
        """
        R = self._matrix

        if sequence == RotationMatrix.Sequence.XYZ:
            # psi = atan2(-R(2,3,:), R(3,3,:))
            # theta = asin(R(1,3,:))
            # phi = atan2(-R(1,2,:), R(1,1,:))
            x = np.arctan2(-R[1, 2], R[2, 2])
            y = np.arcsin(R[0, 2])
            z = np.arctan2(-R[0, 1], R[0, 0])

        elif sequence == RotationMatrix.Sequence.XZY:
            # psi = atan2(R(3,2,:), R(2,2,:))
            # phi = atan2(R(1,3,:), R(1,1,:))
            # theta = asin(-R(1,2,:))
            x = np.arctan2(R[2, 1], R[1, 1])
            y = np.arctan2(R[0, 2], R[0, 0])
            z = np.arcsin(-R[0, 1])

        elif sequence == RotationMatrix.Sequence.YXZ:
            # theta = asin(-R(2,3,:))
            # psi = atan2(R(1,3,:), R(3,3,:))
            # phi = atan2(R(2,1,:), R(2,2,:))
            x = np.arcsin(-R[1, 2])
            y = np.arctan2(R[0, 2], R[2, 2])
            z = np.arctan2(R[1, 0], R[1, 1])

        elif sequence == RotationMatrix.Sequence.YZX:
            # phi = atan2(-R(2,3,:), R(2,2,:))
            # psi = atan2(-R(3,1,:), R(1,1,:))
            # theta = asin(R(2,1,:))
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], R[0, 0])
            z = np.arcsin(R[1, 0])

        elif sequence == RotationMatrix.Sequence.ZXY:
            # theta = asin(R(3,2,:))
            # phi = atan2(-R(3,1,:), R(3,3,:))
            # psi = atan2(-R(1,2,:), R(2,2,:))
            x = np.arcsin(R[2, 1])
            y = np.arctan2(-R[2, 0], R[2, 2])
            z = np.arctan2(-R[0, 1], R[1, 1])

        elif sequence == RotationMatrix.Sequence.ZYX:
            # phi = atan2(R(3,2,:), R(3,3,:))
            # theta = asin(-R(3,1,:))
            # psi = atan2(R(2,1,:), R(1,1,:))
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arcsin(-R[2, 0])
            z = np.arctan2(R[1, 0], R[0, 0])

        else:
            raise NotImplementedError(
                f"Sequence {sequence} not implemented yet."
            )

        if degrees:
            x = np.rad2deg(x)
            y = np.rad2deg(y)
            z = np.rad2deg(z)

        return Vector3(x, y, z)

    def as_array(self) -> np.ndarray:
        return self._matrix

    def to_rodrigues(self) -> np.ndarray:
        """
        Convert the rotation matrix to Rodrigues vector.

        Returns:
            np.ndarray: The Rodrigues vector (3x1).
        """
        theta = np.arccos((np.trace(self._matrix) - 1) / 2)
        if theta == 0:
            return np.zeros((3, 1))
        else:
            rx = (self._matrix[2, 1] - self._matrix[1, 2]) / (2 * np.sin(theta))
            ry = (self._matrix[0, 2] - self._matrix[2, 0]) / (2 * np.sin(theta))
            rz = (self._matrix[1, 0] - self._matrix[0, 1]) / (2 * np.sin(theta))
            r = np.array([[rx], [ry], [rz]])
            return r * theta


class Transformation:
    def __init__(
        self,
        translation: TranslationVector = TranslationVector(0, 0, 0),
        rotation: RotationMatrix = RotationMatrix(np.eye(3)),
    ) -> None:
        self.translation = translation
        self.rotation = rotation
