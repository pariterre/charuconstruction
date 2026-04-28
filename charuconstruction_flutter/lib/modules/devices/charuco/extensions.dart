import 'dart:math';

import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv_dart.dart';

extension VectorExtensions on Vector {
  /// Converts a rotation vector in rodriguez representation to a rotation matrix.
  Mat toMat() => Mat.fromList(length, 1, MatType(MatType.CV_32F), toList());
}

enum CharucoAxis { x, y, z }

enum CharucoAxisSequence { xyz, xzy, yxz, yzx, zxy, zyx }

extension MatrixExtensions on Matrix {
  /// Converts a rotation matrix to a rotation vector in rodriguez representation.
  Mat toMat() => Mat.fromList(
    rowCount,
    columnCount,
    MatType(MatType.CV_32F),
    toList().map((e) => e.toList()).expand((i) => i).toList(),
  );

  static Matrix fromEuler(
    List<(double, CharucoAxis)> angles, {
    bool degrees = true,
  }) {
    final rotationMatrices = angles.map((angle) {
      final (value, axis) = angle;
      final rad = degrees ? value * pi / 180 : value;

      return switch (axis) {
        CharucoAxis.x => Matrix.fromList([
          [1, 0, 0],
          [0, cos(rad), -sin(rad)],
          [0, sin(rad), cos(rad)],
        ]),
        CharucoAxis.y => Matrix.fromList([
          [cos(rad), 0, sin(rad)],
          [0, 1, 0],
          [-sin(rad), 0, cos(rad)],
        ]),
        CharucoAxis.z => Matrix.fromList([
          [cos(rad), -sin(rad), 0],
          [sin(rad), cos(rad), 0],
          [0, 0, 1],
        ]),
      };
    });
    return rotationMatrices.reduce((a, b) => a * b);
  }

  ///
  /// Convert the rotation matrix to Euler angles.
  /// [sequence] The order of the rotations. Default is XYZ.
  /// [degrees] Whether to return the angles in degrees. Default is False (radians).
  /// Returns the rotation angles around the X, Y, and Z axes.
  ///
  Vector toEuler({
    CharucoAxisSequence sequence = CharucoAxisSequence.xyz,
    bool degrees = true,
  }) {
    final (x, y, z) = switch (sequence) {
      // psi = atan2(-R(2,3,:), R(3,3,:))
      // theta = asin(R(1,3,:))
      // phi = atan2(-R(1,2,:), R(1,1,:))
      CharucoAxisSequence.xyz => (
        atan2(-this[1][2], this[2][2]),
        asin(this[0][2]),
        atan2(-this[0][1], this[0][0]),
      ),

      // psi = atan2(R(3,2,:), R(2,2,:))
      // phi = atan2(R(1,3,:), R(1,1,:))
      // theta = asin(-R(1,2,:))
      CharucoAxisSequence.xzy => (
        atan2(this[2][1], this[1][1]),
        atan2(this[0][2], this[0][0]),
        asin(-this[0][1]),
      ),

      // theta = asin(-R(2,3,:))
      // psi = atan2(R(1,3,:), R(3,3,:))
      // phi = atan2(R(2,1,:), R(2,2,:))
      CharucoAxisSequence.yxz => (
        asin(-this[1][2]),
        atan2(this[0][2], this[2][2]),
        atan2(this[1][0], this[1][1]),
      ),

      // phi = atan2(-R(2,3,:), R(2,2,:))
      // psi = atan2(-R(3,1,:), R(1,1,:))
      // theta = asin(R(2,1,:))
      CharucoAxisSequence.yzx => (
        atan2(-this[1][2], this[1][1]),
        atan2(-this[2][0], this[0][0]),
        asin(this[1][0]),
      ),

      // theta = asin(R(3,2,:))
      // phi = atan2(-R(3,1,:), R(3,3,:))
      // psi = atan2(-R(1,2,:), R(2,2,:))
      CharucoAxisSequence.zxy => (
        asin(this[2][1]),
        atan2(-this[2][0], this[2][2]),
        atan2(-this[0][1], this[1][1]),
      ),

      // phi = atan2(R(3,2,:), R(3,3,:))
      // theta = asin(-R(3,1,:))
      // psi = atan2(R(2,1,:), R(1,1,:))
      CharucoAxisSequence.zyx => (
        atan2(this[2][1], this[2][2]),
        asin(-this[2][0]),
        atan2(this[1][0], this[0][0]),
      ),
    };

    return Vector.fromList([x, y, z]) * (degrees ? (180 / pi) : 1);
  }
}

extension MatExtensions on Mat {
  /// Converts a rotation matrix to a rotation vector in rodriguez representation.
  Vector toVector() {
    if (cols != 1) {
      throw ArgumentError('Expected a column vector (Nx1), got ${rows}x$cols');
    }
    return Vector.fromList(List<double>.generate(rows, (i) => at(i, 0)));
  }

  Matrix toMatrix() {
    return Matrix.fromList(toList().cast<List<double>>());
  }
}
