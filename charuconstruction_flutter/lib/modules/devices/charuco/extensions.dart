import 'dart:math';

import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv_dart.dart';

extension VectorExtensions on Vector {
  /// Converts a rotation vector in rodriguez representation to a rotation matrix.
  Mat toMat() => Mat.fromList(length, 1, MatType(MatType.CV_32F), toList());
}

enum CharucoAxis { x, y, z }

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
