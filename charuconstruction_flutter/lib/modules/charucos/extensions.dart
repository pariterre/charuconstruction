import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv_dart.dart';

extension VectorExtensions on Vector {
  /// Converts a rotation vector in rodriguez representation to a rotation matrix.
  Mat toMat() => Mat.fromList(length, 1, MatType(MatType.CV_32F), toList());
}

extension MatrixExtensions on Matrix {
  /// Converts a rotation matrix to a rotation vector in rodriguez representation.
  Mat toMat() => Mat.fromList(
    rowCount,
    columnCount,
    MatType(MatType.CV_32F),
    toList().map((e) => e.toList()).expand((i) => i).toList(),
  );
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
