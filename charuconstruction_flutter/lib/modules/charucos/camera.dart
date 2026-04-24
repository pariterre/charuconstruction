import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv_dart.dart';

enum CameraModels {
  pixel2;

  @override
  String toString() {
    return switch (this) {
      CameraModels.pixel2 => 'pixel2',
    };
  }

  Camera toCamera({bool useVideoParameters = false, bool isVertical = false}) {
    return switch (this) {
      CameraModels.pixel2 => Camera.fromPhone(
        focalLength: 3.4,
        pixelSize: 0.0014,
        sensorWidth: 4032.0,
        sensorHeight:
            3024.0 * 0.75, // TODO Validate why the ratio is applied twice
        distorsionCoefficients: const [-0.05, 0.02, 0.0, 0.0, 0.0],
        useVideoParameters: useVideoParameters,
        isVertical: isVertical,
      ),
    };
  }
}

class Camera {
  ///
  /// Focal length of the camera in pixels.
  ///
  double focalLength;

  ///
  /// Width of the camera sensor in millimeters.
  ///
  double sensorWidth;

  ///
  /// Height of the camera sensor in millimeters.
  ///
  double sensorHeight;

  ///
  /// Camera matrix (3x3).
  ///
  List<List<double>> matrix;
  Mat get matrixAsMat => Mat.from2DList(matrix, MatType(MatType.CV_64F));
  Matrix get matrixAsLinalg => Matrix.fromList(matrix);

  ///
  /// Distortion coefficients (1x5).
  ///
  List<double> distorsionCoefficients;
  Mat get distorsionCoefficientsAsMat => Mat.fromList(
    1,
    distorsionCoefficients.length,
    MatType(MatType.CV_64F),
    distorsionCoefficients,
  );

  ///
  /// Constructor for a phone camera with default parameters.
  /// [focalLength] is the focal length in centimeters (default is 3.4 cm).
  /// [pixelSize] is the pixel size in centimeters (default is 0.0014 cm, which is 1.4 micrometers).
  /// [useVideoParameters] indicates whether the camera is used as a video camera
  /// (which applies a crop factor to the focal length and sensor size assuming 1080p video).
  /// [isVertical] indicates whether the camera is in vertical orientation (which swaps the sensor width and height).
  factory Camera.fromPhone({
    required double focalLength,
    required double pixelSize,
    required double sensorWidth,
    required double sensorHeight,
    List<double> distorsionCoefficients = const [0.0, 0.0, 0.0, 0.0, 0.0],
    bool useVideoParameters = false,
    bool isVertical = false,
  }) {
    var focalLengthInPixels = focalLength / pixelSize;

    if (useVideoParameters) {
      final videoWidth = 3840.0; // 4K video width in pixels (1920x1080 * 2)
      var videoCropFactor = videoWidth / sensorWidth;
      videoCropFactor *= 0.5; // 1080p video crop factor

      focalLengthInPixels *= videoCropFactor;
      sensorWidth *= videoCropFactor;
      sensorHeight *= videoCropFactor;
    }

    if (isVertical) {
      // Swap width and height for vertical orientation
      final temp = sensorWidth;
      sensorWidth = sensorHeight;
      sensorHeight = temp;
    }

    return Camera._(
      focalLength: focalLengthInPixels,
      sensorWidth: sensorWidth,
      sensorHeight: sensorHeight,
      matrix: defaultCameraMatrix(
        focalLengthInPixels,
        sensorWidth,
        sensorHeight,
      ),
      distorsionCoefficients: distorsionCoefficients,
    );
  }

  Camera._({
    required this.focalLength,
    required this.sensorWidth,
    required this.sensorHeight,
    required this.matrix,
    required this.distorsionCoefficients,
  });
}

List<List<double>> defaultCameraMatrix(
  double focalLength,
  double sensorWidth,
  double sensorHeight,
) {
  return [
    [focalLength, 0.0, sensorWidth / 2.0],
    [0.0, focalLength, sensorHeight / 2.0],
    [0.0, 0.0, 1.0],
  ];
}
