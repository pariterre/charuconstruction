import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv_dart.dart';

enum CameraModels {
  pixel2,
  ios;

  @override
  String toString() {
    return switch (this) {
      CameraModels.pixel2 => 'pixel2',
      CameraModels.ios => 'ios',
    };
  }

  static CameraModels fromString(String name) =>
      CameraModels.values.firstWhere((e) => e.toString() == name);

  Camera toCamera({bool useVideoParameters = false, bool isPortrait = false}) {
    return switch (this) {
      CameraModels.pixel2 => Camera.fromPhone(
        name: toString(),
        focalLength: 3.4,
        pixelSize: 0.0014,
        sensorWidth: 4032.0,
        sensorHeight: 3024.0 * 0.75,
        distorsionCoefficients: const [-0.05, 0.02, 0.0, 0.0, 0.0],
        useVideoParameters: useVideoParameters,
        isPortrait: isPortrait,
      ),
      CameraModels.ios => Camera.fromPhone(
        name: toString(),
        focalLength: 3.4,
        pixelSize: 0.0014,
        sensorWidth: 4032.0,
        sensorHeight: 3024.0 * 0.75,
        distorsionCoefficients: const [-0.05, 0.02, 0.0, 0.0, 0.0],
        useVideoParameters: useVideoParameters,
        isPortrait: isPortrait,
      ),
    };
  }
}

class Camera {
  ///
  /// The name of the camera. If CameraModels was used, this corresponds to the
  /// toString() output
  ///
  final String name;

  ///
  /// Whether the camera is in portrait (vertical) orientation or landscape (horizontal) orientation.
  ///
  final bool isPortrait;

  ///
  /// Focal length of the camera in pixels.
  ///
  final double focalLength;

  ///
  /// Width of the camera sensor in millimeters.
  ///
  final double sensorWidth;

  ///
  /// Height of the camera sensor in millimeters.
  ///
  final double sensorHeight;

  ///
  /// Camera matrix (3x3).
  ///
  final List<List<double>> matrix;
  Mat get matrixAsMat => Mat.from2DList(matrix, MatType(MatType.CV_64F));
  Matrix get matrixAsLinalg => Matrix.fromList(matrix);

  ///
  /// Distortion coefficients (1x5).
  ///
  final List<double> distorsionCoefficients;
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
  /// [isPortrait] indicates whether the camera is in portrait (vertical) orientation (which swaps the sensor width and height).
  factory Camera.fromPhone({
    required String name,
    required double focalLength,
    required double pixelSize,
    required double sensorWidth,
    required double sensorHeight,
    List<double> distorsionCoefficients = const [0.0, 0.0, 0.0, 0.0, 0.0],
    bool useVideoParameters = false,
    bool isPortrait = false,
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

    if (isPortrait) {
      // Swap width and height for portrait orientation
      final temp = sensorWidth;
      sensorWidth = sensorHeight;
      sensorHeight = temp;
    }

    return Camera._(
      name: name,
      focalLength: focalLengthInPixels,
      sensorWidth: sensorWidth,
      sensorHeight: sensorHeight,
      matrix: defaultCameraMatrix(
        focalLengthInPixels,
        sensorWidth,
        sensorHeight,
      ),
      distorsionCoefficients: distorsionCoefficients,
      isPortrait: isPortrait,
    );
  }

  Camera._({
    required this.name,
    required this.focalLength,
    required this.sensorWidth,
    required this.sensorHeight,
    required this.matrix,
    required this.distorsionCoefficients,
    required this.isPortrait,
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
