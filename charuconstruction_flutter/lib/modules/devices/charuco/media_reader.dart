import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:logging/logging.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv.dart';

import 'camera.dart';
import 'charuco.dart';
import 'extensions.dart';
import 'frame.dart';

final _logger = Logger("MediaReader");

abstract class MediaReader {
  ///
  /// Read frames from the media source. This method returns a stream of frames t
  /// hat can be listened to for real-time processing.
  ///
  Stream<Frame?> readFrames();

  ///
  /// Dispose of any resources used by the media reader. This should be called
  /// when the reader is no longer needed to free up memory.
  ///
  void dispose();
}

class ImageReader implements MediaReader {
  final Frame frame;

  ImageReader({required String imagePath}) : frame = Frame(imread(imagePath));

  @override
  Stream<Frame?> readFrames() async* {
    yield frame;
    // Terminate the stream after yielding that single frame
  }

  @override
  void dispose() {
    frame.dispose();
  }
}

class WebcamReader implements MediaReader {
  bool _isReading = false;
  bool get _isInitialized =>
      webcamController != null && webcamController!.value.isInitialized;
  final List<CameraDescription> _availableCameras = [];
  CameraController? webcamController;
  final imageBuffer = <CameraImage>[];

  Future<void> initialize() async {
    _availableCameras.addAll(await availableCameras());
    if (_availableCameras.isEmpty) {
      throw Exception("No cameras available");
    }

    webcamController = CameraController(
      _availableCameras.first,
      ResolutionPreset.high,
      enableAudio: false,
      fps: 300,
    );
    await webcamController!.initialize();
  }

  Future<void> startReading() async {
    if (!_isInitialized) {
      throw Exception("Webcam not initialized. Call initialize() first.");
    }
    if (_isReading) return;

    _isReading = true;
    await webcamController!.startImageStream(_addToBuffer);
  }

  Future<void> stopReading() async {
    if (!_isInitialized) {
      throw Exception("Webcam not initialized. Call initialize() first.");
    }
    if (!_isReading) return;
    await webcamController!.stopImageStream();
    _isReading = false;
  }

  @override
  Stream<Frame?> readFrames() async* {
    while (!_isInitialized) {
      await Future.delayed(Duration(milliseconds: 100));
    }

    while (_isInitialized) {
      try {
        if (imageBuffer.isEmpty) {
          await Future.delayed(Duration(milliseconds: 10));
          continue;
        }
        final cameraImage = imageBuffer.removeAt(0);
        final bytes = _toRgb(frame: cameraImage, rotation: 90);
        if (bytes == null) {
          yield null;
          continue;
        }
        final frame = Mat.fromList(
          cameraImage.width,
          cameraImage.height,
          MatType.CV_8UC(3),
          bytes,
        );
        yield Frame(frame);
      } catch (e) {
        yield null;
      }
    }
    yield null;
  }

  void _addToBuffer(CameraImage bytes) {
    imageBuffer.add(bytes);
    if (imageBuffer.length > 10) {
      _logger.warning(
        "Image buffer overflow: ${imageBuffer.length} frames. Dropping oldest frame.",
      );
      imageBuffer.removeAt(0); // Keep the buffer size manageable
    }
  }

  @override
  void dispose() async {
    await stopReading();
    await webcamController?.dispose();
    webcamController = null;
  }

  Uint8List? _toRgb({required CameraImage frame, int rotation = 90}) {
    try {
      return switch (frame.format.group) {
        ImageFormatGroup.yuv420 => Uint8ListExtension.fromYUV420(
          frame,
          rotation: rotation,
        ),
        ImageFormatGroup.bgra8888 => Uint8ListExtension.fromBGRA8888(
          frame,
          rotation: rotation,
        ),
        _ => throw Exception("Unsupported image format: ${frame.format.group}"),
      };
    } catch (e) {
      _logger.severe("Error converting camera image to RGB: $e");
      return null;
    }
  }
}

class CharucoMockReader implements MediaReader {
  ///
  /// The list of Charuco boards to simulate. Each board will be transformed according to the provided transformations.
  ///
  final List<Charuco> charucos;

  ///
  /// A list of transformations for each Charuco board across all frames.
  /// The outer list corresponds to frames, and the inner list corresponds to the
  /// transformations (translation, rotation) for each board in that frame.
  /// Therefore each inner list should have the same length as the number of Charuco boards.
  ///
  final Stream<List<(Vector, Matrix)>> transformations;

  ///
  /// The camera parameters to use for projecting the Charuco boards in the mock frames.
  /// This includes intrinsic parameters like focal length and sensor size.
  ///
  final Camera camera;

  CharucoMockReader({
    required this.charucos,
    required this.camera,
    required this.transformations,
  });

  @override
  Stream<Frame?> readFrames() async* {
    // Move the board further away and rotate the boards and get their images
    await for (final frameTransformations in transformations) {
      // First create a white background which corresponds to a distant wall
      Mat frame = Mat.fromScalar(
        camera.sensorHeight.toInt(),
        camera.sensorWidth.toInt(),
        MatType.CV_8UC(3),
        Scalar(0xFF / 2, 0xFF / 2, 0xFF / 2),
      );

      for (final (charucoTransformation) in frameTransformations) {
        final translation = charucoTransformation.$1;
        final rotation = charucoTransformation.$2;
        final board =
            charucos[frameTransformations.indexOf(charucoTransformation)];
        final (projectedBoard, mask) = _projectBoard(
          charuco: board,
          camera: camera,
          translation: translation,
          rotation: rotation,
        );

        projectedBoard.copyTo(frame, mask: mask);
      }

      // Encode and decode to get a proper Frame object
      final (isSuccess, buf) = imencode(".png", frame);
      if (!isSuccess) break;
      yield Frame(imdecode(buf, IMREAD_COLOR));
    }

    yield null;
  }

  ///
  /// Project the Charuco board image with a given rotation and translation.
  /// [charuco] The Charuco board to project on the image
  /// [camera] The camera parameters to use for projection.
  /// [frame] The Charuco board image to project.
  /// [transformation] The homogenous transformation (4x4 matrix, consisting of
  /// the 3x3 rotation, 3x1 translation, and 1x1 scale) containing translation
  /// and rotation to apply to the board.
  ///
  /// Returns the projected board image and its mask
  ///
  (Mat, Mat) _projectBoard({
    required Charuco charuco,
    required Camera camera,
    required Vector translation,
    required Matrix rotation,
  }) {
    // Scale the translation part of the transformation by the camera's focal
    // length to simulate perspective projection
    final pixelTranslation = translation * camera.focalLength;

    // Drop the third column of rotation and concatenate the translation
    final projectionMatrix = rotation
        .filterColumns((_, index) => index != 2)
        .insertColumns(2, [pixelTranslation]);

    // Remove the third column of the transformation to get a 3x4 matrix for projection
    final hMatrix = camera.matrixAsLinalg * projectionMatrix;

    // Corners of the img
    final height = charuco.boardImage.rows;
    final width = charuco.boardImage.cols;
    final sMatrix = Matrix.fromList([
      [1, 0, -width / 2],
      [0, 1, -height / 2],
      [0, 0, 1],
    ]);

    // Compute the homography and warp the image
    final hFinal = (hMatrix * sMatrix).toMat();

    // Prepare the mask
    final mask = Mat.fromScalar(height, width, MatType.CV_8UC(1), Scalar(0xFF));

    return (
      warpPerspective(cvtColor(charuco.boardImage, COLOR_GRAY2BGR), hFinal, (
        camera.sensorWidth.toInt(),
        camera.sensorHeight.toInt(),
      )),
      warpPerspective(mask, hFinal, (
        camera.sensorWidth.toInt(),
        camera.sensorHeight.toInt(),
      )),
    );
  }

  @override
  void dispose() {}
}

extension Uint8ListExtension on Uint8List {
  // CameraImage BGRA8888 -> PNG
  // Color
  static Uint8List fromBGRA8888(CameraImage image, {int rotation = 0}) {
    final plane = image.planes[0];

    final width = image.width;
    final height = image.height;
    final bytes = plane.bytes;
    final bytesPerRow = plane.bytesPerRow;

    late Uint8List out;
    late int outWidth;
    late int outHeight;

    if (rotation == 90 || rotation == 270) {
      outWidth = height;
      outHeight = width;
    } else {
      outWidth = width;
      outHeight = height;
    }

    out = Uint8List(outWidth * outHeight * 3);

    for (int y = 0; y < height; y++) {
      final rowStart = y * bytesPerRow;

      for (int x = 0; x < width; x++) {
        final i = rowStart + x * 4;

        final b = bytes[i];
        final g = bytes[i + 1];
        final r = bytes[i + 2];

        int newX, newY;

        switch (rotation) {
          case 90:
            newX = height - y - 1;
            newY = x;
            break;
          case 270:
            newX = y;
            newY = width - x - 1;
            break;
          case 180:
            newX = width - x - 1;
            newY = height - y - 1;
            break;
          default:
            newX = x;
            newY = y;
        }

        final outIndex = (newY * outWidth + newX) * 3;

        out[outIndex] = b;
        out[outIndex + 1] = g;
        out[outIndex + 2] = r;
      }
    }

    return out;
  }

  // CameraImage YUV420_888 -> PNG -> Image (compresion:0, filter: none)
  // Black
  static Uint8List fromYUV420(CameraImage image, {int rotation = 90}) {
    final width = image.width;
    final height = image.height;

    final outWidth = (rotation == 90 || rotation == 270) ? height : width;
    final outHeight = (rotation == 90 || rotation == 270) ? width : height;

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final yBuffer = yPlane.bytes;
    final uBuffer = uPlane.bytes;
    final vBuffer = vPlane.bytes;

    final rgb = Uint8List(outWidth * outHeight * 3);

    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel!;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final yIndex = y * yPlane.bytesPerRow + x;
        final uvIndex = (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;

        final Y = yBuffer[yIndex];
        final U = uBuffer[uvIndex];
        final V = vBuffer[uvIndex];

        int R = (Y + 1.402 * (V - 128)).round();
        int G = (Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)).round();
        int B = (Y + 1.772 * (U - 128)).round();

        R = R.clamp(0, 255);
        G = G.clamp(0, 255);
        B = B.clamp(0, 255);

        int newX, newY;

        switch (rotation) {
          case 90:
            newX = height - y - 1;
            newY = x;
            break;
          case 180:
            newX = width - x - 1;
            newY = height - y - 1;
            break;
          case 270:
            newX = y;
            newY = width - x - 1;
            break;
          default:
            newX = x;
            newY = y;
        }

        // Use outWidth instead of width
        final newIndex = (newY * outWidth + newX) * 3;

        rgb[newIndex] = R;
        rgb[newIndex + 1] = G;
        rgb[newIndex + 2] = B;
      }
    }

    return rgb;
  }
}
