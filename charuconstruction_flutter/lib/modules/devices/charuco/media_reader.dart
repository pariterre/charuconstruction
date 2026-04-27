import 'dart:async';

import 'package:camera/camera.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv.dart';

import 'camera.dart';
import 'charuco.dart';
import 'extensions.dart';
import 'frame.dart';

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
  final List<CameraDescription> _availableCameras = [];
  CameraController? webcamController;

  WebcamReader() {
    _initializeWebcam();
  }

  Future<void> _initializeWebcam() async {
    _availableCameras.addAll(await availableCameras());

    if (_availableCameras.isEmpty) {
      throw Exception("No cameras available");
    }

    webcamController = CameraController(
      _availableCameras.first,
      ResolutionPreset.high,
    );
    await webcamController!.initialize();
  }

  @override
  Stream<Frame?> readFrames() async* {
    if (webcamController == null) yield null;

    while (webcamController != null) {
      try {
        final cameraImage = await webcamController!.takePicture();
        final bytes = await cameraImage.readAsBytes();
        yield Frame(imdecode(bytes, IMREAD_COLOR));
      } catch (e) {
        yield null;
      }
    }
    yield null;
  }

  @override
  void dispose() {
    webcamController?.dispose();
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
