import 'package:charuconstruction_flutter/modules/charucos/extensions.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv.dart';

import 'camera.dart';
import 'charuco.dart';
import 'frame.dart';

abstract class MediaReader {
  ///
  /// Read frames from the media source. This method returns a stream of frames t
  /// hat can be listened to for real-time processing.
  ///
  Stream<Frame> readFrames();

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
  Stream<Frame> readFrames() async* {
    yield frame;
    // Terminate the stream after yielding that single frame
  }

  @override
  void dispose() {
    frame.dispose();
  }
}

class VideoReader implements MediaReader {
  final String videoPath;
  late final VideoCapture _capture;

  VideoReader({required this.videoPath})
    : _capture = VideoCapture.fromFile(videoPath);

  @override
  Stream<Frame> readFrames() async* {
    if (!_capture.isOpened) throw Exception('Failed to open video: $videoPath');

    while (true) {
      final (isSuccess, mat) = _capture.read();
      if (!isSuccess) break; // End of video or error

      yield Frame(mat);
    }
  }

  @override
  void dispose() {
    _capture.release();
  }
}

class CharucoMockReader implements MediaReader {
  ///
  /// The number of frames in the mock reader is determined by the length of the transformations list.
  ///
  final int frameCount;

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
  final List<List<(Vector, Matrix)>> transformations;

  ///
  /// The camera parameters to use for projecting the Charuco boards in the mock frames.
  /// This includes intrinsic parameters like focal length and sensor size.
  ///
  final Camera camera;

  CharucoMockReader({
    required this.charucos,
    required this.transformations,
    required this.camera,
  }) : frameCount = transformations.isNotEmpty ? transformations.length : 0 {
    // Sanity checks for angles
    for (var frame in transformations) {
      if (frame.length != charucos.length) {
        throw Exception(
          "Number of transformations must match number of boards in all frames.",
        );
      }
    }
  }

  @override
  Stream<Frame> readFrames() async* {
    // Move the board further away and rotate the boards and get their images
    for (final frameTransformations in transformations) {
      // First create a white background which corresponds to a distant wall
      Mat frame = Mat.fromScalar(
        camera.sensorHeight.toInt(),
        camera.sensorWidth.toInt(),
        MatType.CV_16UC(3),
        Scalar(0xFFFF, 0xFFFF, 0xFFFF),
      );

      for (final charucoTransformation in frameTransformations) {
        final board =
            charucos[frameTransformations.indexOf(charucoTransformation)];
        final projectedBoard = _projectBoard(
          charuco: board,
          camera: camera,
          translation: charucoTransformation.$1,
          rotation: charucoTransformation.$2,
        );

        final (_, mask) = threshold(
          projectedBoard,
          0xFFFF - 1,
          0xFFFF,
          THRESH_BINARY_INV,
        );
        projectedBoard.copyTo(frame, mask: mask);
      }
      // Encode and decode to get a proper Frame object
      final (isSuccess, buf) = imencode(".png", frame);
      if (!isSuccess) break;
      yield Frame(imdecode(buf, IMREAD_COLOR));
      await Future.delayed(const Duration(milliseconds: 100)); // Simulate delay
    }
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
  Mat _projectBoard({
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
    final hFinal = hMatrix * sMatrix;
    return warpPerspective(charuco.boardImage, hFinal.toMat(), (
      camera.sensorWidth.toInt(),
      camera.sensorHeight.toInt(),
    ), borderValue: Scalar(255));
  }

  @override
  void dispose() {}
}



    // def _read_frame(self):



    //     # Encode and decode to get a proper Frame object
    //     success, buf = cv2.imencode(".png", frame)
    //     if not success:
    //         return None
    //     return Frame(cv2.imdecode(buf, cv2.IMREAD_COLOR))

    


