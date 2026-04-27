import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv.dart';

import 'camera.dart';
import 'charuco.dart';
import 'extensions.dart';
import 'frame.dart';

abstract class FrameAnalyser {
  ///
  /// Analyse the frames from the media reader.
  ///
  Future<Frame?> perform(Frame? frame);

  ///
  /// Dispose any resources used by the analyser.
  ///
  void dispose();
}

class CharucoFrameAnalyser extends FrameAnalyser {
  final List<Charuco> charucoBoards;
  final Camera camera;
  final bool ignoreReconstructionError;

  CharucoFrameAnalyser({
    required this.charucoBoards,
    required this.camera,
    this.ignoreReconstructionError = false,
  });

  @override
  Future<Frame?> perform(Frame? frame) async {
    if (frame == null) {
      dispose();
      return null;
    }

    final results = <Charuco, (Vector?, Matrix?)?>{};
    for (var charuco in charucoBoards) {
      results[charuco] =
          (await charuco.detect(
            frame: frame,
            camera: camera,
            ignoreReconstructionError: ignoreReconstructionError,
          )) ??
          (null, null);
    }

    //   # Compute the reconstruction error in degrees
    //   if should_compute_error:
    //       for charuco in charuco_boards:
    //           if charuco not in errors:
    //               errors[charuco] = []

    //           _, rotation = frame_results[charuco]
    //           if rotation is None:
    //               errors[charuco].append(np.ndarray((3, 1)) * np.nan)
    //               continue

    //           sequence = RotationMatrix.Sequence.ZYX
    //           true_values = (
    //               reader.transformations(charuco)
    //               .rotation.to_euler(sequence=sequence, degrees=True)
    //               .as_array()
    //           )
    //           reconstructed_values = rotation.to_euler(
    //               sequence=sequence, degrees=True
    //           ).as_array()
    //           errors[charuco].append(true_values - reconstructed_values)

    // Show the estimated pose for each board on the current frame
    for (final charuco in results.keys) {
      final (translation, rotation) = results[charuco] ?? (null, null);
      if (translation == null || rotation == null) {
        continue;
      }

      drawFrameAxes(
        frame.get(),
        camera.matrixAsMat,
        camera.distorsionCoefficientsAsMat,
        rotation.toMat(),
        translation.toMat(),
        0.1,
      );
    }

    return frame;
  }

  @override
  void dispose() {
    // No resources to dispose in this analyser
  }
}

class FrameAnalyserPipeline extends FrameAnalyser {
  final List<FrameAnalyser> _analysers;

  FrameAnalyserPipeline({required List<FrameAnalyser> analysers})
    : _analysers = analysers;

  @override
  Future<Frame?> perform(Frame? frame) async {
    if (frame == null) {
      dispose();
      return null;
    }

    for (final analyser in _analysers) {
      frame = await analyser.perform(frame);
    }
    return frame;
  }

  @override
  void dispose() {
    for (final analyser in _analysers) {
      analyser.dispose();
    }
  }
}

class VideoSaverAnalyser implements FrameAnalyser {
  final String outputPath;
  late final VideoWriter _writer;

  VideoSaverAnalyser({
    required this.outputPath,
    double fps = 30.0,
    (int, int) frameSize = (1920, 1080),
  }) : _writer = VideoWriter.fromFile(outputPath, 'mp4v', fps, frameSize);

  @override
  Future<Frame?> perform(Frame? frame) async {
    if (frame == null) {
      dispose();
      return null;
    }

    _writer.write(frame.get());
    return frame;
  }

  @override
  void dispose() {
    _writer.release();
  }
}
