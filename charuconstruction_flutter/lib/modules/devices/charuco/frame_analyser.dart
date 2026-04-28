import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv.dart';

import 'camera.dart';
import 'charuco.dart';
import 'extensions.dart';
import 'frame.dart';

enum AvailableFrameAnalysers {
  reconstructCharuco,
  videoSaver;

  @override
  String toString() => switch (this) {
    AvailableFrameAnalysers.reconstructCharuco => 'Reconstruct Charucos',
    AvailableFrameAnalysers.videoSaver => 'Save video',
  };
}

enum AvailableExtraAnalyses { charucosReconstruction }

abstract class FrameAnalyser {
  ///
  /// Analyse the frames from the media reader.
  ///
  Future<(Frame?, Map<AvailableExtraAnalyses, dynamic>?)> perform(
    Frame? frame, {
    Map<AvailableExtraAnalyses, dynamic>? extraAnalyses,
  });

  ///
  /// Dispose any resources used by the analyser.
  ///
  void dispose();
}

class ReconstructCharucoFrameAnalyser extends FrameAnalyser {
  final List<Charuco> charucoBoards;
  final Camera camera;
  final bool ignoreReconstructionError;

  ReconstructCharucoFrameAnalyser({
    required this.charucoBoards,
    required this.camera,
    this.ignoreReconstructionError = false,
  });

  @override
  Future<(Frame?, Map<AvailableExtraAnalyses, dynamic>?)> perform(
    Frame? frame, {
    Map<AvailableExtraAnalyses, dynamic>? extraAnalyses,
  }) async {
    if (frame == null) {
      dispose();
      return (null, extraAnalyses);
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

    extraAnalyses ??= {};
    extraAnalyses[AvailableExtraAnalyses.charucosReconstruction] = results;

    return (frame, extraAnalyses);
  }

  @override
  void dispose() {
    // No resources to dispose in this analyser
  }
}

class FrameAnalyserPipeline extends FrameAnalyser {
  final List<FrameAnalyser> _analysers;
  List<FrameAnalyser> get analysers => _analysers;

  FrameAnalyserPipeline({required List<FrameAnalyser> analysers})
    : _analysers = analysers;

  @override
  Future<(Frame?, Map<AvailableExtraAnalyses, dynamic>?)> perform(
    Frame? frame, {
    Map<AvailableExtraAnalyses, dynamic>? extraAnalyses,
  }) async {
    if (frame == null) {
      dispose();
      return (null, extraAnalyses);
    }

    for (final analyser in _analysers) {
      (frame, extraAnalyses) = await analyser.perform(
        frame,
        extraAnalyses: extraAnalyses,
      );
    }
    return (frame, extraAnalyses);
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
  Future<(Frame?, Map<AvailableExtraAnalyses, dynamic>?)> perform(
    Frame? frame, {
    Map<AvailableExtraAnalyses, dynamic>? extraAnalyses,
  }) async {
    if (frame == null) {
      dispose();
      return (null, extraAnalyses);
    }

    _writer.write(frame.get());
    return (frame, extraAnalyses);
  }

  @override
  void dispose() {
    _writer.release();
  }
}
