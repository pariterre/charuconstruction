import 'package:charuconstruction_flutter/modules/charucos/extensions.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:opencv_dart/opencv.dart';

import 'camera.dart';
import 'charuco.dart';
import 'frame.dart';

abstract class FrameAnalyser {
  ///
  /// Analyse the frames from the media reader.
  ///
  Future<Frame> perform(Frame frame);
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
  Future<Frame> perform(Frame frame) async {
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
    // # charuco.draw_aruco_markers(frame_to_draw)
    // charuco.draw_estimated_pose_axes(
    //     frame=frame_to_draw,
    //     camera=camera,
    //     translation=translation,
    //     rotation=rotation,
    //     axes_length=0.1,
    // )

    return frame;
  }
}

class FrameAnalyserPipeline extends FrameAnalyser {
  final List<FrameAnalyser> _analysers;

  FrameAnalyserPipeline({required List<FrameAnalyser> analysers})
    : _analysers = analysers;

  @override
  Future<Frame> perform(Frame frame) async {
    for (final analyser in _analysers) {
      frame = await analyser.perform(frame);
    }
    return frame;
  }
}


// TODO Implement the flow for saving the video
// class VideoSaverAnalyser extends FrameAnalyser {
//   @override
//   Future<Frame> analyse(Frame frame) async {
//     //   if should_record_video:
//     //       video_frame.frame_from(frame_to_draw)
//     //       video_frame.add_frame_to_recording()
    
//     return frame;
//   }
// }