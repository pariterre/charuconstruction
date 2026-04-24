import 'package:opencv_dart/opencv.dart';

import 'camera.dart';
import 'charuco.dart';
import 'frame.dart';

abstract class FrameAnalyser {
  ///
  /// Analyse the frames from the media reader.
  ///
  Future<Frame> analyse(Frame frame);
}

class CharucoFrameAnalyser extends FrameAnalyser {
  final List<Charuco> charucoBoards;
  final Camera camera;

  CharucoFrameAnalyser({required this.charucoBoards, required this.camera});

  @override
  Future<Frame> analyse(Frame frame) async {
    final results = <Charuco, (Mat?, Mat?)?>{};
    for (var charuco in charucoBoards) {
      results[charuco] =
          (await charuco.detect(frame: frame, camera: camera)) ?? (null, null);
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
        rotation,
        translation,
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