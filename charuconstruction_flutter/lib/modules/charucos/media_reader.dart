import 'dart:io';

import 'package:opencv_dart/opencv.dart';

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
  @override
  Stream<Frame> readFrames() {
    // TODO: implement readFrames
    throw UnimplementedError();
  }

  @override
  void dispose() {}
}
    // def __init__(
    //     self,
    //     boards: Iterable[Charuco],
    //     camera: Camera,
    //     transformations: dict[Charuco, Iterable[Transformation]] = None,
    // ):
    //     """
    //     A mock reader that simulates a video feed of Charuco boards with given transformations.
    //     If no transformations are provided, a controller window is shown to manually adjust the
    //     pose of the boards.
    //     """
    //     self._boards = boards

    //     # Sanity checks for angles
    //     if transformations is None:
    //         self._frame_count = None
    //         self._current_index = None
    //     else:
    //         self._frame_count = (
    //             len(transformations[next(iter(transformations))])
    //             if transformations
    //             else 0
    //         )
    //         self._current_index = 0

    //         for charuco in transformations.keys():
    //             if len(transformations[charuco]) != self._frame_count:
    //                 raise ValueError(
    //                     "Number of transformations must match number of boards in all frames."
    //                 )
    //     self._transformations = transformations

    //     self._camera = camera
    //     self._dynamic_frame_ready = False

    //     super().__init__()


    // @property
    // def frame_count(self) -> int | None:
    //     return self._frame_count

    // def transformations(self, charuco: Charuco) -> Transformation:
    //     """
    //     Return the current transformation for a given Charuco board.
    //     """
        
    //     return self._transformations[charuco][self._current_index - 1]

    // def __iter__(self) -> "CharucoMockReader":
    //     self._current_index = 0
    //     return self

    // def _has_moved(self, _) -> bool:
    //     self._dynamic_frame_ready = True

    // def _read_frame(self):
    //     if self._current_index >= self._frame_count:
    //         return None
    //     # We increment now since the self.transformations uses the current_index - 1
    //     self._current_index += 1

    //     # First create a cv2 image with a white background which corresponds to a
    //     # distant wall
    //     frame = np.full(
    //         (
    //             int(self._camera.sensor_height),
    //             int(self._camera.sensor_width),
    //             3,
    //         ),
    //         255,
    //         dtype=np.uint8,
    //     )

    //     # Move the image further away and rotate the boards and get their images
    //     for board in self._boards:
    //         img = self._project_board(
    //             board.cv2_board_image,
    //             transformation=self.transformations(charuco=board),
    //         )
    //         projected_img = cv2.cvtColor(src=img, code=cv2.COLOR_GRAY2BGR)

    //         masked = projected_img < 255
    //         frame[masked] = projected_img[masked]

    //     # Encode and decode to get a proper Frame object
    //     success, buf = cv2.imencode(".png", frame)
    //     if not success:
    //         return None
    //     return Frame(cv2.imdecode(buf, cv2.IMREAD_COLOR))

    // def _project_board(
    //     self,
    //     img: np.ndarray,
    //     transformation: Transformation,
    // ) -> np.ndarray:
    //     """
    //     Project the Charuco board image with a given rotation and translation.

    //     Parameters:
    //         img (np.ndarray): The Charuco board image to project.
    //         transformation (Transformation): The transformation containing translation and rotation.
    //     """

    //     # Corners of the img
    //     h, w = img.shape[:2]
    //     s_matrix = np.array(
    //         [[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], dtype=np.float32
    //     )

    //     translation = (
    //         transformation.translation.vector * self._camera.focal_length
    //     )
    //     rotation = transformation.rotation.matrix
    //     rt_matrix = np.hstack((rotation, translation))
    //     h_matrix = self._camera.matrix @ np.delete(rt_matrix, 2, 1)
    //     h_final = h_matrix @ s_matrix

    //     return cv2.warpPerspective(
    //         img,
    //         h_final,
    //         (int(self._camera.sensor_width), int(self._camera.sensor_height)),
    //         flags=cv2.INTER_LINEAR,
    //         borderValue=255,
    //     )