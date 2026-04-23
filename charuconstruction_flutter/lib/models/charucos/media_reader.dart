import 'package:charuconstruction_flutter/models/charucos/frame.dart';
import 'package:opencv_dart/opencv.dart';

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
  void dispose() {}
}

class ImageReader extends MediaReader {
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
    super.dispose();
  }
}
