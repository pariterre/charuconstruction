import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dartcv4/dartcv.dart';

class Frame {
  final Mat _image;
  Mat? _grayscaleImage;
  VideoWriter? _videoWriter;

  Frame(Mat image) : _image = image;

  ///
  /// Get the image as a Mat object.
  /// [grayscale] Whether to return the grayscale version of the image.
  /// If true, the grayscale version will be cached for future calls.
  ///
  Mat get({bool grayscale = false}) {
    return grayscale
        ? (_grayscaleImage ??= _convertToGrayscale(_image))
        : _image;
  }

  ///
  /// Dispose of the frame's resources. This should be called when the frame is no longer needed to free up memory.
  ///
  void dispose() {
    _image.dispose();
    _grayscaleImage?.dispose();
  }

  ///
  /// Convert the frame to a displayable format and show it in a window.
  /// The window will be automatically resized to fit the frame while respecting the specified maximum dimensions.
  /// [maxWidth] Maximum width of the displayed window.
  /// [maxHeight] Maximum height of the displayed window.
  /// [grayscale] Whether to show the grayscale version of the frame.
  ///
  /// Returns the frame as a byte array in PNG format, which can be used for
  /// further processing, showing or saving to disk. If the conversion fails, null is returned.
  ///
  Uint8List? toBytes({
    int maxWidth = 800,
    int maxHeight = 600,
    bool grayscale = false,
  }) {
    final frame = get(grayscale: grayscale);

    final height = frame.height;
    final width = frame.width;

    // Compute scaling factor to fit max size
    final scaleW = maxWidth / width;
    final scaleH = maxHeight / height;
    final scale = math.min(scaleW, scaleH); // never upscale

    late final Mat resized;
    if (scale < 1.0) {
      final newWidth = (width * scale).toInt();
      final newHeight = (height * scale).toInt();
      resized = resize(frame, (newWidth, newHeight));
    } else {
      resized = frame;
    }

    final (isSuccess, bytes) = imencode(".png", resized);
    return isSuccess ? bytes : null;
  }

  ///
  /// Close the window displaying the frame.
  ///
  void closeWindow() {
    destroyWindow('frame');
  }

  ///
  /// Start recording a video from the frame. The video will be saved to the specified
  /// filename with the given frames per second (fps).
  /// [filename] The name of the video file to save.
  /// [fps] The frames per second for the video.
  /// Throws an exception if a recording is already in progress.
  ///
  void startRecording(String filename, int fps) {
    if (_videoWriter != null) {
      throw Exception('Recording already in progress');
    }

    final frameSize = (_image.height, _image.width);
    _videoWriter = VideoWriter.fromFile(
      filename,
      'mp4v',
      fps.toDouble(),
      frameSize,
    );
  }

  ///
  /// Add the current frame to the ongoing video recording.
  /// Throws an exception if no recording is in progress.
  ///
  void addFrameToRecording() {
    if (_videoWriter == null) throw Exception('No recording in progress');

    _videoWriter!.write(_image);
  }

  ///
  /// Stop the ongoing video recording and save the file. After calling this method, a new recording can be started.
  /// Throws an exception if no recording is in progress.
  ///
  void stopRecording() {
    if (_videoWriter == null) throw Exception('No recording in progress');

    _videoWriter?.release();
    _videoWriter = null;
  }
}

///
/// Ensure the frame is in grayscale format.
///
Mat _convertToGrayscale(Mat image) {
  // TODO Check how to test for the number of channels in the image
  if (image.channels == 1) {
    return image;
  } else if (image.channels == 3) {
    return cvtColor(image, COLOR_BGR2GRAY);
  } else {
    throw ArgumentError('Unsupported number of channels: ${image.channels}');
  }
}
