import 'package:opencv_dart/opencv_dart.dart';

class Frame {
  final InputArray _image;

  InputArray get({bool grayscale = false}) {
    if (grayscale) {
      throw UnimplementedError(
        "Grayscale conversion is not implemented yet. Please provide a grayscale image directly.",
      );
    } else {
      return _image;
    }
  }

  Frame(InputArray image) : _image = image;
}
