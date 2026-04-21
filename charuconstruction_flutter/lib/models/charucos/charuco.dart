import 'package:charuconstruction_flutter/models/charucos/charuco_resolution.dart';
import 'package:charuconstruction_flutter/utils/math.dart';
import 'package:opencv_dart/opencv_dart.dart';

class Charuco {
  ///
  /// The number of squares in the vertical direction of the charuco board.
  ///
  final int verticalSquaresCount;

  ///
  /// The number of squares in the horizontal direction of the charuco board.
  ///
  final int horizontalSquaresCount;

  ///
  /// The length of the squares in the charuco board, in meters.
  ///
  final double squareLen;

  ///
  /// The length of the markers in the charuco board, in meters.
  /// The markers are the black squares in the charuco board.
  ///
  final double markerLen;

  ///
  /// The resolution of the charuco board, in dots per inch (DPI), i.e. the number
  /// of pixels in one inch on the printed page.
  ///
  final CharucoResolution resolution;

  ///
  /// The Aruco dictionary used for the markers pattern in the charuco board.
  ///
  final ArucoDictionary arucoDict;

  ///
  /// OpenCV Charuco board generated from the dictionary and the parameters of the charuco board.
  ///
  // late final CharucoBoard board;

  ///
  /// Seed for random number generator (will determine the order of aruco markers).
  ///
  final int? seed;

  Charuco({
    required this.verticalSquaresCount,
    required this.horizontalSquaresCount,
    required this.squareLen,
    required this.markerLen,
    required this.resolution,
    ArucoDictionary? arucoDict,
    this.seed,
  }) : arucoDict =
           arucoDict ??
           ArucoDictionary.predefined(PredefinedDictionaryType.DICT_7X7_1000) {
    // Sanity checks
    if (markerLen > squareLen) {
      throw ArgumentError(
        "The marker length cannot be greater than the square length.",
      );
    }

    // Reorder the Aruco markers in the dictionary according to the seed,
    // so that the same seed will always produce the same charuco board.
    _reorderArucoDictionary(
      arucoDict: this.arucoDict,
      squareCount: horizontalSquaresCount * verticalSquaresCount,
      seed: seed,
    );

    // TODO: Complete here with CHARUCO_BOARD creation, using the reordered arucoDict
  }

  static Charuco fromSerialized(Map<String, dynamic> json) {
    return Charuco(
      verticalSquaresCount: json['vertical_squares_count'] as int,
      horizontalSquaresCount: json['horizontal_squares_count'] as int,
      squareLen: json['square_len'] as double,
      markerLen: json['marker_len'] as double,
      resolution: CharucoResolution.fromName(json['resolution'] as String),
      arucoDict: ArucoDictionary.predefined(
        PredefinedDictionaryType.values[json['aruco_dict'] as int],
      ),
      seed: json['seed'] as int,
    );
  }
}

void _reorderArucoDictionary({
  required ArucoDictionary arucoDict,
  required int squareCount,
  int? seed,
}) {
  final random = PythonRandom(seed: seed);
  final arucoIndices = List.generate(squareCount, (i) => i);
  random.shuffle(arucoIndices);

  final original = arucoDict.bytesList;
  final reordered = Mat.zeros(squareCount, original.cols, original.type);
  for (int i = 0; i < squareCount; i++) {
    original.row(arucoIndices[i]).copyTo(reordered.row(i));
  }

  arucoDict.bytesList = reordered;
}
