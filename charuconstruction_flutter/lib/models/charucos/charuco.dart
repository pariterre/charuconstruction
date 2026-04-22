import 'dart:io';

import 'package:charuconstruction_flutter/models/charucos/charuco_resolution.dart';
import 'package:charuconstruction_flutter/models/charucos/frame.dart';
import 'package:charuconstruction_flutter/utils/math.dart';
import 'package:charuconstruction_flutter/utils/misc.dart';
import 'package:dartcv4/dartcv.dart';
import 'package:path_provider/path_provider.dart';

class Charuco {
  ///
  /// The total number of squares in the charuco board.
  ///
  int get totalSquaresCount => verticalSquaresCount * horizontalSquaresCount;

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
  /// Get the total width of the Charuco board in meters.
  ///
  double get widthLen => horizontalSquaresCount * squareLen;

  ///
  /// Get the total height of the Charuco board in meters.
  ///
  double get heightLen => verticalSquaresCount * squareLen;

  ///
  /// Get the ratio of vertical to horizontal squares.
  ///
  double get squareRatio => verticalSquaresCount / horizontalSquaresCount;

  ///
  /// The Aruco dictionary used for the markers pattern in the charuco board.
  ///
  final int arucoDictId;
  final ArucoDictionary arucoDict;

  ///
  /// The resolution of the charuco board, in dots per inch (DPI), i.e. the number
  /// of pixels in one inch on the printed page.
  ///
  final CharucoResolution resolution;

  ///
  /// Get the list of ArUco marker IDs used in the Charuco board.
  ///
  List<int> get markerIds => board.ids.toList();

  ///
  /// OpenCV Charuco board generated from the dictionary and the parameters of the charuco board.
  ///
  late final CharucoBoard board;
  late final Mat boardImage;

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
    required this.arucoDictId,
    this.seed,
  }) : arucoDict = ArucoDictionary.predefined(
         PredefinedDictionaryType.values[arucoDictId],
       ) {
    // Sanity checks
    if (markerLen > squareLen) {
      throw ArgumentError(
        "The marker length cannot be greater than the square length.",
      );
    }

    // Reorder the Aruco markers in the dictionary according to the seed,
    // so that the same seed will always produce the same charuco board.
    _reorderArucoDictionary(
      arucoDict: arucoDict,
      squareCount: horizontalSquaresCount * verticalSquaresCount,
      seed: seed,
    );

    board = CharucoBoard.create(
      (horizontalSquaresCount, verticalSquaresCount),
      squareLen,
      markerLen,
      arucoDict,
    );

    final isHorz = horizontalSquaresCount >= verticalSquaresCount;
    final boardLen =
        resolution.pixelsPerMeter *
        squareLen *
        (isHorz ? horizontalSquaresCount : verticalSquaresCount);
    final imgLen = isHorz
        ? (boardLen, boardLen * squareRatio)
        : (boardLen / squareRatio, boardLen);
    final imgLenAsInt = (imgLen.$1.toInt(), imgLen.$2.toInt());
    boardImage = board.generateImage(imgLenAsInt, marginSize: 0);
  }

  Map<String, dynamic> serialize() {
    return {
      "vertical_squares_count": verticalSquaresCount,
      "horizontal_squares_count": horizontalSquaresCount,
      "square_len": squareLen,
      "marker_len": markerLen,
      "resolution": resolution.name,
      "aruco_dict": arucoDictId,
      "seed": seed,
    };
  }

  static Charuco fromSerialized(Map<String, dynamic> json) {
    return Charuco(
      verticalSquaresCount: json['vertical_squares_count'] as int,
      horizontalSquaresCount: json['horizontal_squares_count'] as int,
      squareLen: json['square_len'] as double,
      markerLen: json['marker_len'] as double,
      resolution: CharucoResolution.fromName(json['resolution'] as String),
      arucoDictId: json['aruco_dict'] as int,
      seed: json['seed'] as int,
    );
  }

  ///
  /// Show the generated Charuco board image in a window.
  ///
  void show() {
    imshow("img", boardImage);
    waitKey(0);
  }

  ///
  /// Save the Charuco board parameters and image to a folder.
  /// [saveFolder] Path to the folder where the board parameters and image will be saved
  /// [override] Whether to override the folder if it already exists.
  ///
  Future<void> save(String saveFolder, {bool override = false}) async {
    // Prepare the save directory
    final baseDir = await getApplicationDocumentsDirectory();
    final saveDir = Directory('${baseDir.path}/$saveFolder');
    if (!override && saveDir.existsSync()) {
      throw FileSystemException(
        "The folder $saveFolder already exists. Use override=true to overwrite.",
      );
    }
    saveDir.createSync(recursive: true);

    // Save the board parameters to a JSON file
    final jsonFile = File('${saveDir.path}/board.json');
    jsonFile.writeAsStringSync(serialize().toString(), flush: true);

    // Save the board image to a PNG file
    imwrite('${saveDir.path}/board.png', boardImage);
  }

  ///
  /// Detect markers and ChArUco corners in the given image.
  /// [frame] OpenCV frame to detect markers and corners from.
  /// [camera] Camera parameters for pose estimation.
  /// [translationInitialGuess] Initial guess for translation vector.
  /// [rotationInitialGuess] Initial guess for rotation matrix.
  void detect({
    required Frame frame,
    // required Camera camera,
    // required TranslationVector? translationInitialGuess,
    // required RotationMatrix? rotationInitialGuess,
  }) {
    final (charucoCorners, charucoIds) =
        _detectMarkerCorners(frame: frame) ?? ([], []);
    if (charucoCorners.isEmpty || charucoIds.isEmpty) return;

    // TODO: Implement initial guesses
    // Initial guesses
    // translation_initial_guess = (
    //     translation_initial_guess.as_array()
    //     if translation_initial_guess is not None
    //     else np.zeros((3, 1))
    // )
    // rotation_initial_guess = (
    //     rotation_initial_guess.to_rodrigues()
    //     if rotation_initial_guess is not None
    //     else np.zeros((3, 1))
    // )

    // was_found, rotation, translation = cv2.aruco.estimatePoseCharucoBoard(
    //     charucoCorners,
    //     charucoIds,
    //     board,
    //     camera.matrix,
    //     camera.distorsion_coefficients,
    //     rotation_initial_guess,
    //     translation_initial_guess,
    // )
    // if not was_found:
    //     return None, None

    // # Reproject the corners to compute the reprojection error and filter out bad detections
    // reprojected_corners, _ = cv2.projectPoints(
    //     self._board.getChessboardCorners()[charuco_ids.flatten()],
    //     rvec=rotation,
    //     tvec=translation,
    //     cameraMatrix=camera.matrix,
    //     distCoeffs=camera.distorsion_coefficients,
    // )
    // error = np.mean(
    //     np.linalg.norm(reprojected_corners - charuco_corners, axis=2)
    // )
    // if error > 5.0:
    //     return None, None

    // # If we get here, the charuco is properly recognized
    // return (
    //     TranslationVector.from_array(translation),
    //     RotationMatrix(cv2.Rodrigues(rotation)[0]),
    // )
  }

  ///
  /// Detect markers and ChArUco corners in the given image.
  ///
  (List<List<Point2f>>, List<int>)? _detectMarkerCorners({
    required Frame frame,
  }) {
    final grayscaleFrame = frame.get(grayscale: true);
    final (corners, ids) = _detectMarkers(frame: grayscaleFrame) ?? ([], []);
    if (corners.isEmpty || ids.isEmpty) return null;

    // // Detect the charuco board
    // final detectorParameters = CharucoDetectorParameters.empty();
    // final detector = CharucoDetector.create(board.board, detectorParameters);
    // final (cornersTp, idsTp, _, _) = detector.detectBoard(frame);

    // corner_counts, charuco_corners, charuco_ids = (
    //     cv2.aruco.interpolateCornersCharuco(
    //         corners, ids, grayscale_frame, charuco._board
    //     )
    // )
    // if corner_counts == 0:
    //     return None

    return null; // TODO return the proper corners and ids
  }

  ///
  /// Detect ArUco markers in the given image, filtering to only include those belonging to the specified Charuco board.
  ///
  (List<Point2f>, List<int>)? _detectMarkers({required InputArray frame}) {
    final detector = ArucoDetector.create(
      arucoDict,
      ArucoDetectorParameters.empty(),
    );
    final (cornersTp, idsTp, _) = detector.detectMarkers(frame);
    if (cornersTp.isEmpty || idsTp.isEmpty) return null;
    if (cornersTp.length != idsTp.length) {
      throw StateError(
        "The number of detected corners and IDs should be the same.",
      );
    }

    //
    // Filter out the markers to only include those that belong to the current Charuco board
    // Strategy :
    //   1 - Remove all the markers which are not in the current board
    //   2 - Keep only unique markers IDs and compute the mid point of that reduced board
    //   3 - Reinsert all the markers which appear multiple times (other boards on screen) and
    //       remove them based on nearest distance to the mid-board position

    // Step 1 - Find all the markers that belong to the current board. It may include
    // markers shared with other boards at that point.
    final ids = <int>[];
    final corners = <Point2f>[];
    for (int i = 0; i < idsTp.length; i++) {
      if (markerIds.contains(idsTp[i])) {
        ids.add(idsTp[i]);
        corners.addAll(cornersTp[i].toList());
      }
    }
    if (ids.isEmpty) return null;

    // Step 2 - Keep only the unique IDs, which are definitely from the current board
    // to compute the mid point of the board.
    final uniqueIndices = ids.singleItemsIndices();
    final pointCount = uniqueIndices.length.toDouble();
    final midPoint = uniqueIndices
        .map((i) => corners[i])
        .fold(
          Point2f(0, 0),
          (weightedSum, corner) => Point2f(
            weightedSum.x + (corner.x / pointCount),
            weightedSum.y + (corner.y / pointCount),
          ),
        );

    // Step 3 - Insert all the markers. For those which appears multiple times
    // (meaning, markers shared across multiple boards), we need to determine which
    // one belongs to the current board by keeping only the one with the nearest
    // distance to the mid-board position
    final idsOfCurrentBoard = <int>[];
    final cornersOfCurrentBoard = <Point2f>[];
    for (int i = 0; i < ids.length; i++) {
      if (idsOfCurrentBoard.contains(ids[i])) {
        // This marker has already been processed, we skip it
        continue;
      }

      if (uniqueIndices.contains(i)) {
        // This is a unique marker, we keep it
        idsOfCurrentBoard.add(ids[i]);
        cornersOfCurrentBoard.add(corners[i]);
        continue;
      }

      // Find all the indices of that marker ID
      final allIndices = [
        for (int j = 0; j < ids.length; j++)
          if (ids[j] == ids[i]) j,
      ];

      var minDistance = double.infinity;
      var bestSoFar = -1;
      for (final index in allIndices) {
        final corner = corners[index];
        final x = corner.x - midPoint.x;
        final y = corner.y - midPoint.y;
        final distanceSquared = x * x + y * y;
        if (distanceSquared < minDistance) {
          minDistance = distanceSquared;
          bestSoFar = index;
        }
      }
      if (i != bestSoFar) {
        // This marker appears multiple times, but the one at index "i" is not the
        // closest to the mid-board position, so we skip it. This is important
        // to preserve the order of the markers
        continue;
      }
      idsOfCurrentBoard.add(ids[bestSoFar]);
      cornersOfCurrentBoard.add(corners[bestSoFar]);
    }

    return (cornersOfCurrentBoard, idsOfCurrentBoard);
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
