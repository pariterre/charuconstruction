import 'dart:async';

import 'package:collection/collection.dart';
import 'package:logging/logging.dart';
import 'package:advance_math/advance_math.dart';

import '../charuco/camera.dart';
import '../charuco/charuco.dart';
import '../charuco/charuco_device.dart';
import '../charuco/extensions.dart';
import '../charuco/frame.dart';
import '../charuco/frame_analyser.dart';
import '../charuco/media_reader.dart';

final _logger = Logger('WebcamCharucos');

enum AvailableDualCharucos {
  dualCharucosFromDevice,
  dualCharucosMocker;

  static AvailableDualCharucos charucoToConstruct =
      AvailableDualCharucos.dualCharucosFromDevice;

  static WebcamCharucos factory() {
    return switch (charucoToConstruct) {
      AvailableDualCharucos.dualCharucosFromDevice => WebcamDualCharucos(),
      AvailableDualCharucos.dualCharucosMocker => MockedDualCharucos(),
    };
  }
}

class WebcamDualCharucos extends WebcamCharucos {
  Matrix _transposedZeroRotationFirst = Matrix.eye(3, isDouble: true);
  Matrix _transposedZeroRotationSecond = Matrix.eye(3, isDouble: true);

  MediaReader? _webcamReader;
  set webcamReader(MediaReader? reader) => _webcamReader = reader;

  @override
  MediaReader? get mediaReader => _webcamReader;

  @override
  int get channelCount => 27; // 2x (3-trans + 9-rot) + 3-euleur

  @override
  List<bool> get channelToShowByDefault =>
      List.filled(12, false) + List.filled(12, false) + List.filled(3, true);

  @override
  Future<void> connect({
    List<Charuco>? charucos,
    Camera? camera,
    List<FrameAnalyser> analysers = const [],
    WebcamResolution resolution = WebcamResolution.medium,
    WebcamFPS fps = WebcamFPS.fps60,
    bool hideVideo = false,
  }) async {
    webcamReader = WebcamReader();

    return await super.connect(
      charucos: charucos,
      camera: camera,
      analysers: analysers,
      resolution: resolution,
      fps: fps,
      hideVideo: hideVideo,
    );
  }

  @override
  Future<void> disconnect() async {
    webcamReader = null;
    return await super.disconnect();
  }

  @override
  Future<(Frame?, Map<AvailableExtraAnalyses, dynamic>?)> pushDataFrame(
    Frame? frame,
  ) async {
    final now = DateTime.now();
    final (analyzedFrame, extraAnalyses) = await super.pushDataFrame(frame);

    if (extraAnalyses?.containsKey(
          AvailableExtraAnalyses.charucosReconstruction,
        ) ??
        false) {
      final charucos =
          extraAnalyses![AvailableExtraAnalyses.charucosReconstruction]!
              as Map<Charuco, (Vector?, Matrix?)?>;
      _pushReconstructedCharucos(now: now, charucos: charucos);
    }

    return (frame, extraAnalyses);
  }

  @override
  Future<void> setZero() async {
    // Aliases
    final frames = data.getData();
    final frameCount = data.length;
    const firstBase = 3;
    const secondBase = 3 + 9 + 3;

    // Compute the mean matrice of the last second using SDV decomposition of mean matrices
    final rotationFirstAsList = List.filled(9, 0.0);
    final rotationSecondAsList = List.filled(9, 0.0);
    for (int i = 0; i < 9; i++) {
      double sumFirst = 0.0;
      double sumSecond = 0.0;

      for (int frame = 0; frame < data.length; frame++) {
        sumFirst += frames[firstBase + i][frame];
        sumSecond += frames[secondBase + i][frame];
      }
      rotationFirstAsList[i] = sumFirst / frameCount;
      rotationSecondAsList[i] = sumSecond / frameCount;
    }
    final rotationFirst = Matrix.fromFlattenedList(rotationFirstAsList, 3, 3);
    final rotationSecond = Matrix.fromFlattenedList(rotationSecondAsList, 3, 3);

    // Setting them to null will make the next frame as the new zero
    _transposedZeroRotationFirst = _orthogonalize(rotationFirst).transpose();
    _transposedZeroRotationSecond = _orthogonalize(rotationSecond).transpose();
  }

  Future<void> _pushReconstructedCharucos({
    required DateTime now,
    required Map<Charuco, (Vector?, Matrix?)?> charucos,
  }) async {
    if (charucos.length != 2) {
      _logger.warning(
        'Expected 2 charucos for dual charuco device, but got ${charucos.length}',
      );
      return;
    }

    final data = charucos.values.toList();
    final translationFirst = data.first?.$1;
    final rotationFirst = data.first?.$2;
    final translationSecond = data.last?.$1;
    final rotationSecond = data.last?.$2;

    if (translationFirst == null ||
        rotationFirst == null ||
        translationSecond == null ||
        rotationSecond == null) {
      _logger.warning(
        'Expected charucos to have data for dual charuco device, but got some with null data',
      );
      return;
    }

    // TODO Introduce anatomical offset?
    final firstZeroed = _transposedZeroRotationFirst * rotationFirst;
    final secondZeroed = _transposedZeroRotationSecond * rotationSecond;

    final transformation = firstZeroed.transpose() * secondZeroed;
    final angles = transformation
        .toEuler(sequence: CharucoAxisSequence.yzx)
        .toList();

    final output =
        translationFirst.toList(growable: false) +
        rotationFirst.flattened.toList(growable: false) +
        translationSecond.toList(growable: false) +
        rotationSecond.flattened.toList(growable: false) +
        angles;
    pushData(now, [
      for (final val in output) val is Complex ? val.sign * val.magnitude : val,
    ]);
  }
}

Matrix _orthogonalize(Matrix m) {
  final svd = m.decomposition.singularValueDecomposition();

  final orthogonal = svd.U * svd.V;
  if (orthogonal.determinant() < 0) {
    return svd.U * Matrix.fromDiagonal([1.0, 1.0, -1.0]) * svd.V;
  } else {
    return orthogonal;
  }
}

class MockedDualCharucos extends WebcamDualCharucos {
  Camera? _internalCamera;
  List<Charuco>? _internalCharucos;

  @override
  set webcamReader(MediaReader? reader) {
    // Hijack the setting of the camera to insert the mocker
    if (reader == null) {
      super.webcamReader = null;
    } else {
      _webcamReader = CharucoMockWebcamReader(
        camera: _internalCamera!,
        charucos: _internalCharucos!,
        transformations: Stream.periodic(
          const Duration(milliseconds: 100),
          _generateValue,
        ),
      );
    }
  }

  @override
  Future<void> connect({
    List<Charuco>? charucos,
    Camera? camera,
    List<FrameAnalyser> analysers = const [],
    WebcamResolution resolution = WebcamResolution.medium,
    WebcamFPS fps = WebcamFPS.fps60,
    bool hideVideo = false,
  }) async {
    if (charucos == null) {
      throw ArgumentError(
        'Charucos must be provided to connect to a CharucoDevice',
      );
    }
    if (camera == null) {
      throw ArgumentError(
        'Camera must be provided to connect to a CharucoDevice',
      );
    }

    _internalCamera = camera;
    _internalCharucos = charucos;
    final output = await super.connect(
      charucos: charucos,
      camera: camera,
      analysers: analysers,
      resolution: resolution,
      fps: fps,
      hideVideo: hideVideo,
    );

    return output;
  }

  @override
  Future<void> disconnect() async {
    _webcamReader = null;
    return await super.disconnect();
  }

  List<(Vector, Matrix)> _generateValue(int value) {
    final data = <(Vector, Matrix)>[];
    for (int i = 0; i < (charucos.length); i++) {
      data.add((
        Vector.fromList([
          Complex(-2.0 + i * 4.0),
          Complex(0.0),
          Complex(4.5 + 0.5 * sin(0.1 * (value + i * 20.0))),
        ]),
        MatrixExtensions.fromEuler([
          (30.0 * sin(0.005 * (3.0 * value + i * 20.0)), CharucoAxis.x),
          (20.0 * cos(0.01 * (4.0 * value + i * 20.0)), CharucoAxis.y),
          (15.0 * sin(0.005 * (5.0 * value + i * 20.0)), CharucoAxis.z),
        ]),
      ));
    }
    return data;
  }
}
