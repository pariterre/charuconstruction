import 'dart:async';
import 'dart:math';

import 'package:charuconstruction_flutter/modules/devices/charuco/frame.dart';
import 'package:logging/logging.dart';
import 'package:ml_linalg/linalg.dart';

import '../charuco/camera.dart';
import '../charuco/charuco.dart';
import '../charuco/charuco_device.dart';
import '../charuco/extensions.dart';
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
  MediaReader? _webcamReader;
  set webcamReader(MediaReader? reader) => _webcamReader = reader;

  @override
  MediaReader? get mediaReader => _webcamReader;

  @override
  Future<void> connect({
    List<Charuco>? charucos,
    Camera? camera,
    List<FrameAnalyser> analysers = const [],
  }) async {
    webcamReader = WebcamReader();

    return await super.connect(
      charucos: charucos,
      camera: camera,
      analysers: analysers,
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
    final rotationFirst = data.first?.$2;
    final rotationSecond = data.last?.$2;

    if (rotationFirst == null || rotationSecond == null) {
      _logger.warning(
        'Expected charucos to have data for dual charuco device, but got some with null data',
      );
      return;
    }

    final transformation = rotationFirst.transpose() * rotationSecond;
    final results = transformation
        .toEuler(sequence: CharucoAxisSequence.yzx)
        .toList();

    pushData(now, results);
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
      _webcamReader = CharucoMockReader(
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
          -2.0 + i * 4.0,
          0.0,
          4.5 + 0.5 * sin(0.1 * (value + i * 20)),
        ]),
        MatrixExtensions.fromEuler([
          (30.0 * sin(0.005 * (3 * value + i * 20)), CharucoAxis.x),
          (20 * cos(0.01 * (4 * value + i * 20)), CharucoAxis.y),
          (15.0 * sin(0.005 * (5 * value + i * 20)), CharucoAxis.z),
        ]),
      ));
    }
    return data;
  }
}
