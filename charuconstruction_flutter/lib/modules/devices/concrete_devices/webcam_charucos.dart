import 'dart:async';
import 'dart:math';

import 'package:ml_linalg/linalg.dart';

import '../charuco/camera.dart';
import '../charuco/charuco.dart';
import '../charuco/charuco_device.dart';
import '../charuco/extensions.dart';
import '../charuco/frame_analyser.dart';
import '../charuco/media_reader.dart';

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

  @override
  MediaReader? get mediaReader => _webcamReader;

  @override
  Future<void> connect({
    List<Charuco>? charucos,
    Camera? camera,
    List<FrameAnalyser> analysers = const [],
  }) async {
    _webcamReader = WebcamReader();

    await _webcamReader!.initialize();
    return await super.connect(
      charucos: charucos,
      camera: camera,
      analysers: analysers,
    );
  }

  @override
  Future<void> disconnect() async {
    _webcamReader = null;
    return await super.disconnect();
  }
}

class MockedDualCharucos extends WebcamDualCharucos {
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

    final output = await super.connect(
      charucos: charucos,
      camera: camera,
      analysers: analysers,
    );

    _webcamReader = CharucoMockReader(
      camera: camera,
      charucos: charucos,
      transformations: Stream.periodic(
        const Duration(milliseconds: 100),
        _generateValue,
      ),
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
