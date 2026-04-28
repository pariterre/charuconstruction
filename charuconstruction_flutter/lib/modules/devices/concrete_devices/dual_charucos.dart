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

  static DualCharucos factory() {
    return switch (charucoToConstruct) {
      AvailableDualCharucos.dualCharucosFromDevice => WebcamDualCharucos(),
      AvailableDualCharucos.dualCharucosMocker => MockedDualCharucos(),
    };
  }
}

abstract class DualCharucos extends CharucoDevice {
  @override
  String get name => "Dual Charucos";

  ///
  /// The [MediaReader] associated with this device. Will be initialized when the device is connected.
  MediaReader get mediaReader;

  ///
  /// The [FrameAnalyser] associated with this device. Will be initialized when the device is connected.
  FrameAnalyser? _analysers;
  FrameAnalyser? get analysers => _analysers;

  @override
  Future<void> connect({
    List<Charuco>? charucos,
    Camera? camera,
    List<FrameAnalyser> analysers = const [],
  }) async {
    final output = await super.connect(charucos: charucos, camera: camera);

    _analysers = FrameAnalyserPipeline(analysers: analysers);

    return output;
  }
}

class WebcamDualCharucos extends DualCharucos {
  late final WebcamReader _webcamReader = WebcamReader();

  @override
  MediaReader get mediaReader => _webcamReader;

  @override
  Future<void> connect({
    List<Charuco>? charucos,
    Camera? camera,
    List<FrameAnalyser> analysers = const [],
  }) async {
    await _webcamReader.initialize();
    return await super.connect(
      charucos: charucos,
      camera: camera,
      analysers: analysers,
    );
  }

  @override
  Future<void> startReading() async {
    await _webcamReader.startReading();
    return await super.startReading();
  }

  @override
  Future<void> stopReading() async {
    await _webcamReader.stopReading();
    return await super.stopReading();
  }

  @override
  Future<void> disconnect() async {
    _webcamReader.dispose();
    return await super.disconnect();
  }
}

class MockedDualCharucos extends DualCharucos {
  MediaReader? _mediaReader;
  @override
  MediaReader get mediaReader => _mediaReader!;

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

    _mediaReader = CharucoMockReader(
      camera: camera,
      charucos: charucos,
      transformations: Stream.periodic(
        const Duration(milliseconds: 100),
        _generateValue,
      ),
    );

    return await super.connect(
      charucos: charucos,
      camera: camera,
      analysers: analysers,
    );
  }

  @override
  Future<void> disconnect() async {
    _mediaReader = null;
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
