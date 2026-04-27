import 'package:ml_linalg/linalg.dart';

import '../charuco/camera.dart';
import '../charuco/charuco.dart';
import '../charuco/charuco_device.dart';
import '../charuco/extensions.dart';
import '../charuco/frame_analyser.dart';
import '../charuco/media_reader.dart';

class DualCharucos extends CharucoDevice {
  @override
  String get name => "Dual Charucos";

  ///
  /// The [MediaReader] associated with this device. Will be initialized when the device is connected.
  MediaReader? _mediaReader;
  MediaReader get mediaReader => _mediaReader!;

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

    _mediaReader = CharucoMockReader(
      camera: this.camera!,
      charucos: this.charucos,
      transformations: List.generate(
        100,
        (value) => [
          (Vector.zero(3), Matrix.identity(3)),
          (
            Vector.fromList([0, 0, 4.5]),
            MatrixExtensions.fromEuler([
              (value * 0.5, CharucoAxis.x),
              (-20.0, CharucoAxis.y),
              (-30.0, CharucoAxis.z),
            ]),
          ),
        ],
      ),
    );

    _analysers = FrameAnalyserPipeline(analysers: analysers);

    return output;
  }
}
