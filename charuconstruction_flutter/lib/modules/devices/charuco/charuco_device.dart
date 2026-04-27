import 'camera.dart';

import 'charuco.dart';

import '../device.dart';

class CharucoDevice extends Device {
  ///
  /// The list of Charuco boards that the device is configured to read.
  final List<Charuco> _charucos = [];
  List<Charuco> get charucos => List.unmodifiable(_charucos);

  ///
  /// The camera used to read the Charuco boards.
  Camera? _camera;
  Camera? get camera => _camera;

  @override
  String? get name => 'Charuco Device';

  @override
  int get channelCount => 9 * _charucos.length;

  @override
  Future<void> connect({List<Charuco>? charucos, Camera? camera}) async {
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
    _camera = camera;

    _charucos.addAll(charucos);

    await super.connect();
  }

  @override
  Future<void> disconnect() async {
    _charucos.clear();
    _camera = null;
    await super.disconnect();
  }
}
