import 'package:charuconstruction_flutter/devices/device.dart';

class DevicesProvider {
  ///
  /// Prepare the singleton instance
  ///
  DevicesProvider._();
  static final DevicesProvider _instance = DevicesProvider._();
  static DevicesProvider get instance => _instance;

  ///
  /// The list of available devices.
  ///
  final List<Device> _devices = [];
  List<Device> get devices => _devices;

  ///
  /// Add a device to the list of available devices.
  ///
  Future<void> add(Device device) async {
    if (_devices.contains(device)) {
      throw Exception('Device already exists in the provider');
    }
    _devices.add(device);
  }
}
