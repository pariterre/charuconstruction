import 'package:charuconstruction_flutter/devices/b24_force_sensor.dart';
import 'package:charuconstruction_flutter/devices/device.dart';
import 'package:charuconstruction_flutter/utils/generic_listener.dart';

const useB24Mocker = bool.fromEnvironment(
  'CHARUCONSTRUCTION_USE_B24_MOCKER',
  defaultValue: false,
);

enum AvailableDevices {
  b24;

  Device _factory() {
    // Setup the device
    final device = switch (this) {
      AvailableDevices.b24 =>
        useB24Mocker ? B24ForceSensorMocker() : B24ForceSensor(),
    };

    // Listen to status changes and notify the provider
    device.onConnectionStatusChanged.listen((_) {
      DevicesProvider.instance.onDeviceStatusChanged.notifyListeners(
        (callback) => callback(this),
      );
    });
    device.onReadingStatusChanged.listen((_) {
      DevicesProvider.instance.onDeviceStatusChanged.notifyListeners(
        (callback) => callback(this),
      );
    });

    // Return the device ready to be used
    return device;
  }
}

class DevicesProvider {
  /// The available devices
  final _devices = {
    for (var device in AvailableDevices.values) device: device._factory(),
  };

  ///
  /// Prepare the singleton instance
  ///
  DevicesProvider._();
  static final DevicesProvider _instance = DevicesProvider._();
  static DevicesProvider get instance => _instance;

  ///
  /// A specific device
  /// [device] is the device to get
  ///
  Device device(AvailableDevices device) => _devices[device]!;

  ///
  /// A notifier to know when a device status changed
  ///
  final onDeviceStatusChanged =
      GenericListener<void Function(AvailableDevices device)>();

  ///
  /// A list of the connected devices
  ///
  List<Device> get connectedDevices =>
      _devices.values.where((d) => d.isConnected).toList();
}
