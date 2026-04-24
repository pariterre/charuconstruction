import 'package:charuconstruction_flutter/modules/data/data.dart';
import 'package:charuconstruction_flutter/utils/generic_listener.dart';

import '../ble/b24_force_sensor.dart';
import '../device.dart';

enum AvailableDevices {
  b24;

  Device _factory() {
    // Setup the device
    final device = switch (this) {
      AvailableDevices.b24 => B24ForceSensor(),
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
  ///
  /// The available devices
  ///
  final _devices = <AvailableDevices, Device>{};

  ///
  /// The data collected from the devices
  ///
  final _data = Data(initialTime: DateTime.now(), isFromLiveData: true);

  ///
  /// Prepare the singleton instance
  ///
  DevicesProvider._() {
    for (final device in AvailableDevices.values) {
      final newDevice = device._factory();

      // Add the device to the list of devices
      _devices[device] = newDevice;

      // Add a reference to the device data in the data pool
      _data.add(device.name, newDevice.data);
      // Reset the initial time for all the devices
      _data.clear(initialTime: DateTime.now());
    }
  }
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
