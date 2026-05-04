import 'package:charuconstruction_flutter/modules/data/data.dart';
import 'package:charuconstruction_flutter/utils/generic_listener.dart';

import '../concrete_devices/available_devices.dart';
import '../device.dart';

class DevicesProvider {
  ///
  /// The available devices
  ///
  final _devices = <AvailableDevices, Device>{};

  ///
  /// The data collected from the devices
  ///
  final _data = Data(initialTime: DateTime.now(), isFromLiveData: true);
  final _recordingData = Data(
    initialTime: DateTime.now(),
    isFromLiveData: true,
  );

  ///
  /// Prepare the singleton instance
  ///
  DevicesProvider._() {
    _data.clear(initialTime: DateTime.now(), keepDevices: false);
    _recordingData.clear(initialTime: DateTime.now(), keepDevices: false);

    for (final device in AvailableDevices.values) {
      final newDevice = device.factory();

      // Add the device to the list of devices
      _devices[device] = newDevice;

      // Add a reference to the device data in the data pool
      _data.add(device.name, newDevice.data);
      _recordingData.add(device.name, newDevice.recordingData);
    }

    // Reset the initial time for all the devices
    clearData();
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

  ///
  /// Save the data collected from the devices
  ///
  Future<void> saveData() async {
    await _recordingData.toFiles();
  }

  ///
  /// Put all the devices in recording mode, if they are not already in recording mode
  ///
  Future<void> startRecording() async {
    final now = DateTime.now();
    for (final device in connectedDevices) {
      if (device.isNotRecording) await device.startRecording(startingTime: now);
    }
  }

  Future<void> stopRecording() async {
    for (final device in connectedDevices) {
      if (device.isRecording) await device.stopRecording();
    }
  }

  ///
  /// Clear the data collected from the devices and reset the initial time
  ///
  Future<void> clearData() async {
    _data.clear(initialTime: DateTime.now(), keepDevices: true);
    _recordingData.clear(initialTime: DateTime.now(), keepDevices: true);
  }

  ///
  /// Set the zero offset for all connected devices
  ///
  Future<void> setZero() async {
    // Make sure all devices are in recording mode
    await startRecording();

    // Show the user that we drop all data prior to the current time
    await clearData();

    // Wait for more than one second of data
    await Future.delayed(Duration(seconds: 1, milliseconds: 500));

    // Set the zero for all connected devices
    for (final device in connectedDevices) {
      await device.setZero();
    }

    // Reset the data so that the new data will be zeroed out
    await clearData();
  }
}
