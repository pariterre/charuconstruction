import 'package:flutter/cupertino.dart';

import '../charuco/charuco_device.dart';
import '../charuco/widgets/media_reader_container.dart';
import '../concrete_devices/b24_force_sensor.dart';
import '../device.dart';
import 'device_data_container.dart';

extension DeviceExtensions on Device {
  ///
  /// A widget to display the device data. Will be implemented by each device.
  ///
  Widget deviceDataContainer({required Device device}) {
    return switch (device) {
      B24ForceSensor b24 => DeviceDataContainer(device: b24),
      WebcamCharucos dualCharucos => CameraFrameContainer(device: dualCharucos),
      _ => Text('No data available for this device'),
    };
  }
}
