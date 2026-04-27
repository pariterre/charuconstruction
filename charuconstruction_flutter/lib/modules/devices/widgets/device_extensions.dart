import 'package:flutter/cupertino.dart';

import '../charuco/frame_analyser.dart';
import '../charuco/widgets/media_reader_container.dart';
import '../concrete_devices/b24_force_sensor.dart';
import '../concrete_devices/dual_charucos.dart';
import '../device.dart';
import 'device_data_container.dart';

extension DeviceExtensions on Device {
  ///
  /// A widget to display the device data. Will be implemented by each device.
  ///
  Widget deviceDataContainer({required Key key, required Device device}) {
    return switch (device) {
      B24ForceSensor b24 => DeviceDataContainer(key: key, device: b24),
      DualCharucos dualCharucos => MediaReaderContainer(
        key: key,
        mediaReader: dualCharucos.mediaReader,
        analyser:
            dualCharucos.analysers ?? FrameAnalyserPipeline(analysers: []),
      ),
      _ => Text('No data available for this device'),
    };
  }
}
