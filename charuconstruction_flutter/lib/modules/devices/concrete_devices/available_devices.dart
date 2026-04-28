import '../device.dart';
import '../providers/devices_provider.dart';
import 'b24_force_sensor.dart';
import 'webcam_charucos.dart';

enum AvailableDevices {
  b24,
  dualCharucos;

  Device factory() {
    // Setup the device
    final device = switch (this) {
      AvailableDevices.b24 => B24ForceSensor(),
      AvailableDevices.dualCharucos => AvailableDualCharucos.factory(),
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
