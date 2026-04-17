class BleDeviceBluetoothOff implements Exception {
  final String message;
  BleDeviceBluetoothOff(this.message);

  @override
  String toString() => 'BleDeviceBluetoothOff: $message';
}

class BleDeviceNotFound implements Exception {
  final String message;
  BleDeviceNotFound(this.message);

  @override
  String toString() => 'BleDeviceNotFound: $message';
}

class BleDeviceCouldNotConnect implements Exception {
  final String message;
  BleDeviceCouldNotConnect(this.message);

  @override
  String toString() => 'BleDeviceCouldNotConnect: $message';
}

class BleDeviceCouldNotDisconnect implements Exception {
  final String message;
  BleDeviceCouldNotDisconnect(this.message);

  @override
  String toString() => 'BleDeviceCouldNotDisconnect: $message';
}

class BleDeviceNotConnected implements Exception {
  final String message;
  BleDeviceNotConnected(this.message);

  @override
  String toString() => 'BleDeviceNotConnected: $message';
}

class CharacteristicNotFound implements Exception {
  final String message;
  CharacteristicNotFound(this.message);

  @override
  String toString() => 'CharacteristicNotFound: $message';
}

class CharacteristicWriteFailed implements Exception {
  final String message;
  CharacteristicWriteFailed(this.message);

  @override
  String toString() => 'CharacteristicWriteFailed: $message';
}
