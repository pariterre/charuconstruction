class BleDeviceBluetoothOff implements Exception {
  final String message;
  BleDeviceBluetoothOff(this.message);

  @override
  String toString() => 'BleDeviceBluetoothOff: $message';
}

class BleCharacteristicNotFound implements Exception {
  final String message;
  BleCharacteristicNotFound(this.message);

  @override
  String toString() => 'CharacteristicNotFound: $message';
}

class BleCharacteristicWriteFailed implements Exception {
  final String message;
  BleCharacteristicWriteFailed(this.message);

  @override
  String toString() => 'CharacteristicWriteFailed: $message';
}
