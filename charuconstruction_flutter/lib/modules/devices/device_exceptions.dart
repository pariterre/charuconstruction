class DeviceNotFound implements Exception {
  final String message;
  DeviceNotFound(this.message);

  @override
  String toString() => 'DeviceNotFound: $message';
}

class DeviceCouldNotConnect implements Exception {
  final String message;
  DeviceCouldNotConnect(this.message);

  @override
  String toString() => 'DeviceCouldNotConnect: $message';
}

class DeviceCouldNotDisconnect implements Exception {
  final String message;
  DeviceCouldNotDisconnect(this.message);

  @override
  String toString() => 'DeviceCouldNotDisconnect: $message';
}

class DeviceNotConnected implements Exception {
  final String message;
  DeviceNotConnected(this.message);

  @override
  String toString() => 'DeviceNotConnected: $message';
}
