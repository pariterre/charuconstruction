import 'dart:async';

import 'package:charuconstruction_flutter/utils/generic_listener.dart';

abstract class Device {
  ///
  /// The time when the [startReading] was called. This can be used to calculate the relative time of subsequent data points.
  /// It can be reset by calling [clearData].
  ///
  DateTime _startingTime = DateTime.now();
  DateTime get startingTime => _startingTime;

  ///
  /// The time vector of the data received from the device. This vector has the same length as the [dataVector].
  ///
  final List<DateTime> _timeVector = [];
  List<DateTime> get timeVector => _timeVector;

  ///
  /// The data vector of the data received from the device. The outer length of this vector has the same length as the [timeVector].
  /// The inner length of this vector depends on the device (i.e. the channel count).
  final List<List<double>> _data = [];
  List<List<double>> get dataVector => _data;

  Future<void> pushData(DateTime timestamp, List<double> values) async {
    _timeVector.add(timestamp);
    _data.add(values);
    await onNewData.notifyListeners((listener) => listener(timestamp, values));
  }

  ///
  /// The name of the device.
  String? get name;

  ///
  /// The MAC address of the device.
  String? get macAddress;

  ///
  /// The number of channels of the device.
  ///
  int get channelCount;

  ///
  /// Whether the device is currently connected.
  ///
  bool get isConnected;
  bool get isDisconnected => !isConnected;
  bool get isNotConnected => isDisconnected;

  final onConnectionStatusChanged =
      GenericListener<void Function(bool isConnected)>();

  ///
  /// Connect to the device. If the device requires a pin, it should be provided as an argument.
  ///
  Future<void> connect();

  ///
  /// Disconnect from the device.
  ///
  Future<void> disconnect();

  ///
  /// Whether the device is currently reading data.
  ///
  bool get isReading;
  bool get isNotReading => !isReading;

  final onReadingStatusChanged =
      GenericListener<void Function(bool isReading)>();

  ///
  /// Start reading data from the device. The device should be configured to send data at this point.
  /// The user must have subscribed to the [onNewData] stream before calling this method, otherwise the data will be lost.
  ///
  Future<void> startReading();

  ///
  /// Stop reading data from the device.
  ///
  Future<void> stopReading();

  void clearData() {
    _startingTime = DateTime.now();
    _timeVector.clear();
    _data.clear();
  }

  ///
  /// Stream of incoming data from the device. The data should be emitted as [timestamp], [values] by the device implementation
  /// when new data is received.
  ///
  final onNewData =
      GenericListener<void Function(DateTime timestamp, List<double> values)>();
}
