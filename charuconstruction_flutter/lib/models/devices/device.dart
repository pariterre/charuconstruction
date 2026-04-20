import 'dart:async';

import 'package:charuconstruction_flutter/models/data/time_series_data.dart';
import 'package:charuconstruction_flutter/utils/generic_listener.dart';

abstract class Device {
  ///
  /// Data holder for the current device
  late final _data = TimeSeriesData(
    channelCount: channelCount,
    isFromLiveData: true,
  );
  TimeSeriesData get data => _data;

  ///
  /// The data vector of the data received from the device. This method is expected
  /// to be called by the device implementation when new data is received.
  /// [timestamp] is the timestamp of the data
  /// [values] is the list of values of the data, in the same order as the channels of the device
  Future<void> pushData(DateTime timestamp, List<double> values) async {
    _data.appendFromJson({
      'data': [
        [timestamp.millisecondsSinceEpoch, values],
      ],
    });
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
    _data.clear();
  }
}
