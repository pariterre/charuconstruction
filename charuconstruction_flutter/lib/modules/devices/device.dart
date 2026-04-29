import 'dart:async';

import 'package:charuconstruction_flutter/modules/data/time_series_data.dart';
import 'package:charuconstruction_flutter/modules/devices/devices.dart';
import 'package:charuconstruction_flutter/utils/generic_listener.dart';
import 'package:logging/logging.dart';

final _logger = Logger('Device');

abstract class Device {
  ///
  /// The name of the device.
  String? get name;

  ///
  /// The number of channels of the device.
  ///
  int get channelCount;

  ///
  /// Whether the device is currently connected.
  bool _isConnected = false;
  bool get isConnected => _isConnected;
  bool get isDisconnected => !isConnected;
  bool get isNotConnected => isDisconnected;

  ///
  /// Whether the device is in reading mode.
  bool _isReading = false;
  bool get isReading => _isReading;
  bool get isNotReading => !isReading;

  ///
  /// Data holder for the current device
  late final data = TimeSeriesData(
    channelCount: channelCount,
    isFromLiveData: true,
  );

  final onConnectionStatusChanged =
      GenericListener<void Function(bool isConnected)>();

  ///
  /// Connect to the device. If the device requires a pin, it should be provided as an argument.
  ///
  Future<void> connect() async {
    clearData(timeOffset: data.startingTime);

    _isConnected = true;

    onConnectionStatusChanged.notifyListeners(
      (listener) => listener(isConnected),
    );
  }

  ///
  /// Disconnect from the device.
  ///
  Future<void> disconnect() async {
    try {
      await stopReading();
    } catch (e) {
      throw DeviceCouldNotDisconnect('Failed to disconnect: $e');
    }

    _isConnected = false;
    onConnectionStatusChanged.notifyListeners(
      (listener) => listener(isConnected),
    );
  }

  final onReadingStatusChanged =
      GenericListener<void Function(bool isReading)>();

  ///
  /// Start reading data from the device. The device should be configured to send data at this point.
  /// The user must have subscribed to the [onNewData] stream before calling this method, otherwise the data will be lost.
  ///
  Future<void> startReading() async {
    if (isNotConnected) {
      throw DeviceNotConnected(
        'Cannot start reading: device is not connected, call connect() to the device first.',
      );
    }

    _isReading = true;
    onReadingStatusChanged.notifyListeners((listener) => listener(isReading));
    _logger.info('Reading data!');
  }

  ///
  /// Stop reading data from the device.
  ///
  Future<void> stopReading() async {
    if (isNotConnected) {
      throw DeviceNotConnected('Cannot stop reading: device is not connected.');
    }

    _isReading = false;
    onReadingStatusChanged.notifyListeners((listener) => listener(isReading));
    _logger.info('Stopped reading data!');
  }

  void clearData({required DateTime? timeOffset}) {
    data.clear(timeOffset: timeOffset);
  }

  ///
  /// Set the current data as the zero of the device.
  ///
  Future<void> setZero();

  ///
  /// The data vector of the data received from the device. This method is expected
  /// to be called by the device implementation when new data is received.
  /// [timestamp] is the timestamp of the data
  /// [values] is the list of values of the data, in the same order as the channels of the device
  ///
  Future<void> pushData(DateTime timestamp, List<double> values) async {
    data.appendFromJson({
      'data': [
        [timestamp.millisecondsSinceEpoch, values],
      ],
    });
  }
}
