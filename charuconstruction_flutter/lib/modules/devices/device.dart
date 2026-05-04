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
  /// Channel to show by default (on graphs)
  ///
  List<bool> get channelToShowByDefault;

  ///
  /// Whether the device is currently connected.
  bool _isConnected = false;
  bool get isConnected => _isConnected;
  bool get isDisconnected => !isConnected;
  bool get isNotConnected => isDisconnected;

  ///
  /// Whether the device is in recording mode.
  bool _isRecording = false;
  bool get isRecording => _isRecording;
  bool get isNotRecording => !isRecording;

  ///
  /// Data holder for the current device
  late final data = TimeSeriesData(
    channelCount: channelCount,
    isFromLiveData: true,
    maxSize: 200,
  );
  late final recordingData = TimeSeriesData(
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
      await stopRecording();
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
  Future<void> startRecording({DateTime? startingTime}) async {
    if (isNotConnected) {
      throw DeviceNotConnected(
        'Cannot start recording: device is not connected, call connect() to the device first.',
      );
    }

    startingTime ??= DateTime.now();
    data.clear(timeOffset: startingTime);
    recordingData.clear(timeOffset: startingTime);

    _isRecording = true;
    onReadingStatusChanged.notifyListeners((listener) => listener(isRecording));
    _logger.info('Reading data!');
  }

  ///
  /// Stop reading data from the device.
  ///
  Future<void> stopRecording() async {
    if (isNotConnected) {
      throw DeviceNotConnected('Cannot stop reading: device is not connected.');
    }

    _isRecording = false;
    onReadingStatusChanged.notifyListeners((listener) => listener(isRecording));
    _logger.info('Stopped recording data!');
  }

  void clearData({required DateTime? timeOffset}) {
    data.clear(timeOffset: timeOffset);
    recordingData.clear(timeOffset: timeOffset);
  }

  ///
  /// Set the current data as the zero of the device. It takes the full data
  /// vector into account. Therefore, clear data should be called before setting the zero
  ///
  Future<void> setZero();

  ///
  /// The data vector of the data received from the device. This method is expected
  /// to be called by the device implementation when new data is received.
  /// [timestamp] is the timestamp of the data
  /// [channels] is the list of values of the data, in the same order as the channels of the device
  ///
  Future<void> pushData(DateTime timestamp, List<double> channels) async {
    data.add([timestamp.millisecondsSinceEpoch], [channels]);
    if (_isRecording) {
      recordingData.add([timestamp.millisecondsSinceEpoch], [channels]);
    }
  }
}
