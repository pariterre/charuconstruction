import 'dart:io';

import 'package:charuconstruction_flutter/utils/generic_listener.dart';

class TimeSeriesData {
  bool isFromLiveData;
  int? _timeOffset; // milliseconds since epoch at the start of the recording
  DateTime? get startingTime => _timeOffset != null
      ? DateTime.fromMillisecondsSinceEpoch(_timeOffset!)
      : null;
  final int _channelCount;
  int get channelCount => _channelCount;

  final List<int> time = []; // In milliseconds since t0
  final List<List<double>> _data;
  List<List<double>> getData() => _data;

  final onNewData = GenericListener<Function()>();

  void clear() {
    resetTime();
    for (var channel in _data) {
      channel.clear();
    }
  }

  void resetTime() {
    time.clear();
    _timeOffset = null;
  }

  int get length => time.length;
  bool get isEmpty => time.isEmpty;
  bool get isNotEmpty => time.isNotEmpty;

  TimeSeriesData({required int channelCount, required this.isFromLiveData})
    : _channelCount = channelCount,
      _data = List.generate(channelCount, (_) => <double>[]);

  ///
  /// Append data from a JSON object, the JSON object should be in the format of:
  /// {
  ///   "data": [
  ///     [time, channel_1_value, channel_2_value, ...],
  ///     [time, channel_1_value, channel_2_value, ...],
  ///     ...
  ///   ]
  /// }
  /// Where time is in milliseconds since the initial time of the data, and the
  /// channel values are floats in the same order as the channels of the device.
  ///
  int appendFromJson(Map<String, dynamic> json) {
    final timeSeries = (json['data'] as List<dynamic>);

    // If this is the first time stamps, we need to set the time offset
    if (timeSeries.isEmpty) return -1;
    bool isNew = _timeOffset == null;
    _timeOffset ??= isFromLiveData ? timeSeries.last[0] : timeSeries.first[0];

    final maxLength = timeSeries.length;
    final newT = timeSeries.map((e) => (e[0] as int) - _timeOffset!).toList();

    // Find the first index where the new time is larger than the last time of t
    int firstNewIndex = isNew
        ? 0
        : newT.indexWhere((value) => value > time.last);
    firstNewIndex = firstNewIndex <= 0 ? 0 : firstNewIndex;
    time.addAll(newT.getRange(firstNewIndex, maxLength));

    // Parse the data for each channel
    for (int channelIndex = 0; channelIndex < channelCount; channelIndex++) {
      _data[channelIndex].addAll(
        timeSeries
            .getRange(firstNewIndex, maxLength)
            .map<double>((e) => e[1][channelIndex]),
      );
    }

    onNewData.notifyListeners((callback) => callback());
    return time.length - (maxLength - firstNewIndex);
  }

  Future<void> toFile(String path, {bool raw = false}) async {
    final file = File(path);
    final sink = file.openWrite();
    final buffer = StringBuffer();

    buffer.writeln(
      'time (s),${List.generate(channelCount, (index) => 'channel$index').join(',')}',
    );

    final data = getData();
    for (int i = 0; i < time.length; i++) {
      buffer.write((time[i] / 1000).toStringAsFixed(4));
      for (int j = 0; j < channelCount; j++) {
        buffer.write(',');
        buffer.write(data[j][i].toStringAsFixed(6));
      }
      buffer.writeln();
    }
    sink.write(buffer.toString());
    await sink.flush();
    await sink.close();
  }

  int dropBefore(double elapsedTime) {
    if (time.isEmpty) return 0;

    final firstIndexToKeep = time.indexWhere((value) => value >= elapsedTime);
    if (firstIndexToKeep == -1) {
      // If we get to the end, we should drop everything
      clear();
    } else {
      time.removeRange(0, firstIndexToKeep);
      for (var channel in _data) {
        channel.removeRange(0, firstIndexToKeep);
      }
    }
    return firstIndexToKeep;
  }

  int dropAfter(double elapsedTime) {
    if (time.isEmpty) return 0;

    final lastIndexToKeep = time.indexWhere((value) => value >= elapsedTime);
    // If we get to the end, we should keep everything, otherwise drop from lastIndexToKeep
    if (lastIndexToKeep != -1) {
      time.removeRange(lastIndexToKeep, time.length);
      for (var channel in _data) {
        channel.removeRange(lastIndexToKeep, channel.length);
      }
    }
    return lastIndexToKeep;
  }

  TimeSeriesData copy({bool isFromLiveData = false}) {
    final copy = TimeSeriesData(
      channelCount: channelCount,
      isFromLiveData: isFromLiveData,
    );
    copy._timeOffset = null;
    copy.time.addAll(time);
    for (int channelIndex = 0; channelIndex < channelCount; channelIndex++) {
      copy._data[channelIndex].addAll(_data[channelIndex]);
    }
    return copy;
  }
}
