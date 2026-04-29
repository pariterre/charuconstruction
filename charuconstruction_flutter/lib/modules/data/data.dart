//
// This file defines the Data class, which is used to store and manipulate the data
// It is an adaptation of the Data class developed for the Neurobiomech Software
// which can be found (22nd of april, 2026) here:
// https://github.com/LabNNL/neurobiomech_software/blob/main/frontend_fundamentals/lib/models/data.dart
//

import 'dart:io';

import 'package:collection/collection.dart';
import 'package:logging/logging.dart';
import 'package:path_provider/path_provider.dart';

import 'time_series_data.dart';

export 'time_series_data.dart';
export 'widgets/data_graph.dart';

final _logger = Logger('Data');

class Data {
  DateTime _initialTime;
  DateTime get initialTime => _initialTime;
  final Map<String, TimeSeriesData> _devices;
  Map<String, TimeSeriesData> get devices => Map.unmodifiable(_devices);

  ///
  /// Constructor
  ///

  ///
  /// Create a new data instance with the given initial time and whether the data is from live data or not
  /// [initialTime] is the initial time of the data, used to compute the time vector of the devices
  /// [isFromLiveData] is whether the data is from live data or not, used to compute the time offset of the devices
  ///
  Data({required DateTime initialTime, required bool isFromLiveData})
    : _initialTime = initialTime,
      _devices = {};

  ///
  /// API METHODS
  ///

  ///
  /// Add a device to the data pool
  ///
  void add(String device, TimeSeriesData data) {
    if (_devices.containsKey(device)) {
      throw Exception('Device with name $device already exists.');
    }
    _devices[device] = data;
  }

  ///
  /// Whether any device are connected and have data
  ///
  bool get isEmpty =>
      _devices.isEmpty || _devices.values.every((device) => device.isEmpty);
  bool get isNotEmpty => !isEmpty;

  ///
  /// Clear the data and reset the initial time if provided
  /// [initialTime] is the new initial time of the data, if not provided, the initial time will not be changed
  /// [keepDevices] is whether to keep the devices in the data pool or not
  ///
  void clear({DateTime? initialTime, bool keepDevices = false}) {
    _initialTime = initialTime ?? _initialTime;

    for (final device in _devices.values) {
      device.clear(timeOffset: _initialTime);
    }

    if (!keepDevices) _devices.clear();
  }

  ///
  /// Append data from a JSON object, the JSON object should be in the format of:
  /// {
  ///   "device_name": {
  ///     "name": "device_name",
  ///     "data": [
  ///       [time, channel_1_value, channel_2_value, ...],
  ///       [time, channel_1_value, channel_2_value, ...],
  ///       ...
  ///     ]
  ///   }
  /// }
  /// The time should be in milliseconds since the initial time of the data, and
  /// the channel values should be in the same order as the channels of the device
  void appendFromJson(Map<String, dynamic> json) {
    for (final data in json.values) {
      final key = _devices.keys.firstWhereOrNull(
        (device) => device == data['name'],
      );
      if (key == null) {
        throw Exception(
          'Device with name ${data['name']} not found in devices list.',
        );
      }

      _devices[key]!.appendFromJson(data['data'] as Map<String, dynamic>);
    }
  }

  ///
  /// Drop data before a certain time, the time should be in the same format as
  /// the time vector of the devices (milliseconds since the initial time of the data)
  ///
  void dropBefore(DateTime t) {
    for (final device in _devices.values) {
      device.dropBefore(
        (t.millisecondsSinceEpoch - initialTime.millisecondsSinceEpoch)
            .toDouble(),
      );
    }
  }

  ///
  /// Drop data after a certain time, the time should be in the same format as
  /// the time vector of the devices (milliseconds since the initial time of the data)
  ///
  void dropAfter(DateTime t) {
    for (final device in _devices.values) {
      device.dropAfter(
        (t.millisecondsSinceEpoch - initialTime.millisecondsSinceEpoch)
            .toDouble(),
      );
    }
  }

  ///
  /// Save the data to files, the files will be saved in a folder named after
  /// the current date and time in the format of "yyyy-MM-dd_HH-mm-ss" in the
  /// Application Documents directory, and each device will be saved in a separate
  /// CSV file named after the device name
  ///
  Future<void> toFiles() async {
    final baseDir = await getApplicationDocumentsDirectory();
    final folderName = DateTime.now()
        .toIso8601String()
        .replaceAll(':', '-')
        .split('.')
        .first;
    final folderPath = '${baseDir.path}/$folderName/';

    // Create the folder if it doesn't exist
    final folder = Directory(folderPath);
    if (!await folder.exists()) {
      await folder.create(recursive: true);
    }
    await Future.wait([
      for (final key in _devices.keys)
        _devices[key]!.toFile('$folderPath/$key.csv'),
    ]);
    _logger.info('Data saved to $folderPath');
  }
}
