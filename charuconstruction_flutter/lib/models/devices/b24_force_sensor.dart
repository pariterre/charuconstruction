import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:charuconstruction_flutter/models/devices/ble/ble_device.dart';
import 'package:logging/logging.dart';
import 'package:charuconstruction_flutter/models/devices/ble/ble_device_interface.dart';

final _logger = Logger('B24ForceSensor');

/// ---------------- ENUMS ----------------

enum B24SampleRateConfiguration {
  stop(0),
  fastest(80),
  batterySaver(5000),
  sleep(10000);

  final int value;
  const B24SampleRateConfiguration(this.value);
}

enum B24ResolutionConfiguration {
  sleep(8),
  batterySaver(8),
  maximum(16);

  final int value;
  const B24ResolutionConfiguration(this.value);
}

/// ---------------- MAIN CLASS ----------------

class B24ForceSensor extends BleDevice {
  B24ForceSensor();

  @override
  Future<void> connect({int? pinNumber, int maxRetries = 10}) {
    if (pinNumber == null) {
      throw ArgumentError('Pin number is required for B24ForceSensor');
    }
    return super.connect(pinNumber: pinNumber, maxRetries: maxRetries);
  }

  ///
  /// Reimplement startReading and stopReading to change the sampling rate on the fly
  ///
  @override
  Future<void> startReading() async {
    // Put the sensor in fast mode
    if (!isReading) {
      await configureResolution(B24ResolutionConfiguration.maximum);
      await configureDataRate(B24SampleRateConfiguration.fastest);
    }

    return super.startReading();
  }

  @override
  Future<void> stopReading() async {
    if (isReading) {
      try {
        // Put the sensor in low power mode
        await configureResolution(B24ResolutionConfiguration.batterySaver);
        await configureDataRate(B24SampleRateConfiguration.batterySaver);
      } catch (e) {
        _logger.warning(
          'Failed to configure sensor for low power mode: $e. Trying to stop reading anyway.',
        );
      }
    }

    return super.stopReading();
  }

  @override
  Future<void> disconnect() async {
    try {
      await stopReading();
    } catch (e) {
      _logger.warning(
        'Failed to stop reading before disconnecting: $e. Trying to disconnect anyway.',
      );
    }

    // Put the sensor in low power mode before disconnecting
    try {
      await configureResolution(B24ResolutionConfiguration.sleep);
      await configureDataRate(B24SampleRateConfiguration.sleep);
    } catch (e) {
      _logger.warning(
        'Failed to configure sensor for low power mode before disconnecting: $e. Trying to disconnect anyway.',
      );
    }
    return super.disconnect();
  }

  /// ---------------- CONFIGURATION ----------------

  Future<void> configureResolution(
    B24ResolutionConfiguration resolution,
  ) async {
    await write(
      _B24Helpers.configuration,
      _B24Helpers.resolution,
      BleDevice.packU8(resolution.value),
    );
  }

  Future<void> configureDataRate(B24SampleRateConfiguration rate) async {
    await write(
      _B24Helpers.configuration,
      _B24Helpers.dataRate,
      BleDevice.packU32(rate.value),
    );
  }

  ///
  /// Interface implementation
  ///
  @override
  int get channelCount => 1;

  @override
  String get deviceNamePrefix => 'B24';

  @override
  String get configurationServiceUuid => _B24Helpers.configuration;

  @override
  String get pinCharacteristicUuid => _B24Helpers.pin;

  @override
  Future<void> updateSubscribeStatus({
    required service,
    required characteristic,
    required bool isSubscribing,
  }) async {
    if (service.uuid.toString() != _B24Helpers.notifications) return;
    if (characteristic.uuid.toString() == _B24Helpers.value) {
      _logger.info(
        '${isSubscribing ? 'Subscribing' : 'Unsubscribing'} to data for characteristic ${characteristic.uuid}',
      );
      if (isSubscribing) {
        characteristic.onValueReceived.listen(_onValue);
        await characteristic.notifications.subscribe();
      } else {
        await characteristic.notifications.unsubscribe();
      }
    } else if (characteristic.uuid.toString() == _B24Helpers.status) {
      _logger.info(
        '${isSubscribing ? 'Subscribing' : 'Unsubscribing'} to status for characteristic ${characteristic.uuid}',
      );
      if (isSubscribing) {
        characteristic.onValueReceived.listen(_onStatus);
        await characteristic.notifications.subscribe();
      } else {
        await characteristic.notifications.unsubscribe();
      }
    }
  }

  /// ---------------- DATA HANDLING ----------------

  Future<void> _onValue(List<int> data) async {
    final timestamp = DateTime.now();

    double value = data.length == 4 ? BleDevice.unpackF32(data) : double.nan;
    await pushData(timestamp, [value]);
  }

  void _onStatus(List<int> data) {
    if (data.isEmpty) return;

    int status = data[0];

    int fast = (status >> 4) & 1;
    int battLow = (status >> 5) & 1;
    int overRange = (status >> 3) & 1;

    _logger.info(
      'STATUS: 0x${status.toRadixString(16)} fast=$fast batt=$battLow over=$overRange',
    );
  }
}

/// ---------------- CONSTANTS ----------------

class _B24Helpers {
  static const configuration = 'a970fd30-a0e8-11e6-bdf4-0800200c9a66';
  static const dataRate = 'a970fd31-a0e8-11e6-bdf4-0800200c9a66';
  static const resolution = 'a970fd32-a0e8-11e6-bdf4-0800200c9a66';
  static const pin = 'a970fd39-a0e8-11e6-bdf4-0800200c9a66';

  static const notifications = 'a9712440-a0e8-11e6-bdf4-0800200c9a66';
  static const status = 'a9712441-a0e8-11e6-bdf4-0800200c9a66';
  static const value = 'a9712442-a0e8-11e6-bdf4-0800200c9a66';
}

class B24MockCharuconstructionBleDevice extends CharuconstructionBleDevice {
  B24MockCharuconstructionBleDevice() : super(deviceId: '00:11:22:33:44:55');

  @override
  String? get name => 'B24 Mocked Device';

  @override
  Future<List<CharuconstructionBleService>> discoverServices() async => [
    CharuconstructionBleService(_B24Helpers.configuration, [
      CharuconstructionBleCharacteristic(_B24Helpers.dataRate, [], []),
      CharuconstructionBleCharacteristic(_B24Helpers.resolution, [], []),
      CharuconstructionBleCharacteristic(_B24Helpers.pin, [], []),
    ]),
    CharuconstructionBleService(_B24Helpers.notifications, [
      CharuconstructionBleCharacteristic(_B24Helpers.status, [], []),
      CharuconstructionBleCharacteristic(
        _B24Helpers.value,
        [],
        [],
        onValueReceivedMock: Stream.periodic(
          Duration(
            milliseconds: 1000 ~/ B24SampleRateConfiguration.fastest.value,
          ),
          (count) => Uint8List.fromList(
            BleDevice.packF32(5000 * (1 + 0.5 * sin(count / 10))),
          ),
        ),
      ),
    ]),
  ];
}
