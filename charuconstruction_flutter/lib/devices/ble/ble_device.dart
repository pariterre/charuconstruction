import 'dart:async';
import 'dart:typed_data';

import 'package:charuconstruction_flutter/devices/ble/ble_exceptions.dart';
import 'package:charuconstruction_flutter/devices/device.dart';
import 'package:logging/logging.dart';
import 'package:universal_ble/universal_ble.dart' as ble;

final _logger = Logger('BleDevice');

abstract class BleDevice extends Device {
  ble.BleDevice? _device;

  @override
  String? get name => _device?.name;

  @override
  String? get macAddress => _device?.deviceId;

  bool get deviceFound => _device != null;
  bool get deviceNotFound => !deviceFound;

  @override
  bool isConnected = false;

  @override
  bool isReading = false;

  ///
  /// API METHODS
  ///
  @override
  Future<void> connect({int? pinNumber, int maxRetries = 10}) async {
    _logger.info('Connecting to ${_device?.name} (${_device?.deviceId})...');
    int retry = 0;
    while (!isConnected && retry < maxRetries) {
      try {
        // If the device is not set, search for it
        if (deviceNotFound) await scan();

        await _device?.connect();
        if (!await _device!.isConnected) throw Exception('Connection failed');

        // Finalize connection
        if (pinNumber != null) await sendPinNumber(pinNumber);
        await _updateSubscribeStatus(isSubscribing: true);
        isConnected = true;
      } catch (e) {
        _logger.warning(
          'Failed to connect ($e), retrying... ($retry/$maxRetries)',
        );
        _device = null; // Reset device to trigger a new scan

        await Future.delayed(Duration(seconds: 1));
        retry++;
      }
    }

    if (!isConnected) {
      throw BleDeviceCouldNotConnect(
        'Failed to connect after $maxRetries attempts.',
      );
    }
  }

  @override
  Future<void> disconnect() async {
    try {
      await stopReading();
      await _updateSubscribeStatus(isSubscribing: false);
      await _device?.disconnect();
      isConnected = false;
    } catch (e) {
      throw BleDeviceCouldNotDisconnect('Failed to disconnect: $e');
    }
  }

  @override
  Future<void> startReading() async {
    if (!isConnected) {
      throw BleDeviceNotConnected(
        'Cannot start reading: device is not connected.',
      );
    }

    isReading = true;
    _logger.info('Reading data!');
  }

  @override
  Future<void> stopReading() async {
    if (!isConnected) {
      throw BleDeviceNotConnected(
        'Cannot stop reading: device is not connected.',
      );
    }

    isReading = false;
    _logger.info('Stopped reading data!');
  }

  Future<void> _updateSubscribeStatus({required bool isSubscribing}) async {
    if (!isConnected) {
      throw BleDeviceNotConnected(
        'Cannot subscribe to notifications: device is not connected.',
      );
    }

    final services = await _device!.discoverServices();
    for (var service in services) {
      for (var characteristic in service.characteristics) {
        updateSubscribeStatus(
          service: service,
          characteristic: characteristic,
          isSubscribing: isSubscribing,
        );
      }
    }
  }

  ///
  /// INTERFACE
  ///

  ///
  /// The name of the device used for filtering during scanning.
  ///
  String get deviceNamePrefix;

  ///
  /// The UUID of the service used for configuration (e.g., sending pin number).
  ///
  String get configurationServiceUuid;

  ///
  /// The UUID of the characteristic used for sending the pin number. This is only
  /// mandatory if the device requires a pin number for connection. If not, this can be
  /// left unimplemented.
  ///
  String get pinCharacteristicUuid => throw UnimplementedError(
    'A pin number was sent while connecting, but pinCharacteristicUuid is not implemented.',
  );

  ///
  /// A callback of the UUID of the characteristics used subscribing to a characteristic. This is only mandatory if
  /// the device has subscribed to notifications (i.e., if notificationsServiceUuid is implemented).
  ///
  /// [service] is the service for which the device is subscribing to notifications. This can be used to perform specific actions when subscribing to certain services (e.g., initializing data structures, etc.).
  /// [characteristic] is the characteristic for which the device is subscribing to notifications.
  /// This can be used to perform specific actions when subscribing to certain characteristics (e.g., initializing data structures, etc.).
  ///
  Future<void> updateSubscribeStatus({
    required ble.BleService service,
    required ble.BleCharacteristic characteristic,
    required bool isSubscribing,
  }) async {
    throw UnimplementedError(
      'Received notification from ${characteristic.uuid}, but updateSubscribeStatus is not implemented.',
    );
  }

  Future<void> scan() async {
    _logger.info('Scanning for BLE devices...');

    // Check Bluetooth is available and powered on
    ble.AvailabilityState state =
        await ble.UniversalBle.getBluetoothAvailabilityState();
    // Start scan only if Bluetooth is powered on
    if (state != ble.AvailabilityState.poweredOn) {
      throw BleDeviceBluetoothOff('Bluetooth is not powered on');
    }

    final device = Completer<ble.BleDevice>();
    ble.UniversalBle.onScanResult = (ble.BleDevice bleDevice) =>
        device.complete(bleDevice);

    // Request permissions in foreground (e.g., during app setup)
    await ble.UniversalBle.requestPermissions();

    await ble.UniversalBle.startScan(
      scanFilter: ble.ScanFilter(withNamePrefix: [deviceNamePrefix]),
    );

    _device = await device.future;

    await ble.UniversalBle.stopScan();

    _logger.info('Found device: $name ($macAddress)');
  }

  Future<void> sendPinNumber(int pin) async {
    await write(configurationServiceUuid, pinCharacteristicUuid, packU32(pin));
  }

  Future<void> write(
    String serviceUuid,
    String charUuid,
    List<int> value,
  ) async {
    if (!isConnected) {
      throw BleDeviceNotConnected(
        'Cannot write to characteristic: device is not connected.',
      );
    }

    final services = await _device!.discoverServices();
    try {
      for (var s in services) {
        if (s.uuid.toString() == serviceUuid) {
          for (var c in s.characteristics) {
            if (c.uuid.toString() == charUuid) {
              await c.write(value, withResponse: true);
              return;
            }
          }
        }
      }
    } catch (e) {
      throw CharacteristicWriteFailed('Failed to write to characteristic: $e');
    }

    throw CharacteristicNotFound('Characteristic not found');
  }

  static List<int> packU32(int x) {
    final b = ByteData(4);
    b.setUint32(0, x, Endian.big);
    return b.buffer.asUint8List();
  }

  static List<int> packU8(int x) {
    return [x & 0xFF];
  }

  static List<int> packF32(double x) {
    final b = ByteData(4);
    b.setFloat32(0, x, Endian.big);
    return b.buffer.asUint8List();
  }

  static double unpackF32(List<int> bytes) {
    final b = ByteData.sublistView(Uint8List.fromList(bytes));
    return b.getFloat32(0, Endian.big);
  }
}
