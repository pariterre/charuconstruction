import 'dart:async';
import 'dart:typed_data';

import 'package:logging/logging.dart';

import '../device.dart';
import '../device_exceptions.dart';
import 'ble_exceptions.dart';
import 'universal_ble_interface.dart';

final _logger = Logger('BleDevice');

abstract class BleDevice extends Device {
  UniversalBleDeviceInterface? _device;

  @override
  String? get name => _device?.name;

  ///
  /// The MAC address of the device.
  String? get macAddress => _device?.deviceId;

  bool get deviceFound => _device != null;
  bool get deviceNotFound => !deviceFound;

  ///
  /// API METHODS
  ///
  @override
  Future<void> connect({int? pinNumber, int maxRetries = 10}) async {
    _logger.info('Connecting to ${_device?.name} (${_device?.deviceId})...');
    int retry = 0;
    bool isConnected = false;

    while (!isConnected && retry < maxRetries) {
      try {
        // If the device is not set, search for it
        if (deviceNotFound) await scan();

        await _device?.connect();
        if (!await _device!.isConnected) throw Exception('Connection failed');

        // Finalize connection
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
      // Rescan for resetting the device as it was before the connection attempts, then throw the exception
      await scan();
      throw DeviceCouldNotConnect(
        'Failed to connect after $maxRetries attempts.',
      );
    }

    // If we get here, we could connect. Now we need to send the pin number
    // for that, the device must know they are connected, so we call super.connect()
    // to update the internal state of the device. However, if the PIN fails,
    // we need to disconnect and reset the state.
    await super.connect();

    // If a pin number was provided, send it and subscribe to notifications. If any of these steps fail, disconnect and retry.
    try {
      if (pinNumber != null) await sendPinNumber(pinNumber);
      await _updateSubscribeStatus(isSubscribing: true);
    } catch (e) {
      _logger.warning('Failed to finalize connection ($e), disconnecting...');

      // If any of the finalization steps fail, disconnect and retry
      try {
        await _updateSubscribeStatus(isSubscribing: false);
      } catch (e) {
        _logger.warning('Failed to unsubscribe from notifications: $e');
      }
      try {
        await _device?.disconnect();
      } catch (e) {
        _logger.warning('Failed to disconnect from device: $e');
      }

      // PIN failure or subscription so we disconnect the device immediately
      try {
        await disconnect();
      } catch (e) {
        // Do nothing if it fails to disconnect
      }
      throw DeviceCouldNotConnect(
        'Failed to finalize connection: device is connected but finalization steps failed.',
      );
    }
  }

  Future<void> _updateSubscribeStatus({required bool isSubscribing}) async {
    if (!isConnected) {
      throw DeviceNotConnected(
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
    required UniversalBleServiceInterface service,
    required UniversalBleCharacteristicInterface characteristic,
    required bool isSubscribing,
  }) async {
    throw UnimplementedError(
      'Received notification from ${characteristic.uuid}, but updateSubscribeStatus is not implemented.',
    );
  }

  Future<void> scan() async {
    _logger.info('Scanning for BLE devices...');

    // Check Bluetooth is available and powered on
    AvailabilityState state =
        await UniversalBleInterface.getBluetoothAvailabilityState();
    // Start scan only if Bluetooth is powered on
    if (state != AvailabilityState.poweredOn) {
      throw BleDeviceBluetoothOff('Bluetooth is not powered on');
    }

    final device = Completer<UniversalBleDeviceInterface>();
    UniversalBleInterface.onScanResult =
        (UniversalBleDeviceInterface bleDevice) => device.complete(bleDevice);

    // Request permissions in foreground (e.g., during app setup)
    await UniversalBleInterface.requestPermissions();

    await UniversalBleInterface.startScan(
      scanFilter: ScanFilter(withNamePrefix: [deviceNamePrefix]),
    );

    _device = await device.future;

    await UniversalBleInterface.stopScan();

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
      throw DeviceNotConnected(
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
      throw BleCharacteristicWriteFailed(
        'Failed to write to characteristic: $e',
      );
    }

    throw BleCharacteristicNotFound('Characteristic not found');
  }

  static List<int> packU32(int x) {
    final b = ByteData(4);
    b.setUint32(0, x, Endian.big);
    return b.buffer.asUint8List();
  }

  static int unpackU32(List<int> bytes) {
    final b = ByteData.sublistView(Uint8List.fromList(bytes));
    return b.getUint32(0, Endian.big);
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
