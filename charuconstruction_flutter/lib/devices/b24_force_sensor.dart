import 'dart:async';
import 'dart:typed_data';
import 'package:universal_ble/universal_ble.dart';

/// ---------------- ENUMS ----------------

enum B24SampleRateConfiguration {
  stop(0),
  fastest(80),
  slow(2000),
  batterySaver(5000),
  slowest(10000);

  final int value;
  const B24SampleRateConfiguration(this.value);
}

enum B24ResolutionConfiguration {
  lowest(8),
  batterySaver(8),
  maximum(16);

  final int value;
  const B24ResolutionConfiguration(this.value);
}

/// ---------------- FORCE SENSOR BASE ----------------

abstract class ForceSensor {
  final StreamController<(double, List<double>)> _controller =
      StreamController.broadcast();

  Stream<(double, List<double>)> get onDataReceived => _controller.stream;

  void _notify(double time, List<double> data) {
    _controller.add((time, data));
  }
}

/// ---------------- MAIN CLASS ----------------

class B24ForceSensor extends ForceSensor {
  BleDevice _device;

  double? _startingTime;
  final List<double> _timeVector = [];
  final List<double> _data = [];

  B24ForceSensor(this._device);

  String get name => _device.name!;
  String get address => _device.deviceId;

  /// ---------------- CONNECTION ----------------

  Future<bool> connect({required int pinNumber, int maxRetries = 10}) async {
    print('Connecting to ${_device.name} (${_device.deviceId})...');
    int retry = 0;
    while (retry < maxRetries) {
      try {
        await _device.connect();
        if (!await _device.isConnected) throw Exception('Connection failed');

        await _sendPinNumber(pinNumber);

        // Subscribe
        await _subscribe();

        print('Connected!');
        return true;
      } catch (e) {
        print('Failed to connect ($e), retrying... ($retry/$maxRetries)');
        await Future.delayed(Duration(seconds: 1));

        _device = await _findDevice();

        retry++;
      }
    }

    print('Failed to connect.');
    return false;
  }

  Future<bool> disconnect() async {
    try {
      await configureResolution(B24ResolutionConfiguration.batterySaver);
      await configureDataRate(B24SampleRateConfiguration.batterySaver);

      await _unsubscribe();
      await _device.disconnect();
      return true;
    } catch (e) {
      print('Disconnect error: $e');
      return false;
    }
  }

  Future<bool> startReading() async {
    if (!(await _device.isConnected)) {
      print('Not connected, please call connect() first.');
      return false;
    }

    // Configure sensor
    await configureResolution(B24ResolutionConfiguration.maximum);
    await configureDataRate(B24SampleRateConfiguration.fastest);

    print('Connected!');
    return true;
  }

  Future<bool> stopReading() async {
    try {
      await configureResolution(B24ResolutionConfiguration.batterySaver);
      await configureDataRate(B24SampleRateConfiguration.slow);

      return true;
    } catch (e) {
      print('Disconnect error: $e');
      return false;
    }
  }

  void clearData() {
    _startingTime = null;
    _timeVector.clear();
    _data.clear();
  }

  List<double> get timeVector => _timeVector;
  List<double> get forceVector => _data;

  /// ---------------- BLUETOOTH DISCOVERY ----------------

  static Future<BleDevice> _findDevice() async {
    print('Scanning for BLE devices...');

    final device = Completer<BleDevice>();
    UniversalBle.onScanResult = (BleDevice bleDevice) =>
        device.complete(bleDevice);

    // Request permissions in foreground (e.g., during app setup)
    await UniversalBle.requestPermissions();

    await UniversalBle.startScan(
      scanFilter: ScanFilter(withNamePrefix: ['B24']),
    );

    final bleDevice = await device.future;

    await UniversalBle.stopScan();

    print('Found device: ${bleDevice.name} (${bleDevice.deviceId})');
    return bleDevice;
  }

  static Future<B24ForceSensor> fromBluetooth({
    int timeoutMs = 1000,
    int maxRetries = 100,
  }) async {
    AvailabilityState state =
        await UniversalBle.getBluetoothAvailabilityState();
    // Start scan only if Bluetooth is powered on
    if (state != AvailabilityState.poweredOn) {
      throw Exception('Bluetooth is not powered on');
    }

    final bleDevice = await _findDevice();
    return B24ForceSensor(bleDevice);
  }

  /// ---------------- CONFIGURATION ----------------

  Future<void> configureResolution(
    B24ResolutionConfiguration resolution,
  ) async {
    await _write(
      _B24Helpers.configuration,
      _B24Helpers.resolution,
      _packU8(resolution.value),
    );
  }

  Future<void> configureDataRate(B24SampleRateConfiguration rate) async {
    await _write(
      _B24Helpers.configuration,
      _B24Helpers.dataRate,
      _packU32(rate.value),
    );
  }

  Future<void> _sendPinNumber(int pin) async {
    await _write(_B24Helpers.configuration, _B24Helpers.pin, _packU32(pin));
  }

  /// ---------------- BLE HELPERS ----------------

  Future<void> _subscribe() async {
    final services = await _device.discoverServices();

    for (var s in services) {
      if (s.uuid.toString() == _B24Helpers.notifications) {
        for (var c in s.characteristics) {
          if (c.uuid.toString() == _B24Helpers.value) {
            c.onValueReceived.listen(_onValue);
            await c.notifications.subscribe();
          } else if (c.uuid.toString() == _B24Helpers.status) {
            await c.notifications.subscribe();
            c.onValueReceived.listen(_onStatus);
          }
        }
      }
    }
  }

  Future<void> _unsubscribe() async {
    final services = await _device.discoverServices();

    for (var s in services) {
      if (s.uuid.toString() == _B24Helpers.notifications) {
        for (var c in s.characteristics) {
          if (c.uuid.toString() == _B24Helpers.value ||
              c.uuid.toString() == _B24Helpers.status) {
            await c.notifications.unsubscribe();
          }
        }
      }
    }
  }

  Future<void> _write(
    String serviceUuid,
    String charUuid,
    List<int> value,
  ) async {
    final services = await _device.discoverServices();

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

    throw Exception('Characteristic not found');
  }

  /// ---------------- DATA HANDLING ----------------

  void _onValue(List<int> data) {
    final now = DateTime.now().microsecondsSinceEpoch / 1e6;

    _startingTime ??= now;

    final t = now - _startingTime!;
    _timeVector.add(t);

    double value = double.nan;

    if (data.length == 4) {
      value = _unpackF32(data);
    }

    _data.add(value);

    _notify(t, [value]);
  }

  void _onStatus(List<int> data) {
    if (data.isEmpty) return;

    int status = data[0];

    int fast = (status >> 4) & 1;
    int battLow = (status >> 5) & 1;
    int overRange = (status >> 3) & 1;

    print(
      'STATUS: 0x${status.toRadixString(16)} fast=$fast batt=$battLow over=$overRange',
    );
  }

  /// ---------------- BYTE HELPERS ----------------

  List<int> _packU32(int x) {
    final b = ByteData(4);
    b.setUint32(0, x, Endian.big);
    return b.buffer.asUint8List();
  }

  List<int> _packU8(int x) {
    return [x & 0xFF];
  }

  double _unpackF32(List<int> bytes) {
    final b = ByteData.sublistView(Uint8List.fromList(bytes));
    return b.getFloat32(0, Endian.big);
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
