import 'dart:typed_data';

import 'package:charuconstruction_flutter/models/devices/b24_force_sensor.dart';
import 'package:universal_ble/universal_ble.dart';
export 'package:universal_ble/universal_ble.dart'
    show AvailabilityState, ScanFilter;

const _useBleMocker = bool.fromEnvironment(
  'CHARUCONSTRUCTION_USE_BLE_MOCKER',
  defaultValue: false,
);

class CharuconstructionBleDevice extends BleDevice {
  CharuconstructionBleDevice({
    required super.deviceId,
    super.name,
    super.rssi,
    super.paired,
    super.services,
    super.isSystemDevice,
    super.manufacturerDataList,
    super.serviceData = const {},
    super.timestamp,
  });

  bool _isConnected = false;

  @override
  String? get name => 'Mocked Device';
  @override
  String get deviceId => '00:11:22:33:44:55';

  Future<void> connect() async {
    await Future.delayed(Duration(seconds: 1));
    _isConnected = true;
  }

  Future<bool> get isConnected async => _isConnected;
  Future<void> disconnect() async {
    _isConnected = false;
  }

  Future<List<CharuconstructionBleService>> discoverServices() async => [];
}

class CharuconstructionBleService extends BleService {
  CharuconstructionBleService(super.uuid, super.characteristics);
  @override
  List<CharuconstructionBleCharacteristic> get characteristics => super
      .characteristics
      .map(
        (c) => CharuconstructionBleCharacteristic(
          c.uuid,
          c.properties,
          c.descriptors,
          onValueReceivedMock: _useBleMocker
              ? (c as CharuconstructionBleCharacteristic).onValueReceivedMock
              : null,
        ),
      )
      .toList();
}

class CharuconstructionBleCharacteristic extends BleCharacteristic {
  Future<void> setNotifyValue(bool value) async {}
  Future<void> write(List<int> value, {bool withResponse = false}) async {
    _useBleMocker
        ? null
        : await BleCharacteristicExtension(
            this,
          ).write(value, withResponse: withResponse);
  }

  Stream<Uint8List>? onValueReceivedMock;
  Stream<Uint8List> get onValueReceived => _useBleMocker
      ? onValueReceivedMock ?? Stream.empty()
      : BleCharacteristicExtension(this).onValueReceived;

  CharuconstructionBleCharacteristic(
    super.uuid,
    super.properties,
    super.descriptors, {
    this.onValueReceivedMock,
  });
  CharacteristicSubscription get notifications => _useBleMocker
      ? CharuconstructionCharacteristicSubscription(
          this,
          CharacteristicProperty.notify,
        )
      : BleCharacteristicExtension(this).notifications;
}

class CharuconstructionCharacteristicSubscription
    extends CharacteristicSubscription {
  final CharuconstructionBleCharacteristic? _characteristic;

  CharuconstructionCharacteristicSubscription(
    super.characteristic,
    super.property,
  ) : _characteristic = _useBleMocker
          ? characteristic as CharuconstructionBleCharacteristic
          : null;

  @override
  Future<void> subscribe({Duration? timeout}) async {
    _useBleMocker
        ? _characteristic!.onValueReceivedMock
        : await super.subscribe(timeout: timeout);
  }

  @override
  Future<void> unsubscribe({Duration? timeout}) async {
    _useBleMocker ? null : await super.unsubscribe(timeout: timeout);
  }
}

class CharuconstructionUniversalBle extends UniversalBle {
  static Future<AvailabilityState> getBluetoothAvailabilityState() async {
    return _useBleMocker
        ? AvailabilityState.poweredOn
        : await UniversalBle.getBluetoothAvailabilityState();
  }

  static Future<void> requestPermissions() async {
    return _useBleMocker ? null : await UniversalBle.requestPermissions();
  }

  static Future<void> startScan({required ScanFilter scanFilter}) async {
    return _useBleMocker
        ? Future.delayed(Duration(seconds: 1)).then(
            (value) => _onScanResult?.call(B24MockCharuconstructionBleDevice()),
          )
        : await UniversalBle.startScan(scanFilter: scanFilter);
  }

  static void Function(CharuconstructionBleDevice scanResult)? _onScanResult;

  static set onScanResult(
    void Function(CharuconstructionBleDevice scanResult)? onScanResult,
  ) {
    if (_useBleMocker) {
      _onScanResult = onScanResult;
    } else {
      UniversalBle.onScanResult = (BleDevice scanResult) {
        print('coucou');
        if (onScanResult != null) {
          onScanResult(scanResult as CharuconstructionBleDevice);
        }
      };
      //onScanResult as void Function(BleDevice scanResult)?;
    }
  }

  static Future<void> stopScan() async {
    return _useBleMocker ? null : await UniversalBle.stopScan();
  }
}
