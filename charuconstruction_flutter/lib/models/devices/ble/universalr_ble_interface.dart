import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:universal_ble/universal_ble.dart';

export 'package:universal_ble/universal_ble.dart'
    show AvailabilityState, ScanFilter;

class UniversalBleInterface extends UniversalBle {
  ///
  /// Set this value to [true] if one wants to use the mocker of the static
  /// methods. This won't affect the [UniversalBleDeviceInterface.isMock] though.
  static bool useMocker = false;

  ///
  /// Use this to register any mock device that should be discoverable when scanning in mock mode.
  static List<UniversalBleDeviceInterface> mockedDevices = [];

  static Future<AvailabilityState> getBluetoothAvailabilityState() async {
    return useMocker
        ? AvailabilityState.poweredOn
        : await UniversalBle.getBluetoothAvailabilityState();
  }

  static Future<void> requestPermissions() async {
    return useMocker ? null : await UniversalBle.requestPermissions();
  }

  static Future<void> startScan({required ScanFilter scanFilter}) async {
    return useMocker
        ? Future.delayed(Duration(seconds: 1)).then((value) {
            final device = mockedDevices.firstWhereOrNull(
              (device) => scanFilter.withNamePrefix.any(
                (filter) => device.name?.startsWith(filter) ?? false,
              ),
            );
            if (device != null) {
              if (_onScanResult != null) _onScanResult!(device);
            }
          })
        : await UniversalBle.startScan(scanFilter: scanFilter);
  }

  static void Function(UniversalBleDeviceInterface scanResult)? _onScanResult;

  static set onScanResult(
    void Function(UniversalBleDeviceInterface scanResult)? onScanResult,
  ) {
    if (useMocker) {
      _onScanResult = onScanResult;
    } else {
      UniversalBle.onScanResult = (BleDevice scanResult) {
        if (onScanResult != null) {
          onScanResult(UniversalBleDeviceInterface._fromRealDevice(scanResult));
        }
      };
      //onScanResult as void Function(BleDevice scanResult)?;
    }
  }

  static Future<void> stopScan() async {
    return useMocker ? null : await UniversalBle.stopScan();
  }
}

class UniversalBleDeviceInterface extends BleDevice {
  final bool isMocker;

  UniversalBleDeviceInterface({
    required super.deviceId,
    this.isMocker = false,
    super.name,
    super.rssi,
    super.paired,
    super.services,
    super.isSystemDevice,
    super.manufacturerDataList,
    super.serviceData = const {},
    super.timestamp,
  });

  UniversalBleDeviceInterface._fromRealDevice(BleDevice device)
    : isMocker = false,
      super(
        deviceId: device.deviceId,
        name: device.name,
        rssi: device.rssi,
        paired: device.paired,
        services: device.services,
        isSystemDevice: device.isSystemDevice,
        manufacturerDataList: device.manufacturerDataList,
        serviceData: device.serviceData,
        timestamp: device.timestamp,
      );

  bool _isConnected = false;
  Future<bool> get isConnected async =>
      isMocker ? _isConnected : await BleDeviceExtension(this).isConnected;

  Future<void> connect() async {
    if (isMocker) {
      await Future.delayed(Duration(seconds: 1));
      _isConnected = true;
    } else {
      await BleDeviceExtension(this).connect();
    }
  }

  Future<void> disconnect() async {
    if (isMocker) {
      await Future.delayed(Duration(seconds: 1));
      _isConnected = false;
    } else {
      await BleDeviceExtension(this).disconnect();
    }
  }

  ///
  /// This should be implemented by the mock devices
  Future<List<UniversalBleServiceInterface>> discoverServices() async => [];
}

class UniversalBleServiceInterface extends BleService {
  final UniversalBleDeviceInterface? _device;

  UniversalBleServiceInterface(
    super.uuid,
    super.characteristics, {
    required UniversalBleDeviceInterface? device,
  }) : _device = device;

  @override
  List<UniversalBleCharacteristicInterface> get characteristics => super
      .characteristics
      .map(
        (c) => UniversalBleCharacteristicInterface(
          c.uuid,
          c.properties,
          c.descriptors,
          onValueReceivedMock: (_device?.isMocker ?? false)
              ? (c as UniversalBleCharacteristicInterface).onValueReceivedMock
              : null,

          onConfigureMock: (_device?.isMocker ?? false)
              ? (c as UniversalBleCharacteristicInterface).onConfigureMock
              : null,
          device: _device,
        ),
      )
      .toList();
}

class UniversalBleCharacteristicInterface extends BleCharacteristic {
  final UniversalBleDeviceInterface? _device;

  void Function(List<int> value, {bool withResponse})? onConfigureMock;

  Stream<Uint8List>? onValueReceivedMock;
  Stream<Uint8List> get onValueReceived => (_device?.isMocker ?? false)
      ? onValueReceivedMock ?? Stream.empty()
      : BleCharacteristicExtension(this).onValueReceived;

  ///
  /// The [onConfigureMock] callback is called when a write operation is performed
  /// on the characteristic in mock mode. It allows to simulate the behavior of
  /// the device when certain values are written to the characteristic.
  /// The [onValueReceivedMock] stream is used to simulate incoming data from
  /// the device when subscribed to notifications in mock mode.
  UniversalBleCharacteristicInterface(
    super.uuid,
    super.properties,
    super.descriptors, {
    this.onConfigureMock,
    this.onValueReceivedMock,
    required UniversalBleDeviceInterface? device,
  }) : _device = device;

  Future<void> write(List<int> value, {bool withResponse = false}) async {
    return (_device?.isMocker ?? false)
        ? (onConfigureMock != null
              ? onConfigureMock!(value, withResponse: withResponse)
              : null)
        : await BleCharacteristicExtension(
            this,
          ).write(value, withResponse: withResponse);
  }

  CharacteristicSubscription get notifications => (_device?.isMocker ?? false)
      ? _CharacteristicSubscriptionMocker(this, CharacteristicProperty.notify)
      : BleCharacteristicExtension(this).notifications;
}

class _CharacteristicSubscriptionMocker extends CharacteristicSubscription {
  _CharacteristicSubscriptionMocker(super.characteristic, super.property);

  @override
  Future<void> subscribe({Duration? timeout}) async {}

  @override
  Future<void> unsubscribe({Duration? timeout}) async {}
}
