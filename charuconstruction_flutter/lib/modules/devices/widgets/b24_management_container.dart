import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../ble/ble_exceptions.dart';
import '../concrete_devices/available_devices.dart';
import '../concrete_devices/b24_force_sensor.dart';
import '../device_exceptions.dart';
import '../providers/devices_provider.dart';

class B24ManagementContainer extends StatefulWidget {
  const B24ManagementContainer({super.key});

  @override
  State<B24ManagementContainer> createState() => _B24ManagementContainerState();
}

class _B24ManagementContainerState extends State<B24ManagementContainer> {
  bool _isBusy = false;

  String? _statusMessage;
  String? _errorMessage;

  late double? _scaling = _device.scaling;
  int? _pinNumber;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Padding(
            padding: const EdgeInsets.only(bottom: 20.0),
            child: SizedBox(
              width: 150,
              child: TextFormField(
                decoration: InputDecoration(labelText: 'Scaling'),
                enabled: !_device.isConnected,
                keyboardType: TextInputType.numberWithOptions(decimal: true),
                inputFormatters: [
                  FilteringTextInputFormatter.allow(RegExp(r'^\d*\.?\d*')),
                ],
                initialValue: _scaling.toString(),
                onChanged: (value) {
                  _scaling = double.tryParse(value);

                  _errorMessage = (_scaling ?? 0) < 0
                      ? 'Invalid scaling'
                      : null;
                  setState(() {});
                },
              ),
            ),
          ),

          Padding(
            padding: const EdgeInsets.only(top: 20.0, bottom: 20.0),
            child: ElevatedButton(
              onPressed: _isBusy ? null : _scan,
              child: Text('Scan for B24 Sensor'),
            ),
          ),

          Padding(
            padding: const EdgeInsets.only(bottom: 20.0),
            child: SizedBox(
              width: 100,
              child: TextField(
                decoration: InputDecoration(labelText: 'Pin Number'),
                keyboardType: TextInputType.number,
                enabled:
                    !_isBusy && _device.deviceFound && !_device.isConnected,
                inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                onChanged: (value) {
                  _pinNumber = int.tryParse(value);

                  _errorMessage = (_pinNumber ?? 0) < 0
                      ? 'Invalid pin number'
                      : null;
                  setState(() {});
                },
              ),
            ),
          ),

          Padding(
            padding: const EdgeInsets.only(bottom: 20.0),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                ElevatedButton(
                  onPressed:
                      _isBusy || (_pinNumber ?? -1) < 0 || _device.isConnected
                      ? null
                      : _connect,
                  child: Text('Connect'),
                ),
                SizedBox(width: 20),
                ElevatedButton(
                  onPressed: _isBusy || _device.isNotConnected
                      ? null
                      : _disconnect,
                  child: Text('Disconnect'),
                ),
              ],
            ),
          ),

          Padding(
            padding: const EdgeInsets.only(bottom: 20.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Status: ${_device.isConnected ? 'Connected' : 'Not Connected'} and ${_device.isRecording ? 'Recording' : 'Not Recording'}',
                ),
                if (_statusMessage != null) Text(_statusMessage!),
                if (_errorMessage != null)
                  Text(_errorMessage!, style: TextStyle(color: Colors.red)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  B24ForceSensor get _device =>
      DevicesProvider.instance.device(AvailableDevices.b24) as B24ForceSensor;

  Future<void> _scan() async {
    setState(() {
      _isBusy = true;
      _statusMessage = 'Scanning for B24 Sensor...';
      _errorMessage = null;
    });

    try {
      await _device.scan();
      if (_device.deviceNotFound) {
        throw DeviceNotFound(
          'B24 Sensor not found. Please make sure it is in pairing mode.',
        );
      }
      _statusMessage = 'B24 Sensor found: ${_device.name}';
      _errorMessage = null;
    } on BleDeviceBluetoothOff {
      _statusMessage = 'Scan failed';
      _errorMessage =
          'Bluetooth is turned off. Please turn it on and try again.';
    } on DeviceNotFound catch (e) {
      _statusMessage = 'Scan failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Scan failed';
      _errorMessage = 'An unexpected error occurred: $e';
    }

    if (!mounted) return;
    setState(() {
      _isBusy = false;
    }); // Update UI to show device found
  }

  Future<void> _connect() async {
    setState(() {
      _isBusy = true;
      _statusMessage = 'Connecting to B24 Sensor...';
      _errorMessage = null;
    });

    try {
      await _device.connect(pinNumber: _pinNumber, scaling: _scaling ?? 1.0);
      _statusMessage = 'Connected to B24 Sensor: ${_device.name}';
    } on DeviceCouldNotConnect catch (e) {
      _statusMessage = 'Connection failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Connection failed';
      _errorMessage = 'An unexpected error occurred: $e';
    }

    if (!mounted) return;
    setState(() {
      _isBusy = false;
    });
  }

  Future<void> _disconnect() async {
    setState(() {
      _isBusy = true;
      _statusMessage = 'Disconnecting from B24 Sensor...';
      _errorMessage = null;
    });

    try {
      await _device.stopRecording();
    } on Exception {
      // If stopping recording fails, we still want to try to disconnect, so we catch the error but don't rethrow it
    }

    try {
      await _device.disconnect();
      _statusMessage = 'Disconnected from B24 Sensor: ${_device.name}';
    } on DeviceCouldNotDisconnect catch (e) {
      _statusMessage = 'Disconnection failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Disconnection failed';
      _errorMessage = 'An unexpected error occurred: $e';
    }

    if (!mounted) return;
    setState(() {
      _isBusy = false;
    });
  }
}
