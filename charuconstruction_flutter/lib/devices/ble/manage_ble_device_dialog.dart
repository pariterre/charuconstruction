import 'package:charuconstruction_flutter/devices/b24_force_sensor.dart';
import 'package:charuconstruction_flutter/devices/ble/ble_exceptions.dart';
import 'package:charuconstruction_flutter/providers/devices_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class BleDeviceManagerPage extends StatefulWidget {
  const BleDeviceManagerPage({super.key});

  @override
  State<BleDeviceManagerPage> createState() => _BleDeviceManagerPageState();
}

class _BleDeviceManagerPageState extends State<BleDeviceManagerPage> {
  AvailableDevices _selectedDevice = AvailableDevices.values.first;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          DropdownButton<AvailableDevices>(
            hint: Text('Select a device to manage'),
            items: AvailableDevices.values.map((device) {
              return DropdownMenuItem(
                value: device,
                child: Text(device.toString().split('.').last),
              );
            }).toList(),
            value: _selectedDevice,
            onChanged: _selectDevice,
          ),
          switch (_selectedDevice) {
            AvailableDevices.b24 => const _B24ManagementScreen(),
          },
        ],
      ),
    );
  }

  void _selectDevice(AvailableDevices? device) {
    if (device == null) return;
    setState(() {
      _selectedDevice = device;
    });
  }
}

class _B24ManagementScreen extends StatefulWidget {
  const _B24ManagementScreen();

  @override
  State<_B24ManagementScreen> createState() => _B24ManagementScreenState();
}

class _B24ManagementScreenState extends State<_B24ManagementScreen> {
  bool _isBusy = false;

  String? _statusMessage;
  String? _errorMessage;

  int? _pinNumber;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 20.0, bottom: 20.0),
          child: ElevatedButton(
            onPressed: _isBusy ? null : _scan,
            child: Text('Scan for B24 Sensor'),
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
              SizedBox(
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
                'Status: ${_device.isConnected ? 'Connected' : 'Not Connected'} and ${_device.isReading ? 'Reading' : 'Not Reading'}',
              ),
              if (_statusMessage != null) Text(_statusMessage!),
              if (_errorMessage != null)
                Text(_errorMessage!, style: TextStyle(color: Colors.red)),
            ],
          ),
        ),

        Padding(
          padding: const EdgeInsets.only(bottom: 20.0),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              ElevatedButton(
                onPressed:
                    _isBusy || _device.isNotConnected || _device.isReading
                    ? null
                    : _startReading,
                child: Text('Start Reading'),
              ),
              SizedBox(width: 20),
              ElevatedButton(
                onPressed:
                    _isBusy || _device.isNotConnected || _device.isNotReading
                    ? null
                    : _stopReading,
                child: Text('Stop Reading'),
              ),
            ],
          ),
        ),
      ],
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
        throw BleDeviceNotFound(
          'B24 Sensor not found. Please make sure it is in pairing mode.',
        );
      }
      _statusMessage = 'B24 Sensor found: ${_device.name}';
      _errorMessage = null;
    } on BleDeviceBluetoothOff {
      _statusMessage = 'Scan failed';
      _errorMessage =
          'B24 Sensor not found. Please make sure it is in pairing mode.';
    } on BleDeviceNotFound catch (e) {
      _statusMessage = 'Scan failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Scan failed';
      _errorMessage = 'An unexpected error occurred: $e';
    }
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
      await _device.connect(pinNumber: 0);
      _statusMessage = 'Connected to B24 Sensor: ${_device.name}';
    } on BleDeviceCouldNotConnect catch (e) {
      _statusMessage = 'Connection failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Connection failed';
      _errorMessage = 'An unexpected error occurred: $e';
    }

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
      await _device.disconnect();
      _statusMessage = 'Disconnected from B24 Sensor: ${_device.name}';
    } on BleDeviceCouldNotDisconnect catch (e) {
      _statusMessage = 'Disconnection failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Disconnection failed';
      _errorMessage = 'An unexpected error occurred: $e';
    }

    setState(() {
      _isBusy = false;
    });
  }

  Future<void> _startReading() async {
    setState(() {
      _isBusy = true;
      _statusMessage = 'Starting to read from B24 Sensor...';
      _errorMessage = null;
    });

    try {
      await _device.startReading();
      _statusMessage = 'Started reading from B24 Sensor: ${_device.name}';
    } on BleDeviceNotConnected catch (e) {
      _statusMessage = 'Start reading failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Start reading failed';
      _errorMessage = 'An unexpected error occurred: $e';
    }

    setState(() {
      _isBusy = false;
    });
  }

  Future<void> _stopReading() async {
    setState(() {
      _isBusy = true;
      _statusMessage = 'Stopping reading from B24 Sensor...';
      _errorMessage = null;
    });

    try {
      await _device.stopReading();
      _statusMessage = 'Stopped reading from B24 Sensor: ${_device.name}';
    } on BleDeviceNotConnected catch (e) {
      _statusMessage = 'Stop reading failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Stop reading failed';
      _errorMessage = 'An unexpected error occurred: $e';
    }

    setState(() {
      _isBusy = false;
    });
  }
}
