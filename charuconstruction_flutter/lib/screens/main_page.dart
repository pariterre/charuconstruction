import 'package:charuconstruction_flutter/modules/devices/devices.dart';
import 'package:charuconstruction_flutter/modules/devices/widgets/device_extensions.dart';
import 'package:flutter/material.dart';

class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  bool _isBusy = false;
  bool _isRecording = false;

  @override
  void initState() {
    super.initState();

    DevicesProvider.instance.onDeviceStatusChanged.listen(
      _onDeviceStatusChanged,
    );
  }

  @override
  void dispose() {
    DevicesProvider.instance.onDeviceStatusChanged.cancel(
      _onDeviceStatusChanged,
    );
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final hasDevices = DevicesProvider.instance.connectedDevices.isNotEmpty;

    return Scaffold(
      appBar: AppBar(
        title: Text('Charuconstruction'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ElevatedButton(
                onPressed: _isBusy || _isRecording ? null : _manageDevices,
                child: Text('Manage sensor devices'),
              ),
              SizedBox(height: 12),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ElevatedButton(
                    onPressed: _isBusy || _isRecording || !hasDevices
                        ? null
                        : _setZero,
                    child: Text('Set zero'),
                  ),
                  SizedBox(width: 20),
                  ElevatedButton(
                    onPressed: _isBusy || _isRecording || !hasDevices
                        ? null
                        : _clearData,
                    child: Text('Clear Data'),
                  ),
                ],
              ),
              SizedBox(height: 12),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ElevatedButton(
                    onPressed: _isBusy || _isRecording || !hasDevices
                        ? null
                        : _startRecording,
                    child: Text('Start Recording'),
                  ),
                  SizedBox(width: 20),
                  ElevatedButton(
                    onPressed: _isBusy || !_isRecording ? null : _stopRecording,
                    child: Text('Stop Recording'),
                  ),
                ],
              ),
              SizedBox(height: 12),
              ...DevicesProvider.instance.connectedDevices.map(
                (device) => Padding(
                  padding: const EdgeInsets.only(top: 20.0),
                  child: device.deviceDataContainer(device: device),
                ),
              ),
              SizedBox(height: MediaQuery.of(context).size.height * 0.75),
            ],
          ),
        ),
      ),
    );
  }

  void _onDeviceStatusChanged(AvailableDevices device) {
    setState(() {});
  }

  Future<void> _manageDevices() async {
    setState(() {
      _isBusy = true;
    });
    await manageDevicesDialog(context);
    if (!mounted) return;
    setState(() {
      _isBusy = false;
    });
  }

  Future<void> _startRecording() async {
    setState(() {
      _isBusy = true;
      _isRecording = true;
    });
    await DevicesProvider.instance.startRecording();

    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text('Recording started!')));
    setState(() {
      _isBusy = false;
    });
  }

  Future<void> _stopRecording() async {
    setState(() {
      _isBusy = true;
      _isRecording = false;
    });
    await DevicesProvider.instance.stopRecording();
    await DevicesProvider.instance.saveData();

    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text('Recording stopped!')));
    setState(() {
      _isBusy = false;
    });
  }

  Future<void> _clearData() async {
    setState(() {
      _isBusy = true;
    });
    await DevicesProvider.instance.clearData();

    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text('Data cleared successfully!')));
    setState(() {
      _isBusy = false;
    });
  }

  Future<void> _setZero() async {
    setState(() {
      _isBusy = true;
    });
    await DevicesProvider.instance.setZero();

    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text('Zero set successfully!')));
    setState(() {
      _isBusy = false;
    });
  }
}
