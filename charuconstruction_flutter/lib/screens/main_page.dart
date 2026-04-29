import 'package:charuconstruction_flutter/modules/devices/devices.dart';
import 'package:charuconstruction_flutter/modules/devices/widgets/device_extensions.dart';
import 'package:flutter/material.dart';

class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
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
                onPressed: _manageDevices,
                child: Text('Manage sensor devices'),
              ),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ElevatedButton(
                    onPressed: _saveData,
                    child: Text('Save Data'),
                  ),
                  SizedBox(width: 20),
                  ElevatedButton(
                    onPressed: _clearData,
                    child: Text('Clear Data'),
                  ),
                ],
              ),
              ...DevicesProvider.instance.connectedDevices.map(
                (device) => Padding(
                  padding: const EdgeInsets.only(top: 20.0),
                  child: device.deviceDataContainer(device: device),
                ),
              ),
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
    await manageDevicesDialog(context);
    if (!mounted) return;
    setState(() {});
  }

  Future<void> _saveData() async {
    await DevicesProvider.instance.saveData();

    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text('Data saved successfully!')));
  }

  Future<void> _clearData() async {
    DevicesProvider.instance.clearData();

    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text('Data cleared successfully!')));
  }
}
