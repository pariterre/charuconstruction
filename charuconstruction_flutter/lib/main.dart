import 'package:charuconstruction_flutter/devices/ble/manage_ble_device_dialog.dart';
import 'package:charuconstruction_flutter/devices/device.dart';
import 'package:charuconstruction_flutter/providers/devices_provider.dart';
import 'package:flutter/material.dart';
import 'package:logging/logging.dart';

void main() {
  // Setup logging
  Logger.root.level = Level.ALL;
  Logger.root.onRecord.listen((record) {
    debugPrint(
      '${record.level.name}: ${record.time}: ${record.loggerName}: ${record.message}',
    );
  });

  runApp(CharuconstructionApp());
}

class CharuconstructionApp extends StatelessWidget {
  const CharuconstructionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: const MainScreen(),
      theme: ThemeData(colorScheme: .fromSeed(seedColor: Colors.deepPurple)),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  Future<void> _manageBleDevices() async {
    await showDialog(
      context: context,
      builder: (context) => Scaffold(
        appBar: AppBar(
          title: Text('Manage BLE Devices'),
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        ),
        body: const BleDeviceManagerPage(),
      ),
    );
    if (!mounted) return;
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Charuconstruction'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ElevatedButton(
              onPressed: _manageBleDevices,
              child: Text('Manage BLE Devices'),
            ),
            ...DevicesProvider.instance.connectedDevices.map(
              (device) => Padding(
                padding: const EdgeInsets.only(top: 20.0),
                child: ShowCurrentValue(key: ValueKey(device), device: device),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ShowCurrentValue extends StatefulWidget {
  const ShowCurrentValue({super.key, required this.device});

  final Device device;

  @override
  State<ShowCurrentValue> createState() => _ShowCurrentValueState();
}

class _ShowCurrentValueState extends State<ShowCurrentValue> {
  @override
  void initState() {
    super.initState();

    widget.device.onNewData.listen(_onNewData);
  }

  @override
  void dispose() {
    widget.device.onNewData.cancel(_onNewData);

    super.dispose();
  }

  void _onNewData(DateTime timestamp, List<double> values) {
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final currentTime = widget.device.timeVector.isNotEmpty
        ? widget.device.timeVector.last
                  .difference(widget.device.startingTime)
                  .inMilliseconds
                  .toDouble() /
              1000
        : null;
    final currentValues = widget.device.dataVector.isNotEmpty
        ? widget.device.dataVector.last
        : <double>[];

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('Current value for ${widget.device.name}: '),
        if (currentTime != null && currentValues.isNotEmpty)
          Text(
            'at ${currentTime.toStringAsFixed(1)}: [${currentValues.map((v) => v.toStringAsFixed(3)).join(', ')}]',
          )
        else
          Text('No data received yet'),
        SizedBox(width: 20),
        ElevatedButton(
          onPressed: () {
            widget.device.clearData();
          },
          child: Text('Clear Data'),
        ),
      ],
    );
  }
}
