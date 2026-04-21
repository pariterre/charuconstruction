import 'dart:convert';
import 'dart:ui';

import 'package:charuconstruction_flutter/models/charucos/charuco.dart';
import 'package:charuconstruction_flutter/models/devices/b24_force_sensor.dart';
import 'package:charuconstruction_flutter/models/devices/ble/manage_ble_device_dialog.dart';
import 'package:charuconstruction_flutter/models/devices/ble/universalr_ble_interface.dart';
import 'package:charuconstruction_flutter/models/devices/device.dart';
import 'package:charuconstruction_flutter/providers/devices_provider.dart';
import 'package:charuconstruction_flutter/widgets/data_graph.dart';
import 'package:flutter/material.dart';
import 'package:logging/logging.dart';

Future<void> main() async {
  Charuco.fromSerialized(
    jsonDecode(
      '{"vertical_squares_count": 6,"horizontal_squares_count": 4,"square_len": 0.018,"marker_len": 0.014,"resolution": "DPI_300","aruco_dict": 15,"seed": 24}',
    ),
  );

  UniversalBleInterface.useMocker = const bool.fromEnvironment(
    'CHARUCONSTRUCTION_USE_BLE_MOCKER',
    defaultValue: false,
  );
  UniversalBleInterface.mockedDevices.add(B24MockCharuconstructionBleDevice());

  // Setup logging
  Logger.root.level = Level.ALL;
  Logger.root.onRecord.listen((record) {
    debugPrint(
      '${record.level.name}: ${record.time}: ${record.loggerName}: ${record.message}',
    );
  });

  runApp(
    MaterialApp(
      home: const CharuconstructionApp(),
      theme: ThemeData(colorScheme: .fromSeed(seedColor: Colors.deepPurple)),
    ),
  );
}

class CharuconstructionApp extends StatefulWidget {
  const CharuconstructionApp({super.key});

  @override
  State<CharuconstructionApp> createState() => _CharuconstructionAppState();
}

class _CharuconstructionAppState extends State<CharuconstructionApp> {
  @override
  void initState() {
    super.initState();

    // Make sure all the devices are disconnected when the app is closed
    AppLifecycleListener(
      onExitRequested: () async {
        // Show a waiting dialog while disconnecting devices
        showDialog(
          context: context,
          barrierDismissible: false,
          builder: (context) => AlertDialog(
            title: Text('Exiting...'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                CircularProgressIndicator(),
                SizedBox(height: 20),
                Text('Disconnecting devices, please wait...'),
              ],
            ),
          ),
        );

        final toWait = <Future>[];
        for (final device in DevicesProvider.instance.connectedDevices) {
          toWait.add(device.disconnect());
        }
        await Future.wait(toWait);

        // close the waiting dialog
        if (mounted) Navigator.of(context).pop();
        return AppExitResponse.exit;
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return const MainScreen();
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
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

  void _onDeviceStatusChanged(AvailableDevices device) {
    setState(() {});
  }

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
}

class ShowCurrentValue extends StatefulWidget {
  const ShowCurrentValue({super.key, required this.device});

  final Device device;

  @override
  State<ShowCurrentValue> createState() => _ShowCurrentValueState();
}

class _ShowCurrentValueState extends State<ShowCurrentValue> {
  late final graphController = DataGraphController(data: widget.device.data);

  @override
  void initState() {
    super.initState();

    widget.device.data.onNewData.listen(_onNewData);
  }

  @override
  void dispose() {
    widget.device.data.onNewData.cancel(_onNewData);

    super.dispose();
  }

  void _onNewData() {
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final time = widget.device.data.time;
    final currentTime = time.isNotEmpty ? time.last.toDouble() / 1000 : null;
    final data = widget.device.data.getData();

    final lastValues = data.map(
      (channel) => channel.isNotEmpty ? channel.last : double.nan,
    );

    return SingleChildScrollView(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('Current value for ${widget.device.name}: '),
          if (currentTime != null)
            Text(
              'at ${currentTime.toStringAsFixed(1)}s: [${lastValues.map((v) => v.toStringAsFixed(3)).join(', ')}]',
            )
          else
            Text('No data received yet'),
          SizedBox(height: 20),
          ElevatedButton(
            onPressed: () {
              widget.device.clearData();
            },
            child: Text('Clear Data'),
          ),
          DataGraph(controller: graphController),
        ],
      ),
    );
  }
}
