import 'dart:convert';
import 'dart:io';

import 'package:charuconstruction_flutter/models/charucos/camera.dart';
import 'package:charuconstruction_flutter/models/charucos/charuco.dart';
import 'package:charuconstruction_flutter/models/charucos/frame_analyser.dart';
import 'package:charuconstruction_flutter/models/charucos/media_reader.dart';
import 'package:charuconstruction_flutter/models/devices/ble/manage_ble_device_page.dart';
import 'package:charuconstruction_flutter/models/devices/device.dart';
import 'package:charuconstruction_flutter/providers/devices_provider.dart';
import 'package:charuconstruction_flutter/widgets/data_graph.dart';
import 'package:charuconstruction_flutter/widgets/media_reader_container.dart';
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
    // TODO Move these temporary stuff to accessors
    final imagePath = '../run/charuco_4x6_24/tata.png';
    final charucoFilePaths = [
      '../run/charuco_4x6_24/board.json',
      '../run/charuco_4x6_42/board.json',
    ];
    final charucoBoards = charucoFilePaths.map((path) {
      final jsonString = File(path).readAsStringSync();
      return Charuco.fromSerialized(jsonDecode(jsonString.toString()));
    }).toList();
    final camera = CameraModels.pixel2.toCamera();

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
                child: _ShowCurrentValue(key: ValueKey(device), device: device),
              ),
            ),
            MediaReaderContainer(
              mediaReader: ImageReader(imagePath: imagePath),
              analyser: CharucoFrameAnalyser(
                charucoBoards: charucoBoards,
                camera: camera,
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

class _ShowCurrentValue extends StatefulWidget {
  const _ShowCurrentValue({super.key, required this.device});

  final Device device;

  @override
  State<_ShowCurrentValue> createState() => _ShowCurrentValueState();
}

class _ShowCurrentValueState extends State<_ShowCurrentValue> {
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
