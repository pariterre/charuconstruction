import 'dart:convert';
import 'dart:io';

import 'package:charuconstruction_flutter/modules/charucos/charucos.dart'
    as charucos;
import 'package:charuconstruction_flutter/modules/devices/devices.dart';
import 'package:flutter/material.dart';
import 'package:ml_linalg/linalg.dart';

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
    final charucoFilePaths = [
      '../run/charuco_4x6_24/board.json',
      '../run/charuco_4x6_42/board.json',
    ];
    final charucoBoards = charucoFilePaths.map((path) {
      final jsonString = File(path).readAsStringSync();
      return charucos.Charuco.fromSerialized(jsonDecode(jsonString.toString()));
    }).toList();
    final camera = charucos.CameraModels.pixel2.toCamera(
      useVideoParameters: true,
    );
    final mediaReader = charucos.CharucoMockReader(
      camera: camera,
      charucos: charucoBoards,
      transformations: List.generate(
        10,
        (value) => [
          (Vector.zero(3), Matrix.identity(3)),
          (
            Vector.fromList([0, 0, 4.5]),
            charucos.MatrixExtensions.fromEuler([
              (value * 2.0, charucos.Axis.x),
              (-20.0, charucos.Axis.y),
              (-30.0, charucos.Axis.z),
            ]),
          ),
        ],
      ),
    );
    final analyser = charucos.FrameAnalyserPipeline(
      analysers: [
        charucos.CharucoFrameAnalyser(
          charucoBoards: charucoBoards,
          camera: camera,
          ignoreReconstructionError: true,
        ),
      ],
    );

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
              ...DevicesProvider.instance.connectedDevices.map(
                (device) => Padding(
                  padding: const EdgeInsets.only(top: 20.0),
                  child: DeviceDataContainer(
                    key: ValueKey(device),
                    device: device,
                  ),
                ),
              ),
              charucos.MediaReaderContainer(
                mediaReader: mediaReader,
                analyser: analyser,
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
}
