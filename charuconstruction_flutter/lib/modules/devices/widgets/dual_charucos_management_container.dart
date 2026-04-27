import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

import '../charuco/camera.dart';
import '../charuco/charuco.dart';
import '../charuco/frame_analyser.dart';
import '../concrete_devices/available_devices.dart';
import '../concrete_devices/dual_charucos.dart';
import '../device_exceptions.dart';
import '../providers/devices_provider.dart';

class DualCharucosManagementContainer extends StatefulWidget {
  const DualCharucosManagementContainer({super.key});

  @override
  State<DualCharucosManagementContainer> createState() =>
      _DualCharucosManagementContainerState();
}

class _DualCharucosManagementContainerState
    extends State<DualCharucosManagementContainer> {
  bool _isBusy = false;

  String? _statusMessage;
  String? _errorMessage;

  late final Set<AvailableFrameAnalysers> _selectedFrameAnalyser;

  @override
  void initState() {
    final analyzers = _device.analysers == null
        ? null
        : _device.analysers is FrameAnalyserPipeline
        ? (_device.analysers as FrameAnalyserPipeline).analysers
        : [_device.analysers!];

    _selectedFrameAnalyser = analyzers == null
        ? {AvailableFrameAnalysers.reconstructCharuco}
        : analyzers.map((analyser) {
            if (analyser is ReconstructCharucoFrameAnalyser) {
              return AvailableFrameAnalysers.reconstructCharuco;
            } else if (analyser is VideoSaverAnalyser) {
              return AvailableFrameAnalysers.videoSaver;
            } else {
              throw StateError(
                'Unknown frame analyser type: ${analyser.runtimeType}',
              );
            }
          }).toSet();

    super.initState();
  }

  String _videoOutputPath = '';

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Checkboxes to select the frame analysers to use, will be passed as arguments to the device when connecting
        ...AvailableFrameAnalysers.values.map((analyser) {
          return SizedBox(
            width: 400,
            child: CheckboxListTile(
              title: Text(analyser.toString()),
              value: _selectedFrameAnalyser.contains(analyser),
              enabled: _device.isNotConnected,
              onChanged: (value) {
                setState(() {
                  if (value!) {
                    _selectedFrameAnalyser.add(analyser);
                  } else {
                    _selectedFrameAnalyser.remove(analyser);
                  }
                });
              },
            ),
          );
        }),

        SizedBox(height: 20),
        if (_selectedFrameAnalyser.contains(AvailableFrameAnalysers.videoSaver))
          SizedBox(
            width: 400,
            child: TextField(
              enabled: _device.isNotConnected,
              decoration: InputDecoration(
                labelText: 'Video output path',
                hintText:
                    'Enter the path to save the video output (e.g. output.mp4)',
                border: OutlineInputBorder(),
              ),
              onChanged: (value) {
                setState(() {
                  _videoOutputPath = value;
                });
              },
            ),
          ),
        SizedBox(height: 20),

        Padding(
          padding: const EdgeInsets.only(bottom: 20.0),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(width: 20),
              ElevatedButton(
                onPressed: _isBusy || _device.isConnected ? null : _connect,
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
                'Status: ${_device.isConnected ? 'Connected' : 'Not Connected'} and ${_device.isReading ? 'Reading' : 'Not Reading'}',
              ),
              if (_statusMessage != null) Text(_statusMessage!),
              if (_errorMessage != null)
                Text(_errorMessage!, style: TextStyle(color: Colors.red)),
            ],
          ),
        ),
      ],
    );
  }

  DualCharucos get _device =>
      DevicesProvider.instance.device(AvailableDevices.dualCharucos)
          as DualCharucos;

  final charucos =
      [
        '../run/charuco_4x6_24/board.json',
        '../run/charuco_4x6_42/board.json',
      ].map((path) {
        final jsonString = File(path).readAsStringSync();
        return Charuco.fromSerialized(jsonDecode(jsonString.toString()));
      }).toList();
  final camera = CameraModels.pixel2.toCamera(useVideoParameters: true);

  Future<void> _connect() async {
    setState(() {
      _isBusy = true;
      _statusMessage = 'Connecting to the Charucos feed...';
      _errorMessage = null;
    });

    if (_selectedFrameAnalyser.contains(AvailableFrameAnalysers.videoSaver)) {
      if (_videoOutputPath.isEmpty) {
        setState(() {
          _isBusy = false;
          _statusMessage = 'Connection to the Charucos feed failed';
          _errorMessage =
              'Video output path is required when using the video saver analyser.';
        });
        return;
      } else if (_videoOutputPath.split('.').last.toLowerCase() != 'mp4') {
        setState(() {
          _isBusy = false;
          _statusMessage = 'Connection to the Charucos feed failed';
          _errorMessage = 'Video output path must have a .mp4 extension.';
        });
        return;
      }
    }

    final analyzers = [
      if (_selectedFrameAnalyser.contains(
        AvailableFrameAnalysers.reconstructCharuco,
      ))
        ReconstructCharucoFrameAnalyser(
          charucoBoards: charucos,
          camera: camera,
          ignoreReconstructionError: true,
        ),
      if (_selectedFrameAnalyser.contains(AvailableFrameAnalysers.videoSaver))
        VideoSaverAnalyser(outputPath: _videoOutputPath),
    ];

    try {
      await _device.connect(
        charucos: charucos,
        camera: camera,
        analysers: analyzers,
      );
      await _device.startReading();
      _statusMessage = 'Connected to the Charucos feed: ${_device.name}';
    } on DeviceCouldNotConnect catch (e) {
      _statusMessage = 'Connection to the Charucos feed failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Connection to the Charucos feed failed';
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
      _statusMessage = 'Disconnecting from the Charucos feed...';
      _errorMessage = null;
    });

    try {
      await _device.stopReading();
      await _device.disconnect();
      _statusMessage = 'Disconnected from the Charucos feed: ${_device.name}';
    } on DeviceCouldNotDisconnect catch (e) {
      _statusMessage = 'Disconnection from the Charucos feed failed';
      _errorMessage = e.toString();
    } catch (e) {
      _statusMessage = 'Disconnection from the Charucos feed failed';
      _errorMessage = 'An unexpected error occurred: $e';
    }

    if (!mounted) return;
    setState(() {
      _isBusy = false;
    });
  }
}
