import 'dart:convert';
import 'dart:io';

import 'package:collection/collection.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../charuco/camera.dart';
import '../charuco/charuco.dart';
import '../charuco/frame_analyser.dart';
import '../charuco/media_reader.dart';
import '../concrete_devices/available_devices.dart';
import '../concrete_devices/webcam_charucos.dart';
import '../device_exceptions.dart';
import '../providers/devices_provider.dart';

String _charucoNameFromPath(String path) =>
    path
        .split('/')
        .firstWhereOrNull((segment) => segment.startsWith('charuco_')) ??
    'Unknown Charuco';

class WebcamDualCharucosManagementContainer extends StatefulWidget {
  const WebcamDualCharucosManagementContainer({super.key});

  @override
  State<WebcamDualCharucosManagementContainer> createState() =>
      _WebcamDualCharucosManagementContainerState();
}

class _WebcamDualCharucosManagementContainerState
    extends State<WebcamDualCharucosManagementContainer> {
  bool _isBusy = false;

  String? _statusMessage;
  String? _errorMessage;

  bool _isInitialized = false;
  late final _availableCharucos = <String>[];

  final List<String> _selectedCharucos = [];
  CameraModels _selectedCamera = Platform.isIOS
      ? CameraModels.iPadPro2016
      : CameraModels.pixel2;
  late final Set<AvailableFrameAnalysers> _selectedFrameAnalyser;

  bool _isPortrait = false;
  WebcamResolution _selectedResolution = WebcamResolution.medium;
  WebcamFPS _selectedFPS = WebcamFPS.fps60;
  bool _filterOutErrors = false;
  bool _showReconstructedCharucosOnFrame = true;
  bool _showOnGrayScale = false;
  bool _hideVideo = false;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    final isFirst = _device.analysers == null;

    final assetManifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final assets = assetManifest.listAssets();

    _availableCharucos.addAll(
      assets
          .where((filePath) => filePath.endsWith('board.json'))
          .map((filePath) => filePath)
          .toList(),
    );

    if (_device.camera != null) {
      _selectedCamera = CameraModels.fromString(_device.camera!.name);
      _isPortrait = _device.camera!.isPortrait;
    }
    if (_device.mediaReader is WebcamReader) {
      final webcamReader = _device.mediaReader as WebcamReader;
      _selectedResolution = webcamReader.resolution;
      _selectedFPS = webcamReader.fps;
      _hideVideo = webcamReader.hideVideo;
    }

    final analyzers = isFirst
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

    final reconstructAnalyser =
        analyzers?.firstWhereOrNull(
              (analyser) => analyser is ReconstructCharucoFrameAnalyser,
            )
            as ReconstructCharucoFrameAnalyser?;
    if (reconstructAnalyser != null) {
      _filterOutErrors = !reconstructAnalyser.ignoreReconstructionError;
      _showReconstructedCharucosOnFrame = reconstructAnalyser.showOnFrame;
      _showOnGrayScale = reconstructAnalyser.showAsGrayscale;
    }

    _selectedCharucos.addAll(
      isFirst
          ? _availableCharucos.map((e) => e).toList()
          : _availableCharucos
                .where(
                  (e) => _device.charucos.any((charuco) {
                    final name =
                        'charuco_${charuco.horizontalSquaresCount}x${charuco.verticalSquaresCount}_${charuco.seed}';

                    return name == _charucoNameFromPath(e);
                  }),
                )
                .toList(),
    );

    if (mounted) {
      setState(() {
        _isInitialized = true;
      });
    }
  }

  String _videoOutputPath = '';

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return Center(child: CircularProgressIndicator());
    }

    return Center(
      child: Column(
        children: [
          SizedBox(height: 20),
          Text('Select the Charuco boards you want to use'),
          ..._availableCharucos.map((charucoPath) {
            final charucoName = _charucoNameFromPath(charucoPath);

            return SizedBox(
              width: 400,
              child: CheckboxListTile(
                title: Text(charucoName),
                value: _selectedCharucos.contains(charucoPath),
                enabled: _device.isNotConnected,
                onChanged: (value) {
                  setState(() {
                    if (value!) {
                      _selectedCharucos.add(charucoPath);
                    } else {
                      _selectedCharucos.remove(charucoPath);
                    }
                  });
                },
              ),
            );
          }),

          SizedBox(height: 20),
          Text('Select the camera'),
          RadioGroup(
            groupValue: _selectedCamera,
            onChanged: (value) {
              setState(() {
                _selectedCamera = value!;
              });
            },
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: CameraModels.values.map((camera) {
                return SizedBox(
                  width: 400,
                  child: RadioListTile<CameraModels>(
                    title: Text(camera.toString()),
                    enabled: _device.isNotConnected,
                    value: camera,
                  ),
                );
              }).toList(),
            ),
          ),

          SizedBox(height: 20),
          SizedBox(
            width: 400,
            child: CheckboxListTile(
              title: Text('Is camera vertical'),
              value: _isPortrait,
              enabled: _device.isNotConnected,
              onChanged: (value) {
                setState(() {
                  _isPortrait = value!;
                });
              },
            ),
          ),

          SizedBox(height: 20),
          Text('Select the resolution'),
          RadioGroup(
            groupValue: _selectedResolution,
            onChanged: (value) {
              setState(() {
                _selectedResolution = value!;
              });
            },
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: WebcamResolution.values.map((resolution) {
                return SizedBox(
                  width: 400,
                  child: RadioListTile<WebcamResolution>(
                    title: Text(resolution.toString()),
                    enabled: _device.isNotConnected,
                    value: resolution,
                  ),
                );
              }).toList(),
            ),
          ),

          SizedBox(height: 20),
          Text('Select the FPS'),
          RadioGroup(
            groupValue: _selectedFPS,
            onChanged: (value) {
              setState(() {
                _selectedFPS = value!;
              });
            },
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: WebcamFPS.values.map((fps) {
                return SizedBox(
                  width: 400,
                  child: RadioListTile<WebcamFPS>(
                    title: Text(fps.toString()),
                    enabled: _device.isNotConnected,
                    value: fps,
                  ),
                );
              }).toList(),
            ),
          ),

          SizedBox(height: 20),
          Text('Select the analysers you want to apply to the Charucos feed'),
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
          if (_selectedFrameAnalyser.contains(
            AvailableFrameAnalysers.reconstructCharuco,
          ))
            Column(
              children: [
                SizedBox(
                  width: 400,
                  child: CheckboxListTile(
                    title: Text(
                      'Reduce reconstruction error by filtering out detections with '
                      'high reprojection error (slower)',
                    ),
                    value: _filterOutErrors,
                    enabled: _device.isNotConnected,
                    onChanged: (value) {
                      setState(() {
                        _filterOutErrors = value!;
                      });
                    },
                  ),
                ),

                SizedBox(height: 20),
                SizedBox(
                  width: 400,
                  child: CheckboxListTile(
                    title: Text('Hide video'),
                    value: _hideVideo,
                    enabled: _device.isNotConnected,
                    onChanged: (value) {
                      setState(() {
                        _hideVideo = value!;
                      });
                    },
                  ),
                ),

                SizedBox(height: 20),
                SizedBox(
                  width: 400,
                  child: CheckboxListTile(
                    title: Text('Show reconstructed charucos on the frame'),
                    value: _hideVideo
                        ? false
                        : _showReconstructedCharucosOnFrame,
                    enabled: !_hideVideo && _device.isNotConnected,
                    onChanged: (value) {
                      setState(() {
                        _showReconstructedCharucosOnFrame = value!;
                      });
                    },
                  ),
                ),

                SizedBox(height: 20),
                SizedBox(
                  width: 400,
                  child: CheckboxListTile(
                    title: Text('Show on grayscale frame'),
                    value: _showOnGrayScale,
                    enabled: !_hideVideo && _device.isNotConnected,
                    onChanged: (value) {
                      setState(() {
                        _showOnGrayScale = value!;
                      });
                    },
                  ),
                ),
              ],
            ),

          SizedBox(height: 20),
          if (_selectedFrameAnalyser.contains(
            AvailableFrameAnalysers.videoSaver,
          ))
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
      ),
    );
  }

  WebcamDualCharucos get _device =>
      DevicesProvider.instance.device(AvailableDevices.dualCharucos)
          as WebcamDualCharucos;

  Future<void> _connect() async {
    final charucosFutures = _selectedCharucos.map((path) async {
      final jsonString = await rootBundle.loadString(path);
      return Charuco.fromSerialized(jsonDecode(jsonString.toString()));
    }).toList();
    final charucos = await Future.wait(charucosFutures);

    final camera = _selectedCamera.toCamera(
      useVideoParameters: true,
      isPortrait: _isPortrait,
    );

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
          ignoreReconstructionError: !_filterOutErrors,
          showOnFrame: _hideVideo ? false : _showReconstructedCharucosOnFrame,
          showAsGrayscale: _showOnGrayScale,
        ),
      if (_selectedFrameAnalyser.contains(AvailableFrameAnalysers.videoSaver))
        VideoSaverAnalyser(outputPath: _videoOutputPath),
    ];

    try {
      await _device.connect(
        charucos: charucos,
        camera: camera,
        analysers: analyzers,
        resolution: _selectedResolution,
        fps: _selectedFPS,
        hideVideo: _hideVideo,
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
