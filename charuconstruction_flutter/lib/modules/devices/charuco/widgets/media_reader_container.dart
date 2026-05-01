import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../charuco_device.dart';
import '../frame.dart';

class CameraFrameContainer extends StatefulWidget {
  const CameraFrameContainer({super.key, required this.device});

  final WebcamCharucos device;

  @override
  State<CameraFrameContainer> createState() => _CameraFrameContainerState();
}

class _CameraFrameContainerState extends State<CameraFrameContainer> {
  Uint8List? _lastFrameBytes;

  @override
  void initState() {
    widget.device.onNewFrame.listen(_onNewFrame);
    super.initState();
  }

  @override
  void dispose() {
    widget.device.onNewFrame.cancel(_onNewFrame);
    super.dispose();
  }

  void _onNewFrame(Frame? frame) {
    if (!mounted) return;
    setState(() {
      _lastFrameBytes = frame?.toBytes(grayscale: true);
    });
  }

  @override
  Widget build(BuildContext context) {
    return FittedBox(
      fit: BoxFit.contain,
      child: _lastFrameBytes == null
          ? const Center(child: CircularProgressIndicator())
          : Image.memory(_lastFrameBytes!, gaplessPlayback: true),
    );
  }
}
