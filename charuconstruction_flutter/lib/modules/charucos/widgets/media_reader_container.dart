import 'dart:typed_data';

import 'package:charuconstruction_flutter/modules/charucos/frame.dart';
import 'package:flutter/material.dart';

import '../frame_analyser.dart';
import '../media_reader.dart';

class MediaReaderContainer extends StatefulWidget {
  const MediaReaderContainer({
    super.key,
    required this.mediaReader,
    required this.analyser,
  });
  final MediaReader mediaReader;
  final FrameAnalyser analyser;

  @override
  State<MediaReaderContainer> createState() => _MediaReaderContainerState();
}

class _MediaReaderContainerState extends State<MediaReaderContainer> {
  @override
  Widget build(BuildContext context) {
    return StreamBuilder(
      stream: widget.mediaReader.readFrames(),
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }

        return FittedBox(
          fit: BoxFit.contain,
          child: SizedBox(
            width: 1920,
            height: 1080,
            child: _MediaReaderStream(
              analyser: widget.analyser,
              frame: snapshot.data!,
            ),
          ),
        );
      },
    );
  }
}

class _MediaReaderStream extends StatefulWidget {
  const _MediaReaderStream({required this.analyser, required this.frame});

  final FrameAnalyser analyser;
  final Frame frame;

  @override
  State<_MediaReaderStream> createState() => _MediaReaderStreamState();
}

class _MediaReaderStreamState extends State<_MediaReaderStream> {
  Uint8List? _lastFrameBytes;

  Future<void> _processFrame(Frame frame) async {
    final processedFrame = await widget.analyser.perform(frame);

    final bytes = processedFrame.toBytes();
    if (bytes != null) {
      setState(() {
        _lastFrameBytes = bytes;
      });
    }
  }

  @override
  void didUpdateWidget(covariant _MediaReaderStream oldWidget) {
    // A new frame has arrived
    _processFrame(widget.frame);
    super.didUpdateWidget(oldWidget);
  }

  @override
  Widget build(BuildContext context) {
    return _lastFrameBytes == null
        ? const Center(child: CircularProgressIndicator())
        : Image.memory(_lastFrameBytes!, gaplessPlayback: true);
  }
}
