import 'package:flutter/material.dart';

import '../frame_analyser.dart';
import '../media_reader.dart';

class MediaReaderContainer extends StatelessWidget {
  const MediaReaderContainer({
    super.key,
    required this.mediaReader,
    required this.analyser,
  });

  final MediaReader mediaReader;
  final FrameAnalyser analyser;

  @override
  Widget build(BuildContext context) {
    return StreamBuilder(
      stream: mediaReader.readFrames(),
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          // A new frame has arrived
          final frame = snapshot.data!;

          return FutureBuilder(
            future: analyser.perform(frame),
            builder: (context, snapshot) {
              if (snapshot.hasData) {
                // The analysis of the frame is complete, we can now display it
                final frame = snapshot.data!;

                final bytes = frame.toBytes();
                if (bytes != null) {
                  return Image.memory(bytes);
                } else {
                  return const Center(
                    child: Text('Failed to convert frame to bytes'),
                  );
                }
              } else if (snapshot.hasError) {
                return Center(child: Text('Error: ${snapshot.error}'));
              } else {
                return const Center(child: CircularProgressIndicator());
              }
            },
          );
        } else if (snapshot.hasError) {
          return Center(child: Text('Error: ${snapshot.error}'));
        } else {
          return const Center(child: CircularProgressIndicator());
        }
      },
    );
  }
}
