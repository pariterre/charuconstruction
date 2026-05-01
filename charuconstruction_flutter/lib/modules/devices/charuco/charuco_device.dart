import 'dart:async';

import 'package:charuconstruction_flutter/utils/generic_listener.dart';
import 'package:logging/logging.dart';

import '../device.dart';
import 'camera.dart';
import 'charuco.dart';
import 'frame.dart';
import 'frame_analyser.dart';
import 'media_reader.dart';

final _logger = Logger('DualCharucos');

abstract class CharucoDevice extends Device {
  ///
  /// The list of Charuco boards that the device is configured to read.
  final List<Charuco> _charucos = [];
  List<Charuco> get charucos => List.unmodifiable(_charucos);

  ///
  /// The camera used to read the Charuco boards.
  Camera? _camera;
  Camera? get camera => _camera;

  @override
  String? get name => 'Charuco Device';

  @override
  int get channelCount => 0;

  @override
  Future<void> connect({List<Charuco>? charucos, Camera? camera}) async {
    if (charucos == null) {
      throw ArgumentError(
        'Charucos must be provided to connect to a CharucoDevice',
      );
    }
    if (camera == null) {
      throw ArgumentError(
        'Camera must be provided to connect to a CharucoDevice',
      );
    }
    _camera = camera;

    _charucos.addAll(charucos);

    await super.connect();
  }

  @override
  Future<void> disconnect() async {
    _charucos.clear();
    _camera = null;
    await super.disconnect();
  }
}

abstract class WebcamCharucos extends CharucoDevice {
  @override
  String get name => "Dual Charucos";

  ///
  /// The [MediaReader] associated with this device. Will be initialized when the device is connected.
  MediaReader? get mediaReader;

  ///
  /// Reading and analyse data
  StreamSubscription? _frameSubscription;
  final onNewFrame = GenericListener<Function(Frame? frame)>();

  ///
  /// The [FrameAnalyser] associated with this device. Will be initialized when the device is connected.
  FrameAnalyser? _analysers;
  FrameAnalyser? get analysers => _analysers;

  @override
  Future<void> connect({
    List<Charuco>? charucos,
    Camera? camera,
    List<FrameAnalyser> analysers = const [],
    WebcamResolution resolution = WebcamResolution.medium,
    WebcamFPS fps = WebcamFPS.fps60,
    bool hideVideo = false,
  }) async {
    final output = await super.connect(charucos: charucos, camera: camera);

    if (mediaReader is WebcamReader) {
      await (mediaReader as WebcamReader).initialize(
        resolution: resolution,
        fps: fps,
        hideVideo: hideVideo,
      );
    } else {
      await mediaReader!.initialize();
    }
    _analysers = FrameAnalyserPipeline(analysers: analysers);

    return output;
  }

  @override
  Future<void> startReading() async {
    if (mediaReader == null) {
      throw StateError('MediaReader must be initialized to start reading');
    }

    if (mediaReader is WebcamReader) {
      await (mediaReader as WebcamReader).startReading(
        isPortrait: camera!.isPortrait,
      );
    } else {
      await mediaReader!.startReading();
    }

    _frameSubscription = mediaReader!.readFrames().listen(
      (frame) => pushDataFrame(frame),
      onDone: () => _logger.info('Finished reading frames'),
      onError: (error) => _logger.severe('Error reading frames: $error'),
    );

    return await super.startReading();
  }

  @override
  Future<void> stopReading() async {
    _frameSubscription?.cancel();
    _frameSubscription = null;
    await mediaReader?.stopReading();
    return await super.stopReading();
  }

  @override
  Future<void> disconnect() async {
    await mediaReader?.dispose();
    return await super.disconnect();
  }

  Future<(Frame?, Map<AvailableExtraAnalyses, dynamic>? extraAnalyses)>
  pushDataFrame(Frame? frame) async {
    final (analysedFrame, extraAnalyses) =
        await _analysers?.perform(frame) ?? (null, null);

    onNewFrame.notifyListeners((listener) => listener(analysedFrame));
    return (analysedFrame, extraAnalyses);
  }
}
