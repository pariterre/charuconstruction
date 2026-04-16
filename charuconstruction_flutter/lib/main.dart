import 'dart:async';
import 'package:charuconstruction_flutter/devices/b24_force_sensor.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(
    MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('B24 Force Sensor')),
        body: Center(
          child: ElevatedButton(
            onPressed: connect,
            child: Text('Connect to Sensor'),
          ),
        ),
      ),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const Placeholder();
  }
}

class ShowStatus extends StatefulWidget {
  const ShowStatus({super.key});

  @override
  State<ShowStatus> createState() => _ShowStatusState();
}

class _ShowStatusState extends State<ShowStatus> {
  @override
  Widget build(BuildContext context) {
    return const Placeholder();
  }
}

void connect() async {
  try {
    final sensor = await B24ForceSensor.fromBluetooth();
    final connected = await sensor.connect(pinNumber: 0);

    if (!connected) {
      print('Failed to connect.');
      return;
    }

    // Listen to incoming data
    final subscription = sensor.onDataReceived.listen((event) {
      final (time, values) = event;
      print(
        't=${time.toStringAsFixed(3)}s  value=${values.map((v) => v.toStringAsFixed(3)).join(', ')}',
      );
    });

    // Wait 10 seconds
    await sensor.startReading();
    print('Streaming data for 10 seconds...\n');
    await Future.delayed(Duration(seconds: 10));

    print('\nStopping...');

    await subscription.cancel();
    await sensor.stopReading();
    await sensor.disconnect();

    print('Disconnected cleanly.');
  } catch (e) {
    print('Error: $e');
  }
}
