import 'dart:async';
import 'package:charuconstruction_flutter/devices/b24_force_sensor.dart';
import 'package:charuconstruction_flutter/providers/devices_provider.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(
    MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('B24 Force Sensor')),
        body: Center(
          child: Column(
            children: [
              ElevatedButton(
                onPressed: _connect,
                child: Text('Connect to Sensor'),
              ),
            ],
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

void _onNewData(DateTime timestamp, List<double> values) {
  print(
    't=${timestamp.toIso8601String()}  value=${values.map((v) => v.toStringAsFixed(3)).join(', ')}',
  );
}

Future<void> _connect() async {
  try {
    final device = B24ForceSensor();
    device.connect(pinNumber: );
    DevicesProvider.instance.add(device);

    // Listen to incoming data
    device.onNewData.listen(_onNewData);
  } catch (e) {
    print('Error: $e');
  }
}

Future<void> _disconnect() async {
  try {
    await sensor.stopReading();
    await sensor.disconnect();
    print('Disconnected cleanly.');
  } catch (e) {
    print('Error during disconnection: $e');
  }
}

Future<void> _startReading() async {
  // Wait 10 seconds
  await sensor.startReading();
  print('Streaming data for 10 seconds...\n');
  await Future.delayed(Duration(seconds: 10));

  print('\nStopping...');

  await subscription.cancel();
  await sensor.stopReading();
  await sensor.disconnect();

  print('Disconnected cleanly.');
}
