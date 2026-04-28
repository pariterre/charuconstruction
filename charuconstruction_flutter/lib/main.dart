import 'dart:ui';

import 'package:charuconstruction_flutter/modules/devices/devices.dart';
import 'package:charuconstruction_flutter/screens/main_page.dart';
import 'package:flutter/material.dart';
import 'package:logging/logging.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  UniversalBleInterface.useMocker = const bool.fromEnvironment(
    'CHARUCONSTRUCTION_USE_BLE_MOCKER',
    defaultValue: false,
  );
  UniversalBleInterface.mockedDevices.add(B24MockCharuconstructionBleDevice());

  final useCharucoMocker = const bool.fromEnvironment(
    'CHARUCONSTRUCTION_USE_CHARUCO_MOCKER',
    defaultValue: false,
  );
  if (useCharucoMocker) {
    AvailableDualCharucos.charucoToConstruct =
        AvailableDualCharucos.dualCharucosMocker;
  }

  // Setup logging
  Logger.root.level = Level.ALL;
  Logger.root.onRecord.listen((record) {
    debugPrint(
      '${record.level.name}: ${record.time}: ${record.loggerName}: ${record.message}',
    );
  });

  runApp(
    MaterialApp(
      home: const CharuconstructionApp(),
      theme: ThemeData(colorScheme: .fromSeed(seedColor: Colors.deepPurple)),
    ),
  );
}

class CharuconstructionApp extends StatefulWidget {
  const CharuconstructionApp({super.key});

  @override
  State<CharuconstructionApp> createState() => _CharuconstructionAppState();
}

class _CharuconstructionAppState extends State<CharuconstructionApp> {
  @override
  void initState() {
    super.initState();

    // Make sure all the devices are disconnected when the app is closed
    AppLifecycleListener(
      onExitRequested: () async {
        // Show a waiting dialog while disconnecting devices
        showDialog(
          context: context,
          barrierDismissible: false,
          builder: (context) => AlertDialog(
            title: Text('Exiting...'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                CircularProgressIndicator(),
                SizedBox(height: 20),
                Text('Disconnecting devices, please wait...'),
              ],
            ),
          ),
        );

        final toWait = <Future>[];
        for (final device in DevicesProvider.instance.connectedDevices) {
          toWait.add(device.disconnect());
        }
        await Future.wait(toWait);

        // close the waiting dialog
        if (mounted) Navigator.of(context).pop();
        return AppExitResponse.exit;
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return const MainPage();
  }
}
