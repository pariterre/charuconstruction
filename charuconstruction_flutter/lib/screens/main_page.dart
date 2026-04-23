import 'package:charuconstruction_flutter/models/devices/ble/manage_ble_device_page.dart';
import 'package:charuconstruction_flutter/models/devices/device.dart';
import 'package:charuconstruction_flutter/providers/devices_provider.dart';
import 'package:charuconstruction_flutter/widgets/data_graph.dart';
import 'package:flutter/material.dart';

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
    return Scaffold(
      appBar: AppBar(
        title: Text('Charuconstruction'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ElevatedButton(
              onPressed: _manageBleDevices,
              child: Text('Manage BLE Devices'),
            ),
            ...DevicesProvider.instance.connectedDevices.map(
              (device) => Padding(
                padding: const EdgeInsets.only(top: 20.0),
                child: ShowCurrentValue(key: ValueKey(device), device: device),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _onDeviceStatusChanged(AvailableDevices device) {
    setState(() {});
  }

  Future<void> _manageBleDevices() async {
    await showDialog(
      context: context,
      builder: (context) => Scaffold(
        appBar: AppBar(
          title: Text('Manage BLE Devices'),
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        ),
        body: const BleDeviceManagerPage(),
      ),
    );
    if (!mounted) return;
    setState(() {});
  }
}

class ShowCurrentValue extends StatefulWidget {
  const ShowCurrentValue({super.key, required this.device});

  final Device device;

  @override
  State<ShowCurrentValue> createState() => _ShowCurrentValueState();
}

class _ShowCurrentValueState extends State<ShowCurrentValue> {
  late final graphController = DataGraphController(data: widget.device.data);

  @override
  void initState() {
    super.initState();

    widget.device.data.onNewData.listen(_onNewData);
  }

  @override
  void dispose() {
    widget.device.data.onNewData.cancel(_onNewData);

    super.dispose();
  }

  void _onNewData() {
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final time = widget.device.data.time;
    final currentTime = time.isNotEmpty ? time.last.toDouble() / 1000 : null;
    final data = widget.device.data.getData();

    final lastValues = data.map(
      (channel) => channel.isNotEmpty ? channel.last : double.nan,
    );

    return SingleChildScrollView(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('Current value for ${widget.device.name}: '),
          if (currentTime != null)
            Text(
              'at ${currentTime.toStringAsFixed(1)}s: [${lastValues.map((v) => v.toStringAsFixed(3)).join(', ')}]',
            )
          else
            Text('No data received yet'),
          SizedBox(height: 20),
          ElevatedButton(
            onPressed: () {
              widget.device.clearData();
            },
            child: Text('Clear Data'),
          ),
          DataGraph(controller: graphController),
        ],
      ),
    );
  }
}
