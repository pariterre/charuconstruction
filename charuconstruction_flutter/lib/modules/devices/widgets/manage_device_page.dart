import 'package:flutter/material.dart';

import '../concrete_devices/available_devices.dart';
import 'b24_management_container.dart';
import 'dual_charucos_management_container.dart';

Future<void> manageDevicesDialog(BuildContext context) async {
  return await showDialog(
    context: context,
    builder: (context) => Dialog(
      child: Scaffold(
        appBar: AppBar(
          title: Text('Manage Devices'),
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        ),
        body: const BleDeviceManagerContainer(),
      ),
    ),
  );
}

class BleDeviceManagerContainer extends StatefulWidget {
  const BleDeviceManagerContainer({super.key});

  @override
  State<BleDeviceManagerContainer> createState() =>
      _BleDeviceManagerContainerState();
}

class _BleDeviceManagerContainerState extends State<BleDeviceManagerContainer> {
  AvailableDevices _selectedDevice = AvailableDevices.values.first;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          DropdownButton<AvailableDevices>(
            hint: Text('Select a device to manage'),
            items: AvailableDevices.values.map((device) {
              return DropdownMenuItem(
                value: device,
                child: Text(device.toString().split('.').last),
              );
            }).toList(),
            value: _selectedDevice,
            onChanged: _selectDevice,
          ),
          switch (_selectedDevice) {
            AvailableDevices.b24 => const B24ManagementContainer(),
            AvailableDevices.dualCharucos =>
              const DualCharucosManagementContainer(),
          },
        ],
      ),
    );
  }

  void _selectDevice(AvailableDevices? device) {
    if (device == null) return;
    setState(() {
      _selectedDevice = device;
    });
  }
}
