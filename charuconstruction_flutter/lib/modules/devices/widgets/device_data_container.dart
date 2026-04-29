import 'package:charuconstruction_flutter/modules/data/data.dart';
import 'package:flutter/material.dart';

import '../device.dart';

class DeviceDataContainer extends StatefulWidget {
  const DeviceDataContainer({super.key, required this.device});

  final Device device;

  @override
  State<DeviceDataContainer> createState() => _DeviceDataContainerState();
}

class _DeviceDataContainerState extends State<DeviceDataContainer> {
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

          DataGraph(controller: graphController),
        ],
      ),
    );
  }
}
