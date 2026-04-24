import 'dart:math';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../time_series_data.dart';

class DataGraphController {
  final TimeSeriesData _data;

  Function()? _updateCallback;

  void _onNewData() {
    if (_updateCallback != null) _updateCallback!();
  }

  DataGraphController({required TimeSeriesData data}) : _data = data {
    data.onNewData.listen(_onNewData);
  }
}

class DataGraph extends StatefulWidget {
  const DataGraph({
    super.key,
    required this.controller,
    this.combineGraphs = false,
  });

  final bool combineGraphs;
  final DataGraphController controller;

  @override
  State<DataGraph> createState() => _DataGraphState();
}

class _DataGraphState extends State<DataGraph> {
  @override
  void initState() {
    super.initState();
    widget.controller._updateCallback = _redraw;
  }

  @override
  void dispose() {
    widget.controller._updateCallback = null;
    super.dispose();
  }

  DateTime _lastRefresh = DateTime.now();
  void _redraw() {
    if (DateTime.now().difference(_lastRefresh).inMilliseconds < 100) return;
    _lastRefresh = DateTime.now();
    if (!mounted) return;
    setState(() {});
  }

  Iterable<LineChartBarData?> _dataToLineBarsData({int? channel}) {
    final TimeSeriesData timeSeries = _timeSeries;

    final time = timeSeries.time;
    return timeSeries.getData().asMap().entries.map(
      (e) =>
          (channel != null && channel == e.key && _showChannels[e.key]) ||
              (channel == null && _showChannels[e.key])
          ? LineChartBarData(
              color: Colors.black,
              spots: e.value
                  .asMap()
                  .entries
                  .map((entry) => FlSpot(time[entry.key] / 1000.0, entry.value))
                  .toList(),
              isCurved: false,
              isStrokeCapRound: false,
              barWidth: 1,
              dotData: const FlDotData(show: false),
            )
          : null,
    );
  }

  TimeSeriesData get _timeSeries => widget.controller._data;

  int get _channelCount => _timeSeries.channelCount;

  bool _combineChannels = true;
  void _onChanged(bool combineChannels) {
    setState(() {
      _combineChannels = combineChannels;
    });
  }

  late final List<bool> _showChannels = List.generate(
    _channelCount,
    (_) => true,
  );
  void _onChannelSelected(int channel, bool newValue) {
    setState(() {
      _showChannels[channel] = newValue;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Column(
          children: [
            const SizedBox(height: 90),
            SizedBox(
              height: 400,
              width: double.infinity,
              child: Padding(
                padding: const EdgeInsets.only(
                  left: 12,
                  bottom: 12,
                  right: 20,
                  top: 20,
                ),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    return _combineChannels
                        ? _buildLineChart(constraints)
                        : ListView.builder(
                            itemCount: _channelCount,
                            itemBuilder: (context, index) =>
                                _showChannels[index]
                                ? Padding(
                                    padding: const EdgeInsets.only(
                                      bottom: 10.0,
                                    ),
                                    child: SizedBox(
                                      height: 90,
                                      child: _buildLineChart(
                                        constraints,
                                        channel: index,
                                      ),
                                    ),
                                  )
                                : const SizedBox(),
                          );
                  },
                ),
              ),
            ),
          ],
        ),
        Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _RadioCombineChannels(
                  combineChannels: _combineChannels,
                  onChanged: _onChanged,
                ),
              ],
            ),
            Align(
              alignment: Alignment.centerRight,
              child: Padding(
                padding: const EdgeInsets.only(right: 12.0),
                child: _ChannelOptionsPopup(
                  onChannelSelected: _onChannelSelected,
                  showChannels: _showChannels,
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildLineChart(BoxConstraints constraints, {int? channel}) {
    var data = _dataToLineBarsData(channel: channel).where((e) => e != null);

    return AspectRatio(
      aspectRatio: 1,
      child: LineChart(
        LineChartData(
          lineTouchData: _lineTouchData(),
          lineBarsData: data.map((e) => e!).toList(),
          titlesData: _titlesData(constraints),
          gridData: const FlGridData(
            show: true,
            drawHorizontalLine: true,
            drawVerticalLine: true,
            horizontalInterval: null,
            verticalInterval: null,
          ),
          borderData: FlBorderData(show: true),
        ),
        duration: Duration.zero,
      ),
    );
  }
}

LineTouchData _lineTouchData() {
  return LineTouchData(
    touchTooltipData: LineTouchTooltipData(
      maxContentWidth: 100,
      getTooltipColor: (touchedSpot) => Colors.grey[800]!,
      getTooltipItems: (touchedSpots) {
        return touchedSpots
            .map(
              (LineBarSpot touchedSpot) => LineTooltipItem(
                '${touchedSpot.x}, ${touchedSpot.y.toStringAsFixed(4)}',
                const TextStyle(color: Colors.white, fontSize: 12),
              ),
            )
            .toList();
      },
    ),
    handleBuiltInTouches: false, // Change this to add the values
    getTouchLineStart: (data, index) => 0,
  );
}

FlTitlesData _titlesData(
  BoxConstraints constraints, {
  TextStyle style = const TextStyle(fontWeight: FontWeight.bold),
}) => FlTitlesData(
  leftTitles: AxisTitles(
    sideTitles: SideTitles(
      showTitles: true,
      getTitlesWidget: (value, meta) => value % 1 != 0
          ? Container()
          : SideTitleWidget(
              meta: meta,
              space: 16,
              child: Text(
                meta.formattedValue,
                style: style.copyWith(
                  fontSize: min(18, 18 * constraints.maxWidth / 300),
                ),
              ),
            ),
      reservedSize: 56,
    ),
    drawBelowEverything: true,
  ),
  rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
  bottomTitles: AxisTitles(
    sideTitles: SideTitles(
      showTitles: true,
      getTitlesWidget: (value, meta) => value % 1 != 0
          ? Container()
          : SideTitleWidget(
              meta: meta,
              space: 16,
              child: Text(
                meta.formattedValue,
                style: style.copyWith(
                  fontSize: min(18, 18 * constraints.maxWidth / 300),
                ),
              ),
            ),
      reservedSize: 36,
      interval: 1,
    ),
    drawBelowEverything: true,
  ),
  topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
);

class _RadioCombineChannels extends StatelessWidget {
  const _RadioCombineChannels({
    required this.combineChannels,
    required this.onChanged,
  });

  final bool combineChannels;
  final Function(bool) onChanged;

  @override
  Widget build(BuildContext context) {
    final channelsName = 'channels';

    return RadioGroup(
      groupValue: combineChannels,
      onChanged: (value) => onChanged(value!),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: min(200, MediaQuery.of(context).size.width / 2),
            child: RadioListTile<bool>(
              title: Text('Combine $channelsName'),
              value: true,
            ),
          ),
          SizedBox(
            width: min(200, MediaQuery.of(context).size.width / 2),
            child: RadioListTile<bool>(
              title: Text('Separate $channelsName'),
              value: false,
            ),
          ),
        ],
      ),
    );
  }
}

class _ChannelOptionsPopup extends StatefulWidget {
  const _ChannelOptionsPopup({
    required this.onChannelSelected,
    required this.showChannels,
  });

  final Function(int index, bool value) onChannelSelected;
  final List<bool> showChannels;

  @override
  State<_ChannelOptionsPopup> createState() => _ChannelOptionsPopupState();
}

class _ChannelOptionsPopupState extends State<_ChannelOptionsPopup> {
  bool _isExpanded = false;
  bool get _canShowSelectAll => widget.showChannels.any((element) => !element);

  void _onExpand() {
    setState(() {
      _isExpanded = !_isExpanded;
    });
  }

  @override
  Widget build(BuildContext context) {
    const indexWidth = 40.0;
    const showWidth = 40.0;

    return SizedBox(
      width: 100.0,
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.black),
        ),
        child: Column(
          children: [
            GestureDetector(
              onTap: _onExpand,
              child: Container(
                width: double.infinity,
                color: Colors.grey,
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: Center(
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text('Channels'),
                      const SizedBox(width: 12),
                      Icon(
                        _isExpanded
                            ? Icons.arrow_drop_up
                            : Icons.arrow_drop_down,
                      ),
                    ],
                  ),
                ),
              ),
            ),
            if (_isExpanded)
              Container(
                color: Colors.grey[100],
                width: double.infinity,
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SizedBox(
                      width: indexWidth,
                      child: Center(child: Text('All')),
                    ),
                    SizedBox(
                      width: showWidth,
                      child: Column(
                        children: [
                          const Text('Show'),
                          Checkbox(
                            onChanged: (_) {
                              final value = _canShowSelectAll;
                              for (
                                var i = 0;
                                i < widget.showChannels.length;
                                i++
                              ) {
                                widget.onChannelSelected(i, value);
                              }
                              setState(() {});
                            },
                            value: !_canShowSelectAll,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            if (_isExpanded)
              ...List.generate(
                widget.showChannels.length,
                (index) => _OptionsBuilder(
                  index: index,
                  indexWidth: indexWidth,
                  showWidth: showWidth,
                  showChannels: widget.showChannels,
                  onChannelSelected: (_) {
                    widget.onChannelSelected(
                      index,
                      !widget.showChannels[index],
                    );
                    setState(() {});
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _OptionsBuilder extends StatefulWidget {
  const _OptionsBuilder({
    required this.index,
    required this.indexWidth,
    required this.showWidth,
    required this.showChannels,
    required this.onChannelSelected,
  });

  final int index;
  final double indexWidth;

  final double showWidth;
  final List<bool> showChannels;
  final Function(bool? value) onChannelSelected;

  @override
  State<_OptionsBuilder> createState() => _OptionsBuilderState();
}

class _OptionsBuilderState extends State<_OptionsBuilder> {
  final _focusNode = FocusNode();

  @override
  void dispose() {
    _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: widget.index % 2 == 0 ? Colors.grey[200] : Colors.grey[100],
      width: double.infinity,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: widget.indexWidth,
            child: Center(child: Text((widget.index + 1).toString())),
          ),
          SizedBox(
            width: widget.showWidth,
            child: Checkbox(
              onChanged: widget.onChannelSelected,
              value: widget.showChannels[widget.index],
            ),
          ),
        ],
      ),
    );
  }
}
