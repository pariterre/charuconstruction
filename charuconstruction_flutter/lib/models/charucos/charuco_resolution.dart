enum CharucoResolution {
  dpi100,
  dpi200,
  dpi300;

  @override
  String toString() {
    return switch (this) {
      CharucoResolution.dpi100 => "dpi_100",
      CharucoResolution.dpi200 => "dpi_200",
      CharucoResolution.dpi300 => "dpi_300",
    };
  }

  int get pixelsPerMeter => switch (this) {
    CharucoResolution.dpi100 => 4724,
    CharucoResolution.dpi200 => 7874,
    CharucoResolution.dpi300 => 11811,
  };

  Map<String, dynamic> serialize() => {'resolution': toString()};

  static CharucoResolution fromName(String name) {
    return switch (name.toLowerCase()) {
      'dpi_100' => CharucoResolution.dpi100,
      'dpi_200' => CharucoResolution.dpi200,
      'dpi_300' => CharucoResolution.dpi300,
      _ => throw ArgumentError('Invalid resolution: $name'),
    };
  }
}
