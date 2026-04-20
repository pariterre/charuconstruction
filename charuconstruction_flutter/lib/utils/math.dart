class PythonRandom {
  final List<int> _mt = List<int>.filled(624, 0);
  int _mti = 625;

  PythonRandom({int? seed}) {
    _pythonSeed(seed ?? DateTime.now().millisecondsSinceEpoch);
  }

  ///
  /// Return a random floating point number in the range [0.0, 1.0)
  ///
  double random() {
    int a = _gen32() >> 5;
    int b = _gen32() >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  ///
  /// Pick a random integer in the range [minValue, maxValue]
  ///
  int randint(int minValue, int maxValue, {bool inclusive = true}) {
    if (maxValue == 0 || minValue >= maxValue) return 0;
    final temporaryMaxValue = maxValue - minValue;

    // Compute the number of bits needed to represent temporaryMaxValue
    int k = temporaryMaxValue.bitLength;
    int r = _gen32() >> (32 - k);

    // If the generated number is out of bounds, we retry (rejection sampling)
    // This ensures a perfectly uniform distribution
    while (r + (inclusive ? 0 : 1) > temporaryMaxValue) {
      r = _gen32() >> (32 - k);
    }

    // Add the minValue to shift the range from [0, temporaryMaxValue) to [minValue, maxValue]
    return r + minValue;
  }

  ///
  /// Shuffle the list in place using the Fisher-Yates algorithm, with the same random generator as Python's random.shuffle.
  ///
  void shuffle(List list) {
    for (int i = list.length - 1; i > 0; i--) {
      // Pick a random index from 0 to i
      int j = randint(0, i + 1, inclusive: false);

      // Échange des éléments
      var temp = list[i];
      list[i] = list[j];
      list[j] = temp;
    }
  }

  void _pythonSeed(int seed) {
    if (seed < 0) seed = -1 * seed;

    // Mimic Python's seeding algorithm for the Mersenne Twister
    _mt[0] = 19650218;
    for (int i = 1; i < 624; i++) {
      _mt[i] =
          (1812433253 * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + i) & 0xffffffff;
    }

    List<int> initKey = [
      seed & 0xffffffff,
      seed >> 32,
    ].where((e) => e != 0 || seed == 0).toList();
    if (seed == 0) initKey = [0];

    int i = 1, j = 0;
    for (int k = (624 > initKey.length ? 624 : initKey.length); k > 0; k--) {
      _mt[i] =
          ((_mt[i] ^ ((_mt[i - 1] ^ (_mt[i - 1] >> 30)) * 1664525)) +
              initKey[j] +
              j) &
          0xffffffff;
      i++;
      j++;
      if (i >= 624) {
        _mt[0] = _mt[623];
        i = 1;
      }
      if (j >= initKey.length) j = 0;
    }
    for (int k = 623; k > 0; k--) {
      _mt[i] =
          ((_mt[i] ^ ((_mt[i - 1] ^ (_mt[i - 1] >> 30)) * 1566083941)) - i) &
          0xffffffff;
      i++;
      if (i >= 624) {
        _mt[0] = _mt[623];
        i = 1;
      }
    }
    _mt[0] = 0x80000000;
  }

  int _gen32() {
    if (_mti >= 624) {
      for (int i = 0; i < 624; i++) {
        int y =
            ((_mt[i] & 0x80000000) | (_mt[(i + 1) % 624] & 0x7fffffff)) &
            0xffffffff;
        _mt[i] = (_mt[(i + 397) % 624] ^ (y >> 1)) & 0xffffffff;
        if ((y & 1) == 1) _mt[i] ^= 0x9908b0df;
      }
      _mti = 0;
    }
    int y = _mt[_mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);
    return y & 0xffffffff;
  }
}
