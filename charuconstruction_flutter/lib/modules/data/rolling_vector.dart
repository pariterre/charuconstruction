class RollingVector<T extends num> {
  ///
  /// The maximum size the vector can hold. Providing "-1" as a value results
  /// in an unbounded vector, i.e. a simple List\<double\>
  final int maxSize;

  ///
  /// The values of the vector
  List<T> _values;
  List<T> get values {
    if (maxSize < 0) return _values;
    if (!_isFilled) return _values.sublist(0, _index);

    final orderedValues = List<T>.filled(maxSize, _zero(), growable: false);
    orderedValues.setRange(0, maxSize - _index, _values, _index);
    orderedValues.setRange(maxSize - _index, maxSize, _values, 0);
    return orderedValues;
  }

  ///
  /// The index of the next value to be added. This is used to determine where to
  /// add the next value and to determine if the vector is filled at least once.
  int _index = 0;

  ///
  /// A flag to indicate if the vector has been filled at least once
  bool _isFilled = false;

  ///
  /// CONSTRUCTOR
  RollingVector(this.maxSize)
    : _values = maxSize >= 0
          ? List.filled(maxSize, _zero(), growable: false)
          : <T>[];

  ///
  /// Clear the vector, resetting all values to 0 and the index to 0.
  void clear() {
    _index = 0;
    _isFilled = false;

    if (maxSize < 0) {
      _values.clear();
    } else {
      _values = List.filled(maxSize, _zero(), growable: false);
    }
  }

  ///
  /// Get the value at a specific index
  T operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError.index(index, this, 'Index out of range');
    }
    if (maxSize < 0) return _values[index];

    if (!_isFilled) return _values[index];
    final actualIndex = (_index - length + index) % maxSize;
    return _values[actualIndex];
  }

  ///
  /// Add a new value to the vector. If the vector is full, the oldest value will be overwritten.
  void add(T value) {
    if (maxSize < 0) {
      _values.add(value);
      return;
    }

    _values[_index] = value;
    _index = (_index + 1) % maxSize;
    if (_index == 0) {
      _isFilled = true;
    }
  }

  void addAll(Iterable<T> values) {
    if (maxSize < 0) {
      _values.addAll(values);
      return;
    }

    final remainingSpace = maxSize - _index;
    final valuesToAdd = values.length;
    if (valuesToAdd >= remainingSpace) {
      _values.setRange(_index, maxSize, values, 0);
      _values.setRange(0, valuesToAdd - remainingSpace, values, remainingSpace);
      _index = valuesToAdd - remainingSpace;
      _isFilled = true;
    } else {
      _values.setRange(_index, _index + valuesToAdd, values);
      _index += valuesToAdd;
    }
  }

  ///
  /// Whether the vector is empty
  bool get isEmpty => values.isEmpty;

  ///
  /// Whether the vector is not empty
  bool get isNotEmpty => !isEmpty;

  ///
  /// The length of the vector, i.e. the number of values currently stored in the vector
  int get length {
    if (maxSize < 0) return _values.length;
    return _isFilled ? maxSize : _index;
  }

  ///
  /// Return the first element of the vector.
  T get first {
    if (isEmpty) {
      throw Exception('Cannot get first element of an empty vector.');
    }

    if (maxSize < 0) return _values.first;
    return _isFilled ? _values[_index] : _values[0];
  }

  ///
  /// Return the last element of the vector.
  T get last {
    if (isEmpty) {
      throw Exception('Cannot get last element of an empty vector.');
    }

    if (maxSize < 0) return _values.last;
    return _isFilled
        ? _values[(_index - 1 + maxSize) % maxSize]
        : _values[_index - 1];
  }

  ///
  /// Find index based on a condition.
  int indexWhere(bool Function(T value) test) {
    if (maxSize < 0) return _values.indexWhere(test);

    final firstIndex = _isFilled ? _index : 0;
    for (int i = 0; i < length; i++) {
      final actualIndex = (firstIndex + i) % maxSize;
      if (test(_values[actualIndex])) return i;
    }
    return -1;
  }

  ///
  /// Drop before a certain index, i.e. remove all values before the given index.
  void dropBefore(int index) {
    if (index <= 0 || index >= length) {
      throw RangeError.index(index, this, 'Index out of range');
    }

    if (maxSize < 0) {
      _values = _values.sublist(index);
      return;
    }

    final firstIndex = _isFilled ? _index : 0;
    final newValues = List<T>.filled(maxSize, _zero(), growable: false);
    for (int i = 0; i < length - index; i++) {
      final actualIndex = (firstIndex + index + i) % maxSize;
      newValues[i] = _values[actualIndex];
    }
    _values = newValues;
    _index = length - index;
    _isFilled = false;
  }

  ///
  /// The average of the values in the vector. If the vector is empty, returns 0.
  double get average {
    if (!_isFilled && _index == 0) return 0.0; // No values added yet

    final count = _isFilled ? maxSize : _index;
    final sum = values.fold(0.0, (a, b) => a + b);
    return sum / count;
  }
}

///
/// Zero as T
T _zero<T extends num>() => (T == double ? 0.0 : 0) as T;
