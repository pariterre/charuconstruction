import 'package:charuconstruction_flutter/utils/math.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('Test next int from 0 to 10', () {
    final random = PythonRandom(seed: 123);
    expect(random.randint(0, 10), 0);
    expect(random.randint(0, 10), 4);
    expect(random.randint(0, 10), 1);
    expect(random.randint(0, 10), 6);
    expect(random.randint(0, 10), 4);
  });

  test('Test next int from 10 to 20', () {
    final random = PythonRandom(seed: 123);
    expect(random.randint(10, 20), 10);
    expect(random.randint(10, 20), 14);
    expect(random.randint(10, 20), 11);
    expect(random.randint(10, 20), 16);
    expect(random.randint(10, 20), 14);
  });

  test('Test next double', () {
    final random = PythonRandom(seed: 123);
    expect(random.random(), 0.052363598850944326);
    expect(random.random(), 0.08718667752263232);
    expect(random.random(), 0.4072417636703983);
    expect(random.random(), 0.10770023493843905);
    expect(random.random(), 0.9011988779516946);
  });

  test('Test negative seed', () {
    final random = PythonRandom(seed: -123);
    expect(random.random(), 0.052363598850944326);
    expect(random.random(), 0.08718667752263232);
    expect(random.random(), 0.4072417636703983);
    expect(random.random(), 0.10770023493843905);
    expect(random.random(), 0.9011988779516946);
  });

  test('Test shuffle', () {
    final random = PythonRandom(seed: 123);
    final results = List.generate(10, (index) => index);
    random.shuffle(results);
    expect(results, [8, 7, 5, 9, 2, 3, 6, 1, 4, 0]);
  });
}
