extension ListExtension<T> on List<T> {
  ///
  /// Returns the indices of the items that are unique in the list.
  /// For example, if the list is [1, 2, 3, 2, 4], the unique items are [1, 3, 4]
  /// and their indices are [0, 2, 4].
  List<int> singleItemsIndices() {
    final uniqueIndices = <int>{};
    for (int i = 0; i < length; i++) {
      if (indexOf(this[i]) == lastIndexOf(this[i])) {
        uniqueIndices.add(i);
      }
    }
    return uniqueIndices.toList();
  }
}
