from typing import Any, Callable, List


class Listener:
    def __init__(self):
        self._listeners: List[Callable[[Any], None]] = []

    def __contains__(self, item):
        return item in self._listeners

    def add_listener(self, callback: Callable[[Any], None]) -> None:
        """
        Register a callback function to be called whenever new data is received from the sensor.

        Args:
            callback (function): A function that takes a single argument (the new data) and returns None.
        """
        self._listeners.append(callback)

    def notify_listeners(self, data: Any) -> None:
        """
        Notify all registered listeners with the new data.

        Args:
            data: The new data to be passed to the listeners.
        """
        for callback in self._listeners:
            callback(data)

    def remove_listener(self, callback: Callable[[Any], None]) -> None:
        """
        Unregister a previously registered callback function.

        Args:
            callback (function): The callback function to be removed.
        """
        if callback in self._listeners:
            self._listeners.remove(callback)
