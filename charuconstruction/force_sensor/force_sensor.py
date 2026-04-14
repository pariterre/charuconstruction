from abc import ABC, abstractmethod
from functools import partial

from matplotlib import pyplot as plt
import numpy as np

from .listener import Listener


class ForceSensor(ABC):
    def __init__(self):
        super().__init__()

        self._on_data_received = Listener()
        self._on_status_received = Listener()

        self._live_plot_axes = None

    @property
    @abstractmethod
    def name() -> str:
        """
        Return the name of the force sensor type.

        Returns:
            str: The name of the force sensor type.
        """
        pass

    @property
    def address(self) -> str:
        """
        Return the Bluetooth address of the force sensor.

        Returns:
            str: The Bluetooth address of the force sensor.
        """
        pass

    @abstractmethod
    def start_reading(self, *args, **kwargs) -> bool:
        """
        Start reading data from the force sensor. The device starts streaming data
        to the connected client via the on_data_received callback.

        Returns:
            bool: True if the reading was successfully started, False otherwise.
        """

        pass

    @abstractmethod
    def stop_reading(self, *args, **kwargs) -> bool:
        """
        Stop reading data from the force sensor.

        Returns:
            bool: True if the reading was successfully stopped, False otherwise.
        """
        pass

    @abstractmethod
    def clear_data(self) -> bool:
        """
        Clear the data received from the sensor since the starting of the current reading session, or since clear_data was last called.

        Returns:
            bool: True if the data were successfully cleared, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def time_vector(self) -> np.ndarray:
        """
        Return the time vector corresponding to full data batches received from the sensor since the starting
        of the current reading session, or since clear_data was last called.

        Returns:
            np.ndarray: The time vector corresponding to full data batches received from the sensor since the starting
            of the current reading session, or since clear_data was last called.
        """
        pass

    @property
    @abstractmethod
    def force_vector(self) -> np.ndarray:
        """
        Return the force vector corresponding to full data batches received from the sensor since the starting
        of the current reading session, or since clear_data was last called. The data are arranged in a 2D array
        of shape (n_samples, n_dimensions), matching the shape of time_vector.

        Returns:
            np.ndarray: The force vector corresponding to full data batches received from the sensor since the starting
            of the current reading session, or since clear_data was last called. The data are arranged in a 2D array
            of shape (n_samples, n_dimensions), matching the shape of time_vector.

        """
        pass

    def on_data_received(self, callback, cancel: bool = False) -> None:
        if cancel:
            if callback not in self._on_data_received:
                raise ValueError(
                    "Callback not found in on_data_received listeners."
                )
            self._on_data_received.remove_listener(callback)

        else:
            if callback in self._on_data_received:
                raise ValueError(
                    "Callback already registered in on_data_received listeners."
                )
            self._on_data_received.add_listener(callback)

    def on_status_received(self, callback, cancel: bool = False) -> None:
        if cancel:
            if callback not in self._on_status_received:
                raise ValueError(
                    "Callback not found in on_status_received listeners."
                )
            self._on_status_received.remove_listener(callback)
        else:
            if callback in self._on_status_received:
                raise ValueError(
                    "Callback already registered in on_status_received listeners."
                )
            self._on_status_received.add_listener(callback)

    def start_live_plot(self) -> None:
        """
        Start a live plot of the force data received from the sensor. The plot is updated whenever new data are received.

        Returns:
            None
        """
        self._live_plot_axes = _prepare_figure()
        self.on_data_received(partial(_update_figure, ax=self._live_plot_axes))


def _prepare_figure():
    plt.ion()
    plt.figure(figsize=(10, 6))
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("B24 Force Sensor Data")
    plt.show()


def _update_figure(data, ax: plt.Axes):
    if not plt.fignum_exists(1):
        return

    time_vector, force_vector = data
    plt.clf()
    plt.plot(time_vector, force_vector)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("B24 Force Sensor Data")
    plt.draw()
    plt.pause(0.001)
