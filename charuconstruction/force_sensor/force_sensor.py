from abc import ABC, abstractmethod
from multiprocessing import Event, Process, Queue
import time

import matplotlib.pyplot as plt
import numpy as np

from .listener import Listener


class ForceSensor(ABC):
    def __init__(self):
        super().__init__()

        self._on_data_received = Listener()
        self._on_status_received = Listener()

        # Queue for live plotting
        self._figure_queue = Queue()
        self._figure_closed_event = Event()

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
        # Start a live plot in another process to avoid blocking the main thread
        process = Process(
            target=_prepare_figure,
            args=(self._figure_queue, self._figure_closed_event),
        )
        process.start()

        self.on_data_received(self._update_figure)

    def _update_figure(self, data):
        self._figure_queue.put(data)

    def wait_for_plot_close(self):
        """
        Wait until the live plot is closed before returning. This should be called after stop_reading to ensure that the program does not exit before the user has closed the plot.

        Returns:
            None
        """

        self._figure_closed_event.wait()


def _prepare_figure(queue, closed_event, frame_per_second: float = 20) -> None:
    def on_close(_):
        closed_event.set()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.mpl_connect("close_event", on_close)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_title("B24 Force Sensor Data")

    x_data = []
    y_data = []
    (line,) = ax.plot([], [])
    plt.show(block=False)

    last_draw_time = time.time()
    while not closed_event.is_set():
        if not queue.empty():
            data = queue.get()

            x_data.append(data[0])
            y_data.append(data[1])

        now = time.time()
        if now - last_draw_time >= 1.0 / frame_per_second:
            line.set_xdata(x_data)
            line.set_ydata(y_data)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            last_draw_time = now
