from abc import ABC, abstractmethod

from .listener import Listener


class ForceSensor(ABC):
    def __init__(self):
        super().__init__()

        self.on_data_received = Listener()
        self.on_status_received = Listener()

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
