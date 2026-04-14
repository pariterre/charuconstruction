import asyncio
from enum import Enum
import logging
import struct
import time
from typing import TYPE_CHECKING

from bleak import BleakClient, BleakScanner
import numpy as np

from .force_sensor import ForceSensor

if TYPE_CHECKING:
    from bleak.backends.device import BLEDevice


_logger = logging.getLogger(__name__)


class B24SampleRateConfiguration(Enum):
    # Time between samples in milliseconds (ms)
    STOP = 0
    FASTEST = 80
    BATTERY_SAVER = 5000  # TODO Change this to 1000?
    SLOWEST = 10_000


class B24ResolutionConfiguration(Enum):
    # Number of bits of resolution (affects max data rate)
    LOWEST = 8
    BATTERY_SAVER = 8
    MAXIMUM = 16


class B24ForceSensor(ForceSensor):
    def __init__(self, device: BLEDevice):
        super().__init__()

        self._device = device
        self._client: BleakClient | None = None

        self._is_started = False
        self._starting_time: float | None = None
        self._time_vector = np.ndarray((0,))
        self._data = np.ndarray((0, 1))

    @property
    def name(self) -> str:
        return self._device.name

    @property
    def address(self) -> str:
        return self._device.address

    async def start_reading(
        self,
        pin_number: int,
        max_retries: int = 10,
    ) -> bool:
        """
        Parameters
        ----------
        pin_number: int
            A unique PIN number to send to the sensor for authentication.
            This should be a non-negative integer.
        max_retries: int
            The maximum number of connection attempts before giving up.
        """
        # Start the connection on a separate task to allow for retries without blocking the main thread
        asyncio.create_task(
            self._start_reading(pin_number=pin_number, max_retries=max_retries)
        )
        while self._client is None or not self._client.is_connected:
            await asyncio.sleep(0.1)
        return True

    async def _start_reading(self, pin_number: int, max_retries: int) -> bool:
        if self._client is not None and self._client.is_connected:
            _logger.warning(
                "Already connected to a B24 sensor. Stop reading before starting again."
            )
            return False

        _logger.info(f"Connecting to B24 sensor...")
        retry_count = 0
        while retry_count < max_retries:
            self._is_started = False
            try:
                async with BleakClient(
                    address_or_ble_device=self._device, timeout=30.0
                ) as self._client:
                    await self._send_pin_number(pin_number=pin_number)
                    _logger.info(f"Connected to B24 sensor")
                    self._is_started = True

                    # Configure the sensor for maximum resolution and fastest data rate
                    await self.configure_resolution(
                        B24ResolutionConfiguration.MAXIMUM
                    )
                    await self.configure_data_rate(
                        sample_rate=B24SampleRateConfiguration.FASTEST
                    )

                    # Subscribe to notifications for value and status updates
                    await self._client.start_notify(
                        _B24Helpers.B24Services.VALUE.value, self._on_value
                    )
                    await self._client.start_notify(
                        _B24Helpers.B24Services.STATUS.value, self._on_status
                    )

                    # Keep the connection alive until stop_reading is called
                    while self._is_started:
                        await asyncio.sleep(1.0)
                    self._client = None
                    return True

            except Exception:
                _logger.warning(
                    f"Could not connect to B24 sensor, retrying in 1 second..."
                )
                await asyncio.sleep(1)
                retry_count += 1

        _logger.error(
            f"Failed to connect to B24 sensor after {max_retries} attempts. Please ensure the sensor is nearby and advertising via Bluetooth."
        )
        return False

    async def stop_reading(self) -> bool:
        if self._client is None or not self._client.is_connected:
            _logger.warning("Not connected to any B24 sensor.")
            return False

        _logger.info("Stopping reading from B24 sensor and disconnecting...")
        # Stop notifications and reset sensor configuration to battery saver mode before disconnecting
        await self.configure_resolution(
            B24ResolutionConfiguration.BATTERY_SAVER
        )
        await self.configure_data_rate(B24SampleRateConfiguration.BATTERY_SAVER)

        # Stop notifications before disconnecting
        await self._client.stop_notify(_B24Helpers.B24Services.VALUE.value)
        await self._client.stop_notify(_B24Helpers.B24Services.STATUS.value)

        self._is_started = False
        _logger.info("Disconnected from B24 sensor.")
        return True

    def clear_data(self):
        self._starting_time = None
        self._time_vector = np.ndarray((0,))
        self._data = np.ndarray((0, 1))
        return True

    @property
    def time_vector(self) -> np.ndarray:
        return self._time_vector

    @property
    def force_vector(self) -> np.ndarray:
        return self._data

    @classmethod
    async def from_bluetooth(
        cls, timeout_s: float = 1.0, max_retries: int = 100
    ) -> B24ForceSensor:
        """
        Discover nearby B24 sensors via Bluetooth and return an instance of B24ForceSensor.
        """

        _logger.info("Looking for nearby B24 sensors via Bluetooth...")
        try_count = 0
        while try_count < max_retries:
            if try_count % 10 == 0:
                _logger.info(
                    f"Scanning for B24 sensors (attempt {try_count + 1}/{max_retries})..."
                )
            devices = await BleakScanner.discover(timeout=timeout_s)
            b24 = [
                d for d in devices if (d.name or "").upper().startswith("B24")
            ]
            if b24:
                _logger.info(f"Found B24 sensor")
                return cls(device=b24[0])
            try_count += 1

        _logger.error(
            f"No B24 sensor found nearby after {max_retries} attempts. "
            f"Please ensure the sensor is nearby and advertising via Bluetooth."
        )
        raise RuntimeError("No B24 sensor found nearby after maximum retries.")

    async def configure_resolution(
        self, resolution: B24ResolutionConfiguration
    ):
        """
        Configure the resolution (number of bits) for the B24 sensor.
        Note that higher resolution may limit the maximum data rate.
        """
        await self._client.write_gatt_char(
            _B24Helpers.B24Services.RESOLUTION.value,
            _B24Helpers._pack_u8(resolution.value),
            response=True,
        )

    async def configure_data_rate(
        self, sample_rate: B24SampleRateConfiguration
    ):
        """
        Configure the data rate (time between samples in ms) for the B24 sensor.
        """

        await self._client.write_gatt_char(
            _B24Helpers.B24Services.DATA_RATE.value,
            _B24Helpers._pack_u32_be(sample_rate.value),
            response=True,
        )

        await self._client.write_gatt_char(
            _B24Helpers.B24Services.RESOLUTION.value,
            _B24Helpers._pack_u8(16),
            response=True,
        )

    async def _send_pin_number(self, pin_number: int) -> bool:
        """
        Send a PIN number so the sensor can connect.

        Parameters
        ----------
        pin_number: int
            A unique PIN number to send to the sensor for authentication.
            This should be a non-negative integer.
        """

        response = await self._client.write_gatt_char(
            _B24Helpers.B24Services.PIN.value,
            _B24Helpers._pack_u32_be(pin_number),
            response=True,
        )
        return response

    def _on_value(self, _, data: bytearray):
        now = time.perf_counter()
        if self._starting_time is None:
            self._starting_time = now

        self._time_vector = np.concatenate(
            (self._time_vector, [now - self._starting_time])
        )

        data = (
            _B24Helpers._unpack_f32_be(bytes(data))
            if len(data) == 4
            else float("nan")
        )
        self._data = np.concatenate((self._data, [[data]]), axis=0)

        self._on_data_received.notify_listeners(
            data=(self._time_vector[-1], self._data[-1, :])
        )

    def _on_status(self, _, data: bytearray):
        status = data[0] if data else 0
        fast_mode_flag = (status >> 4) & 0x01
        batt_low = (status >> 5) & 0x01
        over_range = (status >> 3) & 0x01
        print(
            f"STATUS: 0x{status:02X}  fast={fast_mode_flag} batt_low={batt_low} over={over_range}"
        )


# --- UUIDs (B24 Telemetry Technical Manual) ---


class _B24Helpers:
    class B24Services(Enum):
        # Configuration characteristics (write-only)
        TELEMETRY = "a970fd30-a0e8-11e6-bdf4-0800200c9a66"
        DATA_RATE = "a970fd31-a0e8-11e6-bdf4-0800200c9a66"
        RESOLUTION = "a970fd32-a0e8-11e6-bdf4-0800200c9a66"
        PIN = "a970fd39-a0e8-11e6-bdf4-0800200c9a66"

        # Notification characteristics (notify-only)
        STATUS = "a9712441-a0e8-11e6-bdf4-0800200c9a66"
        VALUE = "a9712442-a0e8-11e6-bdf4-0800200c9a66"
        UNITS = "a9712443-a0e8-11e6-bdf4-0800200c9a66"

    # --- Helpers: B24 uses MSB-first in examples => big-endian packing ---
    @staticmethod
    def _pack_u32_be(x: int) -> bytes:
        return struct.pack(">I", int(x))

    @staticmethod
    def _pack_u8(x: int) -> bytes:
        return struct.pack(">B", int(x))

    @staticmethod
    def _unpack_f32_be(b: bytes) -> float:
        return struct.unpack(">f", b)[0]
