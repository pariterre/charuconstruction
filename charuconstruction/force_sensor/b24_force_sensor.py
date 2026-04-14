import asyncio
from enum import Enum
import struct
import time
from typing import TYPE_CHECKING

from bleak import BleakClient, BleakScanner

from .force_sensor import ForceSensor

if TYPE_CHECKING:
    from bleak.backends.device import BLEDevice


class B24SampleRateConfiguration(Enum):
    # Time between samples in milliseconds (ms)
    STOP = 0
    FASTEST = 80
    BATTERY_SAVER = 5000
    SLOWEST = 10000


class B24ResolutionConfiguration(Enum):
    # Number of bits of resolution (affects max data rate)
    LOWEST = 8
    BATTERY_SAVER = 8
    MAXIMUM = 16


global _device_index
_device_index = -1


class B24ForceSensor(ForceSensor):
    def __init__(self, device: BLEDevice):
        # Set a unique pin index for this sensor instance (used for configuration)
        global _device_index
        self._device_index = _device_index
        _device_index += 1

        super().__init__()

        self._last_t = None
        self._device = device
        # Client will be created and connected in start_reading, and closed when done
        self._client: BleakClient | None = None

    async def start_reading(self, duration_s: float) -> bool:
        async with BleakClient(self.address) as coucou:
            self._client = coucou
            self._initialize_connection()

            # Configure the sensor for maximum resolution and fastest data rate
            await self.configure_resolution(B24ResolutionConfiguration.MAXIMUM)
            await self.configure_data_rate(
                sample_rate=B24SampleRateConfiguration.FASTEST
            )

            # # Read the units code (e.g., N, kgf) from the sensor and print it
            # units_raw = await self._client.read_gatt_char(UUID_UNITS)
            # units_code = units_raw[0] if units_raw else None
            # print(f"Units code: {units_code}")

            # Subscribe to notifications for value and status updates
            await self._client.start_notify(UUID_VALUE, self._on_value)
            await self._client.start_notify(UUID_STATUS, self._on_status)

            print(f"Streaming {duration_s}s…")
            await asyncio.sleep(duration_s)

            # Stop notifications and reset sensor configuration to battery saver mode before disconnecting
            await self.configure_resolution(
                B24ResolutionConfiguration.BATTERY_SAVER
            )
            await self.configure_data_rate(
                B24SampleRateConfiguration.BATTERY_SAVER
            )

            # Stop notifications
            await self._client.stop_notify(UUID_VALUE)
            await self._client.stop_notify(UUID_STATUS)

    @property
    def name(self) -> str:
        return self._device.name

    @property
    def address(self) -> str:
        return self._device.address

    def stop_reading(self) -> bool:
        pass

    @classmethod
    async def from_bluetooth(cls, timeout_s: float = 10.0) -> B24ForceSensor:
        """
        Discover nearby B24 sensors via Bluetooth and return an instance of B24ForceSensor.
        """

        devices = await BleakScanner.discover(timeout=timeout_s)
        b24 = [d for d in devices if (d.name or "").upper().startswith("B24")]
        if not b24:
            raise RuntimeError("No B24 sensor found nearby.")
        return cls(device=b24[0])

    async def configure_resolution(
        self, resolution: B24ResolutionConfiguration
    ):
        """
        Configure the resolution (number of bits) for the B24 sensor.
        Note that higher resolution may limit the maximum data rate.
        """
        await self._client.write_gatt_char(
            _B24Helpers.B24ConfigurationServices.RESOLUTION.value,
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
            UUID_DATA_RATE,
            _B24Helpers._pack_u32_be(sample_rate.value),
            response=True,
        )

        await self._client.write_gatt_char(
            UUID_RESOLUTION, _B24Helpers._pack_u8(16), response=True
        )

    async def _initialize_connection(self):
        await self._device.write_gatt_char(
            _B24Helpers.B24ConfigurationServices.PIN.value,
            _B24Helpers._pack_u32_be(self._device_index),
            response=True,
        )

    def _on_value(self, _, data: bytearray):
        now = time.perf_counter()
        value = (
            _B24Helpers._unpack_f32_be(bytes(data))
            if len(data) == 4
            else float("nan")
        )
        hz = (1.0 / (now - self._last_t)) if self._last_t is not None else None
        self._last_t = now
        if hz is None:
            print(f"{now:.6f}  value={value:.6g}")
        else:
            print(f"{now:.6f}  value={value:.6g}  recv≈{hz:.2f} Hz")

    def _on_status(self, _, data: bytearray):
        status = data[0] if data else 0
        fast_mode_flag = (status >> 4) & 0x01
        batt_low = (status >> 5) & 0x01
        over_range = (status >> 3) & 0x01
        print(
            f"STATUS: 0x{status:02X}  fast={fast_mode_flag} batt_low={batt_low} over={over_range}"
        )


# --- UUIDs (B24 Telemetry Technical Manual) ---
UUID_CFG_PIN = "a970fd39-a0e8-11e6-bdf4-0800200c9a66"  # Uint32
UUID_DATA_RATE = "a970fd31-a0e8-11e6-bdf4-0800200c9a66"  # Uint32 (ms)
UUID_RESOLUTION = "a970fd32-a0e8-11e6-bdf4-0800200c9a66"  # Uint8

UUID_STATUS = "a9712441-a0e8-11e6-bdf4-0800200c9a66"  # Uint8
UUID_VALUE = "a9712442-a0e8-11e6-bdf4-0800200c9a66"  # Float (IEEE754)
UUID_UNITS = "a9712443-a0e8-11e6-bdf4-0800200c9a66"  # Uint8


class _B24Helpers:
    class B24ConfigurationServices(Enum):
        PIN = "a970fd2f-a0e8-11e6-bdf4-0800200c9a66"
        TELEMETRY = "a970fd30-a0e8-11e6-bdf4-0800200c9a66"
        DATA_RATE = "a970fd31-a0e8-11e6-bdf4-0800200c9a66"
        RESOLUTION = "a970fd32-a0e8-11e6-bdf4-0800200c9a66"

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
