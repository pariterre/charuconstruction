import asyncio
from enum import Enum
import struct
import time
from bleak import BleakClient, BleakScanner

# --- UUIDs (B24 Telemetry Technical Manual) ---
UUID_CFG_PIN = "a970fd39-a0e8-11e6-bdf4-0800200c9a66"  # Uint32
UUID_DATA_RATE = "a970fd31-a0e8-11e6-bdf4-0800200c9a66"  # Uint32 (ms)
UUID_RESOLUTION = "a970fd32-a0e8-11e6-bdf4-0800200c9a66"  # Uint8

UUID_STATUS = "a9712441-a0e8-11e6-bdf4-0800200c9a66"  # Uint8
UUID_VALUE = "a9712442-a0e8-11e6-bdf4-0800200c9a66"  # Float (IEEE754)
UUID_UNITS = "a9712443-a0e8-11e6-bdf4-0800200c9a66"  # Uint8


class B24ConfigurationServices(Enum):
    TELEMETRY = "a970fd30-a0e8-11e6-bdf4-0800200c9a66"
    DATA_RATE = "a970fd31-a0e8-11e6-bdf4-0800200c9a66"
    RESOLUTION = "a970fd32-a0e8-11e6-bdf4-0800200c9a66"


class SampleRateConfiguration(Enum):
    # Time between samples in milliseconds (ms)
    STOP = 0
    FASTEST = 80
    BATTERY_SAVER = 5000
    SLOWEST = 10000


class ResolutionConfiguration(Enum):
    # Number of bits of resolution (affects max data rate)
    LOWEST = 8
    BATTERY_SAVER = 8
    MAXIMUM = 16


# --- Helpers: B24 uses MSB-first in examples => big-endian packing ---
def pack_u32_be(x: int) -> bytes:
    return struct.pack(">I", int(x))


def pack_u8(x: int) -> bytes:
    return struct.pack(">B", int(x))


def unpack_f32_be(b: bytes) -> float:
    return struct.unpack(">f", b)[0]


async def find_b24(timeout_s: float = 5.0):
    devices = await BleakScanner.discover(timeout=timeout_s)
    # Filtre simple sur le nom "B24" (le nom par défaut est "B24" dans le manuel)
    b24 = [d for d in devices if (d.name or "").upper().startswith("B24")]
    return b24


async def configure_resolution(
    client: BleakClient, resolution: ResolutionConfiguration
):
    """
    Configure the resolution (number of bits) for the B24 sensor.
    Note that higher resolution may limit the maximum data rate.
    """
    await client.write_gatt_char(
        UUID_RESOLUTION, pack_u8(resolution.value), response=True
    )


async def configure_data_rate(
    client: BleakClient,
    sample_rate: SampleRateConfiguration,
    config_pin: int = 0,
):
    """
    Configure the data rate (time between samples in ms) for the B24 sensor.
    """
    # IMPORTANT: écrire le Configuration PIN en premier, rapidement après connexion
    await client.write_gatt_char(
        UUID_CFG_PIN, pack_u32_be(config_pin), response=True
    )

    await client.write_gatt_char(
        UUID_DATA_RATE, pack_u32_be(sample_rate.value), response=True
    )

    await client.write_gatt_char(UUID_RESOLUTION, pack_u8(16), response=True)


async def stream_notifications(
    address: str, config_pin: int = 0, duration_s: float = 10.0
):
    """
    Stream des mesures via notifications (le plus fiable pour ne pas perdre de points).
    Affiche aussi une estimation de la fréquence réellement reçue.
    """
    async with BleakClient(address) as client:
        # Configure d'abord (PIN + Data Rate=80ms + Resolution=16)
        await configure_resolution(client, ResolutionConfiguration.MAXIMUM)
        await configure_data_rate(
            client,
            sample_rate=SampleRateConfiguration.FASTEST,
            config_pin=config_pin,
        )

        # Lire l'unité (optionnel)
        units_raw = await client.read_gatt_char(UUID_UNITS)
        units_code = units_raw[0] if units_raw else None
        print(f"Units code: {units_code}")

        last_t = None

        def on_value(_, data: bytearray):
            nonlocal last_t
            now = time.perf_counter()
            value = (
                unpack_f32_be(bytes(data)) if len(data) == 4 else float("nan")
            )
            hz = (1.0 / (now - last_t)) if last_t is not None else None
            last_t = now
            if hz is None:
                print(f"{now:.6f}  value={value:.6g}")
            else:
                print(f"{now:.6f}  value={value:.6g}  recv≈{hz:.2f} Hz")

        def on_status(_, data: bytearray):
            status = data[0] if data else 0
            fast_mode_flag = (status >> 4) & 0x01
            batt_low = (status >> 5) & 0x01
            over_range = (status >> 3) & 0x01
            print(
                f"STATUS: 0x{status:02X}  fast={fast_mode_flag} batt_low={batt_low} over={over_range}"
            )

        await client.start_notify(UUID_VALUE, on_value)
        await client.start_notify(UUID_STATUS, on_status)

        print(f"Streaming {duration_s}s…")
        await asyncio.sleep(duration_s)

        await configure_resolution(
            client, ResolutionConfiguration.BATTERY_SAVER
        )
        await configure_data_rate(
            client,
            sample_rate=SampleRateConfiguration.BATTERY_SAVER,
            config_pin=config_pin,
        )

        await client.stop_notify(UUID_VALUE)
        await client.stop_notify(UUID_STATUS)


async def main():
    b24 = await find_b24(10.0)
    if not b24:
        print("Aucun B24 trouvé. Réessaie en rapprochant le module.")
        return

    dev = b24[0]
    print(f"Trouvé: name={dev.name} address={dev.address}")

    # PIN par défaut = 0 d’après le manuel
    await stream_notifications(dev.address, config_pin=0, duration_s=3.0)


if __name__ == "__main__":
    asyncio.run(main())
