import asyncio
import logging

from charuconstruction import B24ForceSensor

_logger = logging.getLogger(__name__)


async def connect_b24_sensor(max_retries: int = 10) -> B24ForceSensor:
    force_sensor = None
    retry_count = 0
    while force_sensor is None and retry_count < max_retries:
        _logger.info(
            f"Scanning for B24 sensors (attempt {retry_count + 1}/{max_retries})…"
        )
        try:
            force_sensor = await B24ForceSensor.from_bluetooth()
        except RuntimeError as e:
            force_sensor = None
            retry_count += 1
    if force_sensor is None:
        _logger.error(
            f"No B24 sensor found after {max_retries} attempts. "
            f"Please ensure the sensor is nearby and advertising via Bluetooth."
        )
        raise RuntimeError("Failed to connect to B24 sensor.")

    _logger.info("B24 sensor connected successfully.")
    return force_sensor


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    force_sensor = await connect_b24_sensor()
    await force_sensor.start_reading(duration_s=10.0)


if __name__ == "__main__":
    asyncio.run(main())
