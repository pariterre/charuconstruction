import asyncio
import logging
from matplotlib import pyplot as plt

from charuconstruction import B24ForceSensor

_logger = logging.getLogger(__name__)


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    force_sensor = await B24ForceSensor.from_bluetooth()
    force_sensor.start_live_plot()
    force_sensor.on_data_received(
        lambda data: _logger.info(
            f"Received data: time: {data[0]}, force: {data[1]}"
        )
    )
    await force_sensor.start_reading(pin_number=0)

    # Keep the program running for 10 seconds to receive data from the sensor
    await asyncio.sleep(10)

    # Stop reading from the sensor and exit
    await force_sensor.stop_reading()

    # Wait until the figure is closed before exiting the program
    if plt.fignum_exists(1):
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    asyncio.run(main())
