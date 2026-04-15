import logging
import time

from charuconstruction import B24ForceSensor

_logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    force_sensor = B24ForceSensor.from_bluetooth()
    force_sensor.start_live_plot()
    force_sensor.on_data_received(
        lambda data: _logger.info(
            f"Received data: time: {data[0]}, force: {data[1]}"
        )
    )
    force_sensor.start_reading(pin_number=0)

    # Keep the program running for 10 seconds to receive data from the sensor
    time.sleep(10)

    # Stop reading from the sensor and exit
    force_sensor.stop_reading()

    # Wait for the live plot to be closed before exiting the program
    _logger.info("Waiting for the live plot to be closed...")
    force_sensor.wait_for_plot_close()


if __name__ == "__main__":
    main()
