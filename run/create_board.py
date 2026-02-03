import os
from pathlib import Path
import json

from charuconstruction import Charuco


def main():
    params = json.loads(os.environ["CHARUCO_PARAMETERS"])

    for param in params:
        charuco = Charuco(**param)
        charuco.save(
            save_folder=Path(
                f"charuco_{charuco.vertical_squares_count}x{charuco.horizontal_squares_count}_{charuco.random_seed}",
            ),
            override=True,
        )
        charuco.show()


if __name__ == "__main__":
    main()
