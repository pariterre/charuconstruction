import os
import json

from charuconstruction import Charuco


def main():
    params = json.loads(os.environ["CHARUCO_PARAMETERS"])

    for param in params:
        charuco = Charuco(**param)
        charuco.show(
            save_path=f"charuco_{charuco.vertical_squares_count}x{charuco.horizontal_squares_count}_{param['seed']}.png"
        )


if __name__ == "__main__":
    main()
