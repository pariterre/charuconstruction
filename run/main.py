import os
from pathlib import Path

from charuconstruction import Charuco, CharucoMockReader, Frame, Camera
import numpy as np


def main():
    camera = Camera.default(focal_length=500.0, sensor_width=1000.0, sensor_height=800.0)

    charuco_boards: list[Charuco] = []
    angles_list: list[list[float]] = []
    for i, folder_name in enumerate(os.environ["CHARUCOS"].split(",")):
        charuco_boards.append(Charuco.load(Path(folder_name), camera=camera))
        if i == 0:
            angles_list.append(list(np.linspace(0, 30, 101)))
        else:
            angles_list.append(list(np.linspace(0, 0, 101)))

    # Detect markers for each Charuco board at each frame
    reader = CharucoMockReader(boards=charuco_boards, angles=angles_list, camera=camera)
    should_continue = True
    initial_guess = {}
    for frame in reader:
        frame_to_draw = Frame(frame.get())
        for charuco in charuco_boards:
            frame_to_draw, results = charuco.detect(frame_to_draw, initial_guess=initial_guess.get(charuco))
            initial_guess[charuco] = results

        should_continue = frame_to_draw.show(wait_time=None)
        if not should_continue:
            break

    if should_continue:
        frame_to_draw.show(wait_time=None)
    reader.destroy()


if __name__ == "__main__":
    main()
