import os
from pathlib import Path

from charuconstruction import Charuco, CharucoMockReader, Frame


def main():
    charuco_boards: list[Charuco] = []
    for folder_name in os.environ["CHARUCOS"].split(","):
        charuco_boards.append(Charuco.load(Path(folder_name)))

    # Detect markers for each Charuco board at each frame
    reader = CharucoMockReader(board1=charuco_boards[0], board2=charuco_boards[1], angles=range(0, 15, 1))
    for frame in reader:
        frame_to_draw = Frame(frame.get())
        for charuco in charuco_boards:
            frame_to_draw = charuco.detect(frame_to_draw)

        should_continue = frame_to_draw.show()
        if not should_continue:
            break

    reader.destroy()


if __name__ == "__main__":
    main()
