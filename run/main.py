import os
from pathlib import Path

from charuconstruction import Charuco, ImageReader, Frame


def main():
    charuco_boards: list[Charuco] = []
    for folder_name in os.environ["CHARUCOS"].split(","):
        charuco_boards.append(Charuco.load(Path(folder_name)))

    # Detect markers for each Charuco board at each frame
    reader = ImageReader(image_path="./charuco_frame.png")
    for frame in reader:
        frame_to_draw = Frame(frame.get())
        for charuco in charuco_boards:
            frame_to_draw = charuco.detect(frame_to_draw)
        frame_to_draw.show()

    reader.destroy()


if __name__ == "__main__":
    main()
