import os
import json

from charuconstruction import Charuco, ImageReader, Frame


def main():
    charuco_boards: list[Charuco] = []
    for param in json.loads(os.environ["CHARUCO_PARAMETERS"]):
        charuco_boards.append(Charuco(**param))

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
