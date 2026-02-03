import os
from pathlib import Path

from charuconstruction import (
    Charuco,
    CharucoMockReader,
    Frame,
    Camera,
    Transformation,
    TranslationVector,
    RotationMatrix,
    Vector3,
)
import numpy as np


def main():
    camera = Camera.default_12mpx_camera()
    frame_count = 100

    # Load the boards
    charuco_boards: list[Charuco] = []
    for i, folder_name in enumerate(os.environ["CHARUCOS"].split(",")):
        charuco_boards.append(Charuco.load(Path(folder_name)))

    # Create some motion for each board
    transformations_list: list[list[Transformation]] = []
    for i in range(frame_count):
        transformations: list[Transformation] = []
        for j in range(len(charuco_boards)):
            translation = 75 * j
            rotation = i * 0.1 * (1 if j == 0 else -1)
            transformations.append(
                Transformation(
                    translation=TranslationVector(translation, 0, 0.5),
                    rotation=RotationMatrix.from_euler(
                        Vector3(0.0, rotation, 0.0), degrees=True
                    ),
                )
            )
        transformations_list.append(transformations)

    # Detect markers for each Charuco board at each frame
    reader = CharucoMockReader(
        boards=charuco_boards,
        transformations=transformations_list,
        camera=camera,
    )
    should_continue = True
    initial_guess = {}
    errors = {}
    for index, frame in enumerate(reader):
        frame_to_draw = Frame(frame.get())
        for charuco in charuco_boards:
            frame_to_draw, results = charuco.detect(
                frame_to_draw,
                initial_guess=initial_guess.get(charuco),
                camera=camera,
            )

            # Prepare the initial guess for the next frame
            initial_guess[charuco] = results

            # Compute the reconstruction error in degrees
            true_values = (
                transformations_list[index][0]
                .rotation.to_euler(
                    sequence=RotationMatrix.Sequence.ZYX, degrees=True
                )
                .as_array()
            )
            reconstructed_values = (
                RotationMatrix.from_euler(Vector3.from_array(results[1]))
                .to_euler(sequence=RotationMatrix.Sequence.ZYX, degrees=True)
                .as_array()
                if results[1] is not None
                else np.ndarray((3, 1)) * np.nan
            )
            if charuco not in errors:
                errors[charuco] = []
            errors[charuco].append(true_values - reconstructed_values)

        should_continue = frame_to_draw.show(wait_time=None)
        if not should_continue:
            break

    if should_continue:
        frame_to_draw.show(wait_time=None)

    reader.destroy()


if __name__ == "__main__":
    main()
