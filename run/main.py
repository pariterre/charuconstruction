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


def simulate_reader(
    camera: Camera,
    charuco_boards: list[Charuco],
) -> CharucoMockReader:
    frame_count = 100

    # Create some motion for each board
    transformations: dict[Charuco, list[Transformation]] = {
        charuco: [] for charuco in charuco_boards
    }
    for i in range(frame_count):
        for j in range(len(charuco_boards)):
            base_translation = -0.4 if j == 0 else -0.7
            translation = i * 0.02 * (1 if j == 0 else -1)
            rotation = i * 1 * (1 if j == 0 else -1)
            transformations[charuco_boards[j]].append(
                Transformation(
                    translation=TranslationVector(
                        base_translation + translation, -0.2 - translation, -1.5
                    ),
                    rotation=RotationMatrix.from_euler(
                        Vector3(rotation, rotation, rotation),
                        degrees=True,
                        sequence=RotationMatrix.Sequence.ZYX,
                    ),
                )
            )

    # Detect markers for each Charuco board at each frame
    return CharucoMockReader(
        boards=charuco_boards,
        transformations=transformations,
        camera=camera,
    )


def main():
    # Load the material used for the experiment
    camera = Camera.default_iphone_camera()
    charuco_boards: list[Charuco] = []
    for folder_name in os.environ["CHARUCOS"].split(","):
        charuco_boards.append(Charuco.load(Path(folder_name)))

    # Simulate a reader (since we don't have real media for now)
    reader = simulate_reader(camera, charuco_boards)

    should_continue = True
    initial_guess = {}
    errors: dict[Charuco, list[np.ndarray]] = {}
    for frame_index, frame in enumerate(reader):
        frame_to_draw = Frame(frame.get())
        for charuco_index, charuco in enumerate(charuco_boards):
            frame_to_draw, results = charuco.detect(
                frame_to_draw,
                initial_guess=initial_guess.get(charuco),
                camera=camera,
            )

            # Prepare the initial guess for the next frame
            initial_guess[charuco] = results

            # Compute the reconstruction error in degrees
            real_transformation = reader.transformations[charuco][frame_index]
            true_values = real_transformation.rotation.to_euler(
                sequence=RotationMatrix.Sequence.ZYX, degrees=True
            ).as_array()
            reconstructed_values = (
                RotationMatrix(results[1])
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

    # Print the mean error for each Charuco board
    for charuco, error_list in errors.items():
        error_array = np.array(error_list).squeeze()
        mean_error = np.nanmean(np.abs(error_array), axis=0)
        std_error = np.nanstd(error_array, axis=0)
        print(
            f"Mean reconstruction error for Charuco board "
            f"{charuco.horizontal_squares_count}x{charuco.vertical_squares_count} "
            f"with seed {charuco.random_seed}: "
            f"Roll: {mean_error[0]:.2f}° ± {std_error[0]:.2f}°, Pitch: {mean_error[1]:.2f}° ± {std_error[1]:.2f}°, Yaw: {mean_error[2]:.2f}° ± {std_error[2]:.2f}°"
        )


if __name__ == "__main__":
    main()
