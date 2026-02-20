import os
from pathlib import Path

from charuconstruction import (
    Charuco,
    CharucoWithDynamicStates,
    CharucoMockReader,
    ImageReader,
    VideoReader,
    LiveVideoReader,
    Frame,
    Camera,
    CameraModel,
    Transformation,
    TranslationVector,
    RotationMatrix,
    Vector3,
)
import numpy as np

# TODO Add force sensor
# TODO Add capability to detect joint angles (calibrate first?)
# TODO Add capability to see if joint angles are on target
# TODO Add capability to graph force against joint angles (phase diagram?)


def simulate_reader(
    camera: Camera,
    charuco_boards: list[Charuco],
    use_gui: bool = False,
) -> CharucoMockReader:

    # Create some motion for each board
    transformations: dict[Charuco, list[Transformation]] = {
        charuco: [] for charuco in charuco_boards
    }
    if use_gui:
        transformations = None
    else:
        frame_count = 50
        for i in range(frame_count):
            for j in range(len(charuco_boards)):
                base_translation = -0.6 if j == 0 else 0.6
                translation = i * 0.01 * (1 if j == 0 else -1) * 0
                rotation = i * 1 * (1 if j == 0 else -1)
                transformations[charuco_boards[j]].append(
                    Transformation(
                        translation=TranslationVector(
                            base_translation + translation, 0, 2.5
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
        boards=charuco_boards, transformations=transformations, camera=camera
    )


def main():
    # Load the material used for the experiment
    charuco_boards: list[Charuco] = []
    for folder_name in os.environ["CHARUCOS"].split(","):
        charuco_boards.append(
            CharucoWithDynamicStates.from_charuco(
                Charuco.load(Path(folder_name))
            )
        )

    # Simulate a reader (since we don't have real media for now)
    data_type = os.environ.get("DATA_TYPE", "simulated").lower()
    camera_model = os.environ.get("CAMERA_MODEL")
    if camera_model is None:
        raise ValueError("CAMERA_MODEL environment variable must be set.")
    camera_model = CameraModel(camera_model.lower())

    if data_type == "simulated":
        camera = camera_model.to_camera(
            use_video_parameters=True, is_vertical=False
        )
        reader = simulate_reader(
            camera,
            charuco_boards,
            use_gui=os.environ.get("WITH_GUI", "true").lower() == "true",
        )
        automatic_frame = reader.with_gui
    elif data_type == "video":
        camera = camera_model.to_camera(
            use_video_parameters=True, is_vertical=False
        )
        video_path = os.environ["VIDEO_PATH"] = os.environ.get("VIDEO_PATH")
        if video_path is None:
            raise ValueError(
                "VIDEO_PATH environment variable must be set for video data type."
            )
        reader = VideoReader(video_path=Path(video_path))
        automatic_frame = True
    elif data_type == "photos":
        camera = camera_model.to_camera(
            use_video_parameters=False, is_vertical=True
        )
        photos_path = os.environ["PHOTOS_PATH"] = os.environ.get("PHOTOS_PATH")
        if photos_path is None:
            raise ValueError(
                "PHOTOS_PATH environment variable must be set for photos data type."
            )
        reader = ImageReader(
            image_path=[Path(p) for p in photos_path.split(",")]
        )
        automatic_frame = False
    elif data_type == "live":
        camera = camera_model.to_camera(
            use_video_parameters=True, is_vertical=False
        )
        live_ip = os.environ["LIVE_IP"] = os.environ.get("LIVE_IP")
        if live_ip is None:
            raise ValueError(
                "LIVE_IP environment variable must be set for live data type."
            )
        live_ip_parts = live_ip.split(":")
        if len(live_ip_parts) != 2:
            raise ValueError("LIVE_IP must be in the format 'IP_ADDRESS:PORT'.")
        camera_ip, camera_port = live_ip_parts[0], int(live_ip_parts[1])
        reader = LiveVideoReader(camera_ip=camera_ip, camera_port=camera_port)
        automatic_frame = True
    else:
        raise ValueError(
            f"Invalid DATA_TYPE: {data_type}. Must be 'simulated', 'video', 'photos', or 'live'."
        )

    should_record_video = (
        os.environ.get("RECORD_VIDEO", "false").lower() == "true"
    )
    if should_record_video:
        video_save_path = Path(os.environ.get("RECORD_PATH"))
        if video_save_path is None:
            raise ValueError(
                "RECORD_PATH environment variable must be set when RECORD_VIDEO is true."
            )
        video_save_path.parent.mkdir(parents=True, exist_ok=True)
        video_frame = None

    is_visible = False

    should_compute_error = isinstance(reader, CharucoMockReader)
    if should_compute_error:
        errors: dict[Charuco, list[np.ndarray]] = {}

    for frame in reader:
        if should_record_video and video_frame is None:
            video_frame = frame
            video_frame.start_recording(video_save_path)

        frame_results: dict[
            Charuco, tuple[TranslationVector | None, RotationMatrix | None]
        ] = {}
        for charuco in charuco_boards:
            frame_results[charuco] = charuco.detect(frame=frame, camera=camera)

        # Compute the reconstruction error in degrees
        if should_compute_error:
            for charuco in charuco_boards:
                if charuco not in errors:
                    errors[charuco] = []

                _, rotation = frame_results[charuco]
                if rotation is None:
                    errors[charuco].append(np.ndarray((3, 1)) * np.nan)
                    continue

                sequence = RotationMatrix.Sequence.ZYX
                true_values = (
                    reader.transformations(charuco)
                    .rotation.to_euler(sequence=sequence, degrees=True)
                    .as_array()
                )
                reconstructed_values = rotation.to_euler(
                    sequence=sequence, degrees=True
                ).as_array()
                errors[charuco].append(true_values - reconstructed_values)

        # Show the estimated pose for each board on the current frame
        frame_to_draw = Frame(frame.get().copy())
        for charuco in charuco_boards:
            translation, rotation = frame_results[charuco]
            if translation is None or rotation is None:
                continue

            charuco.draw_aruco_markers(frame_to_draw)
            charuco.draw_estimated_pose_axes(
                frame=frame_to_draw,
                camera=camera,
                translation=translation,
                rotation=rotation,
                axes_length=0.1,
            )

        if should_record_video:
            video_frame.frame_from(frame_to_draw)
            video_frame.add_frame_to_recording()

        is_visible = frame_to_draw.show(
            wait_time=(1 if automatic_frame else None)
        )
        if not is_visible:
            break

    if (
        isinstance(reader, CharucoMockReader)
        and not automatic_frame
        and is_visible
    ):
        frame_to_draw.show(wait_time=None)

    if should_record_video:
        video_frame.stop_recording()
        print(f"Video saved to {video_save_path}")
    reader.destroy()

    # Print the mean error for each Charuco board
    if should_compute_error:
        for charuco, error_list in errors.items():
            error_array = np.array(error_list).squeeze()
            if not np.isfinite(error_array).any():
                continue
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
