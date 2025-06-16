from pathlib import Path

import biorbd
import ezc3d
from orthotics_reconstruction import leg_model
from matplotlib import pyplot as plt
import numpy as np


def to_legend(dof_names: list[str]) -> list[str]:
    name_maps = {
        "RotX": "Flexion (-)/Extension (+)",
        "RotY": "Abduction (-)/Adduction (+)",
        "RotZ": "Internal (+)/External (-) Rotation",
    }
    out = []
    for name in dof_names:
        for key, value in name_maps.items():
            if key in name:
                out.append(value)
                break
        else:
            out.append(name)
    return out


def to_path_name(dof_names: list[str]) -> list[str]:
    name_maps = {
        "RotX": "Flexion_Extension",
        "RotY": "Abduction_Adduction",
        "RotZ": "Internal_External_Rotation",
    }
    out = []
    for name in dof_names:
        for key, value in name_maps.items():
            if key in name:
                out.append(value)
                break
        else:
            out.append(name)
    return out


def show_model(model: biorbd.Model, q: np.ndarray | None = None, markers: np.ndarray | None = None):
    import bioviz

    viz = bioviz.Viz(loaded_model=model)
    if q is not None:
        viz.load_movement(q)
    if markers is not None:
        viz.load_experimental_markers(markers)
    viz.exec()


def get_forces(c3d_file_path: str) -> np.ndarray:
    c3d = ezc3d.c3d(c3d_file_path, extract_forceplat_data=True)
    rate_ratio = c3d["header"]["analogs"]["frame_rate"] / c3d["header"]["points"]["frame_rate"]
    if int(rate_ratio) != rate_ratio:
        raise ValueError("The frame rates of the analogs and points do not match.")
    rate_ratio = int(rate_ratio)
    return c3d.data["platform"][0]["force"][:, ::rate_ratio]


def main(
    show_static: bool = False,
    show_reconstructions: bool = True,
    results_folder: str = "",
    force_threshold: float = 40.0,
):
    # Create the kinematic model
    model = leg_model()
    if show_static:
        show_model(model)

    # Make sure the results folder exists
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    # Reconstruct the kinematics
    data_folder = "../data/pilote/"
    material_conditions = ["elastic", "platform"]
    angle_conditions = ["000", "030"]
    for material_condition in material_conditions:
        for angle_condition in angle_conditions:
            for c3d_file_path in Path(data_folder).glob(f"{material_condition}_leg{angle_condition}_*.c3d"):
                print(f"Processing {c3d_file_path}")

                t, q, _, _ = biorbd.extended_kalman_filter(model, str(c3d_file_path))
                if show_reconstructions:
                    show_model(model, q=q, markers=str(c3d_file_path))

                # Get the forces
                forces = get_forces(str(c3d_file_path))
                initial_forces = forces[:, 0][:, None]
                t_indices = np.linalg.norm(forces - initial_forces, axis=0) > force_threshold
                t_indices = np.where(t_indices)[0]
                t_indices = slice(t_indices[0], t_indices[-1] + 1)

                # Plot the forces against dof for the last segment (SHANK)
                root_count = model.nbRoot()
                shank_dof_names = [dof.to_string() for dof in model.nameDof()[root_count:]]
                q_shank = q[root_count:, :]
                shank_legend = to_legend(dof_names=shank_dof_names)
                shank_path = to_path_name(dof_names=shank_dof_names)

                plt.figure(f"Joint Angles Over Time for {material_condition} {angle_condition}", figsize=(10, 6))
                plt.title(f"Joint Angles Over Time for {material_condition} {angle_condition}")
                plt.plot(t[t_indices], q_shank[:, t_indices].T * 180 / np.pi, label=to_legend(dof_names=shank_legend))
                plt.legend()
                plt.xlabel("Time (s)")
                plt.ylabel("Angle (degrees)")
                plt.grid()
                plt.savefig(f"{results_folder}/joint_angles_{material_condition}_{angle_condition}.png")

                for i in range(q_shank.shape[0]):
                    plt.figure(
                        f"Force X-axis against {shank_legend[i]} for {material_condition} {angle_condition}",
                        figsize=(10, 6),
                    )
                    plt.title(f"Force X-axis against {shank_legend[i]} for {material_condition} {angle_condition}")
                    plt.plot(forces[0, t_indices], q_shank[i, t_indices] * 180 / np.pi)
                    plt.xlabel("Force X-axis (N)")
                    plt.ylabel(f"{shank_legend[i]} (degrees)")
                    plt.grid()
                    plt.savefig(
                        f"{results_folder}/forceX_vs_{shank_path[i]}_{material_condition}_{angle_condition}.png"
                    )

    plt.show()


if __name__ == "__main__":
    main(show_static=False, show_reconstructions=False, results_folder="results")
