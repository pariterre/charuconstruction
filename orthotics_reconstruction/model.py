import biorbd
import biobuddy as bb
import numpy as np


def shank_vertical(data, model):
    """
    Compute the vertical axis
    """
    return cond_mid(data, model) + np.array([[0, 0, 0.3, 0]]).T


def cond_mid(data, model):
    """
    Compute the midpoint between the medial and lateral condyles
    """
    return np.mean((data["cond_med"] + data["cond_lat"]) / 2, axis=1, keepdims=True)


def mal_mid(data, model):
    """
    Compute the midpoint between the medial and lateral malleoli
    """
    return np.mean((data["mal_med"] + data["mal_lat"]) / 2, axis=1, keepdims=True)


def leg_model() -> biorbd.Model:
    model = bb.BiomechanicalModel()
    model.segments.append(
        bb.Segment(
            name="root",
            rotations=bb.Rotations.XYZ,
            translations=bb.Translations.XYZ,
            segment_coordinate_system=bb.SegmentCoordinateSystem(
                origin=shank_vertical,
                first_axis=bb.Axis(name=bb.Axis.Name.X, start="cond_med", end="cond_lat"),
                second_axis=bb.Axis(name=bb.Axis.Name.Z, start=cond_mid, end=shank_vertical),
                axis_to_keep=bb.Axis.Name.X,
            ),
        )
    )
    model.segments["root"].add_marker(
        bb.Marker("VERTICAL", shank_vertical, parent_name="root", is_technical=False, is_anatomical=True)
    )
    model.segments["root"].add_marker(bb.Marker("thigh1", parent_name="root"))
    model.segments["root"].add_marker(bb.Marker("thigh2", parent_name="root"))
    model.segments["root"].add_marker(bb.Marker("thigh3", parent_name="root"))
    model.segments["root"].add_marker(bb.Marker("thigh4", parent_name="root"))
    model.segments["root"].add_marker(bb.Marker("cond_lat", parent_name="root", is_technical=False, is_anatomical=True))
    model.segments["root"].add_marker(bb.Marker("cond_med", parent_name="root", is_technical=False, is_anatomical=True))

    model.segments.append(
        bb.SegmentReal(
            name="SHANK",
            parent_name="root",
            rotations=bb.Rotations.XYZ,
            segment_coordinate_system=bb.SegmentCoordinateSystem(
                origin=cond_mid,
                first_axis=bb.Axis(name=bb.Axis.Name.X, start="mal_med", end="mal_lat"),
                second_axis=bb.Axis(name=bb.Axis.Name.Z, start=mal_mid, end=cond_mid),
                axis_to_keep=bb.Axis.Name.X,
            ),
        )
    )
    model.segments["SHANK"].add_marker(bb.Marker("leg1", parent_name="SHANK"))
    model.segments["SHANK"].add_marker(bb.Marker("leg2", parent_name="SHANK"))
    model.segments["SHANK"].add_marker(bb.Marker("leg3", parent_name="SHANK"))
    model.segments["SHANK"].add_marker(bb.Marker("leg4", parent_name="SHANK"))
    model.segments["SHANK"].add_marker(
        bb.Marker("mal_lat", parent_name="SHANK", is_technical=False, is_anatomical=True)
    )
    model.segments["SHANK"].add_marker(
        bb.Marker("mal_med", parent_name="SHANK", is_technical=False, is_anatomical=True)
    )

    data = bb.C3dData("../data/pilote/static.c3d")
    real = model.to_real(data=data)
    real.to_biomod("temporary.bioMod")

    return biorbd.Model("temporary.bioMod")
