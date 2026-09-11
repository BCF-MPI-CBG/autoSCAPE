"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from napari_autoscape.models.leica import (
    Point,
    Vertex,
)

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


def _apply_meta_transform(data: Any, meta: dict) -> Any:
    """Apply napari transform metadata to vertex coordinates."""
    if not isinstance(data, Sequence) or len(data) == 0:
        return data

    ndim = meta.get("ndim")
    if ndim is None:
        return data

    scale = np.asarray(meta.get("scale", np.ones(ndim)), dtype=float)
    translate = np.asarray(meta.get("translate", np.zeros(ndim)), dtype=float)
    affine = meta.get("affine")

    if scale.shape != (ndim,):
        scale = np.resize(scale, ndim).astype(float)
    if translate.shape != (ndim,):
        translate = np.resize(translate, ndim).astype(float)

    transformed_shapes = []
    for shape in data:
        arr = np.asarray(shape, dtype=float)
        if arr.size == 0:
            transformed_shapes.append(arr)
            continue

        arr = arr * scale
        arr = arr + translate

        if affine is not None:
            affine_matrix = np.asarray(affine, dtype=float)
            if affine_matrix.shape == (ndim, ndim):
                arr = arr @ affine_matrix.T
            elif affine_matrix.shape == (ndim + 1, ndim + 1):
                hom = np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)
                arr = hom @ affine_matrix.T
                arr = arr[:, :ndim]

        transformed_shapes.append(arr)

    return transformed_shapes


def write_single_shape_leica(path: str, data: Any, meta: dict) -> list[Point]:
    """
    Write a single shape to a file.
    Parameters
    ----------
    path : str
        The path to the file to write to.
    data : Any
        The data to write.
    meta : dict
        The metadata to write.

    Returns
    -------
    list[str]
        A list of paths that were written.
    """

    data = _apply_meta_transform(data, meta)

    centers = np.stack([pos.mean(axis=0) for pos in data])
    centers[:, 1] *= -1

    points = []
    for i, pos in enumerate(centers):
        ndim = len(pos)
        if ndim == 3:
            point = Point(
                Name=f"Point_{i}",
                Identifier=meta.name,
                Verticies=[Vertex(Z=pos[0], Y=pos[1], X=pos[2])],
            )
        elif ndim == 2:
            point = Point(
                Name=f"Point_{i}",
                Identifier=meta.name,
                Verticies=[Vertex(Y=pos[0], X=pos[1])],
            )

        points.append(point)

    return points


def write_single_shape(path: str, data: Any, meta: dict) -> list[str]:
    """
    Write a single shape to a file.
    Parameters
    ----------
    path : str
        The path to the file to write to.
    data : Any
        The data to write.
    meta : dict
        The metadata to write.

    Returns
    -------
    list[str]
        A list of paths that were written.
    """

    data = _apply_meta_transform(data, meta)

    centers = np.stack([pos.mean(axis=0) for pos in data])
    centers[:, 1] *= -1

    #    heigths = [pos.max(axis=0)[1] - pos.min(axis=0)[1] for pos in data]
    #    widths = [pos.max(axis=0)[2] - pos.min(axis=0)[2] for pos in data]

    #    centers[:, 1] += np.mean(heigths) / 2
    #    centers[:, 2] -= np.mean(widths) / 2
    config = _generate_pos_config(centers)

    # Save to a JSON file
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def _generate_pos_config(positions: "napari.types.PointsData"):  # noqa F821
    """
    Generate a configuration dictionary for the positions.

    Parameters
    ----------
    positions : napari.types.PointsData
        The positions to generate the configuration for.
    Returns
    -------
    dict
        The configuration dictionary (i.e., the pos-file).
    """

    # Extract unique x and y values to determine grid structure
    unique_x = sorted(set(pos[2] for pos in positions))  # noqa C401
    unique_y = sorted(set(pos[1] for pos in positions))  # noqa C401

    # Create a mapping from x and y values to grid columns and rows
    x_to_col = {x: col for col, x in enumerate(unique_x)}
    y_to_row = {y: row for row, y in enumerate(unique_y)}

    # Prepare the configuration dictionary
    config = {
        "VERSION": 3,
        "ID": "Micro-Manager XY-position list",
        "POSITIONS": [],
    }

    # Populate the positions
    for pos in positions:
        z, y, x = pos
        position_entry = {
            "GRID_COL": x_to_col[x],
            "GRID_ROW": y_to_row[y],
            "DEVICES": [
                {
                    "DEVICE": "XYStage:XY:31",
                    "AXES": 2,
                    "Y": float(y),
                    "X": float(x),
                    "Z": 0,
                },
                {
                    "DEVICE": "ZStage:Z:32",
                    "AXES": 1,
                    "Y": 0,
                    "X": float(z),
                    "Z": 0,
                },
            ],
            "PROPERTIES": {
                "OverlapUmX": "138.2400",
                "OverlapUmY": "138.2400",
                "OverlapPixelsX": "384",
                "OverlapPixelsY": "384",
            },
            "DEFAULT_Z_STAGE": "ZStage:Z:32",
            "LABEL": f"1-Pos_{x_to_col[x]:03d}_{y_to_row[y]:03d}",
            "DEFAULT_XY_STAGE": "XYStage:XY:31",
        }
        config["POSITIONS"].append(position_entry)

    return config
