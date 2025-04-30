"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union
import json
import numpy as np

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]

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

    centers = np.stack([pos.mean(axis=0) for pos in data])
    config = _generate_pos_config(centers)

    # Save to a JSON file
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def _generate_pos_config(positions: "napari.types.PointsData"):
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
    unique_x = sorted(set(pos[2] for pos in positions))
    unique_y = sorted(set(pos[1] for pos in positions))

    # Create a mapping from x and y values to grid columns and rows
    x_to_col = {x: col for col, x in enumerate(unique_x)}
    y_to_row = {y: row for row, y in enumerate(unique_y)}

    # Prepare the configuration dictionary
    config = {
        "VERSION": 3,
        "ID": "Micro-Manager XY-position list",
        "POSITIONS": []
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
                    "Z": 0
                },
                {
                    "DEVICE": "ZStage:Z:32",
                    "AXES": 1,
                    "Y": 0,
                    "X": 1955.9799999999998,
                    "Z": float(z)
                }
            ],
            "PROPERTIES": {
                "OverlapUmX": "138.2400",
                "OverlapUmY": "138.2400",
                "OverlapPixelsX": "384",
                "OverlapPixelsY": "384"
            },
            "DEFAULT_Z_STAGE": "ZStage:Z:32",
            "LABEL": f"1-Pos_{x_to_col[x]:03d}_{y_to_row[y]:03d}",
            "DEFAULT_XY_STAGE": "XYStage:XY:31"
        }
        config["POSITIONS"].append(position_entry)

    return config