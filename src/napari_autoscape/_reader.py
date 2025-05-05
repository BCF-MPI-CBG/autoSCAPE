"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

import os
from pathlib import Path

import numpy as np


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not os.path.isdir(path):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    import bioio_tifffile
    from bioio import BioImage

    tif_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tif")
    ]
    metadata_file = os.path.join(path, "metadata.txt")

    # get metadata from first image
    Image = BioImage(
        os.path.abspath(os.path.join(path, tif_files[0])),
        reader=bioio_tifffile.Reader,
    )

    # stack arrays into single array
    data = np.squeeze(
        [BioImage(f, reader=bioio_tifffile.Reader).data for f in tif_files]
    )
    metadata = _load_pos_file(metadata_file)

    folder_name = Path(path).stem
    grid_col = int(folder_name.split("_")[-2])
    grid_row = int(folder_name.split("_")[-1])
    tile_metadata = _get_tile_metadata(metadata, grid_col, grid_row)

    scale = [
        Image.physical_pixel_sizes.Z,
        Image.physical_pixel_sizes.Y,
        Image.physical_pixel_sizes.X,
    ]
    scale = [s if s is not None else 10 for s in scale]
    translate = (
        float(tile_metadata["DeviceCoordinatesUm"]["ZStage:Z:32"][0]),
        -float(tile_metadata["DeviceCoordinatesUm"]["XYStage:XY:31"][1]),
        float(tile_metadata["DeviceCoordinatesUm"]["XYStage:XY:31"][0]),
    )

    add_kwargs = {
        "scale": scale,
        "translate": translate,
        "metadata": tile_metadata,
        "blending": "additive",
    }

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def _load_pos_file(file_path: Path) -> dict:
    import json

    # Read the file content
    with open(file_path) as file:
        json_data = file.read()

    # Parse the JSON data
    json_data = json.loads(json_data)

    return json_data


def _get_tile_metadata(data: dict, grid_col: int, grid_row: int) -> dict:
    """
    Get the tile metadata for a specific grid column and row of a pos file

    Parameters
    ----------
    data : dict
        The parsed JSON data from the pos file.
    grid_col : int
        The grid column of the tile.
    grid_row : int
        The grid row of the tile.

    Returns
    -------
    dict
        The tile metadata for the specified grid column and row.
    """
    for position in data["Summary"]["InitialPositionList"]:
        if (
            position["GridColumnIndex"] == grid_col
            and position["GridRowIndex"] == grid_row
        ):
            return position
    return None
