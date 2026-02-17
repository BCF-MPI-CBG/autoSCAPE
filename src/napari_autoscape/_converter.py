"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

import os
from pathlib import Path

import numpy as np

def convert_to_ome_zarr(path):

    import dask.array as da
    import yaml
    from dask_image import imread
    from ome_zarr.image import NgffImage, NgffMultiscales
    from ome_zarr_models._v06.coordinate_transforms import Translation

    # like numpy.mean, but maintains dtype
    def mean_dtype(arr, **kwargs):
        return np.mean(arr, **kwargs).astype(arr.dtype)

    tif_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tif")
    ]

    metadata_file = os.path.join(path, "metadata.txt")
    with open(metadata_file) as f:
        metadata = yaml.safe_load(f)["Summary"]

    # index positionlist by label
    positionlist = metadata["InitialPositionList"]
    positionlist = {position["Label"]: position for position in positionlist}
    position = positionlist[Path(path).stem]

    # stack arrays into single array and make sure array is 3d
    array = da.stack(
        [imread.imread(f) for f in tif_files],
    ).squeeze()

    z_scale = metadata["z-step_um"] if metadata["z-step_um"] != 0 else 1.0
    y_scale = metadata["PixelSize_um"]
    x_scale = metadata["PixelSize_um"]

    size_y = array.shape[1] * y_scale
    size_x = array.shape[2] * x_scale
    size_z = array.shape[0] * z_scale

    omero = {
        "channels": [
            {
                "active": True,
                "color": "ffffff",
                "label": Path(path).stem,
                "window": {
                    "start": 0,
                    "end": 2**16 // 2,
                    "min": 0,
                    "max": 2**16,
                },
            }
        ]
    }

    ngff_image = NgffImage(
        data=array,
        scale={'z': z_scale, 'y': y_scale, 'x': x_scale},
        dims=["z", "y", "x"],
        axes_units={'z': 'micrometer', 'y': 'micrometer', 'x': 'micrometer'},
        name=Path(path).stem,
    )

    translation = Translation(
        name="translation",
        input="physical",
        output="translated",
        translation=(
            float(position["DeviceCoordinatesUm"]["ZStage:Z:32"][0]) - size_z // 2,
            -float(position["DeviceCoordinatesUm"]["XYStage:XY:31"][1]) - size_y // 2,
            float(position["DeviceCoordinatesUm"]["XYStage:XY:31"][0]) - size_x // 2,
        )
    )

    ngff_ms = NgffMultiscales(
        ngff_image,
        coordinateTransformations=[translation]
    )

    return ngff_ms
