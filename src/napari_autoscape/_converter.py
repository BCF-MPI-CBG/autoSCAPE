"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

import os
from pathlib import Path

import numpy as np

def convert_single_tile_to_ome_zarr(path):

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
            - float(position["DeviceCoordinatesUm"]["XYStage:XY:31"][1]) + size_y // 2,
            float(position["DeviceCoordinatesUm"]["XYStage:XY:31"][0]) - size_x // 2,
        )
    )

    ngff_ms = NgffMultiscales(
        ngff_image,
        coordinateTransformations=[translation]
    )

    return ngff_ms

def convert_tiles_to_stitched_ome_zarr(path: str):
    import glob
    import yaml
    import tqdm
    from dask_image import imread

    from multiview_stitcher import msi_utils
    from multiview_stitcher import spatial_image_utils as si_utils
    from multiview_stitcher import fusion

    from ome_zarr.image import NgffImage, NgffMultiscales
    from ome_zarr_models._v06.coordinate_transforms import Translation, CoordinateSystem, Axis

    folders = [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    metadata = yaml.safe_load(open(os.path.join(path, folders[0], 'metadata.txt'), 'r'))['Summary']

    positionlist = metadata['InitialPositionList']
    positionlist = {position['Label']: position for position in positionlist}

    tile_translations = []
    tile_arrays = []
    z_values = []
    for label, position in tqdm.tqdm(positionlist.items()):
        z_scale = metadata['z-step_um'] if metadata['z-step_um'] != 0 else 1.0
        y_scale = metadata['PixelSize_um']
        x_scale = metadata['PixelSize_um']

        # read single tile as array
        array = imread.imread(
            glob.glob(os.path.join(path, label, '*.tif'))[0]
        )

        size_z = np.round(array.shape[0] * z_scale)
        size_y = np.round(array.shape[1] * y_scale)
        size_x = np.round(array.shape[2] * x_scale)

        tile_translations.append(
            {
                "y": -float(position["DeviceCoordinatesUm"]["XYStage:XY:31"][1]) + size_y // 2,
                "x": float(position["DeviceCoordinatesUm"]["XYStage:XY:31"][0]) - size_x // 2
                }
        )
        z_values.append(float(position["DeviceCoordinatesUm"]["ZStage:Z:32"][0]))
        tile_arrays.append(array)
    z_offset = np.median(z_values)

    y_min = min([translation['y'] for translation in tile_translations])
    x_min = min([translation['x'] for translation in tile_translations])

    msims = []
    for tile_array, tile_translation in zip(tile_arrays, tile_translations):
        sim = si_utils.get_sim_from_array(
            tile_array,
            dims=["c", "y", "x"],
            scale={"y": y_scale, "x": x_scale},
            translation=tile_translation,
            transform_key="stage_metadata",
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))

    fused_sim = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key="stage_metadata"
    ).squeeze().pad(x=1, y=1, constant_values=0)

    ngff_image_overview = NgffImage(
        fused_sim.data[None, :],
        dims=["z", "y", "x"],
        scale={"z": float(z_scale), "y": float(y_scale), "x": float(x_scale)},
        axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
        name=Path(path).stem
        )

    ngff_multiscales_overview = NgffMultiscales(
        ngff_image_overview,
        scale_factors=(2, 4, 8, 16, 32)
    )

    new_transform = Translation(
        input=ngff_multiscales_overview.metadata.coordinateSystems[0].name,
        output="translated",
        translation=(
            z_offset,
            y_min,
            x_min
        )
    )

    new_cs = CoordinateSystem(
        name="translated",
        axes=(Axis(name="z", type="space", unit="micrometer"),
              Axis(name="y", type="space", unit="micrometer"),
              Axis(name="x", type="space", unit="micrometer"))
    )

    ngff_multiscales_overview.metadata = ngff_multiscales_overview.metadata.model_copy(
        update={
            "coordinateTransformations": (new_transform,),
            "coordinateSystems": (ngff_multiscales_overview.metadata.coordinateSystems[0], new_cs)
        }
    )

    return ngff_multiscales_overview