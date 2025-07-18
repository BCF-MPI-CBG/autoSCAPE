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
    from napari_ome_zarr._reader import napari_get_reader as ome_zarr_reader

    if path.endswith('.ome.zarr'):
        return None

    ome_zarr_file = Path(path).parent / ('converted_' + Path(path).stem + '.ome.zarr')
    if not os.path.exists(ome_zarr_file):
        ome_zarr_file = convert_to_ome_zarr(path)

    layer_data = ome_zarr_reader(ome_zarr_file)()

    # set napari view options
    layer_data[0][1]['blending'] = 'additive'
    layer_data[0][1]['depiction'] = 'plane'

    plane_parameters = {
        'normal': (1, 0, 0),
        'thickness': 10,
    }
    layer_data[0][1]['plane'] = plane_parameters
    layer_data[0][1]['colormap'] = 'gray'
    return layer_data


def convert_to_ome_zarr(path):

    from skimage import io
    import yaml
    import dask.array as da
    import ome_zarr
    from ome_zarr.writer import write_multiscale
    import zarr

    # like numpy.mean, but maintains dtype
    def mean_dtype(arr, **kwargs):
        return np.mean(arr, **kwargs).astype(arr.dtype)

    tif_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tif")
    ]

    metadata_file = os.path.join(path, "metadata.txt")
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)['Summary']

    # index positionlist by label
    positionlist = metadata['InitialPositionList']
    positionlist = {position['Label']: position for position in positionlist}
    position = positionlist[Path(path).stem]

    # stack arrays into single array and make sure array is 3d
    array = da.stack([
        da.from_array(io.imread(f)) for f in tif_files],
    )

    # make multiscale
    scales = [array]
    for _ in range(3):
        scales.append(
            da.coarsen(mean_dtype, scales[-1], {scales[-1].ndim - 2: 2, scales[-1].ndim - 1: 2}, trim_excess=True)
        )

    z_scale = metadata['z-step_um'] if metadata['z-step_um'] != 0 else 1.0
    y_scale = metadata['PixelSize_um']
    x_scale = metadata['PixelSize_um']


    size_z = np.round(array.shape[0] * z_scale)
    size_y = np.round(array.shape[1] * y_scale)
    size_x = np.round(array.shape[2] * x_scale)

    coordtfs = [
        [
            {
                'type': 'scale',
                'scale': [z_scale * (2 ** i), y_scale * (2 ** i), x_scale * (2 ** i)],
                },
            {
                'type': 'translation',
                'translation': [
                    float(
                        position["DeviceCoordinatesUm"]["ZStage:Z:32"][0]
                    ) - size_z // 2,
                    -float(
                        position["DeviceCoordinatesUm"]["XYStage:XY:31"][1]
                    ) - size_y // 2,
                    float(
                       position["DeviceCoordinatesUm"]["XYStage:XY:31"][0]
                    ) - size_x // 2,],
                }
        ]
        for i in range(4)
    ]

    axes = [
        {
            'name': 'z',
            'type': 'space',
            'unit': 'micrometer'
            },
        {
            'name': 'y',
            'type': 'space',
            'unit': 'micrometer'
            },
        {
            'name': 'x',
            'type': 'space',
            'unit': 'micrometer'
            }
    ]

    ome = {
        "channels": [
            {
                "active": True,
                "color": "ffffff",
                "label": Path(path).stem,
                "window": {
                    "start": 0,
                    "end": 2**16 // 2,
                    "min": 0,
                    "max": 2**16
                }
            }
        ]
    }

    target = Path(path).parent / ('converted_' + Path(path).stem + '.ome.zarr')
    store = ome_zarr.io.parse_url(target, mode="w").store
    root = zarr.group(store=store)

    write_multiscale(
        pyramid=scales,
        group=root,
        axes=axes,
        coordinate_transformations=coordtfs,
        name=Path(path).stem,
    )
    root.attrs['omero'] = ome

    return target