from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import numpy as np
from napari.layers import Shapes


def tesellate_area(
        selection_area: Shapes,
        overlap_percent: int = 20,
        fov_size_x: float = 553.0,
        fov_size_z: float = 553.0) -> "napari.types.LayerDataTuple":
    """
    Tessellate the area defined by the selection_area into tiles.
    
    The tiles are defined by the size of the field of view (fov_size_x,
    fov_size_z) and the overlap percentage (overlap_percent).

    Parameters
    ----------
    selection_area : napari.types.Shapes
        The selection area to tessellate.
    overlap_percent : int
        The percentage of overlap between tiles.
    fov_size_x : float
        The size of the field of view in the x direction.
    fov_size_z : float
        The size of the field of view in the z direction.

    Returns
    -------
    napari.types.LayerDataTuple
        A tuple containing the tiles and their properties.
    """
    import napari
    rectangle_data = selection_area.data[0]
    scale = selection_area.scale

    fov_size_x = fov_size_x / scale[2]
    fov_size_y = fov_size_z / scale[1]

    z = rectangle_data[0][0]
    center_z = z * scale[0]  # z coordinate is constant

    min_x = rectangle_data[:, 2].min() - fov_size_x / 2
    max_x = rectangle_data[:, 2].max() + fov_size_x / 2
    min_y = rectangle_data[:, 1].min() - fov_size_y / 2
    max_y = rectangle_data[:, 1].max() + fov_size_y / 2

    # tesellate the area given the overlap percentage
    overlap_x = overlap_percent * fov_size_x / 100

    n_tiles_x = int((max_x - min_x) / (fov_size_x - overlap_x))
    n_tiles_y = int((max_y - min_y) / (fov_size_y - overlap_x))

    tiles = []
    for i in range(0, n_tiles_x):
        for j in range(n_tiles_y, -1, -1):
            # Calculate the center of the tile
            center_x = min_x + i * (fov_size_x - overlap_x) + fov_size_x / 2
            center_y = min_y + j * (fov_size_y - overlap_x) + fov_size_y / 2

            x1 = center_x - fov_size_x / 2
            x2 = center_x + fov_size_x / 2
            y1 = center_y - fov_size_y / 2
            y2 = center_y + fov_size_y / 2

            # Create a list of corner coordinates
            corners = np.asarray([
                (center_z, y1, x1),
                (center_z, y1, x2),
                (center_z, y2, x2),
                (center_z, y2, x1)
            ])
            
            corners[:, 1] *= scale[1]
            corners[:, 2] *= scale[2]
            
            tiles.append(corners)

    viewer = napari.current_viewer()
    if 'SCAPE_fields' in viewer.layers:
        viewer.layers['SCAPE_fields'].edge_width = 25

    tuple_data = (
        tiles, 
        {
            'edge_color': 'blue',
            'face_color': 'transparent',
            'edge_width': 25,
            'name': 'SCAPE_fields'
        },
        'shapes'
    )

    return tuple_data