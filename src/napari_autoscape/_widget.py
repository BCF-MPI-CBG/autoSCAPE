from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import numpy as np
from napari.layers import Shapes, Points


def tesellate_area(
    selection_area: Shapes,
    overlap_percent: int = 20,
    fov_size_x: float = 332.0,
    fov_size_y: float = 332.0,
) -> "napari.types.LayerDataTuple":
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
    fov_size_x [um] : float
        The size of the field of view in the x direction.
    fov_size_z [um]: float
        The size of the field of view in the z direction.

    Returns
    -------
    napari.types.LayerDataTuple
        A tuple containing the tiles and their properties.
    """
    import napari

    rectangle_data = selection_area.data[0] * selection_area.scale

    z = rectangle_data[0][0]
    center_z = z

    min_x = rectangle_data[:, 2].min() - fov_size_x / 2
    max_x = rectangle_data[:, 2].max() + fov_size_x / 2
    min_y = rectangle_data[:, 1].min() - fov_size_y / 2
    max_y = rectangle_data[:, 1].max() + fov_size_y / 2

    # tesellate the area given the overlap percentage
    overlap_x = overlap_percent * fov_size_x / 100

    n_tiles_x = int((max_x - min_x) / (fov_size_x - overlap_x))
    n_tiles_y = int((max_y - min_y) / (fov_size_y - overlap_x))

    tiles = []
    for i in range(n_tiles_x):
        for j in range(n_tiles_y, -1, -1):
            # Calculate the center of the tile
            center_x = min_x + i * (fov_size_x - overlap_x) + fov_size_x / 2
            center_y = min_y + j * (fov_size_y - overlap_x) + fov_size_y / 2

            x1 = center_x - fov_size_x / 2
            x2 = center_x + fov_size_x / 2
            y1 = center_y - fov_size_y / 2
            y2 = center_y + fov_size_y / 2

            # Create a list of corner coordinates
            corners = np.asarray(
                [
                    (center_z, y1, x1),
                    (center_z, y1, x2),
                    (center_z, y2, x2),
                    (center_z, y2, x1),
                ]
            )

            tiles.append(corners)

    viewer = napari.current_viewer()
    if "SCAPE_fields" in viewer.layers:
        viewer.layers["SCAPE_fields"].edge_width = 25

    tuple_data = (
        tiles,
        {
            "edge_color": "blue",
            "face_color": "transparent",
            "edge_width": 5,
            "name": "SCAPE_fields",
        },
        "shapes",
    )

    return tuple_data


def fit_focus_plane(
        focus_points: Points,
        query_locations: Shapes,
        order: int = 1) -> 'napari.types.LayerDataTuple':
    
    positions = focus_points.data * focus_points.scale
    query_points = np.stack([np.mean(loc, axis=0) for loc in query_locations.data])

    # Fit focus plane
    poly_model = _fit_polynomial_surface(positions, order)
    z_pred = poly_model(query_points[:, 2], query_points[:, 1])

    new_locations = []

    for i, loc in enumerate(query_locations.data):
        # Create a new location with the predicted z value
        new_loc = loc.copy()
        new_loc[:, 0] = z_pred[i]
        new_locations.append(new_loc)

    return (
        new_locations,
        {
            "features": {'focus_positions': z_pred},
            "name": f"Focus Plane (order={order})",
            "edge_color": "focus_positions",
            "face_color": "transparent",
            "edge_width": 5,
        },
        "shapes",
    )


def _fit_polynomial_surface(points, order):
    """
    Fit a polynomial surface to a set of 3D points.

    Parameters:
    points (np.ndarray): Nx3 array of points with format [z, y, x].
    order (int): Order of the polynomial.

    Returns:
    Polynomial: A polynomial model that approximates z given x and y.
    """
    # Extract x, y, z coordinates
    z = points[:, 0]
    y = points[:, 1]
    x = points[:, 2]

    # Create a grid of polynomial terms
    xx, yy = np.meshgrid(np.arange(order + 1), np.arange(order + 1))
    xx = xx.ravel()
    yy = yy.ravel()

    # Filter terms where the sum of the exponents is <= order
    mask = (xx + yy) <= order
    xx = xx[mask]
    yy = yy[mask]

    # Create the design matrix
    X = np.column_stack([(x**xx_i) * (y**yy_i) for xx_i, yy_i in zip(xx, yy)])

    # Solve for the coefficients
    coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)

    # Create a polynomial model
    def poly_model(x, y):
        z_pred = np.zeros_like(x)
        for coef, xx_i, yy_i in zip(coeffs, xx, yy):
            z_pred += coef * (x**xx_i) * (y**yy_i)
        return z_pred

    return poly_model