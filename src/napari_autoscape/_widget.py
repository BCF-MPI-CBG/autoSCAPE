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


def create_well_grid(
        fiducial_layer: Points,
        n_columns: int = 28,
        n_rows: int = 32,
        microwell_diameter_um: float = 200
        ) -> Shapes:
    """
    Create a grid of wells based on fiducial positions.
    
    Parameters:
        fiducial_positions: Coordinates of fiducials.
        ncolumns: Number of columns in the microwell plate.
        nrows: Number of rows in the microwell plate.
        microwell_size_um: Size of a single microwell in micrometers.
        
    Returns:
        Well positions as a numpy array.
    """
    from scipy.spatial import distance, KDTree
    import pandas as pd

    # Calculate row and column vectors
    fiducials = fiducial_layer.data * fiducial_layer.scale
    fiducial_indeces = [0, 1, 2, 3]

    # Distance between fiducials and set radius
    dist_matrix = distance.cdist(fiducials, fiducials, 'euclidean')

    max_dist = np.max(dist_matrix[dist_matrix > 0])
    radius = 9/10 * max_dist
    tree = KDTree(fiducials)

    # find point with 4 neighbors
    fiducial_ids = {}

    for i in range(len(fiducials)):
        neighbors = tree.query_ball_point(fiducials[i], radius)
        if len(neighbors) >= 4:
            fiducial_ids['center'] = i
            break

    # find south fiducial to center
    distances = dist_matrix[fiducial_ids['center']]
    fiducial_ids['south'] = np.argwhere(distances == sorted(distances)[1])[0][0]
    fiducial_indeces = [index for index in fiducial_indeces if index not in fiducial_ids.values()]

    # find fiductial on the west-hand side of the vector from south to center
    south_fiducial_coords = fiducials[fiducial_ids['south']]
    center_fiducial_coords = fiducials[fiducial_ids['center']]

    vector1 = (south_fiducial_coords - center_fiducial_coords)
    vector2 = (fiducials[fiducial_indeces[0]] - center_fiducial_coords)
    angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))) / np.pi * 180

    if angle < 180:
        fiducial_ids['west'] = fiducial_indeces[0]
        fiducial_ids['east'] = fiducial_indeces[1]
    else:
        fiducial_ids['west'] = fiducial_indeces[1]
        fiducial_ids['east'] = fiducial_indeces[0]

    fiducials = {key: fiducials[fiducial_ids[key]] for key in fiducial_ids.keys()}

    # # there are four possible orientations of the fiducials: 0, 90, 180, 270 degrees
    # if fiducials['west'][1] < fiducials['center'][1] and fiducials['east'][1] > fiducials['center'][1]:
    #     orientation = 90
    # elif fiducials['center'][1] < fiducials['west'][1] and fiducials['center'][1] < fiducials['east'][1]:
    #     orientation = 180
    # elif fiducials['center'][1] > fiducials['west'][1] and fiducials['center'][1] > fiducials['east'][1]:
    #     orientation = 0
    # elif fiducials['west'][1] > fiducials['center'][1] and fiducials['east'][1] < fiducials['center'][1]:
    #     orientation = 270
    
    row_vector = (fiducials['east'] - fiducials['west']) / (n_columns - 1)

    col_vector1 = _rotate_vector(row_vector[1:], -120)
    col_vector1 = np.insert(col_vector1, 0, 0)
    col_vector2 = _rotate_vector(row_vector[1:], -60)
    col_vector2 = np.insert(col_vector2, 0, 0)

    row_vector1 = np.stack([fiducials['west'], row_vector])
    col_vector1 = np.stack([fiducials['west'], col_vector1])
    col_vector2 = np.stack([fiducials['west'], col_vector2])

    well_positions = []
    rows = []
    cols = []
    for j in range(n_rows):

        if j % 4 == 1 or j % 4 == 3:
            delta = col_vector1
        else:
            delta = np.zeros_like(col_vector1)

        starting_position = fiducials['west'] + j//2 * col_vector1[1:] + j//2 * col_vector2[1:] + delta[1:]
        for i in range(n_columns):
            well_positions.append(starting_position + i * row_vector1[1:])
            rows.append(j)
            cols.append(i)

    well_positions = np.stack(well_positions).squeeze()
    hexagons = [_hexagon_around_point(p, size=microwell_diameter_um/2) for p in well_positions]

    features = pd.DataFrame({
        'row': rows,
        'column': cols,
    })

    text = {
        'string': 'WELL ({row}, {column})',
        'anchor:': 'center',
        'color': 'red',
    }

    return Shapes(
        hexagons,
        edge_color='red',
        face_color='transparent',
        name='Well grid',
        features=features,
        shape_type='polygon',
        text=text,
        edge_width=2)


def _rotate_vector(vector, angle):
    """Rotate a vector by a given angle in degree."""
    angle = np.deg2rad(angle)  # Convert angle to radians
    R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]])
    return R @ vector    

def _hexagon_around_point(point, size):
    """Generate coordinates of a hexagon around a point."""
    # consider that points are zyx and disregard z
    z_value = point[0]  # Save z coordinate 
    point = point[1:]  # Take only x and y coordinates
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6  # 6 angles for hexagon, offset by 30 degrees
    points =  point + size * np.column_stack((np.cos(angles), np.sin(angles)))

    # put z back in
    points = np.insert(points, 0, z_value, axis=1)  # Insert z coordinate back
    return points
