from pathlib import Path

def load_pos_file(file_path: Path) -> dict:
    import json
    # Read the file content
    with open(file_path, 'r') as file:
        json_data = file.read()

    # Parse the JSON data
    json_data = json.loads(json_data)

    return json_data


def get_tile_metadata(data: dict, grid_col: int, grid_row: int) -> dict:
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
    for position in data['POSITIONS']:
        if position['GRID_COL'] == grid_col and position['GRID_ROW'] == grid_row:
            return position
    return None