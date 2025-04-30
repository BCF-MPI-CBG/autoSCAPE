from pathlib import Path

def test_reader(make_napari_viewer):

    viewer = make_napari_viewer()

    # get path to this file
    path = Path(__file__).parent /'1-Pos_000_006'
    viewer.open(path, plugin='napari-autoscape')

    assert len(viewer.layers) == 1