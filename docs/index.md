# Autoscape

Read, write and process data acquired with a SCAPE (Swept Confocally-Aligned Planar Excitation) microscope to speed up acquisition.

----------------------------------

`autoscape` helps plan and automate high-throughput SCAPE acquisitions, for example of multiwell plates. It provides napari widgets for acquisition planning and position-list export, plus library/notebook functions for converting vendor tile/stage data into OME-Zarr and for automatically detecting regions and focus planes with trained models.

## Features

Additional functionality (used via the Python API and the notebooks in [docs/notebooks](docs/notebooks)):

- Conversion of Leica composite/tile-scan data and ASI/Micro-Manager tile stacks + stage position metadata into stitched, multiscale OME-Zarr datasets (using [bioio] and [multiview-stitcher]).
- Building and serializing Leica LAS X stage-overview region (`.rgn`) files, so detected positions can be re-imported into the acquisition software.
- Optional deep-learning extras for:
  - YOLO-based sliding-window object detection to find samples/wells in large overview images, with per-detection feature extraction for use with [napari-clusters-plotter].
  - A CNN focus-detection model that automatically predicts the best-focus z-slice for a detected region, as an alternative to manual focus-point fitting.

See the example notebooks in [docs/notebooks](docs/notebooks) for end-to-end ASI/Leica conversion, YOLO detection, and focus-detection workflows.

## Installation

You can install `autoscape` via [uv]. First clone the repository:

    git clone https://github.com/bcf-mpi-cbg/autoscape.git
    cd autoscape

    uv sync .

To use the optional YOLO/focus-detection models, install the `dl` extra:

    uv pip install -e ".[dl]"

## License

Distributed under the terms of the [BSD-3] license,
"autoscape" is free and open source software