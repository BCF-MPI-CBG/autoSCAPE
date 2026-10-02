# autoSCAPE

This project provides functionality that converts and processes data acquired with a SCAPE (Swept Confocally-Aligned Planar Excitation) microscope, speeding up high-throughput acquisitions such as multiwell-plate screens.

The core workflow takes the raw tile images and stage positions written out by the acquisition software (ASI/Micro-Manager or Leica LAS X), stitches the tiles together using their stage coordinates, and writes the result as a multiscale OME-Zarr dataset ready to explore in napari. On top of this conversion pipeline, `autoscape` also provides napari widgets for acquisition planning (tile grids, well grids, focus-plane fitting) and stage-position export, as well as optional deep-learning models for automated region and focus detection.

## Installation

Install `autoscape` with [pip]:

```bash
pip install autoscape
```

To use the optional YOLO/focus-detection models, install the `dl` extra:

```bash
pip install "autoscape[dl]"
```

## Contents

Start with the conversion notebooks below to see the end-to-end workflow for turning a raw SCAPE acquisition into a stitched OME-Zarr dataset:

- [Convert ASI SCAPE acquisitions](notebooks/convert_ASI_SCAPE.ipynb)
- [Convert Leica SCAPE acquisitions](notebooks/convert_leica_SCAPE.ipynb)

Additional notebooks cover downstream analysis built on top of converted data:

- [Apply YOLO detection and autofocus](notebooks/apply_yolo.ipynb)
- [Paper figures](notebooks/paper_screenshots.ipynb)

## Links

- Source code: [github.com/BCF-MPI-CBG/autoSCAPE](https://github.com/BCF-MPI-CBG/autoSCAPE)
- Issues: [github.com/BCF-MPI-CBG/autoSCAPE/issues](https://github.com/BCF-MPI-CBG/autoSCAPE/issues)

[napari]: https://napari.org
[pip]: https://pypi.org/project/pip/

