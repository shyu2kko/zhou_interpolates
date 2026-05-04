# Benchmarking spatial interpolation methods for brain maps
Authors: Yigu Zhou, Vincent Bazinet, Bratislav Misic

This repository contains scripts and functions to reproduce results in "Benchmarking spatial interpolation methods for brain maps".

## Repository Structure

Notebooks that contains main analyses and figures are stored here. Each notebook calls scripts and functions inside `code`, and files inside `data`.

### `code`

This folder contains scripted runs and wrappers for interpolation functions.

`/interpmodules` contains deterministic (`deterministic.py`) and spatially-informed stochastic (`geospatial.py`) interpolation functions, as well as benchmark metrics (`metrics.py`) and utility functions (`helpers.py`) that support them.
`*config.py` files specify global variables such as project directory, analysis parameters, transform matrices, etc.
`getdata_*.py` or `.ipynb` contain code for handling GRF, empirical surface, and empirical volume data.
`run_*.py` or `.sh` contain code to set up and run interpolation for all combinations of data modality/characteristics 
`res1_*` notebooks contain code to generate figures from analyses with GRF
`res2_*` notebooks contain code to generate figures from analyses with empirical surface maps (Neuromaps)
`usecase_*` notebooks contain code to generate figures from analyses with empirical volumetric data (iEEG or microarray)


### `data`

This folder contains GRF, empirical surface and volume maps (from Neuromaps), each stored as a Pyvista.PolyData object.

`/sampling` contains data matrixes that shuffle training and testing samples with the spherical or midthckness template surfaces.
`/yee_transformed-points` contains MNI coordinates from Yee et al., 2025 for the microarray data

## Requirements

**Environment.** Python 3.11.5, GNU bash 5.1.16 \
**Software.** The experiments presented utilize a number of published and openly available packages for generation, processing, and analysis of spatial data.

GSTools \
MGWR \
PyKrige \
Pyvista \
Scikit-Gstat \
Scikit-Image