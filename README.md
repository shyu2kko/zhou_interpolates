# Benchmarking spatial interpolation methods for brain maps
Authors: Yigu Zhou, Vincent Bazinet, Bratislav Misic

This repository contains scripts and functions to reproduce results in "Benchmarking spatial interpolation methods for brain maps".

## Repository Structure

Notebooks that contains main analyses and figures are stored here. Each notebook calls scripts and functions inside `code`, and files inside `data`.

### `code`

This folder contains scripted runs and wrappers for interpolation functions.

`/interpmodules` contains spatially-agnostic (`deterministic`) and spatially-informed (`geospatial`) interpolation functions.


### `data`

This folder contains GRF, empirical surface and volume maps (from Neuromaps), each stored as a Pyvista.PolyData object.

`/sampling` contains data matrixes that shuffle training and testing samples with the spherical or midthckness template surfaces.

## Requirements

**Environment.** Python 3.11.5, GNU bash 5.1.16 \
**Software.** The experiments presented utilize a number of published and openly available packages for generation, processing, and analysis of spatial data.

GSTools \
MGWR \
PyKrige \
Pyvista \
Scikit-Gstat \
Scikit-Image