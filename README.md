# Benchmarking spatial interpolation methods for brain maps
Authors: Yigu Zhou, Vincent Bazinet, Bratislav Misic

This repository contains scripts and functions to reproduce results in ["Benchmarking spatial interpolation methods for brain maps"](https://doi.org/10.64898/2026.04.27.721143). The raw results files are too large for upload to this repository, but all input data and code to handle those data are available to reproduce results. See below for an outline:

## Repository Structure

Notebooks that contains main analyses and figures are stored here. Each notebook calls scripts and functions inside [**/code**](code/), and files inside [**/data**](data/).

### /code

This folder contains scripted runs and wrappers for interpolation functions.

- [**/interpmodules**](code/interpmodules/) contains [deterministic](code/interpmodules/deterministic.py) and spatially-informed [stochastic](code/interpmodules/stochastic.py) interpolation functions, as well as benchmark [metrics](code/interpmodules/metrics.py) and [helper](code/interpmodules/helpers.py) functions that support them.

- The various ***config** files specify global variables such as project directory, analysis parameters, transform matrices, etc.

- The Python files or notebooks with prefix **getdata_** contain code for handling GRF, empirical surface, and empirical volume data.

- The Python or Bash scripts with prefix **run_** contain code to set up and run interpolation for all combinations of data modality/characteristics 

- Notebooks with prefix **res1_** contain code to generate figures from analyses with GRF

- Notebooks with prefix **res2_** contain code to generate figures from analyses with empirical surface maps (Neuromaps)

- Notebooks with prefix **use_case_** contain code to generate figures from analyses with empirical volumetric data (iEEG or microarray)


### /data

This folder contains GRF, empirical surface and volume maps (from Neuromaps), each stored as a Pyvista.PolyData object.

- [**/sampling**](data/sampling/) contains data matrixes that shuffle training and testing samples with the spherical or midthckness template surfaces. They can be re-generated with the **getdata_** scripts.

- [**/yee_transformed-points**](data/yee_transformed-points/) contains MNI coordinates from [Yee et al., 2025](https://doi.org/10.1101/2025.06.02.656812), fetched from its [related public repository](https://github.com/CoBrALab/AllenHumanGeneMNI/tree/master/transformed-points)

## Requirements

**Environment.** Python 3.11.5, GNU bash 5.1.16 \
**Software.** The experiments presented utilize a number of published and openly available packages for generation, processing, and analysis of spatial data.

GSTools \
MGWR \
PyKrige \
Pyvista \
Scikit-Gstat \
Scikit-Image