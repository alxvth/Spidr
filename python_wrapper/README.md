# Spidr wrapper

Creates python bindings for the SpidrLib, based on [this](https://github.com/pybind/cmake_example) pybind11 example.

## Installation
Make sure, that the external pybind11 submodule is present. That should be the case if you cloned the entire repo with the `--recurse-submodule` option. To build and install the python wrapper use (after navigating into this folder in a python shell):

```bash
pip install . --use-feature=in-tree-build
```

## Usage

See `example/example.py` for t-SNE, UMAP and MDS examples with a synthetic data set.