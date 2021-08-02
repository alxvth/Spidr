# Spatial Information in Dimensionality Reduction (Spidr)

Introduces spatial neighborhood information in dimensionality reduction for high-dimensional images. 
Extends t-SNE such that similarities are based on a point's spatial neighborhood instead of only the high-dimensional point itself.

To clone the repo and its external submodules (hnswlib, glfw, spdlog):

```git clone --recurse-submodule https://github.com/alxvth/Spidr/```

Currently, only builds on Windows, tested with Visual Studio 2017. Use cmake for setting up the project:
```
mkdir build
cd build
cmake .. -G "Visual Studio 15 2017 Win64"
```

The standard cpp implementation uses the A-tSNE implementation from the [HDILib](https://github.com/biovault/HDILib) and [Hnswlib](https://github.com/nmslib/hnswlib) for approximated nearest neighbor search. Other DR techniques might also be used, as shown in the python example below.

## Usage

See `example/SpidrExample.cpp` for a example on how to use the library in cpp.

See `python_wrapper` for install intructions and a usage example on how to use the library in python. The example showcases spatially informed t-SNE, UMAP and MDS embeddings.

## Dependencies
Not all dependencies are included in this repo (see submodules in `external/`), some need to be downloaded/installed by yourself. 
Make sure to adjust your system variables respectively:
- [HDILibSlim](https://github.com/alxvth/HDILib-slim) (build and install the library and define the system variable `HDILIBSLIM_ROOT` pointing to the install DIR for cmake to automatically find the library)
- [Boost](https://www.boost.org/) (headers-only [Boost.Histogram](https://www.boost.org/doc/libs/1_73_0/libs/histogram/doc/html/index.html), define the system variable `BOOST_INCLUDEDIR = D:\..\boost\boost_1_7X_0\` for cmake to find it automatically)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (headers-only, install it anyways following these [eigen instructions](https://gitlab.com/libeigen/eigen/-/blob/master/INSTALL) so that cmake can find it automatically)
- OpenMP
