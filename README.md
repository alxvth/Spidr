# Spatial Information in Dimensionality Reduction (Spidr)

To clone the repo and some dependencies (hnsw lib):

```git clone --recurse-submodule https://github.com/alxvth/Spidr/```

Intergrated for easy usage in the [hdps](https://github.com/hdps/SpidrPlugin) framework.

## Other Dependencies
Not all dependencies are included in this repo, some need to be downloaded/installed by yourself. 
Make sure to adjust your system variables respectively:
- [Boost](https://www.boost.org/) (headers-only basically just [Boost.Histogram](https://www.boost.org/doc/libs/1_73_0/libs/histogram/doc/html/index.html), define the system variable `BOOST_INCLUDEDIR = D:\..\boost\boost_1_7X_0\` for cmake to find it automatically)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (headers-only, define the system variable `Eigen3_DIR = D:\...\eigen\cmake` after following these [eigen instructions](https://gitlab.com/libeigen/eigen/-/blob/master/INSTALL), otherwise cmake might not find it automatically)
- OpenMP
