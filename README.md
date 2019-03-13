# CountDist

CountDist is a way to get the separations between pairs of galaxies, including potentially both the true and observed separations parallel and perpendicular to the line of sight. It can then be used to get pair counts by histogramming the separations, but is useful for determining how the true separations depend on those observed for photometric uncertainty corrections.

## Installation

Much of the pair processing for CountDist is written in C++ with python wrappers, although there is also pure python functionality. Therefore, a C++ compiler is required, with the C++17 standard available. Optionally, OpenMP can be used to run the pair processing in parallel, with the option set by using a >1 value for the option flag `--omp=` (python) or `-Domp_num_threads=` (CMake). There is also the ability to install older versions of the code (not recommended due to lack of documentation) using the option flag `--vers=` (python) or `-Dvers=` (CMake). The python setup.py script also calls CMake, so everything can be compiled using the python setup script.

To only compile the C++ code, use CMake (it is recommended to run in a separate build directory):

```shell
cd CountDist
mkdir build
cd build
cmake .. [-Domp_num_threads=0] [-Dvers=4]
cmake --build
```

To install with the python code, the setup.py script can be used with flags:

```shell
python setup.py install [--version=4] [--omp=0]
```

All other standard setup.py install flags, such as `--prefix=`, are also available.

The setup.py script also includes a `clean` command to remove the extra directories created during the python installation.

**_Important note:_** The name of the package for importing in python is actually `countdist2` rather than `countdist`. This is due to a previous version which will be removed completely in the future, at which point the name will become `countdist`, but for now be sure to know what to import!
