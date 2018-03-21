# CountDist

CountDist is a way to get the separations between pairs of galaxies, including potentially both the true and observed separations parallel and perpendicular to the line of sight. It can then be used to get pair counts by histogramming the separations, but is useful for determining how the true separations depend on those observed for photometric uncertainty corrections.

## Installation

The majority of CountDist is written in C++ with python functions for the user interface. Therefore, you must have a C++ compiler installed. Optionally, OpenMP can be used if installed to run in parallel, and this is set by using a non-zero value on the option flag "omp=" for the python install or by either a non-zero value for "OMP=" or an environment variable named "OMP" during the make call if compiling the C++ code directly. However, the setup.py script includes a call to the Makefile, so this will be done using the python install. Additionally, the results are written to a database, which requires [SQLite3](https://www.sqlite.org/index.html) to be installed as well as the SQLite3 C++ libraries.

To only compile the C++ code, use the Makefile:

```shell
make VERS=3 OMP=2 all
```

To install with the python code, the setup.py script can be used with flags:

```shell
python setup.py install --version=3 --omp=2
```

All other standard setup.py install flags, such as `--prefix=`, are also available.

The setup.py script also includes two additional commands, `uninstall` to completely uninstall the package including from the `PYTHONPATH`, and `clean`, to remove the extra directories created during the python installation.
