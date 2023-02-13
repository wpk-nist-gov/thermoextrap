# `thermoextrap`: Thermodynamic Extrapolation/Interpolation Library
This repository contains code used and described in:

Monroe, J. I.; Hatch, H. W.; Mahynski, N. A.; Shell, M. S.; Shen, V. K. Extrapolation and Interpolation Strategies for Efficiently Estimating Structural Observables as a Function of Temperature and Density. J. Chem. Phys. 2020, 153 (14), 144101. https://doi.org/10.1063/5.0014282.


# Overview

If you find this code useful in producing published works, please provide an appropriate citation.

Code included here can be used to perform thermodynamic extrapolation and
interpolation of observables calculated from molecular simulations. This allows
for more efficient use of simulation data for calculating how observables change
with simulation conditions, including temperature, density, pressure, chemical
potential, or force field parameters. Users are highly encourage to work through
the Jupyter Notebook tutorial (Ideal_Gas_Example.ipynb) presenting examples for
a variety of different observable functional forms. We only guarantee that this
code is functional for the test cases we present here or for which it has
previously been applied Additionally, the code may be in continuous development
at any time. Use at your own risk and always check to make sure the produced
results make sense. If bugs are found, please report them. If specific features
would be helpful just let us know and we will be happy to work with you to come
up with a solution.


# Status

This package is actively used by the author.  Please feel free to create a pull request for wanted features and suggestions!


# Installation

`thermoextrap` may be installed with either (recommended)
```bash
conda install -c wpk-nist thermoextrap
```
or
```bash
pip install thermoextrap
```

If you use pip, then you can include additional dependencies using
```bash
pip install thermoextrap[all]
```

If you install `thermoextrap` with conda, there are additional optional dependencies that take some care for installation.  We recommend installing the following via `pip`, as the verisons on the conda/conda-forge channels are often a bit old.
```bash
pip install tensorflow tensorflow-probability gpflow
```


# Documentation

Documentation can be found at
For a deeper dive, look at the [documentation](https://pages.nist.gov/thermo-extrap/)


## License

This is free software.  See [LICENSE](LICENSE).

## Related work

This package extensively uses the ``cmomy`` package to handle central comoments.  See [here](https://github.com/usnistgov/cmomy).


## Contact

The authors can be reached at wpk@nist.gov.

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
