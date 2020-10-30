
<!-- README.md is generated from README.Rmd. Please edit that file -->

# MIDASpy<img src='MIDASpy_logo.png' align="right" height="139" /></a>

<!-- badges: start -->

<!-- [![CRAN status](https://www.r-pkg.org/badges/version/dplyr)](https://cran.r-project.org/package=dplyr) -->

<!-- [![R build status](https://github.com/tidyverse/dplyr/workflows/R-CMD-check/badge.svg)](https://github.com/tidyverse/dplyr/actions?workflow=R-CMD-check) -->

<!-- [![Codecov test coverage](https://codecov.io/gh/tidyverse/dplyr/branch/master/graph/badge.svg)](https://codecov.io/gh/tidyverse/dplyr?branch=master) -->

<!-- [![R build status](https://github.com/tidyverse/dplyr/workflows/R-CMD-check/badge.svg)](https://github.com/tidyverse/dplyr/actions) -->

<!-- badges: end -->

## Overview

**MIDASpy** is a Python class for multiply imputing missing data using MIDAS, a deep learning method based on denoising autoencoder neural networks. The **MIDASpy** algorithm offers significant accuracy and efficiency advantages over other multiple imputation strategies, particularly when applied to large datasets with complex features. In addition to implementing the algorithm, the class contains functions for processing data before and after model training, running imputation model diagnostics, generating multiple completed datasets, and estimating regression models on these datasets.

For an implementation in R, see our **rMIDAS** [repository](https://github.com/MIDASverse/rMIDAS).

## Background on MIDAS

For more information on the MIDAS method, see:

Lall, Ranjit, and Thomas Robinson. 2020. “Applying the MIDAS Touch: How to Handle Missing Values in Large and Complex Data.” APSA Preprints. https://doi.org/10.33774/apsa-2020-3tk40-v3

## Installation

To install via pip, input the following command into the terminal:  
`pip install MIDASpy`

The latest development version (potentially unstable) can be installed
via the terminal with: `pip install
git+https://github.com/MIDASverse/MIDASpy.git`

MIDAS requires:

  - Python (\>=3.5)
  - Numpy (\>=1.5)
  - Pandas (\>=0.19)
  - Tensorflow (\>= 1.10) – **NOTE**: 2.X is not yet supported for
    performing imputations
  - Matplotlib

Tensorflow also has a number of requirements, particularly if GPU
acceleration is desired. See <https://www.tensorflow.org/install/> for
details.

## Example

For a simple demonstration of **MIDASpy**, see our [Jupyter Notebook example](https://github.com/MIDASverse/MIDASpy/blob/master/Examples/midas_demo.ipynb). 

## Version 1.0.2 (September 2020)

*v1.0.2 patches MIDASpy to allow for packaging to PyPi, and updates the
package name.*

Key changes in 1.0:

  - Minor, mainly cosmetic, changes to the underlying source code.
  - Renamed ‘categorical\_columns’ argument in build\_model() to
    ‘binary\_columns’ to avoid confusion
  - Added plotting arguments to overimputation() method to suppress
    intermediary overimputation plots (plot\_main) and all plots
    (skip\_plot).
  - Changed overimputation() plot titles, labels and legends
  - Added tensorflow 2.0 version check on import, returns custom error
  - Fixed seed-setting bug in earlier versions

## Previous versions

*Alpha 0.2:*

Variational autoencoder enabled. More flexibility in model
specification, although defaulting to a simple mirrored system. Deeper
analysis tools within .overimpute() for checking fit on continuous
values. Constructor code deconflicted. Individual output specification
enabled for very large datasets.

Key added features:

  - Variational autoencoder capacity added, including encoding to and
    sampling from latent space

Planned features:

  - Time dependence handling through recurrent cells
  - Improving the pipeline methods for very large datasets
  - Tensorboard integration
  - Dropout scaling
  - A modified constructor that can generate embeddings for better
    interpolation of features
  - R support

Wish list:

  - Smoothing for time series (LOESS?)
  - Informative priors?

*Alpha 0.1:*

  - Basic functionality feature-complete.
  - Support for mixed categorical and continuous data types
  - An “additional data” pipeline, allowing data that may be relevant to
    the imputation to be included (without being included in error
    generating statistics)
  - Simplified calibration for model complexity through the
    “overimputation” function, including visualization of
    reconstructed features
  - Basic large dataset functionality

**NOTE**: An earlier version of the software is stored
[here](https://github.com/Oracen/MIDAS).
