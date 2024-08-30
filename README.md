
# MIDASpy<img src='https://raw.githubusercontent.com/MIDASverse/MIDASpy/master/MIDASpy_logo.png' align="right" height="139" /></a>

[![PyPI Latest Release](https://img.shields.io/pypi/v/midaspy.svg)](https://pypi.org/project/midaspy/)
[![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/midaspy/)
[![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)](https://lifecycle.r-lib.org/articles/stages.html)
[![CI Linux](https://github.com/edvinskis/MIDASpy/actions/workflows/testlinux.yml/badge.svg)](https://github.com/edvinskis/MIDASpy/actions/workflows/testlinux.yml)
[![CI macOS](https://github.com/edvinskis/MIDASpy/actions/workflows/testmacos.yml/badge.svg)](https://github.com/edvinskis/MIDASpy/actions/workflows/testmacos.yml)
[![CI Windows](https://github.com/edvinskis/MIDASpy/actions/workflows/testwindows.yml/badge.svg)](https://github.com/edvinskis/MIDASpy/actions/workflows/testwindows.yml)

## Overview

**MIDASpy** is a Python package for multiply imputing missing data using
deep learning methods. The **MIDASpy** algorithm offers significant
accuracy and efficiency advantages over other multiple imputation
strategies, particularly when applied to large datasets with complex
features. In addition to implementing the algorithm, the package contains
functions for processing data before and after model training, running
imputation model diagnostics, generating multiple completed datasets,
and estimating regression models on these datasets.

For an implementation in R, see our **rMIDAS** repository
[here](https://github.com/MIDASverse/rMIDAS).

## Background and suggested citations

For more information on MIDAS, the method underlying the software, see:

Lall, Ranjit, and Thomas Robinson. 2022. "The MIDAS Touch: Accurate and Scalable Missing-Data Imputation with Deep Learning." _Political Analysis_ 30, no. 2: 179-196. doi:10.1017/pan.2020.49. [Published version](https://ranjitlall.github.io/assets/pdf/Lall%20and%20Robinson%202022%20PA.pdf). [Accepted version](http://eprints.lse.ac.uk/108170/1/Lall_Robinson_PA_Forthcoming.pdf).

Lall, Ranjit, and Thomas Robinson. 2023. "Efficient Multiple Imputation for Diverse Data in Python and R: MIDASpy and rMIDAS." _Journal of Statistical Software_ 107, no. 9: 1-38. doi:10.18637/jss.v107.i09. [Published version](https://ranjitlall.github.io/assets/pdf/Lall%20and%20Robinson%202023%20JSS.pdf).

## Installation

To install via pip, enter the following command into the terminal:  
`pip install MIDASpy`

The latest development version (potentially unstable) can be installed
via the terminal with:  
`pip install git+https://github.com/MIDASverse/MIDASpy.git`

MIDAS requires:

  - Python (>=3.6; <3.11)
  - Numpy (>=1.5)
  - Pandas (>=0.19)
  - TensorFlow (<2.12)
  - Matplotlib
  - Statmodels
  - Scipy
  - TensorFlow Addons (<0.20)

Tensorflow also has a number of requirements, particularly if GPU acceleration is desired. See <https://www.tensorflow.org/install/> for details.

## Examples

For a simple demonstration of **MIDASpy**, see our Jupyter Notebook
[examples](https://github.com/MIDASverse/MIDASpy/blob/master/Examples/).

## Contributing to MIDASpy

Interested in contributing to **MIDASpy**? We are looking to hire a research assistant to work part-time (flexibly) to help us build out new features and integrate our software with existing machine learning pipelines. You would be paid the standard research assistant rate at the University of Oxford. To apply, please send your CV (or a summary of relevant skills/experience) to ranjit.lall@sjc.ox.ac.uk.


## Version 1.4.0 (August 2024)

- Adds support for non-negative output columns, with a `positive_columns` argument


## Version 1.3.1 (October 2023)

- Minor update to reflect publication of accompanying article in Journal of Statistical Software
- Further updates to make documentation and URLs consistent, including removing unused metadata

## Version 1.2.4 (August 2023)

- Adds support for Python 3.9 and 3.10
- Addresses deprecation warnings and other minor bug fixes
- Resolves dependency issues and includes an updated `setup.py` file
- Adds GitHub Actions workflows that trigger automatic tests on the latest Ubuntu, macOS, and Windows for Python versions 3.7 to 3.10 each time a push or pull request is made to the main branch
- An additional Jupyter Notebook example that demonstrates the core functionalities of **MIDASpy**

## Version 1.2.3 (December 2022)

*v1.2.3 adds support for installation on Apple Silicon hardware (i.e. M1 and M2 Macs).*

## Version 1.2.2 (July 2022)

*v1.2.2 makes minor efficiency changes to the codebase. Full details are available in the Release logs.*

## Version 1.2.1 (January 2021)

*v1.2.1 adds new pre-processing functionality and a multiple imputation regression function.*

Users can now automatically preprocess binary and categorical columns prior to running the MIDAS algorithm using `binary_conv()` and `cat_conv()`.

The new `combine()` function allows users to run regression analysis across the complete data, following Rubin’s combination rules.

## Previous versions

*Version 1.1.1 (October 2020)*

Key changes:

  - Update adds **full Tensorflow 2.X support**:

    - Users can now run the MIDAS algorithm in TensorFlow 2.X (TF1 support
    retained)

    - Tidier handling of random seed setting across both TensorFlow and
    NumPy
    
  - Fixes a minor dependency bug
  
  - Other minor bug fixes

*Version 1.0.2 (September 2020)*

Key changes:

  - Minor, mainly cosmetic, changes to the underlying source code.
  - Renamed ‘categorical\_columns’ argument in build\_model() to
    ‘binary\_columns’ to avoid confusion
  - Added plotting arguments to overimputation() method to suppress
    intermediary overimputation plots (plot\_main) and all plots
    (skip\_plot).
  - Changed overimputation() plot titles, labels and legends
  - Added tensorflow 2.0 version check on import
  - Fixed seed-setting bug in earlier versions

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
