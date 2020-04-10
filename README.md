# MIDAS (Multiple Imputation with Denoising Autoencoders)

MIDAS is a Python class for multiply imputing missing values based on neural network methods that is particularly well suited to large and complex data. The software implements a new approach to multiple imputation that involves introducing an additional portion of missingness into the dataset, attempting to reconstruct this portion with a type of unsupervised neural network known as a denoising autoencoder, and using the resulting model to draw imputations of originally missing values. These steps are implemented with a fast, scalable, and flexible algorithm that expands both the quantity and the range of data that can be analyzed with multiple imputation. To help users optimize the algorithm for their specific application, MIDAS offers a variety of user-friendly tools for calibrating and validating the imputation model.


Installation
------------

To install via pip, input the following command into the terminal:  
`pip install git+https://github.com/ranjitlall/MIDAS.git`


MIDAS requires
- Python (>=3.5)
- Numpy (>=1.5)
- Pandas (>=0.19)
- Tensorflow (>= 1.10; 2.X coming)
- Matplotlib

Tensorflow also has a number of requirements, particularly if GPU acceleration is desired. See https://www.tensorflow.org/install/ for details.

Version 1.0 (April 2020)
---------
Minor, mainly cosmetic, changes to the underlying source code.

Key changes
- Renamed 'categorical_columns' argument in build_model() to 'binary_columns' to avoid confusion
- Added plotting arguments to overimputation() method to suppress intermediary overimputation plots (plot_main) and all plots (skip_plot).
- Changed overimputation() plot titles, labels and legends
- Added tensorflow 2.0 version check on import, returns custom error
- Fixed seed-setting bug in earlier versions


Previous versions
-----------------

*Alpha 0.2:*

Variational autoencoder enabled. More flexibility in model specification, although defaulting to a simple mirrored system. Deeper analysis tools within .overimpute() for checking fit on continuous values. Constructor code deconflicted. Individual output specification enabled for very large datasets.

Key added features
- Variational autoencoder capacity added, including encoding to and sampling from latent space

Planned features:
- Time dependence handling through recurrent cells
- Improving the pipeline methods for very large datasets
- Tensorboard integration
- Dropout scaling
- A modified constructor that can generate embeddings for better interpolation of features
- R support

Wish list:
- Smoothing for time series (LOESS?)
- Informative priors?

*Alpha 0.1:*

Basic functionality feature-complete.
- Support for mixed categorical and continuous data types
- An "additional data" pipeline, allowing data that may be relevant to the imputation to be included (without being included in error generating statistics)
- Simplified calibration for model complexity through the "overimputation" function, including visualization of reconstructed features
- Basic large dataset functionality





