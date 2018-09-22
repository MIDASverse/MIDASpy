# MIDAS - Multiple Imputation with Denoising Autoencoders

MIDAS draws on recent advances in deep learning to deliver a fast, scalable, and high-performance solution for multiply imputing missing data. MIDAS employs a class of unsupervised neural networks known as denoising autoencoders, which are capable of producing complex yet robust reconstructions of partially corrupted inputs. To enhance their efficiency and accuracy while preserving their robustness, these networks are trained with the recently developed technique of Monte Carlo dropout, which is mathematically equivalent to approximate Bayesian inference in deep Gaussian processes. Preliminary tests indicate that, in addition to handling larger datasets than existing multiple imputation algorithms, MIDAS generates more accurate and precise imputed values in ordinary  applications.


Installation
------------

To install via pip, input the following command into the terminal:  
`pip install git+https://github.com/Oracen/MIDAS.git`


MIDAS requires
- Python (>=3.5, 2.X coming)
- Numpy (>=1.5)
- Pandas (>=0.19)
- Tensorflow (>= 1.10)
- Matplotlib

Tensorflow also has a number of requirements, particularly if GPU acceleration is desired. See https://www.tensorflow.org/install/ for details.


ALPHA 0.2
---------
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

Previous versions
-----------------

Alpha 0.1:

Basic functionality feature-complete.
- Support for mixed categorical and continuous data types
- An "additional data" pipeline, allowing data that may be relevant to the imputation to be included (without being included in error generating statistics)
- Simplified calibration for model complexity through the "overimputation" function, including visualization of reconstructed features
- Basic large dataset functionality





