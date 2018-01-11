# MIDAS - Multiple Imputation with Denoising Autoencoders

Missing data is a widespread problem in machine learning. Bayesian inference is a robust solution to imputing missing values, particularly if multiple imputations are used to model the uncertainty regarding said values. Unfortunately, most existing MI programs are slow due the the sequential nature of the calculations. While some can be parallelised, not all MI programs have this functionality. Generally, these programs scale in geometric time with both size of dataset and missingness.

When trained with Monte Carlo dropout, a neural network is capable of approximate Bayesian inference (specifically, VI) while scaling in linear time relative to dataset, and regardless of missingness. Compared to existing methods, neural network-based MI is a high performance, scalable machine learning solutions capable of distributed training and inference. Denoising autoencoders represent one implementation of a neural network, which is trained to reconstruct a corrupted input. With some manipulation, they can be coerced to handle missing data. This is why MIDAS was created. MIDAS allows the most advanced algorithms in machine learning to be applied to the imputation of missing data, leveraging the principle of Bayesian nonparametrics to minimise bias and inaccuracy in imputed values.

Installation
------------

MIDAS requires
- Python (>=2.7 or >=3.5)
- Numpy (>=1.8.2)
- Pandas (>=0.19.2)
- Tensorflow (>= 1.3)
- Matplotlib

Tensorflow also has a number of requirements, particularly if GPU acceleration is desired. See https://www.tensorflow.org/install/ for details.

Currently, installation via pip/conda is not supported. For this early period, simply download the MIDAS.py script into the project working directory and call from there.


ALPHA 0.1
---------

Basic functionality feature-complete. Additional features will be added, but MIDAS is ready for most users. Next release TBD.


Current features:
- Support for mixed categorical and continuous data types
- An "additional data" pipeline, allowing data which may be relevant to the imputation to be included, without being included in error generating statistics
- Simplified calibration for model complexity through the "overimputation" function, including visualisation of reconstructed features
- Basic large dataset functionality
 
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



