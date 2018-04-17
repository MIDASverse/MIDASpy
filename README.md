# MIDAS - Multiple Imputation with Denoising Autoencoders

Strategies for analyzing missing data have become increasingly sophisticated in recent years, most notably with the growing popularity of best-practice multiple imputation (MI) methods. However, existing algorithms for implementing MI tend to suffer from limited efficiency and scalability, rendering them poorly suited to large datasets. MIDAS draws on recent advances in deep learning to deliver a fast, scalable, and high-performance implementation of MI. MIDAS employs a class of unsupervised neural networks known as denoising autoencoders, which can produce complex and robust reconstructions of partially corrupted inputs. To enhance their accuracy and efficiency while preserving their robustness, these networks are trained with the technique of Monte Carlo dropout, which is equivalent to approximate Bayesian inference in deep Gaussian processes. Preliminary tests indicate that, in addition to successfully handling larger datasets than existing MI algorithms, MIDAS generates more accurate and precise imputed values in ordinary statistical applications.


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



