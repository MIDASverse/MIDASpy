# AEImputer

Missing data is a widespread problem in machine learning. Bayesian inference is a robust solution to imputing missing values, particularly if multiple imputations are used to model the uncertainty regarding said values. Unfortunately, most existing MI programs are slow due the the sequential nature of the calculations. While some can be parallelised, not all MI programs have this functionality. Additionally, EM-based MI cannot scale to larger datasets, as the computation time scales geometrically with number of features.

By comparison, neural networks are high performance, scalable machine learning solutions capable of distributed training and inference. When trained with Monte Carlo dropout, a neural network is capable of approximate Bayesian inference while being trained in linear time. Denoising autoencoders represent one implementation of a neural network, which is trained to reconstruct a corrupted input. With some manipulation, they can be coerced to handle missing data. This is why MIDAS was created. MIDAS allows the most advanced algorithms in machine learning to be applied to the imputation of missing data, leveraging the principle of Bayesian nonparametrics to minimise bias and inaccuracy in imputed values.

ALPHA 0.1
Basic functionality feature-complete. Additional features will be added, but MIDAS is ready for most users. Next release TBD.


Current features:

  -Support for categorical and continuous data types
  
  -An "additional data" pipeline, allowing data which may be relevant to the imputation to be included, without being included in error       generating statistics
  
  -Simplified calibration for model complexity through the "overimputation" function
 
Planned features:

  -Time dependence handling through recurrent cells
  
  -API for SQL/Hadoop queries for big data pipelines
  
  -Tensorboard integration
  
  -Dropout scaling
  
  -A modified constructor that can generate embeddings for better interpolation of features
 
Wish list:

  -Smoothing for time series (LOESS?)
  
  -Informative priors?


# A brief disclaimer:

This project is developed essentially as a learning process. I have no formal coding training, so this has been quite the learning curve. By which of course I mean I was flying by the seat of my pants, essentially learning by solving problems. This is not to forestall criticism, but rather to invite it. Should you stumble across it, please feel free to contribute or highlight areas that need improvement. I know I have probably commited some egregious coding sins (for loops!), but my nonexistent coding education means I simply did the best I could with the tools I had. The better this algorithm is, the more the scholarly/data science community stands to benefit - which is the ultimate goal. Therefore, the more criticism and advice, the better.

