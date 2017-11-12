# AEImputer

Missing data is a widespread problem in machine learning. Bayesian inference is a robust solution to imputing missing values, particularly if multiple imputations are used to model the uncertainty regarding said values. Unfortunately, most existing MI programs are slow due the the sequential nature of the calculatiosn. While some can be parallelised, not all MI programs have this functionality. Additionally, they cannot scale to larger datasets, as the computation time scales geometrically with features.

Neural networks are high performance, scalable machine learning solutions capable of distributed training and inference. When trained with Monte Carlo dropout, they can approximate a Bayesian posterior. Denoising autoencoders represent one implementation of a neural network, which is trained to reconstruct a corrupted input. With some manipulation, they can be coerced to handle missing data. Enter MIDAS. MIDAS allows the most advanced algorithms in machine learning to be applied to the imputation of missing data. 

This project is developed essentially as a learning process. Should you stumble across it, please feel free to contribute. I know I have commited some egregious coding sins (expanding items in a for loop!), but my nonexistent coding education means I simply did the best I could with the tools I had. The better this algorithm is, the more the scholarly/data science community stands to benefit. At the end of the day, that's what this is all about.

