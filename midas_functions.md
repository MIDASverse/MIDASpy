# Guide to the methods and arguments of MIDAS

Model construction first requires an instantiation of MIDAS. The model then needs to be constructed and trained before imputations can be generated. Calibration is optional, but strongly recommeded.

This class doesn't explicitly return values. Values are either stored internally, files are saved remotely or methods yield rather than returning. The key attribute is .output_list when samples are generated.

#### Instantiation:

- Midas()

#### Model construction:

- .build_model()
- .build_model_pipeline()

#### Model calibration:

- .overimpute()

#### Model training:

- .train_model()
- .train_model_pipeline()

#### Imputation generation:

- .batch_generate_samples()
- .batch_yield_samples()
- .generate_samples()
- .yield_samples()
- .yield_samples_pipeline()

---

### Midas()

- layer_structure= \[256, 256, 256\]
- learn_rate= 1e-4
- input_drop= 0.8
- train_batch = 16
- savepath= 'tmp/MIDAS'
- seed= None
- loss_scale= 1
- init_scale= 1
- softmax_adj= 1

Initialiser. Called separately to 'build_model' to allow for out-of-memory datasets. All key hyperparameters are entered at this stage, as the model construction methods only deal with the dataset.

#### Args:
- **layer_structure:** List of integers. The number of nodes in each layer of the network (default = [256, 256, 256], denoting a three-layer network with 256 nodes per layer). Larger networks can learn more complex data structures but require longer training and are more prone to overfitting.

- **learn_rate:** Float. The learning rate $\gamma$ (default = 0.0001), which controls the size of the weight adjustment in each training epoch. In general, higher values reduce training time at the expense of less accurate results.

- **input_drop:** Float between 0 and 1. The probability of corruption for input columns in training mini-batches (default = 0.8). Higher values increase training time but reduce the risk of overfitting. In our experience, values between 0.7 and 0.95 deliver the best performance.

- **train_batch:** Integer. The number of observations in training mini-batches (default = 16). Common choices are 8, 16, 32, 64, and 128; powers of 2 tend to enhance memory efficiency. In general, smaller sizes lead to faster convergence at the cost of greater noise and thus less accurate estimates of the error gradient. Where memory management is a concern, they should be favored.

- **savepath:** String. The location to which the trained model will be saved.

- **seed:** Integer. The value to which Python's pseudo-random number generator is initialized. This enables users to ensure that data shuffling, weight and bias initialization, and missingness indicator vectors are reproducible.

- **loss_scale:** Float. A constant by which the RMSE loss functions are multiplied (default = 1). This hyperparameter performs a similar function to the learning rate. If loss during training is very large, increasing its value can help to prevent overtraining.

- **init_scale:** Float. The numerator of the variance component of Xavier Initialisation equation (default = 1). In very deep networks, higher values may help to prevent extreme gradients (though this problem is less common with ELU activation functions).

- **softmax_adj:** Float. A constant by which the cross-entropy loss functions are multiplied (default = 1). This hyperparameter is the equivalent of loss_scale for categorical variables. If cross-entropy loss falls at a consistently faster rate than RMSE during training, a lower value may help to redress this imbalance.

- **vae_layer:** Boolean. Specifies whether to include a variational autoencoder layer in the network (default = False), one of the key diagnostic tools included in midas. If set to True, variational autoencoder hyperparameters must be specified via a number of additional arguments.

- **latent_space_size:** Integer. The number of normal dimensions used to parameterize the latent space.

- **vae_sample_var:** Float. The sampling variance of the normal distributions used to parameterize the latent space.

- **vae_alpha:** Float. The strength of the prior imposed on the Kullback-Leibler divergence term in the variational autoencoder loss functions.

- **kld_min:**  Float. The minimum value of the Kullback-Leibler divergence term in the variational autoencoder loss functions.

---

### .build_model()

- imputation_target
- categorical_columns= None
- softmax_columns= None
- unsorted= True
- additional_data = None
- verbose= True

This method is called to construct the neural network that is the heart of MIDAS. This includes the assignment of loss functions to the appropriate data types.

THIS FUNCTION MUST BE CALLED BEFORE ANY TRAINING OR IMPUTATION OCCURS. Failing to do so will simply raise an error.

The categorical columns should be a list of column names. Softmax columns should be a list of lists of column names. This will allow the model to dynamically assign cost functions to the correct variables. If, however, the data comes pre-sorted, 'arranged' can be set to "True", in which case the arguments can be passed in as integers of size, ie. shape[1] attributes for each of the relevant categories.

In other words, if you're experienced at using MIDAS and understand how its indexing works, pre-sort your data and pass in the integers so specifying reindexing values doesn't become too onerous.

Alternatively, list(df.columns.values) will output a list of column names, which can be easily implemented in the 'for' loop which constructs your dummy variables.

#### Args:
- **imputation_target:** DataFrame. The name of the incomplete input dataset. Upon being read in, the dataset will be appropriately formatted and stored for training.

- **binary_columns:** List of names. A list of  all binary variables in the input dataset.

- **softmax_columns:** List of lists. The outer list should include all non-binary categorical variables in the input dataset. Each inner list should contain the mutually exclusive set of possible classes for each of these variables.

- **unsorted:** Boolean. Specifies whether the input dataset has been pre-ordered in terms of variable type (default = True, denoting no sorting). If set to False, binary_columns and softmax_columns should be a list of integers denoting shape attributes for each category.

- **additional_data:** DataFrame. Data that should be included in the imputation model but are not required for later analyses. Such data will not be formatted, rearranged, or included in the loss functions, reducing training time.

- **verbose:** Boolean. Specifies whether to print messages to the terminal (default = True).

---

### .build_model_pipeline()

- data_sample
- categorical_columns= None
- softmax_columns= None
- unsorted= True
- additional_data_sample= None
- verbose= True
- crossentropy_adj= 1
- loss_scale = 1

This function is for integration with databasing or any dataset that needs to be batched into memory. The data sample is simply there to allow the original constructor to be recycled. The head of the data should be sufficient to build the imputation model. The input pipeline itself should pre-scale the data, and code null values as type np.nan. The pipeline ought to output a Pandas DataFrame. If additional data will be passed in, then the return must be a list of two DataFrames. The columns of the dataframe will be re-arranged so that error functions are efficiently generated.

IT IS IMPERATIVE that this ordering is respected. Design the input batching function accordingly.

The categorical columns should be a list of column names. Softmax columns should be a list of lists of column names. This will allow the model to dynamically assign cost functions to the correct variables. If, however, the data comes pre-sorted, arranged can be set to "true", in which case the arguments can be passed in as integers of size, ie. shape[1] attributes for each of the relevant categories.

In other words, pre-sort your data and pass in the integers, so indexing dynamically doesn't become too difficult. Alternatively, list(df.columns.values) will output a list of column names, which can be easily implemented in the 'for' loop which constructs your dummy variables.

#### Args:
- **data_sample:** DataFrame. The head of the data that will be fed in via a batching pipeline. This sample is just used to enforce indexing and to allow code recyling.

- **categorical_columns:** List of names. Specifies the binary (ie. non-exclusive categories) to be imputed. If unsorted = False, this value can be an integer

- **softmax_columns:** List of lists. Every inner list should contain column names. Each inner list should represent a set of mutually exclusive categories, such as current day of the week. if unsorted = False, this should be a list of integers.

- **unsorted:** Boolean. Specifies to MIDAS that data has been pre-sorted, and indices can simply be appended to the size index.

- **additional_data:** DataFrame. Any data that shoud be included in the imputation model, but is not required from the output. By passing data here, the data will neither be rearranged nor will it generate a cost function. This reduces the regularising effects of multiple loss functions, but reduces both networksize requirements and training time.

- **verbose:** Boolean. Set to False to suppress messages printing to terminal.

---

### .overimpute()

- spikein = 0.1
- training_epochs= 100
- report_ival = 10
- report_samples = 32
- plot_all= True
- verbose= True
- verbosity_ival= 1
- spike_seed= 42
- excessive= False

This function spikes in additional missingness, so that known values can be used to help adjust the complexity of the model. As conventional train/validation splits can still lead to autoencoders overtraining, the method for limiting complexity is overimputation and early stopping. This gives an estimate of how the model will react to unseen variables.

Error is defined as RMSE for continuous variables, and classification error for binary and categorical variables (ie. 1 - accuracy). Note that this means that binary classification is inherently dependent on a selection threshold of 0.5, and softmax accuracy will naturally decrease as a function of the number of classes within the model. All three will be affected by the degree of imbalance within the dataset.

The accuracy measures provided here may not be ideal for all problems, but they are generally appropriate for selecting optimum complexity. Should the lines denoting error begin to trend upwards, this indicates overtraining and is a sign that the training_epochs parameter to the .train_model() method should be capped before this point.

The actual optimal point may differ from that indicated by the .overimpute() method for two reasons:
- The loss that is spiked in reduces the overall data available to the algorithm to learn the patterns inherent, so there should be some improvement in performance when .train_model() is called. If this is a concern, then it should be possible to compare the behaviour of the loss figure between .train_model() and .overimpute().
- The missingness inherent to the data may depend on some unobserved factor.
In this case, the bias in the observed data may lead to inaccurate inference.

It is worth visually inspecting the distribution of the overimputed values against imputed values (using plot_all) to ensure that they fall within a sensible range.

#### Args:

- **spikein:** Float, between 0 and 1. The proportion of observed values in the input dataset to be randomly removed (default = 0.1).

- **training_epochs:** Integer. The number of overimputation training epochs (default = 100). Selecting a low value increases the risk that trends in the loss metrics have not stabilized by the end of training, in which case additional epochs may be necessary.

- **report_ival:** Integer. The number of overimputation training epochs between calculations of loss (default = 10). Shorter intervals provide a more granular view of model performance but slow down the overimputation process.

- **report_samples:** The number of Monte Carlo samples drawn from the estimated missing-data posterior for loss calculations (default = 32). A larger number increases overimputation runtime and may thus necessitate a lower value of report_ival.

- **plot_vars:** Boolean. Specifies whether to plot the distribution of original versus overimputed values (default = True). This takes the form of a density plot for continuous variables and a barplot for categorical variables (showing proportions of each class).

- **plot_main:** Boolean. Specifies whether to display the main graphical output (overimputation error during training) at every reporting interval (default = True). If set to False, it will only appear at the end of the overimputation training process. Error values are still shown at each report_ival.

- **skip_plot:** Boolean. Specifies whether to suppress the main graphical output (default = False). This may be desirable when users are conducting multiple overimputation exercises sequentially and are primarily interested in the console output.

- **verbose:** Boolean. Prints out messages, including loss, to the terminal (default = True).

- **verbosity_ival:** Integer. The number of overimputation training epochs between messages (default = True).

- **spike_seed:** Integer. The value to which Python's pseudo-random number generator is initialized for the missingness spike-in. This is separate to the seed specified in the Midas() call.

- **excessive:** Boolean. Specifies whether to print aggregate mini-batch loss to the terminal (default = False). This argument differs from the .train\_model()'s excessive argument, which prints individual mini-batch loss. This allows users to check for unusual imputations, which may be helpful if loss is not declining during overimputation training.

---

### .train_model()

- training_epochs= 100
- verbose= True
- verbosity_ival= 1
- excessive= False
                  
This is the standard method for optimising the model's parameters. Must be called before imputation can be performed. The model is automatically saved upon conclusion of training

#### Args:

- **training_epochs:** Integer. The number of complete cycles (forward passes) through the network during training (default = 100).

- **verbose:** Boolean. Specifies whether to print messages to the terminal during training, including loss values (default = True).

- **verbosity_ival:** Integer. The number of training epochs between messages (default = 1).

- **excessive:** Boolean. Specifies whether to print loss for each mini-batch to the terminal (default = False), which can help with troubleshooting.

---

### .train_model_pipeline()

- input_pipeline
- training_epochs= 100
- verbose= True
- verbosity_ival= 1
- excessive= False

This is the alternative method for optimising the model's parameters when input data must be batched into memory. Must be called before imputation can be performed. The model will then be saved to the specified directory.

#### Args:
      
- **input_pipeline:** Function which yields a pre-processed and scaled DataFrame from the designated source, be it a server or large flat file.

- **training_epochs:** Integer. The number of epochs the model will run for

- **verbose:** Boolean. Prints out messages, including loss

- **verbosity_ival:** Integer. This number determines the interval between messages.

- **excessive:** Boolean. Used for troubleshooting, this argument will cause the cost of each minibatch to be printed to the terminal.

----

### .batch_generate_samples()

- m= 50
- b_size= 256
- verbose= True
 
Method used to generate a set of m imputations to the .output_list attribute. Imputations are stored within a list in memory, and can be accessed in any order. As batch generation implies very large datasets, this method is only provided for internal troubleshooting.

This function is for a dataset large enough to be stored in memory, but too large to be passed into the model in its entirety. This may be due to GPU memory limitations, or just the size of the model

If a model has been pre-trained, on subsequent runs this function can be directly called without having to train first. An 'if' statement checking the default save location is useful for this.

#### Args:
- **m:** Integer. Number of imputations to generate.

- **b_size:** Integer. Number of data entries to process at once. For managing wider datasets, smaller numbers may be required.

- **verbose:** Boolean. Prints out messages.

---

### .batch_yield_samples()

- m= 50
- b_size= 256
- verbose= True

Method used to generate a set of m imputations via the 'yield' command, allowing imputations to be used in a 'for' loop'

This function is for a dataset large enough to be stored in memory, but too large to be passed into the model in its entirety. This may be due to GPU memory limitations, or just the size of the model or dataset.

If a model has been pre-trained, on subsequent runs this function can be directly called without having to train first. An 'if' statement checking the default save location is useful for this.

#### Args:
- **m:** Integer. Number of imputations to generate.

- **b_size:** Integer. Number of data entries to process at once. For managing wider datasets, smaller numbers may be required.

- **verbose:** Boolean. Prints out messages.

---

### .generate_samples()

- m= 50
- verbose= True

Method used to generate a set of m imputations to the .output_list attribute. Imputations are stored within a list in memory, and can be accessed in any order.

If a model has been pre-trained, on subsequent runs this function can be directly called without having to train first. An 'if' statement checking the default save location is useful for this.

#### Args:
- **m:** Integer. The number of completed datasets to produce (default = 50)

- **verbose:** Boolean. Specifies whether to print messages to the terminal (default = True).

---

### .yield_samples()

- m= 50
- verbose= True

Method used to generate a set of m imputations via the 'yield' command, allowing imputations to be used in a 'for' loop.

If a model has been pre-trained, on subsequent runs this function can be directly called without having to train first. An 'if' statement checking the default save location is useful for this.

#### Args:

- **m:** Integer. Number of imputations to generate.

- **verbose:** Boolean. Prints out messages.

---

### .yield_samples_pipeline()

- verbose= False

As it's impossible to know the specifics of the pipeline, this method simply cycles through all data provided by the input function. The number of imputations can be specified by the user, depending on their needs. The size of the output DataFrame depends on the size specified by the input function that was passed to 'train_model_pipeline'.

#### Args:

- **verbose: Prints out messages

