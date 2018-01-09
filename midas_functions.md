# Guide to the methods and arguments of MIDAS

Model construction first requires an instantiation of MIDAS. The model then needs to be constructed and trained before imputations can be generated. Calibration is optional, but strongly recommeded.

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

layer_structure= [256, 256, 256]
learn_rate= 1e-4
input_drop= 0.8
train_batch = 16
savepath= 'tmp/MIDAS'
seed= None
loss_scale= 1
init_scale= 1
softmax_adj= 1

Initialiser. Called separately to 'build_model' to allow for out-of-memory datasets. All key hyperparameters are entered at this stage, as the model construction methods only deal with the dataset.

#### Args:
- **layer_structure:** List of integers. Specifies the pattern by which the model contstruction methods will instantiate a neural network. For reference, the default layer structure is a three-layer network, with each layer containing 256 units. Larger networks can learn more complex representations of data, but also require longer and longer training times. Due to the large degree of regularisation used by MIDAS, making a model "too big" is less of a problem than making it "too small". If training time is relatively quick, but you want improved performance, try increasing the size of the layers or, as an alternative, add more layers. (More layers generally corresponds to learning more complex relationships.) As a rule of thumb, I keep the layers to powers of two - not only does this narrow my range of potential size values to search through, but apparently also helps with network assignment to memory.

- **learn_rate:** Float. This specifies the default learning rate behaviour for the training stage. In general, larger numbers will give faster training, smaller numbers more accurate results. If a cost is exploding (ie. increasing rather than decreasing), then the first solution tried ought to be reducing the learning rate.

- **input_drop:** Float between 0 and 1. The 'keep' probability of each input column per training batch. A higher value will allow more data into MIDAS per draw, while a lower number seems to render the aggregated posterior more robust to bias from the data. (This effect will require further investigation, but the central tendency of the posterior seems to fall closer to the true value.) Empirically, a number between 0.7 and 0.95 works best. Numbers close to 1 reduce or eliminate the regularising benefits of input noise, as well as converting the model to a simple neural network regression.

- **train_batch:** Integer. The batch size of each training pass. Larger batches mean more stable loss, as biased sampling is less likely to present an issue. However, the noise generated from smaller batches, as well as the greater number of training updates per epoch, allows the algorithm to converge to better optima. Research suggests the ideal number lies between 8 and 512 observations per batch. I've found that 16 seems to be ideal, and would only consider reducing it on enormous datasets where memory management is a concern.

- **savepath:** String. Specifies the location to which Tensorflow will save the trained model.

- **seed:** Integer. Initialises the pseudorandom number generator to a set value. Important if you want your results to be reproducible.

- **loss_scale:** Float. Instantiates a constant to multiply the loss function by. If there are a large number of losses, this is a method that can be used to attempt to prevent overtraining while also allowing for a larger learning rate. With general SGD, this would be equivalent to a modifier of the learn rate, but this interacts differently with AdaM due to its adaptive learn rate. Useful only in some circumstances.

- **init_scale:** Float. MIDAS is initialised with a variant of Xavier initialisation, where a numerator of 1 is used instead of a 6. For deeper networks, larger values might be useful to prevent dying gradients - although with ELU activations, this is less of a concern. Can be reduced if early training is characterised by exploding gradients.

- **softmax_adj:** Float. As categorical clusters each require its own softmax cost function, they quickly outnumber the other variables. Should continuous or binary variables not train well, while softmax error is smoothly decreasing, try using a number less than 1 to scale the loss of the softmaxes down. A useful rule of thumb seems to be 1/(number of softmaxes) to use the averaged softmax loss.

---

### .build_model()

imputation_target
categorical_columns= None
softmax_columns= None
unsorted= True
additional_data = None
verbose= True

This method is called to construct the neural network that is the heart of MIDAS. This includes the assignment of loss functions to the appropriate data types.

THIS FUNCTION MUST BE CALLED BEFORE ANY TRAINING OR IMPUTATION OCCURS. Failing to do so will simply raise an error.

The categorical columns should be a list of column names. Softmax columns should be a list of lists of column names. This will allow the model to dynamically assign cost functions to the correct variables. If, however, the data comes pre-sorted, 'arranged' can be set to "True", in which case the arguments can be passed in as integers of size, ie. shape[1] attributes for each of the relevant categories.

In other words, if you're experienced at using MIDAS and understand how its indexing works, pre-sort your data and pass in the integers so specifying reindexing values doesn't become too onerous.

Alternatively, list(df.columns.values) will output a list of column names, which can be easily implemented in the 'for' loop which constructs your dummy variables.

#### Args:
- **imputation_target:** DataFrame. Any data specified here will be rearranged and stored for the subsequent imputation process. The data must be preprocessed before it is passed to build_model.

- **categorical_columns:** List of names. Specifies the binary (ie. non-exclusive categories) to be imputed. If unsorted = False, this value can be an integer

- **softmax_columns:** List of lists. Every inner list should contain column names. Each inner list should represent a set of mutually exclusive categories, such as current day of the week. if unsorted = False, this should be a list of integers.

- **unsorted:** Boolean. Specifies to MIDAS that data has been pre-sorted, and indices can simply be appended to the size index.

- **additional_data:** DataFrame. Any data that shoud be included in the imputation model, but is not required from the output. By passing data here, the data will neither be rearranged nor will it generate a cost function. This reduces the regularising effects of multiple loss functions, but reduces both networksize requirements and training time.

- **verbose:** Boolean. Set to False to suppress messages printing to terminal.

---

### .build_model_pipeline()

data_sample
categorical_columns= None
softmax_columns= None
unsorted= True
additional_data_sample= None
verbose= True
crossentropy_adj= 1
loss_scale = 1

This function is for integration with databasing or any dataset that needs to be batched into memory. The data sample is simply there to allow the original constructor to be recycled. The head of the data should be sufficient to build the imputation model. The input pipeline itself should pre-scale the data, and code null values as type np.nan. The pipeline ought to output a Pandas DataFrame. If additional data will be passed in, then the return must be a list of two DataFrames. The columns of the dataframe will be re-arranged so that error functions are efficiently generated.

IT IS IMPERITIVE that this ordering is respected. Design the input batching function accordingly.

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

spikein = 0.1
training_epochs= 100
report_ival = 10
report_samples = 32
plot_all= True
verbose= True
verbosity_ival= 1
spike_seed= 42
excessive= False

This function spikes in additional missingness, so that known values can be used to help adjust the complexity of the model. As conventional train/validation splits can still lead to autoencoders overtraining, the method for limiting complexity is overimputation and early stopping. This gives an estimate of how the model will react to unseen variables.

Error is defined as RMSE for continuous variables, and classification error for binary and categorical variables (ie. 1 - accuracy). Note that this means that binary classification is inherently dependent on a selection threshold of 0.5, and softmax accuracy will naturally decrease as a function of the number of classes within the model. All three will be affected by the degree of imbalance within the dataset.

The accuracy measures provided here may not be ideal for all problems, but they are generally appropriate for selecting optimum complexity. Should the lines denoting error begin to trend upwards, this indicates overtraining and is a sign that the training_epochs parameter to the .train_model() method should be capped before this point.

The actual optimal point may differ from that indicated by the .overimpute() method for two reasons:
- The loss that is spiked in reduces the overall data available to the algorithm to learn the patterns inherent, so there should be some improvement in performance when .train_model() is called. If this is a concern, then it should be possible to compare the behaviour of the loss figure between .train_model() and .overimpute().
- The missingness inherent to the data may depend on some unobserved factor.
In this case, the bias in the observed data may lead to inaccurate inference.

It is worth visually inspecting the distribution of the overimputed values against imputed values (using plot_all) to ensure that they fall within a sensible range.

#### Args:

- **spikein:** Float, between 0 and 1. The proportion of total values to remove from the dataset at random. As this is a random selection, the sample should be representative. It should also equally capture known and missing values, therefore this sample represents the percentage of known data to remove. If concerns about sampling remain, adjusting this number or changing the seed can allow for validation. Larger numbers mean greater amounts of removed data, which may mean estimates of optimal training time might be skewed. This can be resolved by lowering the learning rate and aiming for a window.

- **training_epochs:** Integer. Specifies the number of epochs model should be trained for. It is often worth specifying longer than expected to ensure that the model does not overtrain, or that another, better, optimum exists given slightly longer training time.

- **report_ival:** Integer. The interval between sampling from the posterior of the model. Smaller intervals mean a more granular view of convergence, but also drastically slow training time.

- **report_samples:** The number of Monte-Carlo samples drawn for each check of the posterior at report_ival. Greater numbers of samples means a longer runtime for overimputation. For low numbers of samples, the impact will be reduced, though for large numbers of Monte-Carlo samples, report_ival will need to be adjusted accordingly. I recommend a number between 5 and 25, depending on the complexity of the data.

- **plot_all:** Generates plots of the distribution of spiked in values v. the mean of the imputations. Continuous values have a density plot, categorical values a bar plot representing proportions. Only the mean is plotted at this point for simplicity's sake.

- **verbose:** Boolean. Prints out messages, including loss

- **verbosity_ival:** Integer. This number determines the interval between messages.

- **spike_seed:** A different seed, separate to the one used in the main call, used to initialise the RNG for the missingness spike-in.

- **excessive:** Unlike .train_model()'s excessive arg, this argument prints the entire batch output to screen. This allows for inspection for unusual values appearing, useful if the model's accuracy will not reduce.

---

