# ==============================================================================
#
#   888b     d888 8888888 8888888b.        d8888  .d8888b.  
#   8888b   d8888   888   888  "Y88b      d88888 d88P  Y88b 
#   88888b.d88888   888   888    888     d88P888 Y88b.      
#   888Y88888P888   888   888    888    d88P 888  "Y888b.   
#   888 Y888P 888   888   888    888   d88P  888     "Y88b. 
#   888  Y8P  888   888   888    888  d88P   888       "888 
#   888   "   888   888   888  .d88P d8888888888 Y88b  d88P 
#   888       888 8888888 8888888P" d88P     888  "Y8888P"  
#
# --- Multiple Imputation using Denoising Autoencoders
# Copyright 2020 Ranjit Lall, Alex Stenlake and Thomas Robinson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse

class tfVersionError(Exception):
  pass

if tf.__version__[0] == '2':
  raise tfVersionError("midas v1.0 is currently only compatible with TensorFlow 1.X") 
elif tf.__version__[0] == '1':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Midas(object):
  """
  Welcome, and thank you for downloading your new script! Thank you for choosing
  MIDAS, the missing-data solution of the present - today!

  Japes aside, a few points to note.

  For now, all input must take the form of a Pandas DataFrame. Pandas is the only
  library with the kind of flexible indexing that I required. While it might not
  be as fast as a pure Numpy-based solution, it is still quite fast - and allows
  for mask-based indexing and reindexing.

  All categorical variables have to be converted to onehot before the algorithm
  can process anything. I might integrate this process into the workflow later,
  should pandas get a reverse_dummies function that would allow the reverse
  transform. This will also change when I implement embedding for sparse variables.

  Neural networks are unpredictable models, where things as small as scaling method
  may have major implications for inference. Try to scale to between 0 and 1, or
  between -1 and 1, as this will help the weight updates be more gradual (and thus
  the learned representation will be more accurate). By the same token, feel
  free to experiment.

  The general form of a call to MIDAS takes the following form:

    imputer.build_model(data)
    imputer.generate_samples()
    for dataset in imputer.output_list:
      print(dataset)

  Of course, the specifics of how the data are generated and accessed depends on
  the intended application.

  Feedback, contributions and criticism equally welcomed.

  https://github.com/Oracen/MIDAS
  """

  def __init__(self,
               layer_structure= [256, 256, 256],
               learn_rate= 1e-4,
               input_drop= 0.8,
               train_batch = 16,
               savepath= 'tmp/MIDAS',
               seed= None,
               output_layers= 'reversed',
               loss_scale= 1,
               init_scale= 1,
               vae_layer= False,
               individual_outputs= False,
               manual_outputs= False,
               output_structure= [16, 16, 32],
               latent_space_size = 4,
               cont_adj= 1.0,
               binary_adj= 1.0,
               softmax_adj= 1.0,
               dropout_level = 0.5,
               weight_decay = 'default',
               vae_alpha = 1.0,
               act = tf.nn.elu,
               vae_sample_var = 1.0,
               noise_type = 'bernoulli',
               kld_min = 0.01
               ):
    """
    Initialiser. Called separately to 'build_model' to allow for out-of-memory
    datasets. All key hyperparameters are entered at this stage, as the model
    construction methods only deal with the dataset.

    Args:
      layer_structure: List of integers. Specifies the pattern by which the model
      contstruction methods will instantiate a neural network. For reference, the
      default layer structure is a three-layer network, with each layer containing
      256 units. Larger networks can learn more complex representations of data,
      but also require longer and longer training times. Due to the large degree
      of regularisation used by MIDAS, making a model "too big" is less of a problem
      than making it "too small". If training time is relatively quick, but you
      want improved performance, try increasing the size of the layers or, as an
      alternative, add more layers. (More layers generally corresponds to learning
      more complex relationships.) As a rule of thumb, I keep the layers to powers
      of two - not only does this narrow my range of potential size values to
      search through, but apparently also helps with network assignment to memory.

      learn_rate: Float. This specifies the default learning rate behaviour for
      the training stage. In general, larger numbers will give faster training,
      smaller numbers more accurate results. If a cost is exploding (ie. increasing
      rather than decreasing), then the first solution tried ought to be reducing
      the learning rate.

      input_drop: Float between 0 and 1. The 'keep' probability of each input
      column per training batch. A higher value will allow more data into MIDAS
      per draw, while a lower number seems to render the aggregated posterior
      more robust to bias from the data. (This effect will require further
      investigation, but the central tendency of the posterior seems to fall
      closer to the true value.) Empirically, a number between 0.7 and 0.95 works
      best. Numbers close to 1 reduce or eliminate the regularising benefits of
      input noise, as well as converting the model to a simple neural network
      regression.

      train_batch: Integer. The batch size of each training pass. Larger batches
      mean more stable loss, as biased sampling is less likely to present an issue.
      However, the noise generated from smaller batches, as well as the greater
      number of training updates per epoch, allows the algorithm to converge to
      better optima. Research suggests the ideal number lies between 8 and 512
      observations per batch. I've found that 16 seems to be ideal, and would
      only consider reducing it on enormous datasets where memory management is
      a concern.

      savepath: String. Specifies the location to which Tensorflow will save the
      trained model.

      seed: Integer. Initialises the pseudorandom number generator to a set
      value. Important if you want your results to be reproducible.

      loss_scale: Float. Instantiates a constant to multiply the loss function
      by. If there are a large number of losses, this is a method that can be used
      to attempt to prevent overtraining while also allowing for a larger learning
      rate. With general SGD, this would be equivalent to a modifier of the learn
      rate, but this interacts differently with AdaM due to its adaptive learn
      rate. Useful only in some circumstances.

      init_scale: Float. MIDAS is initialised with a variant of Xavier initialisation,
      where a numerator of 1 is used instead of a 6. For deeper networks, larger
      values might be useful to prevent dying gradients - although with ELU
      activations, this is less of a concern. Can be reduced if early training
      is characterised by exploding gradients.

      softmax_adj: Float. As categorical clusters each require its own softmax
      cost function, they quickly outnumber the other variables. Should continuous
      or binary variables not train well, while softmax error is smoothly decreasing,
      try using a number less than 1 to scale the loss of the softmaxes down. A
      useful rule of thumb seems to be 1/(number of softmaxes) to use the averaged
      softmax loss.

    Returns:
      Self



    """
    if type(layer_structure) == list:
      self.layer_structure = layer_structure
    else:
      raise ValueError("Layer structure must be specified within a list")
    if type(output_layers) == list:
      self.output_layers = output_layers

    elif output_layers == 'reversed':
      self.output_layers = layer_structure.copy()
      self.output_layers.reverse()
    else:
      raise ValueError("Please specify correct output layer structure")
    self.learn_rate = learn_rate
    self.input_drop = input_drop
    self.model_built = False
    self.savepath = savepath
    self.model = None
    self.additional_data = None
    self.train_batch = train_batch
    self.seed = seed
    self.input_is_pipeline = False
    self.input_pipeline = None
    self.vae_layer = vae_layer
    self.loss_scale = loss_scale
    self.init_scale = init_scale
    self.individual_outputs = individual_outputs
    self.manual_outputs = manual_outputs
    self.vae_sample_var = vae_sample_var
    self.latent_space_size = latent_space_size
    self.dropout_level = dropout_level
    self.prior_strength = vae_alpha
    self.kld_min = kld_min
    if weight_decay == 'default':
      self.weight_decay = 'default'
    elif type(weight_decay) == float:
      self.weight_decay = weight_decay
    else:
      raise ValueError("Weight decay argument accepts either 'standard' (string) "\
                       "or floating point")


    if type(output_structure) == int:
      self.output_structure = [output_structure]*3
    elif (individual_outputs == True) | (len(output_structure) ==3):
      self.output_structure = output_structure
    else:
      raise TypeError("The output transform assignment must take the form of "\
                      "an integer, a list of three elements (cont, bin, cat), "\
                      "or individual values must be specified.")
    self.cont_adj = cont_adj
    self.binary_adj = binary_adj
    self.softmax_adj = softmax_adj
    self.act = act
    self.noise_type = noise_type

  def _batch_iter(self,
                  train_data,
                  na_mask,
                  b_size = 16):
    """
    Function for handling the batch feeds for training loops
    """
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, train_data.shape[0] - b_size + 1, b_size):
      excerpt = indices[start_idx:start_idx + b_size]
      if self.additional_data is None:
        yield train_data[excerpt], na_mask[excerpt]
      else:
        yield train_data[excerpt], na_mask[excerpt], self.additional_data.values[excerpt]

  def _batch_iter_output(self,
                  train_data,
                  b_size = 256):
    """
    Identical to _batch_iter(), although designed for a single datasource
    """

    indices = np.arange(train_data.shape[0])
    for start_idx in range(0, train_data.shape[0], b_size):
      excerpt = indices[start_idx:start_idx + b_size]
      if self.additional_data is None:
        yield train_data[excerpt]
      else:
        yield train_data[excerpt], self.additional_data.values[excerpt]
  def _batch_iter_zsample(self,
                          data,
                          b_size = 256):
    """
    Identical to _batch_iter(), although designed for sampling from latent
    """
    indices = np.arange(data.shape[0])
    for start_idx in range(0, data.shape[0], b_size):
      excerpt = indices[start_idx:start_idx + b_size]
      yield data[excerpt]

  def _build_layer(self,
                   X,
                   weight_matrix,
                   bias_vec,
                   dropout_rate = 0.5,
                   output_layer= False):
    """
    Constructs layers for the build function
    """
    X_tx = tf.matmul(tf.nn.dropout(X, dropout_rate), weight_matrix) + bias_vec
    if output_layer:
      return X_tx
    else:
      return self.act(X_tx)

  def _build_variables(self,
                       weights,
                       biases,
                       num_in,
                       num_out,
                       scale= 1):
    """
    Custom initialiser for a weights, using a variation on Xavier initialisation
    with smaller starting weights. Allows for faster convergence on low learn
    rates, useful in the presence of multiple loss functions
    """
    weights.append(tf.Variable(tf.truncated_normal([num_in, num_out],
                                                   mean = 0,
                                                   stddev = scale / np.sqrt(num_in + num_out))))
    biases.append(tf.Variable(tf.zeros([num_out]))) #Bias can be zero
    return weights, biases

  def _sort_cols(self,
                 data,
                 subset):
    """
    This function is used to sequence the columns of the dataset, so as to be in
    the order [Continuous data], [Binary data], [Categorical data]. It simply
    rearranges a column, done functionally to minimise memory overhead
    """
    data_1 = data[subset]
    data_0 = data.drop(subset, axis= 1)
    chunk = data_1.shape[1]
    return pd.concat([data_0, data_1], axis= 1), chunk



  def build_model(self,
                imputation_target,
                binary_columns= None,
                softmax_columns= None,
                unsorted= True,
                additional_data = None,
                verbose= True,
                ):
    """
    This method is called to construct the neural network that is the heart of
    MIDAS. This includes the assignment of loss functions to the appropriate
    data types.

    THIS FUNCTION MUST BE CALLED BEFORE ANY TRAINING OR IMPUTATION OCCURS. Failing
    to do so will simply raise an error.

    The categorical columns should be a list of column names. Softmax columns
    should be a list of lists of column names. This will allow the model to
    dynamically assign cost functions to the correct variables. If, however,
    the data comes pre-sorted, arranged can be set to "true", in which case
    the arguments can be passed in as integers of size, ie. shape[1] attributes
    for each of the relevant categories.

    In other words, if you're experienced at using MIDAS and understand how its
    indexing works, pre-sort your data and pass in the integers so specifying
    reindexing values doesn't become too onerous.

    Alternatively, list(df.columns.values) will output a list of column names,
    which can be easily implemented in the 'for' loop which constructs your dummy
    variables.

    Args:
      imputation_target: DataFrame. Any data specified here will be rearranged
      and stored for the subsequent imputation process. The data must be
      preprocessed before it is passed to build_model.

      binary_columns: List of names. Specifies the binary (ie. non-exclusive
      categories) to be imputed. If unsorted = False, this value can be an integer

      softmax_columns: List of lists. Every inner list should contain column names.
      Each inner list should represent a set of mutually exclusive categories,
      such as current day of the week. if unsorted = False, this should be a list
      of integers.

      unsorted: Boolean. Specifies to MIDAS that data has been pre-sorted, and
      indices can simply be appended to the size index.

      additional_data: DataFrame. Any data that shoud be included in the imputation
      model, but is not required from the output. By passing data here, the data
      will neither be rearranged nor will it generate a cost function. This reduces
      the regularising effects of multiple loss functions, but reduces both network
      size requirements and training time.

      verbose: Boolean. Set to False to suppress messages printing to terminal.

      Returns:
        Self

    """
    if not isinstance(imputation_target, pd.DataFrame):
      raise TypeError("Input data must be in a DataFrame")
    if imputation_target.isnull().sum().sum() == 0:
      raise ValueError("Imputation target contains no missing values. Please"\
                       " ensure missing values are encoded as type np.nan")
    self.original_columns = imputation_target.columns
    cont_exists = False
    cat_exists = False
    in_size = imputation_target.shape[1]
    if additional_data is not None:
      add_size = additional_data.shape[1]
    else:
      add_size = 0

    # Establishing indices for cost function
    size_index = []
    if binary_columns is not None:
      if unsorted:
        imputation_target, chunk = self._sort_cols(imputation_target,
                                                   binary_columns)
        size_index.append(chunk)
      else:
        size_index.append(binary_columns)
      cat_exists = True
    if softmax_columns is not None:
      if unsorted:
        for subset in softmax_columns:
          imputation_target, chunk = self._sort_cols(imputation_target,
                                                    subset)
          size_index.append(chunk)
      else:
        for digit in softmax_columns:
          size_index.append(digit)
    if sum(size_index) < in_size:
      chunk = in_size - sum(size_index)
      size_index.insert(0, chunk)
      cont_exists = True
      if not sum(size_index) == in_size:
        raise ValueError("Sorting columns has failed")
    if verbose:
      print("Size index:", size_index)

    #Commit some variables to the instance of the class
    self.size_index = size_index
    if not self.input_is_pipeline:
      self.na_matrix = imputation_target.notnull().astype(np.bool)
    self.imputation_target = imputation_target.fillna(0)
    if additional_data is not None:
      self.additional_data = additional_data.fillna(0)

    #Build graph
    tf.reset_default_graph()
    self.graph = tf.Graph()
    with self.graph.as_default():
      if self.seed is not None:
        tf.set_random_seed(self.seed)

      #Placeholders
      self.X = tf.placeholder(tf.float32, [None, in_size])
      self.na_idx = tf.placeholder(tf.bool, [None, in_size])
      if additional_data is not None:
        self.X_add = tf.placeholder(tf.float32, [None, add_size])
      if self.vae_layer:
        self.latent_inputs = tf.placeholder(tf.float32, [None, self.latent_space_size])

      #Build list for determining input and output structures
      struc_list = self.layer_structure.copy()
      struc_list.insert(0, in_size + add_size)
      outputs_struc = []
      for n in range(len(size_index)):
        if n == 0:
          if cont_exists:
            outputs_struc += ["cont"]*size_index[n]
          elif cat_exists:
            outputs_struc += ["bin"]*size_index[n]

          else:
            outputs_struc += [size_index[n]]

        elif n == 1:
          if cont_exists and cat_exists:
            outputs_struc += ["bin"]*size_index[n]

          else:
            outputs_struc += [size_index[n]]
        else:
          outputs_struc += [size_index[n]]

      if self.manual_outputs == True:
        output_layer_size = np.sum(self.output_structure)
        output_layer_structure = self.output_structure
      else:
        output_layer_structure = []
        for item in outputs_struc:
          if item == "cont":
            output_layer_structure.append(self.output_structure[0])
          if item == "bin":
            output_layer_structure.append(self.output_structure[1])
          if type(item) == int:
            output_layer_structure.append(self.output_structure[2])
          output_layer_size = np.sum(output_layer_structure)

      #Instantiate and initialise variables
      _w = []
      _b = []
      _zw = []
      _zb = []
      _ow = []
      _ob = []

      #Input, denoising
      for n in range(len(struc_list) -1):
        _w, _b = self._build_variables(weights= _w, biases= _b,
                                       num_in= struc_list[n],
                                       num_out= struc_list[n+1],
                                       scale= self.init_scale)
      if self.vae_layer:
        mapped_dist = tf.distributions.Normal(tf.constant(0.),
                                              tf.constant(self.vae_sample_var))
#        mapped_dist = tf.distributions.StudentT(tf.constant(3.0),
#                                                tf.constant(0.0),
#                                                tf.constant(1.0))
        #Latent state, variance
        _zw, _wb = self._build_variables(weights= _zw, biases= _zb,
                                         num_in= struc_list[-1],
                                         num_out= self.latent_space_size*2,
                                         scale= self.init_scale)
        _zw, _wb = self._build_variables(weights= _zw, biases= _zb,
                                         num_in= self.latent_space_size,
                                         num_out= self.output_layers[0],
                                         scale= self.init_scale)

      t_l = len(self.output_layers)
      #Output, specialisation
      assert len(output_layer_structure) == len(outputs_struc)
      output_split = []
      if self.individual_outputs:
        self.output_layers.append(output_layer_size)
        for n in range(t_l):
          _ow, _ob = self._build_variables(weights= _ow, biases= _ob,
                                         num_in= self.output_layers[n],
                                         num_out= self.output_layers[n+1],
                                         scale= self.init_scale)
        for n in range(len(outputs_struc)):
          if type(outputs_struc[n]) == str:
            _ow, _ob = self._build_variables(weights= _ow, biases= _ob,
                                             num_in= output_layer_structure[n],
                                             num_out= 1,
                                             scale= self.init_scale)
            output_split.append(1)
          elif type(outputs_struc[n]) == int:
            _ow, _ob = self._build_variables(weights= _ow, biases= _ob,
                                             num_in= output_layer_structure[n],
                                             num_out= outputs_struc[n],
                                             scale= self.init_scale)
            output_split.append(outputs_struc[n])
      else:
        self.output_layers.append(in_size)
        for n in range(t_l):
          _ow, _ob = self._build_variables(weights= _ow, biases= _ob,
                                           num_in= self.output_layers[n],
                                           num_out= self.output_layers[n+1])
        for n in range(len(outputs_struc)):
          if type(outputs_struc[n]) == str:
            output_split.append(1)
          elif type(outputs_struc[n]) == int:
            output_split.append(outputs_struc[n])

      #Build the neural network. Each layer is determined by the struc list
      def denoise(X):
        #Input tx
        for n in range(len(struc_list) -1):
          if n == 0:
            if self.noise_type == 'bernoulli':
              X = self._build_layer(X, _w[n], _b[n],
                                    dropout_rate = self.input_drop)
            elif self.noise_type == 'gaussian':
              X = X + tf.distributions.Normal(loc=tf.constant(0.),
                        scale = tf.constant(self.input_drop)).sample(sample_shape= tf.shape(X))
              X = self._build_layer(X, _w[n], _b[n],
                                    dropout_rate = self.input_drop)
          else:
            X = self._build_layer(X, _w[n], _b[n],
                                  dropout_rate = self.dropout_level)
        return X

      if self.vae_layer:
        def to_z(X):
          #Latent tx
          X = self._build_layer(X, _zw[0], _zb[0], dropout_rate = self.dropout_level,
                                   output_layer= True)
          x_mu, x_log_sigma = tf.split(X, [self.latent_space_size]*2, axis=1)
          return x_mu, x_log_sigma

        def from_z(z):
          #Joint transform
          X = self._build_layer(z, _zw[1], _zb[1], dropout_rate= 1)
          return X

        def vae(X, output=False):
          x_mu, x_log_sigma = to_z(X)
          if output:
            reparam_z = mapped_dist.sample(sample_shape= tf.shape(x_mu))
#            reparam_z = tf.random_normal(tf.shape(x_mu))
          else:
            reparam_z = tf.random_normal(tf.shape(x_mu))
          z = x_mu + reparam_z * tf.exp(x_log_sigma)
          kld = tf.maximum(tf.reduce_mean(1 + 2*x_log_sigma*x_mu**2 - tf.exp(2-x_log_sigma),
                                       axis=1)*self.prior_strength * - 0.5,
            self.kld_min)
          X = from_z(z)
          return X, kld

      if self.individual_outputs:
        def decode(X):
          for n in range(t_l):
            X = self._build_layer(X, _ow[n], _ob[n], dropout_rate= self.dropout_level)
          #Output tx
          base_splits = tf.split(X, output_layer_structure, axis=1)
          decombined = []
          for n in range(len(outputs_struc)):
            decombined.append(self._build_layer(base_splits[n], _ow[n+t_l], _ob[n+t_l],
                                                dropout_rate = self.dropout_level,
                                                output_layer= True))
          return decombined

      else:
        def decode(X):
          for n in range(t_l):
            if n == t_l-1:
              X = self._build_layer(X, _ow[n], _ob[n],
                                    dropout_rate = self.dropout_level,
                                    output_layer= True)
            else:
              X = self._build_layer(X, _ow[n], _ob[n],
                                    dropout_rate = self.dropout_level)
          decombined = tf.split(X, output_split, axis=1)
          return decombined

      if self.vae_layer:
        def decode_z(z):
          X = from_z(z)
          X = decode(X)
          return X

      #Determine which imputation function is to be used. This is constructed to
      #take advantage of additional data provided.
      if additional_data is not None:
        encoded = denoise(tf.concat([self.X, self.X_add], axis= 1))
      else:
        encoded = denoise(self.X)

      if self.vae_layer:
        perturb, kld = vae(encoded)
        perturb_out, _ = vae(encoded, True)
        pred_split = decode(perturb)
        out_split = decode(perturb_out)
      else:
        pred_split = decode(encoded)

      #Output functions
      cost_list = []
      self.output_types = []

      #Build L2 loss and KL-Divergence
      if self.weight_decay == 'default':
        lmbda = 1/self.imputation_target.shape[0]
      else:
        lmbda = self.weight_decay
#      if self.vae_layer:
#        l2_penalty = tf.multiply(tf.reduce_mean(
#            [tf.nn.l2_loss(w) for w in _w]+\
#            [tf.nn.l2_loss(w) for w in _zw]+\
#            [tf.nn.l2_loss(w) for w in _ow]
#            ), lmbda)
#      else:
#        l2_penalty = tf.multiply(tf.reduce_mean(
#            [tf.nn.l2_loss(w) for w in _w]+\
#            [tf.nn.l2_loss(w) for w in _ow]
#            ), lmbda)

      #Assign cost and loss functions
      na_split = tf.split(self.na_idx, output_split, axis=1)
      true_split = tf.split(self.X, output_split, axis=1)
      for n in range(len(outputs_struc)):
        na_adj = tf.cast(tf.count_nonzero(na_split[n]),tf.float32)\
        /tf.cast(tf.size(na_split[n]),tf.float32)
        if outputs_struc[n] == 'cont':
          if 'rmse' not in self.output_types:
            self.output_types.append('rmse')
          cost_list.append(tf.sqrt(
              tf.losses.mean_squared_error(tf.boolean_mask(true_split[n], na_split[n]),
                                           tf.boolean_mask(pred_split[n], na_split[n])\
                                           ))*self.cont_adj * na_adj)
        elif outputs_struc[n] == 'bin':
          if 'bacc' not in self.output_types:
            self.output_types.append('bacc')
          cost_list.append(
              tf.losses.sigmoid_cross_entropy(tf.boolean_mask(true_split[n], na_split[n]),
                                              tf.boolean_mask(pred_split[n], na_split[n]))\
              *self.binary_adj * na_adj)
        elif type(outputs_struc[n]) == int:
          self.output_types.append('sacc')
          cost_list.append(tf.losses.softmax_cross_entropy(
              tf.reshape(tf.boolean_mask(true_split[n], na_split[n]), [-1, outputs_struc[n]]),
              tf.reshape(tf.boolean_mask(pred_split[n], na_split[n]), [-1, outputs_struc[n]]))\
              *self.softmax_adj *na_adj)

      def output_function(out_split):
        output_list = []
        #Break outputs into their parts
        for n in range(len(outputs_struc)):
          if outputs_struc[n] == 'cont':
            output_list.append(out_split[n])
          elif outputs_struc[n] == 'bin':
            output_list.append(tf.nn.sigmoid(out_split[n]))
          elif type(outputs_struc[n]) == int:
            output_list.append(tf.nn.softmax(out_split[n]))
        return tf.concat(output_list, axis= 1)


      self.outputs_struc = outputs_struc
      if self.vae_layer:
        self.output_op = output_function(out_split)
        self.joint_loss = tf.reduce_mean(tf.reduce_sum(cost_list) + kld)# + l2_penalty)
        self.encode_to_z = to_z(encoded)
        self.gen_from_z_sample = output_function(decode_z(mapped_dist.sample(
            sample_shape= tf.shape(self.latent_inputs))))
        self.gen_from_z_inputs = output_function(decode_z(self.latent_inputs))

      else:
        self.output_op = output_function(pred_split)
        self.joint_loss = tf.reduce_mean(tf.reduce_sum(cost_list))# + l2_penalty)

      optim = tf.contrib.opt.AdamWOptimizer(lmbda, self.learn_rate)
      self.train_step = optim.minimize(self.joint_loss)
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()

    self.model_built = True
    if verbose:
      print()
      print("Computation graph constructed")
      print()
    return self

  def train_model(self,
                  training_epochs= 100,
                  verbose= True,
                  verbosity_ival= 1,
                  excessive= False):
    """
    This is the standard method for optimising the model's parameters. Must be
    called before imputation can be performed.

    Args:
      training_epochs: Integer. Number of cycles through complete dataset.
      verbose: Boolean. Prints out messages, including loss

      verbosity_ival: Integer. This number determines the interval between
      messages.

      excessive: Boolean. Used for troubleshooting, this argument will cause the
      cost of each batch to be printed to the terminal.

    Returns:
      Self. Model is automatically saved upon reaching specified number of epochs

    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")

    if self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept pipeline data, either"\
                           " use 'train_model_pipeline' method or rebuild model "\
                           "with in-memory dataset.")

    if self.seed is not None:
      np.random.seed(self.seed)
    feed_data = self.imputation_target.values
    na_loc = self.na_matrix.values
    with tf.Session(graph= self.graph) as sess:
      sess.run(self.init)
      if verbose:
        print("Model initialised")
        print()
      for epoch in range(training_epochs):
        count = 0
        run_loss = 0
        for batch in self._batch_iter(feed_data, na_loc, self.train_batch):
          if np.sum(batch[1]) == 0:
            continue
          feedin = {self.X: batch[0], self.na_idx: batch[1]}
          if self.additional_data is not None:
            feedin[self.X_add] = batch[2]
          loss, _ = sess.run([self.joint_loss, self.train_step],
                             feed_dict= feedin)
          if excessive:
            print("Current cost:", loss)
          count +=1
          if not np.isnan(loss):
            run_loss += loss
        if verbose:
          if epoch % verbosity_ival == 0:
            print('Epoch:', epoch, ", loss:", str(run_loss/count))
      if verbose:
        print("Training complete. Saving file...")
      save_path = self.saver.save(sess, self.savepath)
      if verbose:
        print("Model saved in file: %s" % save_path)
      return self

  def generate_samples(self,
                       m= 50,
                       verbose= True):
    """
    Method used to generate a set of m imputations to the .output_list attribute.
    Imputations are stored within a list in memory, and can be accessed in any
    order.

    If a model has been pre-trained, on subsequent runs this function can be
    directly called without having to train first. An 'if' statement checking
    the default save location is useful for this.

    Args:
      m: Integer. Number of imputations to generate.

      verbose: Boolean. Prints out messages.
    Returns:
      Self
    """

    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")

    if self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept pipeline data, either"\
                           " use 'pipeline_yield_samples' method or rebuild model "\
                           "with in-memory dataset.")
    self.output_list = []
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      if verbose:
        print("Model restored.")
      for n in range(m):
        feed_data = self.imputation_target.values
        feedin = {self.X: feed_data}
        if self.additional_data is not None:
          feedin[self.X_add] = self.additional_data
        y_out = pd.DataFrame(sess.run(self.output_op,
                                             feed_dict= feedin),
                                columns= self.imputation_target.columns)
        output_df = self.imputation_target.copy()
        output_df[np.invert(self.na_matrix)] = y_out[np.invert(self.na_matrix)]
        self.output_list.append(output_df)
    return self

  def yield_samples(self,
                    m= 50,
                    verbose= True):
    """
    Method used to generate a set of m imputations via the 'yield' command, allowing
    imputations to be used in a 'for' loop'

    If a model has been pre-trained, on subsequent runs this function can be
    directly called without having to train first. An 'if' statement checking
    the default save location is useful for this.

    Args:
      m: Integer. Number of imputations to generate.

      verbose: Boolean. Prints out messages.

    Returns:
      Self
    """

    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")

    if self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept pipeline data, either"\
                           " use 'pipeline_yield_samples' method or rebuild model "\
                           "with in-memory dataset.")
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      if verbose:
        print("Model restored.")
      for n in range(m):
        feed_data = self.imputation_target.values
        feedin = {self.X: feed_data}
        if self.additional_data is not None:
          feedin[self.X_add] = self.additional_data
        y_out = pd.DataFrame(sess.run(self.output_op,
                                           feed_dict= feedin),
                                columns= self.imputation_target.columns)
        output_df = self.imputation_target.copy()
        output_df[np.invert(self.na_matrix)] = y_out[np.invert(self.na_matrix)]
        yield output_df
    return self

  def batch_generate_samples(self,
                             m= 50,
                             b_size= 256,
                             verbose= True):
    """
    Method used to generate a set of m imputations to the .output_list attribute.
    Imputations are stored within a list in memory, and can be accessed in any
    order. As batch generation implies very large datasets, this method is only
    provided for completeness' sake.

    This function is for a dataset large enough to be stored in memory, but
    too large to be passed into the model in its entirety. This may be due to
    GPU memory limitations, or just the size of the model

    If a model has been pre-trained, on subsequent runs this function can be
    directly called without having to train first. An 'if' statement checking
    the default save location is useful for this.

    Args:
      m: Integer. Number of imputations to generate.

      b_size: Integer. Number of data entries to process at once. For managing
      wider datasets, smaller numbers may be required.

      verbose: Boolean. Prints out messages.

    Returns:
      Self
    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")

    if self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept pipeline data, either"\
                           " use 'pipeline_yield_samples' method or rebuild model "\
                           "with in-memory dataset.")
    self.output_list = []
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      if verbose:
        print("Model restored.")
      for n in range(m):
        feed_data = self.imputation_target.values
        minibatch_list = []
        for batch in self._batch_iter_output(feed_data, b_size):
          if self.additional_data is not None:
            feedin = {self.X: batch[0], self.X_add: batch[1]}
          else:
            feedin = {self.X: batch}
          y_batch = pd.DataFrame(sess.run(self.output_op,
                                        feed_dict= feedin),
                               columns= self.imputation_target.columns)
          minibatch_list.append(y_batch)
        y_out = pd.DataFrame(pd.concat(minibatch_list, ignore_index= True),
                             columns= self.imputation_target.columns)
        output_df = self.imputation_target.copy()
        output_df[np.invert(self.na_matrix)] = y_out[np.invert(self.na_matrix)]
        self.output_list.append(output_df)
    return self

  def batch_yield_samples(self,
                             m= 50,
                             b_size= 256,
                             verbose= True):
    """
    Method used to generate a set of m imputations via the 'yield' command, allowing
    imputations to be used in a 'for' loop'

    This function is for a dataset large enough to be stored in memory, but
    too large to be passed into the model in its entirety. This may be due to
    GPU memory limitations, or just the size of the model

    If a model has been pre-trained, on subsequent runs this function can be
    directly called without having to train first. An 'if' statement checking
    the default save location is useful for this.

    Args:
      m: Integer. Number of imputations to generate.

      b_size: Integer. Number of data entries to process at once. For managing
      wider datasets, smaller numbers may be required.

      verbose: Boolean. Prints out messages.

    Returns:
      Self    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")

    if self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept pipeline data, either"\
                           " use 'pipeline_yield_samples' method or rebuild model "\
                           "with in-memory dataset.")
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      if verbose:
        print("Model restored.")
      for n in range(m):
        feed_data = self.imputation_target.values
        minibatch_list = []
        for batch in self._batch_iter_output(feed_data, b_size):
          if self.additional_data is not None:
            feedin = {self.X: batch[0], self.X_add: batch[1]}
          else:
            feedin = {self.X: batch}
          y_batch = pd.DataFrame(sess.run(self.output_op,
                                        feed_dict= feedin),
                               columns= self.imputation_target.columns)
          minibatch_list.append(y_batch)
        y_out = pd.DataFrame(pd.concat(minibatch_list, ignore_index= True),
                             columns= self.imputation_target.columns)
        output_df = self.imputation_target.copy()
        output_df[np.invert(self.na_matrix)] = y_out[np.invert(self.na_matrix)]
        yield output_df
    return self

  def overimpute(self,
                 spikein = 0.1,
                 training_epochs= 100,
                 report_ival = 10,
                 report_samples = 32,
                 plot_all= True,
                 verbose= True,
                 verbosity_ival= 1,
                 spike_seed= 42,
                 cont_kdes = False,
                 excessive= False,
                 plot_block = True
                 ):
    """
    This function spikes in additional missingness, so that known values can be
    used to help adjust the complexity of the model. As conventional train/
    validation splits can still lead to autoencoders overtraining, the method for
    limiting complexity is overimputation and early stopping. This gives an
    estimate of how the model will react to unseen variables.

    Error is defined as RMSE for continuous variables, and classification error
    for binary and categorical variables (ie. 1 - accuracy). Note that this means
    that binary classification is inherently dependent on a selection threshold
    of 0.5, and softmax accuracy will automatically decrease as a function of the
    number of classes within the model. All three will be affected by the degree
    of imbalance within the dataset.

    The accuracy measures provided here may not be ideal for all problems, but
    they are generally appropriate for selecting optimum complexity. Should the
    lines denoting error begin to trend upwards, this indicates overtraining and
    is a sign that the training_epochs parameter to the .train_model() method should
    be capped before this point.

    The actual optimal point may differ from that indicated by the .overimpute()
    method for two reasons:
      -The loss that is spiked in reduces the overall data available to the algorithm
      to learn the patterns inherent, so there should be some improvement in performance
      when .train_model() is called. If this is a concern, then it should be possible
      to compare the behaviour of the loss figure between .train_model() and
      .overimpute().
      -The missingness inherent to the data may depend on some unobserved factor.
      In this case, the bias in the observed data may lead to inaccurate inference.

    It is worth visually inspecting the distribution of the overimputed values
    against imputed values (using plot_all) to ensure that they fall within a
    sensible range.

    Args:
      spikein: Float, between 0 and 1. The proportion of total values to remove
      from the dataset at random. As this is a random selection, the sample should
      be representative. It should also equally capture known and missing values,
      therefore this sample represents the percentage of known data to remove.
      If concerns about sampling remain, adjusting this number or changing the
      seed can allow for validation. Larger numbers mean greater amounts of removed
      data, which may mean estimates of optimal training time might be skewed.
      This can be resolved by lowering the learning rate and aiming for a window.

      training_epochs: Integer. Specifies the number of epochs model should be
      trained for. It is often worth specifying longer than expected to ensure
      that the model does not overtrain, or that another, better, optimum exists
      given slightly longer training time.

      report_ival: Integer. The interval between sampling from the posterior of
      the model. Smaller intervals mean a more granular view of convergence,
      but also drastically slow training time.

      report_samples: The number of Monte-Carlo samples drawn for each check of
      the posterior at report_ival. Greater numbers of samples means a longer
      runtime for overimputation. For low numbers of samples, the impact will be
      reduced, though for large numbers of Monte-Carlo samples, report_ival will
      need to be adjusted accordingly. I recommend a number between 5 and 25,
      depending on the complexity of the data.

      plot_all: Generates plots of the distribution of spiked in values v. the
      mean of the imputations. Continuous values have a density plot, categorical
      values a bar plot representing proportions. Only the mean is plotted at this
      point for simplicity's sake.

      verbose: Boolean. Prints out messages, including loss

      verbosity_ival: Integer. This number determines the interval between
      messages.

      spike_seed: A different seed, separate to the one used in the main call,
      used to initialise the RNG for the missingness spike-in.

      excessive: Unlike .train_model()'s excessive arg, this argument prints the
      entire batch output to screen. This allows for inspection for unusual values
      appearing, useful if the model's accuracy will not reduce.
      
      plot_block: Boolean. Determines whether code execution should be blocked 
      while overimputation plot is shown (default = True). To allow code to run 
      uninterrupted, set both plot_block AND plot_all to False.

    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")

    if self.input_is_pipeline:
      raise AttributeError("Overimputation not currently supported for models"\
                           " which use a pipeline function for input.")
    #These values simplify control flow used later for error calculation and
    #visualisation of convergence.
    if cont_kdes & (plot_all == False):
      raise ValueError("Cannot plot KDEs if plot_all is False")

    if excessive:
      import time
    rmse_in = False
    sacc_in = False
    bacc_in = False
    if 'rmse' in self.output_types:
      rmse_in = True
    if 'sacc' in self.output_types:
      def sacc(true, pred, spike): #Softmax accuracy
        a = np.argmax(true, 1)
        b = np.argmax(pred, 1)
        return np.sum(a[spike.flatten()] == b[spike.flatten()]) / np.sum(spike)
      def findcatname(strlist):
        return strlist[0][:([min([x[0]==elem for elem in x]) \
                         for x in zip(*strlist)]+[0]).index(0)]
      sacc_in = True

    if 'bacc' in self.output_types:
      def bacc(true, pred, spike):
        pred = (pred > 0.5).astype(np.int_)
        return np.sum(true[spike] == pred[spike]) / np.sum(spike)
      bacc_in = True

    feed_data = self.imputation_target.copy()
    na_loc = self.na_matrix
    np.random.seed(spike_seed)
    n_softmax = 0 #Necessary to derive the average classification error

    #Pandas lacks an equivalent to tf.split, so this is used to divide columns
    #for their respective error metrics
    break_list = list(np.cumsum(self.size_index))
    break_list.insert(0, 0)

    #Generate spike-in
    spike = []
    for n in range(len(self.size_index)):
      if self.output_types[n] == 'sacc':
        temp_spike = pd.Series(np.random.choice([True, False],
                                                size= self.imputation_target.shape[0],
                                                p= [spikein, 1-spikein]))

        spike.append(pd.concat([temp_spike]*self.size_index[n], axis=1))
        n_softmax += 1

      else:
        spike.append(pd.DataFrame(np.random.choice([True, False],
                                        size= [self.imputation_target.shape[0],
                                               self.size_index[n]],
                                        p= [spikein, 1-spikein])))
    spike = pd.concat(spike, axis= 1)
    spike.columns = self.imputation_target.columns
    spike[np.invert(na_loc)] = False
    feed_data[spike] = 0
    feed_data =  feed_data.values
    na_loc[spike] = False
    spike = spike.values
    na_loc = na_loc.values

    #Initialise lists for plotting
    s_rmse = []
    a_rmse = []
    s_bacc = []
    a_bacc = []
    s_sacc = []
    a_sacc = []
    with tf.Session(graph= self.graph) as sess:
      sess.run(self.init)
      print("Model initialised")
      print()
      for epoch in range(training_epochs + 1):
        count = 0
        run_loss = 0
        for batch in self._batch_iter(feed_data, na_loc, self.train_batch):
          if np.sum(batch[1]) == 0:
            continue
          feedin = {self.X: batch[0], self.na_idx: batch[1]}
          if self.additional_data is not None:
            feedin[self.X_add] = batch[2]
          if excessive:
            out, loss, _ = sess.run([self.output_op, self.joint_loss, self.train_step],
                             feed_dict= feedin)
            print("Current cost:", loss)
            print(out)
            time.sleep(5)
          else:
            loss, _ = sess.run([self.joint_loss, self.train_step],
                             feed_dict= feedin)
          count +=1

          if not np.isnan(loss):
            run_loss += loss
        if verbose:
          if epoch % verbosity_ival == 0:
            print('Epoch:', epoch, ", loss:", str(run_loss/count))

        if epoch % report_ival == 0:
          """
          For each report interval, generate report_samples worth of imputations
          and measure both individual and aggregate error values
          """
          #Initialise losses
          single_rmse = 0
          single_sacc = 0
          single_bacc = 0
          first =  True
          if cont_kdes:
            plot_first = True

          for sample in range(report_samples):

            minibatch_list = []
            for batch in self._batch_iter_output(feed_data, self.train_batch):
              feedin = {self.X: batch}
              if self.additional_data is not None:
                feedin = {self.X: batch[0]}
                feedin[self.X_add] = batch[1]
              else:
                feedin = {self.X: batch}
              y_batch = pd.DataFrame(sess.run(self.output_op,
                                              feed_dict= feedin),
                                      columns= self.imputation_target.columns)
              minibatch_list.append(y_batch)
            y_out = pd.DataFrame(pd.concat(minibatch_list, ignore_index= True),
                                 columns= self.imputation_target.columns)
            if cont_kdes:
              if 'rmse' in self.output_types:
                for n in range(self.size_index[0]):
                  plt.figure(n+1)
                  t_t = self.imputation_target.iloc[:,n]
                  t_p = y_out.iloc[:,n]
                  t_s = spike[:,n]
                  if plot_first:
                    t_p[t_s].plot(kind= 'density', color= 'k', alpha= 0.5, label='Single imputation')
                  else:
                    t_p[t_s].plot(kind= 'density', color= 'k', alpha= 0.5, label='_nolegend_')
                plot_first = False

            #Calculate individual imputation losses
            for n in range(len(self.size_index)):
              temp_pred = y_out.iloc[:,break_list[n]:break_list[n+1]]
              temp_true = self.imputation_target.iloc[:,break_list[n]:break_list[n+1]]
              temp_spike = spike[:,break_list[n]:break_list[n+1]]
              if self.output_types[n] == 'sacc':
                temp_spike = temp_spike[:,0]
                single_sacc += (1 - sacc(temp_true.values,
                                         temp_pred.values, temp_spike)) / n_softmax

              elif self.output_types[n] == 'rmse':
                single_rmse += np.sqrt(mse(temp_true[temp_spike],
                                           temp_pred[temp_spike]))
              else:
                single_bacc += 1 - bacc(temp_true.values, temp_pred.values, temp_spike)

            if first:
              running_output = y_out
              first= False
            else:
              running_output += y_out
          single_rmse = single_rmse / report_samples
          single_sacc = single_sacc / report_samples
          single_bacc = single_bacc / report_samples
          y_out = running_output / report_samples

          #Calculate aggregate imputation losses
          agg_rmse = 0
          agg_sacc = 0
          agg_bacc = 0
          for n in range(len(self.size_index)):
            temp_pred = y_out.iloc[:,break_list[n]:break_list[n+1]]
            temp_true = self.imputation_target.iloc[:,break_list[n]:break_list[n+1]]
            temp_spike = spike[:,break_list[n]:break_list[n+1]]
            if self.output_types[n] == 'sacc':
              temp_spike = temp_spike[:,0]
              if plot_all:
                temp_pred[temp_spike].mean().plot(kind= 'bar',
                         label= 'Imputed values (mean)', color ='C0')
                temp_true[temp_spike].mean().plot(kind= 'bar', alpha= 0.5,
                         color= 'r', align= 'edge', label= 'Removed observed values (mean)')
                temp_true_name = findcatname(temp_true[temp_spike].columns)[:-1]
                plt.title('Overimputation density plot: '+temp_true_name+' (categorical)')
                plt.xlabel(temp_true_name)
                plt.ylabel('Proportion')
                plt.legend()
                plt.show()
              agg_sacc += (1 - sacc(temp_true.values, temp_pred.values,
                                   temp_spike)) / n_softmax
            elif self.output_types[n] == 'rmse':
              if plot_all:
                for n_rmse in range(len(temp_pred.columns)):
                  plt.figure(n_rmse+1)
                  t_p = temp_pred.iloc[:,n_rmse]
                  t_t = temp_true.iloc[:,n_rmse]
                  t_s = temp_spike[:,n_rmse]
                  t_p[t_s].plot(kind= 'density', label= 'Imputed values (mean)')
                  t_t[t_s].plot(kind= 'density', color= 'r', label= 'Removed observed values')
                  t_t.plot(kind='kde', color= 'g', label= 'All observed values')
                  hyp_output = pd.concat([t_t[np.invert(t_s)], t_p[t_s]])
                  hyp_output.plot(kind='kde', color= 'm', label = 'Completed data')
                  plt.title('Overimputation density plot: ' + \
                            temp_pred.columns[n_rmse] + ' (continuous)')
                  plt.xlabel(temp_pred.columns[n_rmse])
                  plt.ylabel('Density')
                  plt.legend()
                plt.show()

              agg_rmse += np.sqrt(mse(temp_true[temp_spike],
                                         temp_pred[temp_spike]))
            else:
              if plot_all:
                temp_pred[temp_spike].mean().plot(kind= 'bar',
                         label= 'Imputed values',
                         color= 'C0')
                temp_true[temp_spike].mean().plot(kind= 'bar', alpha= 0.5,
                         color= 'r', align= 'edge', label= 'Observed values')
                plt.title('Overimputation binary proportions')
                plt.xlabel('Variables')
                plt.ylabel('Proportion')
                plt.legend()
                plt.show()
              agg_bacc += 1 - bacc(temp_true.values, temp_pred.values, temp_spike)

          #Plot losses depending on which loss values present in data
          if rmse_in:
            s_rmse.append(single_rmse)
            a_rmse.append(agg_rmse)
            print("Individual RMSE on spike-in:", single_rmse)
            print("Aggregated RMSE on spike-in:", agg_rmse)
            plt.plot(s_rmse, 'k-', label= "Individual RMSE")
            plt.plot(a_rmse, 'k--', label= "Aggregated RMSE")
            min_sr = min(s_rmse)
            min_ar = min(a_rmse)
            plt.plot([min_sr]*len(s_rmse), 'r:')
            plt.plot([min_ar]*len(a_rmse), 'r:')
            plt.plot(s_rmse.index(min(s_rmse)),
                   min_sr, 'rx')
            plt.plot(a_rmse.index(min(a_rmse)),
                   min_ar, 'rx')
          if sacc_in:
            s_sacc.append(single_sacc)
            a_sacc.append(agg_sacc)
            print("Individual error on softmax spike-in:", single_sacc)
            print("Aggregated error on softmax spike-in:", agg_sacc)
            plt.plot(s_sacc, 'g-', label= "Individual classification error")
            plt.plot(a_sacc, 'g--', label= "Aggregated classification error")
            min_ss = min(s_sacc)
            min_as = min(a_sacc)
            plt.plot([min_ss]*len(s_sacc), 'r:')
            plt.plot([min_as]*len(a_sacc), 'r:')
            plt.plot(s_sacc.index(min(s_sacc)),
                   min_ss, 'rx')
            plt.plot(a_sacc.index(min(a_sacc)),
                   min_as, 'rx')
          if bacc_in:
            s_bacc.append(single_bacc)
            a_bacc.append(agg_bacc)
            print("Individual error on binary spike-in:", single_bacc)
            print("Aggregated error on binary spike-in:", agg_bacc)
            plt.plot(s_bacc, 'b-', label= "Individual binary error")
            plt.plot(a_bacc, 'b--', label= "Aggregated binary error")
            min_sb = min(s_bacc)
            min_ab = min(a_bacc)
            plt.plot([min_sb]*len(s_bacc), 'r:')
            plt.plot([min_ab]*len(a_bacc), 'r:')
            plt.plot(s_bacc.index(min(s_bacc)),
                   min_sb, 'rx')
            plt.plot(a_bacc.index(min(a_bacc)),
                   min_ab, 'rx')

          #Complete plots
          plt.title("Overimputation error during training")
          plt.ylabel("Error")
          plt.legend()
          plt.ylim(ymin= 0)
          plt.xlabel("Reporting interval")
          
          if not (plot_all or plot_block):
          	plt.show(block=False)
          else:
          	plt.show()
          
      if not (plot_all or plot_block):
          	plt.show()
      print("Overimputation complete. Adjust complexity as needed.")
      return self

  def build_model_pipeline(self,
                           data_sample,
                           binary_columns= None,
                           softmax_columns= None,
                           unsorted= True,
                           additional_data_sample= None,
                           verbose= True,
                           crossentropy_adj= 1,
                           loss_scale = 1):
    """
    This function is for integration with databasing or any dataset that needs
    to be batched into memory. The data sample is simply there to allow the
    original constructor to be recycled. The head of the data should be sufficient
    to build the imputation model. The input pipeline itself should pre-scale
    the data, and code null values as type np.nan. The pipeline ought to output
    a Pandas DataFrame. If additional data will be passed in, then the return must
    be a list of two DataFrames. The columns of the dataframe will be re-arranged
    so that error functions are efficiently generated.

    IT IS IMPERITIVE that this ordering is respected. Design the input batching
    function accordingly.

    The categorical columns should be a list of column names. Softmax columns
    should be a list of lists of column names. This will allow the model to
    dynamically assign cost functions to the correct variables. If, however,
    the data comes pre-sorted, arranged can be set to "true", in which case
    the arguments can be passed in as integers of size, ie. shape[1] attributes
    for each of the relevant categories.

    In other words, pre-sort your data and pass in the integers, so indexing
    dynamically doesn't become too difficult. Alternatively, list(df.columns.values)
    will output a list of column names, which can be easily implemented in the
    'for' loop which constructs your dummy variables.
    """
    self.input_is_pipeline = True
    b_c = binary_columns
    s_c = softmax_columns
    us = unsorted
    a_d = additional_data_sample
    vb = verbose
    cea = crossentropy_adj
    l_s = loss_scale

    self.build_model(data_sample, b_c, s_c, us, a_d, vb, cea, l_s)

    return self

  def train_model_pipeline(self,
                           input_pipeline,
                           training_epochs= 100,
                           verbose= True,
                           verbosity_ival= 1,
                           excessive= False):
    """
    This is the alternative method for optimising the model's parameters when input
    data must be batched into memory. Must be called before imputation can be
    performed. The model will then be saved to the specified directory

    Args:
      input_pipeline: Function which yields a pre-processed and scaled DataFrame
      from the designated source, be it a server or large flat file.

      training_epochs: Integer. The number of epochs the model will run for

      verbose: Boolean. Prints out messages, including loss

      verbosity_ival: Integer. This number determines the interval between
      messages.

      excessive: Boolean. Used for troubleshooting, this argument will cause the
      cost of each batch to be printed to the terminal.

    Returns:
      Self. Model is automatically saved upon reaching specified number of epochs

    """
    self.input_pipeline = input_pipeline
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")
    if not self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept locally-stored data,"\
                           "either use 'train_model' method or rebuild model "\
                           "with the 'build_model_pipeline' method.")

    if self.seed is not None:
      np.random.seed(self.seed)
    with tf.Session(graph= self.graph) as sess:
      sess.run(self.init)
      if verbose:
        print("Model initialised")
        print()
      for epoch in range(training_epochs):
        count = 0
        run_loss = 0

        for feed_data in input_pipeline:
          if self.additional_data is None:
            if not isinstance(feed_data, pd.DataFrame):
              raise TypeError("Input data must be in a DataFrame")
            na_loc = feed_data.notnull().astype(bool).values
            feedin = {self.X: feed_data.values,
                      self.na_idx: na_loc}
          else:
            if not isinstance(feed_data, list):
              raise TypeError("Input should be a list of two DataFrames, with "\
                              "index 0 containing the target imputation data, and"\
                              " the data at index 1 containing additional data")
            if len(feed_data) != 2:
              raise TypeError("Input should be a list of two DataFrames, with "\
                              "index 0 containing the target imputation data, and"\
                              " the data at index 1 containing additional data")
            if not isinstance(feed_data[0], pd.DataFrame):
              raise TypeError("Input data must be in a DataFrame")
            if not isinstance(feed_data[1], pd.DataFrame):
              raise TypeError("Additional data must be in a DataFrame")
            na_loc = feed_data[0].notnull().astype(bool).values
            feedin = {self.X: feed_data[0].fillna(0).values,
                      self.X_add: feed_data[1].fillna(0).values,
                      self.na_idx: na_loc}

          if np.sum(na_loc) == 0:
            continue
          loss, _ = sess.run([self.joint_loss, self.train_step],
                             feed_dict= feedin)
          if excessive:
            print("Current cost:", loss)
          count +=1
          if not np.isnan(loss):
            run_loss += loss
        if verbose:
          if epoch % verbosity_ival == 0:
            print('Epoch:', epoch, ", loss:", str(run_loss/count))
      if verbose:
        print("Training complete. Saving file...")
      save_path = self.saver.save(sess, self.savepath)
      if verbose:
        print("Model saved in file: %s" % save_path)
    return self

  def yield_samples_pipeline(self,
                             verbose= False):
    """
    As its impossible to know the specifics of the pipeline, this method simply
    cycles through all data provided by the input function. The number of imputations
    can be specified by the user, depending on their needs.

    Args:
      verbose: Prints out messages

    Yields:
      A 'DataFrame' of the size specified by the input function passed to the
      'train_model_pipeline' method.

    Returns:
      Self

    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")
    if not self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept locally-stored data,"\
                           "either use 'train_model' method or rebuild model "\
                           "with the 'build_model_pipeline' method.")

    if self.seed is not None:
      np.random.seed(self.seed)
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      if verbose:
        print("Model restored.")

      for feed_data in self.inpinput_pipeline:
        if self.additional_data is None:
          if not isinstance(feed_data, pd.DataFrame):
            raise TypeError("Input data must be in a DataFrame")
          na_loc = feed_data.notnull().astype(bool).values
          feedin = {self.X: feed_data.fillna(0).values}
        else:
          if not isinstance(feed_data, list):
            raise TypeError("Input should be a list of two DataFrames, with "\
                            "index 0 containing the target imputation data, and"\
                            " the data at index 1 containing additional data")
          if len(feed_data) != 2:
            raise TypeError("Input should be a list of two DataFrames, with "\
                            "index 0 containing the target imputation data, and"\
                            " the data at index 1 containing additional data")
          if not isinstance(feed_data[0], pd.DataFrame):
            raise TypeError("Input data must be in a DataFrame")
          if not isinstance(feed_data[1], pd.DataFrame):
            raise TypeError("Additional data must be in a DataFrame")
          na_loc = feed_data[0].notnull().astype(bool).values
          feedin = {self.X: feed_data[0].fillna(0).values,
                    self.X_add: feed_data[1].fillna(0).values}
          feed_data = feed_data[0]
        na_loc = feed_data.notnull().astype(bool).values

        y_out = pd.DataFrame(sess.run(self.output_op,feed_dict= feedin),
                                columns= self.imputation_target.columns)
        output_df = self.imputation_target.copy()
        output_df[np.invert(na_loc)] = y_out[np.invert(na_loc)]
        yield output_df

    return self

  def sample_from_z(self,
                    sample_size= 256,
                    verbose= True):
    """
    Method used to generate new samples by drawing on the default Student-T(3)
    sampling distribution. In effect, generates new data samples.
    Arguments:

      sample_size: Integer. Number of sample observations to draw at once.

      verbose: Boolean. Prints out messages.

    Returns:
      Sampled_output
    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")
    if not self.vae_layer:
      raise AttributeError("The model must include a VAE layer to be used to generate"\
                           " new observations from a latent distribution")
    if self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept pipeline data, either"\
                           " use 'pipeline_yield_samples' method or rebuild model "\
                           "with in-memory dataset.")
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      if verbose:
        print("Model restored.")
      feedin = {self.latent_inputs: np.zeros([sample_size, self.latent_space_size])}
      out = sess.run(self.gen_from_z_sample, feed_dict= feedin)
      sampled_output = pd.DataFrame(out,
        columns= self.imputation_target.columns)
    return sampled_output

  def transform_from_z(self,
                       data,
                       b_size= 256,
                       verbose= True):
    """
    Method used to generate new samples by drawing on the default Student-T(3)
    sampling distribution. In effect, generates new data samples.
    Arguments:

      data: Pandas dataframe or numpy array, as wide as latent_space_size. These
      numbers can be sampled from some distribution, or can be structured vectors
      to enable sweeping through the data space.

      b_size: Integer. Number of data entries to process at once. For managing
      larger input datasets, smaller numbers may be required.

      verbose: Boolean. Prints out messages.

    Returns:
      Generated_output
    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")
    if not self.vae_layer:
      raise AttributeError("The model must include a VAE layer to be used to generate"\
                           " new observations from a latent distribution")
    if self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept pipeline data, either"\
                           " use 'pipeline_yield_samples' method or rebuild model "\
                           "with in-memory dataset.")
    assert data.shape[1] == self.latent_space_size
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      if verbose:
        print("Model restored.")
      feed_data = data
      minibatch_list = []
      for batch in self._batch_iter_zsample(feed_data, b_size):
        feedin = {self.latent_inputs: batch}
        y_batch = pd.DataFrame(sess.run(self.gen_from_z_inputs,
                                      feed_dict= feedin),
                             columns= self.imputation_target.columns)
        minibatch_list.append(y_batch)
      generated_output = pd.DataFrame(pd.concat(minibatch_list, ignore_index= True),
                           columns= self.imputation_target.columns)
    return generated_output

  def inputs_to_z(self,
                  b_size= 256,
                  verbose= True):
    """
    Method used for transforming imputation_target into a latent representation
    for analysis. Can be used for observing how data behaves in a lower dimensional
    space, etc.

    Args:
      m: Integer. Number of imputations to generate.

      b_size: Integer. Number of data entries to process at once. For managing
      wider datasets, smaller numbers may be required.

      verbose: Boolean. Prints out messages.

    Returns:
      Self, z_mu, z_log_sigma
    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model"\
                           " can be trained")
    if not self.vae_layer:
      raise AttributeError("The model must include a VAE layer to be used to encode"\
                           " the dataset into the latent space")

    if self.input_is_pipeline:
      raise AttributeError("Model was constructed to accept pipeline data, either"\
                           " use 'pipeline_yield_samples' method or rebuild model "\
                           "with in-memory dataset.")
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      if verbose:
        print("Model restored.")
      feed_data = self.imputation_target.values
      mu_list = []
      sigma_list = []
      for batch in self._batch_iter_output(feed_data, b_size):
        if self.additional_data is not None:
          feedin = {self.X: batch[0], self.X_add: batch[1]}
        else:
          feedin = {self.X: batch}
        batch_mu, batch_sigma = sess.run(self.encode_to_z,
                                      feed_dict= feedin)
        batch_mu = pd.DataFrame(batch_mu)
        batch_sigma = pd.DataFrame(batch_sigma)
        mu_list.append(batch_mu)
        sigma_list.append(batch_sigma)
    x_mu = pd.concat(mu_list, ignore_index= True)
    x_log_sigma = pd.concat(sigma_list, ignore_index= True)
    return x_mu, x_log_sigma

  def change_imputation_target(self, new_target, additional_data= None):
    """
    Helper method to allow for imputed dataset to be hotswapped. MIDAS is not
    designed with such a function in mind, but this should allow for more flexible
    workflows.
    """
    if type(self.imputation_target) != type(new_target):
      raise ValueError("New target must be of same type as original target dataset")
    if type(self.imputation_target) == pd.core.series.Series:
      if self.imputation_target.name != new_target.name:
        raise ValueError("Ensure input series are from same source")
    elif type(self.imputation_target) == pd.core.frame.DataFrame:
      test_1 = new_target.shape[1] == self.imputation_target.shape[1]
      test_2 = new_target.columns.isin(self.imputation_target.columns).sum() \
      == new_target.shape[1]
      if not test_1 & test_2:
        raise ValueError("New target must have same columns as original target dataframe")
      if self.additional_data is not None:
        test_1 = new_target.shape[1] == self.additional_data.shape[1]
        test_2 = additional_data.columns.isin(self.additional_data.columns).sum() \
        == additional_data.shape[1]
        if not test_1 & test_2:
          raise ValueError("New target must have same columns as original target dataframe")
    else:
      raise ValueError("Target must be Pandas dataframe or series")
    self.imputation_target = new_target.copy()
    if self.additional_data is not None:
      self.additional_data = additional_data.copy()
      self.additional_data.fillna(0, inplace= True)
    self.na_matrix = self.imputation_target.notnull().astype(np.bool)
    self.imputation_target.fillna(0, inplace= True)
    return self




