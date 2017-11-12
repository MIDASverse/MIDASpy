# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:18:52 2017

@author: Alex
"""

import numpy as np
import pandas as pd
import tensorflow as tf

class MIDAS(object):
  """
  All categorical variables have to be converted to onehot before the
  algorithm can process everything. I might integrate this process into
  the workflow later, should pandas get a reverse_dummies function that
  would allow the reverse transform. For now, all input must take the
  form of a pandas dataframe
  """

  def __init__(self,
               layer_structure= [256, 256, 256],
               learn_rate= 1e-4,
               input_drop= 0.8,
               savepath= 'tmp/MIDAS',
               query_input= False #Placeholder
               ):
    self.layer_structure = layer_structure
    self.learn_rate = learn_rate
    self.input_drop = input_drop
    self.model_built = False
    self.savepath = savepath
    self.model = None
    self.query_input = query_input


  def _batch_iter(self,
                  train_data,
                  na_mask,
                  b_size = 16):
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, train_data.shape[0] - b_size + 1, b_size):
      excerpt = indices[start_idx:start_idx + b_size]
      yield train_data[excerpt], na_mask[excerpt]

  def _batch_iter_output(self,
                  train_data,
                  b_size = 256):
    indices = np.arange(train_data.shape[0])

    for start_idx in range(0, train_data.shape[0], b_size):
      excerpt = indices[start_idx:start_idx + b_size]
      yield train_data[excerpt]

  def _build_layer(self,
                   X,
                   weight_matrix,
                   bias_vec,
                   dropout_rate= 0.5,
                   output_layer= False):
    X_tx = tf.matmul(tf.nn.dropout(X, dropout_rate), weight_matrix) + bias_vec
    if output_layer:
      return X_tx
    else:
      return tf.nn.elu(X_tx)

  def _build_variables(self,
                       weights,
                       biases,
                       num_in,
                       num_out):
    weights.append(tf.Variable(tf.truncated_normal([num_in, num_out],
                                                   mean = 0,
                                                   stddev = 1 / np.sqrt(num_in + num_out))))
    biases.append(tf.Variable(tf.zeros([num_out]))) #Bias can be zero
    return weights, biases

  def _sort_cols(self, data, subset):
    data_1 = data[subset]
    data_0 = data.drop(subset, axis= 1)
    chunk = data_1.shape[1]
    return pd.concat([data_0, data_1], axis= 1), chunk

  def build_model(self,
                imputation_target,
                categorical_columns= None,
                softmax_columns= None,
                unsorted= True,
                verbose= True):
    """
    The categorical columns should be a list of column names. Softmax columns
    should be a list of lists of column names. This will allow the model to
    dynamically assign cost functions to the correct variables. If, however,
    the data comes pre-sorted, arranged can be set to "true", in which case
    the arguments can be passed in as integers of size, ie. shape[1] attributes
    for each of the relevant categories.

    In other words, pre-sort your data and pass in the integers, so indexing
    dynamically doesn't become too difficult.
    """
    if not isinstance(imputation_target, pd.DataFrame):
      raise TypeError("Input data must be contained within a DataFrame")

    self.original_columns = imputation_target.columns
    cont_exists = False
    cat_exists = False
    in_size = imputation_target.shape[1]

    # Establishing indices for cost function
    size_index = []
    if categorical_columns is not None:
      if unsorted:
        imputation_target, chunk = self._sort_cols(imputation_target,
                                                   categorical_columns)
        size_index.append(chunk)
      else:
        size_index.append(categorical_columns)
      cat_exists = True
    if softmax_columns is not None:
      if unsorted:
        for subset in softmax_columns:
          imputation_target, chunk = self.sort_cols(imputation_target,
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

    self.na_matrix = imputation_target.notnull().astype(bool).values
    self.imputation_target = imputation_target.fillna(0)

    #Build graph
    tf.reset_default_graph()
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.X = tf.placeholder('float', [None, in_size])
      self.na_idx = tf.placeholder(tf.bool, [None, in_size])

      struc_list = self.layer_structure.copy()
      struc_list.insert(0, in_size)
      struc_list.append(in_size)
      _w = []
      _b = []

      for n in range(len(struc_list) -1):
        _w, _b = self._build_variables(weights= _w, biases= _b,
                                       num_in= struc_list[n],
                                       num_out= struc_list[n+1])

      def impute(X):
        for n in range(len(struc_list) -1):
          if n == 0:
            X = self._build_layer(X, _w[n], _b[n],
                                  dropout_rate = self.input_drop)
          elif (n+1) == (len(struc_list) -1):
            X = self._build_layer(X, _w[n], _b[n],
                                  output_layer = True)
          else:
            X = self._build_layer(X, _w[n], _b[n])
        return X
      self.impute = impute(self.X)

      #Output functions
      output_list = []
      cost_list = []

      for n in range(len(size_index)):
        na_temp = tf.split(self.na_idx, size_index, axis= 1)[n]
        pred_temp = tf.split(self.impute, size_index, axis= 1)[n]
        true_temp = tf.split(self.X, size_index, axis= 1)[n]
        p_t = tf.boolean_mask(pred_temp, na_temp)
        t_t = tf.boolean_mask(true_temp, na_temp)

        if n == 0:
          if cont_exists:
            output_list.append(pred_temp)
            cost_list.append(tf.reduce_mean(tf.squared_difference(t_t, p_t)))
          elif cat_exists:
            output_list.append(tf.nn.sigmoid(pred_temp))
            cost_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels= t_t, logits= p_t)))
          else:
            output_list.append(tf.nn.softmax(pred_temp))
            cost_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels= t_t, logits= p_t)))
        elif n == 1:
          if cont_exists and cat_exists:
            output_list.append(tf.sigmoid(pred_temp))
            cost_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels= t_t, logits= p_t)))
          else:
            output_list.append(tf.nn.softmax(pred_temp))
            cost_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels= t_t, logits= p_t)))
        else:
          output_list.append(tf.nn.softmax(pred_temp))
          cost_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels= t_t, logits= p_t)))

      self.output_op = tf.concat(output_list, axis= 1)
      self.joint_loss = tf.reduce_sum(cost_list)

      self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.joint_loss)
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()

    self.model_built = True
    print()
    print("Computation graph constructed")
    print()
    return self

  def train_model(self,
                  training_epochs= 100,
                  batch_size= 16,
                  verbose= True):
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
    feed_data = self.imputation_target.copy().values
    na_loc = self.na_matrix.copy()
    with tf.Session(graph= self.graph) as sess:
      sess.run(self.init)
      print("Model initialised")
      print()
      for epoch in range(training_epochs):
        count = 0
        run_loss = 0
        for batch in self._batch_iter(feed_data, na_loc, batch_size):
          batch_data, batch_mask = batch
          loss, _ = sess.run([self.joint_loss, self.train_step],
                             feed_dict= {self.X: batch_data,
                                         self.na_idx: batch_mask})
          count +=1
          run_loss += loss
        if verbose:
          print('Epoch:', epoch, ", loss:", str(run_loss/count))
      print("Training complete. Saving file...")
      save_path = self.saver.save(sess, self.savepath)
      print("Model saved in file: %s" % save_path)
      return self

  def generate_samples(self,
                       m= 50):
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
    self.output_list = []
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      print("Model restored.")
      for n in range(m):
        feed_data = self.imputation_target.copy().values
        y_out = pd.DataFrame(sess.run(self.output_op,
                                             feed_dict= {self.X: feed_data}),
                                columns= self.imputation_target.columns)
        output_df = self.imputation_target.copy()
        output_df[np.invert(self.na_matrix)] = y_out[np.invert(self.na_matrix)]
        self.output_list.append(output_df)
    return self

  def yield_samples(self,
                    m= 50):
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      print("Model restored.")
      for n in range(m):
        feed_data = self.imputation_target.copy().values
        y_out = pd.DataFrame(sess.run(self.output_op,
                                           feed_dict= {self.X: feed_data}),
                                columns= self.imputation_target.columns)
        output_df = self.imputation_target.copy()
        output_df[np.invert(self.na_matrix)] = y_out[np.invert(self.na_matrix)]
        yield output_df
    return self

  def batch_generate_samples(self,
                             m= 50,
                             b_size= 256):
    """
    This function is for a dataset large enough to be stored in memory, but
    too large to be passed into the model in its entirety. This may be due to
    GPU memory limitations, or just the size of the model
    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
    self.output_list = []
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      print("Model restored.")
      for n in range(m):
        feed_data = self.imputation_target.copy().values
        minibatch_list = []
        for batch in self._batch_iter_output(feed_data, b_size):
          y_batch = pd.DataFrame(sess.run(self.output_op,
                                        feed_dict= {self.X: batch}),
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
                             b_size= 256):
    """
    This function is for a dataset large enough to be stored in memory, but
    too large to be passed into the model in its entirety. This may be due to
    GPU memory limitations, or just the size of the model
    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
    with tf.Session(graph= self.graph) as sess:
      self.saver.restore(sess, self.savepath)
      print("Model restored.")
      for n in range(m):
        feed_data = self.imputation_target.copy().values
        minibatch_list = []
        for batch in self._batch_iter_output(feed_data, b_size):
          y_batch = pd.DataFrame(sess.run(self.output_op,
                                        feed_dict= {self.X: batch}),
                               columns= self.imputation_target.columns)
          minibatch_list.append(y_batch)
        y_out = pd.DataFrame(pd.concat(minibatch_list, ignore_index= True),
                             columns= self.imputation_target.columns)
        output_df = self.imputation_target.copy()
        output_df[np.invert(self.na_matrix)] = y_out[np.invert(self.na_matrix)]
        yield output_df
    return self





















