# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:18:52 2017

@author: Alex
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse

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
               train_batch = 16,
               seed= None,
               query_input= False #Placeholder
               ):
    self.layer_structure = layer_structure
    self.learn_rate = learn_rate
    self.input_drop = input_drop
    self.model_built = False
    self.savepath = savepath
    self.model = None
    self.query_input = query_input
    self.additional_data = None
    self.train_batch = train_batch
    self.seed = None


  def _batch_iter(self,
                  train_data,
                  na_mask,
                  b_size = 16):
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
    indices = np.arange(train_data.shape[0])

    for start_idx in range(0, train_data.shape[0], b_size):
      excerpt = indices[start_idx:start_idx + b_size]
      if self.additional_data is None:
        yield train_data[excerpt]
      else:
        yield train_data[excerpt], self.additional_data.values[excerpt]

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
                       num_out,
                       scale= 1):
    weights.append(tf.Variable(tf.truncated_normal([num_in, num_out],
                                                   mean = 0,
                                                   stddev = scale / np.sqrt(num_in + num_out))))
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
                additional_data = None,
                verbose= True,
                crossentropy_adj= 1,
                loss_scale = 1):
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
      raise TypeError("Input data must be in a DataFrame")
    if imputation_target.isnull().sum().sum() == 0:
      raise ValueError("Imputation target contains no missing values. Please ensure missing values are encoded as type np.nan")
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
    self.size_index = size_index
    self.na_matrix = imputation_target.notnull().astype(bool)
    self.imputation_target = imputation_target.fillna(0)
    if additional_data is not None:
      self.additional_data = additional_data.fillna(0)

    #Build graph
    tf.reset_default_graph()
    self.graph = tf.Graph()
    with self.graph.as_default():
      if self.seed is not None:
        tf.set_random_seed(self.seed)
      self.X = tf.placeholder('float', [None, in_size])
      self.na_idx = tf.placeholder(tf.bool, [None, in_size])
      if additional_data is not None:
        self.X_add = tf.placeholder('float', [None, add_size])
      struc_list = self.layer_structure.copy()
      struc_list.insert(0, in_size + add_size)
      struc_list.append(in_size)
      _w = []
      _b = []

      for n in range(len(struc_list) -1):
        _w, _b = self._build_variables(weights= _w, biases= _b,
                                       num_in= struc_list[n],
                                       num_out= struc_list[n+1],
                                       scale= crossentropy_adj)

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

      if additional_data is not None:
        self.impute = impute(tf.concat([self.X, self.X_add], axis= 1))
      else:
        self.impute = impute(self.X)

      #Output functions
      output_list = []
      cost_list = []
      self.output_types = []

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
            self.output_types.append('rmse')
          elif cat_exists:
            output_list.append(tf.nn.sigmoid(pred_temp))
            cost_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels= t_t, logits= p_t)))
            self.output_types.append('bacc')

          else:
            p_t = tf.reshape(p_t, [-1, size_index[n]])
            t_t = tf.reshape(t_t, [-1, size_index[n]])

            output_list.append(tf.nn.softmax(pred_temp))
            cost_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels= t_t, logits= p_t)))
            self.output_types.append('sacc')

        elif n == 1:
          if cont_exists and cat_exists:
            output_list.append(tf.sigmoid(pred_temp))
            cost_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels= t_t, logits= p_t)))
            self.output_types.append('bacc')

          else:
            p_t = tf.reshape(p_t, [-1, size_index[n]])
            t_t = tf.reshape(t_t, [-1, size_index[n]])

            output_list.append(tf.nn.softmax(pred_temp))
            cost_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels= t_t, logits= p_t)))
            self.output_types.append('sacc')
        else:
          p_t = tf.reshape(p_t, [-1, size_index[n]])
          t_t = tf.reshape(t_t, [-1, size_index[n]])

          output_list.append(tf.nn.softmax(pred_temp))
          cost_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels= t_t, logits= p_t)/self.train_batch))
          self.output_types.append('sacc')


      #loss_agg = tf.reshape(tf.concat(1, cost_list), [-1, len(size_index)])
      self.output_op = tf.concat(output_list, axis= 1)

      self.joint_loss = tf.reduce_sum(cost_list) * loss_scale

      self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.joint_loss)
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
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
    if self.seed is not None:
      np.seed(self.seed)
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
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
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
        output_df[np.invert(self.na_matrix.values)] = y_out[np.invert(self.na_matrix.values)]
        self.output_list.append(output_df)
    return self

  def yield_samples(self,
                    m= 50,
                    verbose= True):
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
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
        output_df[np.invert(self.na_matrix.values)] = y_out[np.invert(self.na_matrix.values)]
        yield output_df
    return self

  def batch_generate_samples(self,
                             m= 50,
                             b_size= 256,
                             verbose= True):
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
        output_df[np.invert(self.na_matrix.values)] = y_out[np.invert(self.na_matrix.values)]
        self.output_list.append(output_df)
    return self

  def batch_yield_samples(self,
                             m= 50,
                             b_size= 256,
                             verbose= True):
    """
    This function is for a dataset large enough to be stored in memory, but
    too large to be passed into the model in its entirety. This may be due to
    GPU memory limitations, or just the size of the model
    """
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
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
        output_df[np.invert(self.na_matrix.values)] = y_out[np.invert(self.na_matrix.values)]
        yield output_df
    return self

  def overimpute(self,
                 spikein = 0.1,
                 training_epochs= 100,
                 report_ival = 10,
                 report_samples = 32,
                 verbose= True,
                 verbosity_ival= 1,
                 spike_seed= 42,
                 excessive= False
                 ):
    rmse_in = False
    sacc_in = False
    bacc_in = False
    if not self.model_built:
      raise AttributeError("The computation graph must be built before the model can be trained")
    if 'rmse' in self.output_types:
      rmse_in = True
    if 'sacc' in self.output_types:
      def sacc(true, pred, spike):
        a = np.argmax(true, 1)
        b = np.argmax(pred, 1)
        return np.sum(a[spike.flatten()] == b[spike.flatten()]) / np.sum(spike)
      sacc_in = True

    if 'bacc' in self.output_types:
      def bacc(true, pred, spike):
        pred = (pred > 0.5).astype(np.int_)
        return np.sum(true[spike] == pred[spike]) / np.sum(spike)
      bacc_in = True



    feed_data = self.imputation_target.copy()
    na_loc = self.na_matrix
    np.random.seed(spike_seed)
    n_softmax = 0
    break_list = list(np.cumsum(self.size_index))
    break_list.insert(0, 0)
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
          out, loss, _ = sess.run([self.output_op, self.joint_loss, self.train_step],
                             feed_dict= feedin)
          if excessive:
            print("Current cost:", loss)
            print(out)
          count +=1

          if not np.isnan(loss):
            run_loss += loss
        if verbose:
          if epoch % verbosity_ival == 0:
            print('Epoch:', epoch, ", loss:", str(run_loss/count))

        if epoch % report_ival == 0:
          single_rmse = 0
          single_sacc = 0
          single_bacc = 0
          first =  True
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
          for n in range(len(self.size_index)):
            temp_pred = y_out.iloc[:,break_list[n]:break_list[n+1]]
            temp_true = self.imputation_target.iloc[:,break_list[n]:break_list[n+1]]
            temp_spike = spike[:,break_list[n]:break_list[n+1]]
            if self.output_types[n] == 'sacc':
              temp_spike = temp_spike[:,0]
              agg_sacc = (1 - sacc(temp_true.values, temp_pred.values,
                                   temp_spike)) / n_softmax
            elif self.output_types[n] == 'rmse':
              agg_rmse = np.sqrt(mse(temp_true[temp_spike],
                                         temp_pred[temp_spike]))
            else:
              agg_bacc = 1 - bacc(temp_true.values, temp_pred.values, temp_spike)

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
            plt.plot(s_sacc, 'g-', label= "Individual softmax error")
            plt.plot(a_sacc, 'g--', label= "Aggregated softmax error")
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

          plt.title("Spike-in error levels as training progresses")
          plt.ylabel("Error (see documentation for details)")
          plt.legend()
          plt.ylim(ymin= 0)
          plt.xlabel("Report interval")
          plt.show()

      print("Overimputation complete. Adjust complexity as needed.")
      return self



















