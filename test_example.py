# -*- coding: utf-8 -*-
"""
This code is still heavily placeholder, with a lot of boilerplate code borrowed
from various open source repositories. Feel free to play around with hyperparams,
but obviously silly things (such as a 2048-0-1024 architecture) will cause the
code to throw errors

Created on Sun Jun 18 14:01:02 2017

@author: Alex
"""
from support_functions import (dataprep, log_transform, unlog_transform,
                               batch_iter, trn_test_split)
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



"""Parameters for the model"""
filepath = "/tmp/"
checkpoint_delay = 100
patience = 500
test_proportion = 0.1
batch_size = 128
display_step = 10
init_mean = 0.0
example_dataset_large = True  #Recommended, as the smaller dataset was mainly for debug
log_tx = False

"""Regularisation params"""
input_dropout = 0.5
model_dropout = 0.5
learn_rate = 3e-4
l2_lam = 1e-5

"""Architecture params"""
n_hidden_1 = 2048
n_hidden_2 = 1048
n_hidden_3 = 512
n_hidden_4 = 16
n_hidden_5 = 0
n_hidden_6 = 0











"""Input"""
tf.reset_default_graph() # Seeems to prevent errors


if example_dataset_large:  # The dropped columns contain no datapoints
  raw = pd.read_csv("GEMAsia.csv", na_values= "..")
  raw = raw.iloc[0:10950,:].drop(["Time Code", "Country Code",
                'J.P. Morgan Emerging Markets Bond Index(EMBI+),,,, [EMBIGI]',
                'CPI Price, % y-o-y, median weighted, seas. adj., [CPTOTSAXMZGY]',
                'GDP,constant 2010 LCU,millions,seas. adj., [NYGDPMKTPSAKN]',
                'GDP,constant 2010 US$,millions,seas. adj., [NYGDPMKTPSAKD]',
                'GDP,current LCU,millions,seas. adj., [NYGDPMKTPSACN]',
                'GDP,current US$,millions,seas. adj., [NYGDPMKTPSACD]'], 1)
else:
  raw = pd.read_csv("GEM.csv", na_values= "NA")
  raw = raw.iloc[0:2650,:].drop(["Time Code", "Country Code",
                'J.P. Morgan Emerging Markets Bond Index(EMBI+),,,, [EMBIGI]'], 1)

"""Normalising and scaling"""
scaler = MinMaxScaler()

raw, na_raw = dataprep(raw, 'Country', 'Time')
na_array = na_raw.as_matrix()
if log_tx:
  raw, unlogged_mins = log_transform(raw)
data = scaler.fit_transform(raw)
data[np.invert(na_array)] = 0

"""Placeholders, splits and transformed inputs"""

n_input = data.shape[1]
train, train_na, test, test_na = trn_test_split(data, na_array, test_proportion)

X = tf.placeholder('float', [None, n_input], name= 'X')
na_idx = tf.placeholder(tf.bool, [None, n_input], name= 'na_idx')
input_noise = tf.placeholder('float', name = 'input_noise')
drop_rate = tf.placeholder('float', name = 'drop_rate')

"""Computation graph"""
weights = {}
biases = {}

weights['E_1'] = tf.Variable(tf.truncated_normal([n_input, n_hidden_1],
                                              mean= init_mean,
                                              stddev= (1/ np.sqrt(n_input + n_hidden_1))))
weights['D_1'] = tf.Variable(tf.truncated_normal([n_hidden_1, n_input],
                                              mean= init_mean,
                                              stddev= (1/ np.sqrt(n_input + n_hidden_1))))

biases['E_1'] = tf.Variable(tf.zeros([n_hidden_1]))
biases['D_1'] = tf.Variable(tf.zeros(n_input))

weights['E_2'] = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_1 + n_hidden_2))))
weights['D_2'] = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_1 + n_hidden_2))))
biases['E_2'] = tf.Variable(tf.zeros([n_hidden_2]))
biases['D_2'] = tf.Variable(tf.zeros([n_hidden_1]))

weights['E_3'] = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_2 + n_hidden_3))))
weights['D_3'] = tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_2 + n_hidden_3))))
biases['E_3'] = tf.Variable(tf.zeros([n_hidden_3]))
biases['D_3'] = tf.Variable(tf.zeros([n_hidden_2]))

weights['E_4'] = tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_3 + n_hidden_4))))
weights['D_4'] = tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_3 + n_hidden_4))))
biases['E_4'] = tf.Variable(tf.zeros([n_hidden_4]))
biases['D_4'] = tf.Variable(tf.zeros([n_hidden_3]))

weights['E_5'] = tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_4 + n_hidden_5))))
weights['D_5'] = tf.Variable(tf.truncated_normal([n_hidden_5, n_hidden_4],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_4 + n_hidden_5))))
biases['E_5'] = tf.Variable(tf.zeros([n_hidden_5]))
biases['D_5'] = tf.Variable(tf.zeros([n_hidden_4]))

weights['E_6'] = tf.Variable(tf.truncated_normal([n_hidden_5, n_hidden_6],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_5 + n_hidden_6))))
weights['D_6'] = tf.Variable(tf.truncated_normal([n_hidden_6, n_hidden_5],
                                          mean= init_mean,
                                          stddev= (1/ np.sqrt(n_hidden_5 + n_hidden_6))))
biases['E_6'] = tf.Variable(tf.zeros([n_hidden_6]))
biases['D_6'] = tf.Variable(tf.zeros([n_hidden_5]))


def encoder(x):
  model = tf.nn.dropout(x, input_noise) ##Input masking
  model = tf.nn.tanh(tf.matmul(model, weights['E_1']) + biases['E_1'])

  if n_hidden_2 != 0:
    model = tf.nn.dropout(model, drop_rate)
    model = tf.nn.tanh(tf.matmul(model, weights['E_2']) + biases['E_2'])

  if n_hidden_3 != 0:
    model = tf.nn.dropout(model, drop_rate)
    model = tf.nn.tanh(tf.matmul(model, weights['E_3']) + biases['E_3'])

  if n_hidden_4 != 0:
    model = tf.nn.dropout(model, drop_rate)
    model = tf.nn.tanh(tf.matmul(model, weights['E_4']) + biases['E_4'])

  if n_hidden_5 != 0:
    model = tf.nn.dropout(model, drop_rate)
    model = tf.nn.tanh(tf.matmul(model, weights['E_5']) + biases['E_5'])

  if n_hidden_6 != 0:
    model = tf.nn.dropout(model, drop_rate)
    model = tf.nn.tanh(tf.matmul(model, weights['E_6']) + biases['E_6'])

  return model

def decoder(model):
  if n_hidden_6 != 0:
    model = tf.nn.tanh(tf.matmul(model, weights['D_6']) + biases['D_6'])
    model = tf.nn.dropout(model, drop_rate)

  if n_hidden_5 != 0:
    model = tf.nn.tanh(tf.matmul(model, weights['D_5']) + biases['D_5'])
    model = tf.nn.dropout(model, drop_rate)

  if n_hidden_4 != 0:
    model = tf.nn.tanh(tf.matmul(model, weights['D_4']) + biases['D_4'])
    model = tf.nn.dropout(model, drop_rate)

  if n_hidden_3 != 0:
    model = tf.nn.tanh(tf.matmul(model, weights['D_3']) + biases['D_3'])
    model = tf.nn.dropout(model, drop_rate)

  if n_hidden_2 != 0:
    model = tf.nn.tanh(tf.matmul(model, weights['D_2']) + biases['D_2'])
    model = tf.nn.dropout(model, drop_rate)

  model = tf.matmul(model, weights['D_1']) + biases['D_1']

  return model

encode_op = encoder(X)
decode_op = decoder(encode_op)

y_pred = decode_op
y_true = X

cost = tf.reduce_mean(tf.pow(tf.subtract(tf.boolean_mask(y_true, na_idx),
                        tf.boolean_mask(y_pred, na_idx)), 2))

l2_loss = (tf.nn.l2_loss(weights['E_1']) +
           tf.nn.l2_loss(weights['E_2']) +
           tf.nn.l2_loss(weights['E_3']) +
           tf.nn.l2_loss(weights['E_4']) +
           tf.nn.l2_loss(weights['E_5']) +
           tf.nn.l2_loss(weights['E_6']) +
           tf.nn.l2_loss(weights['D_1']) +
           tf.nn.l2_loss(weights['D_2']) +
           tf.nn.l2_loss(weights['D_3']) +
           tf.nn.l2_loss(weights['D_4']) +
           tf.nn.l2_loss(weights['D_5']) )

loss = cost + (l2_lam * l2_loss)

train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


"""Training and imputation loop"""

"""A lot of the output here is for my own benchmarking, diagnosis and monitoring.
Seeing as this is an experimental release, I've left it in so people can test
their own data more easily. Subsequent versions designed for general use will
be much more streamlined. Returns a maximum likelihood estimation"""

def train_model(save_checkpoint= False, load_checkpoint= False):

  with tf.Session() as sess:
    sess.run(init)
    print("Model init...")
    if load_checkpoint:
      tf.reset_default_graph()
      saver.restore(sess, filepath)
      print("Model restored.")

    best = 1.0
    last = 1.0
    last_test = 1.0
    c_imp = 1.0
    epochs_since_new_best = 0
    previous_longest = 0
    rms_corr_error = 1
    best_rms_corr_error = 1
    corr_error_min = 0

    prev_last_1 = 0
    prev_last_2 = 0
    prev_last_3 = 0
    prev_last_4 = 0
    prev_last_5 = 0
    prev_last_test_1 = 0
    prev_last_test_2 = 0
    prev_last_test_3 = 0
    prev_last_test_4 = 0
    prev_last_test_5 = 0
    print("Running averages initialised")

    # Training cycle
    for epoch in range(15000):
      c = 0
      batch_count = 0
      for batch in batch_iter(train, train_na):
        x_batch, na_batch = batch
        _, c_batch = sess.run([train_step, cost],
                              feed_dict= {X: x_batch,
                                          na_idx: na_batch,
                                          input_noise: input_dropout,
                                          drop_rate: model_dropout})
        c += c_batch
        batch_count += 1

      c = c / batch_count
      last += c
      c_test = sess.run(cost, feed_dict= {X: test,
                                          na_idx: test_na,
                                          input_noise: 1,
                                          drop_rate: 1})

      last_test += c_test

      if c_test < best:
        if previous_longest < epochs_since_new_best:
          previous_longest = epochs_since_new_best
        best = c_test
        epochs_since_new_best = 0
        print("New minimum test loss: ", str(np.sqrt(best) * 100), "%")


        if epoch > 0: #For sanity and time
          corrupted_error = 0
          print("Testing reconstruction error:")
          for n in range(test.shape[1]): # Unsupervised error check
            corrupted = test.copy()
            corrupted[:, n] = 0
            y_corr = sess.run(y_pred, feed_dict= {X: corrupted,
                                                  na_idx: test_na,
                                                  input_noise: 1,
                                                  drop_rate: 1})
            y_corr[np.invert(test_na)] = 0
            corrupted_error += mean_squared_error(test[:,n], y_corr[:,n])
          rms_corr_error = np.sqrt(corrupted_error / test.shape[1])
          print("Reconstruction loss: ", str(rms_corr_error * 100), "%")
          print()
          if rms_corr_error < best_rms_corr_error:
            print("    New minimum logged.")
            print()
            print("Updating imputed values")
            best_rms_corr_error = rms_corr_error
            corr_error_min = epoch
            y_imp, c_imp = sess.run([y_pred, cost], feed_dict= {X: data,
                                          na_idx: na_array,
                                          input_noise: 1,
                                          drop_rate: 1})
            print("Imputed loss: ", str(np.sqrt(c_imp) * 100), "%")
            print()
            if save_checkpoint and epoch > checkpoint_delay:
              print("Checkpoint!")
              print()
              save_path = saver.save(sess, filepath)
              print("Model saved in file: %s" % save_path)
              print()
      else:
        epochs_since_new_best += 1

      # Display logs per epoch step
      if epoch % display_step == 0:
        last = last / display_step
        trend = (prev_last_1 +
                 prev_last_2 +
                 prev_last_3 +
                 prev_last_4 +
                 prev_last_5) / 5

        trend = last - trend

        print("Epoch:", '%04d' % (epoch))
             # "cost=", "{:.9f}".format(c))
        print("Cost:", c)
        print("Encode loss:", str(np.sqrt(c) * 100), "%")
        if trend > 0:
          print("Trend:", trend, "  XXXX")
        else:
          print("Trend:", trend)
        print()

        prev_last_5 = prev_last_4
        prev_last_4 = prev_last_3
        prev_last_3 = prev_last_2
        prev_last_2 = prev_last_1
        prev_last_1 = last
        last = 0


        last_test = last_test / display_step
        trend_test = (prev_last_test_1 +
                      prev_last_test_2 +
                      prev_last_test_3 +
                      prev_last_test_4 +
                      prev_last_test_5) / 5

        trend_test = last_test - trend_test

        print("  Test loss:", str(np.sqrt(c_test) * 100), "%")
        print("  Best test loss:", str(np.sqrt(best) * 100), "%")
        if trend_test > 0:
          print("  Test cost trend:", trend_test, "  XXXX")
        else:
          print("  Test cost trend:", trend_test)
        print()
        print("    Lowest for", epochs_since_new_best,
              "epochs, previous record:", previous_longest)
        print("    Best imputed loss: ", str(np.sqrt(c_imp) * 100), "%")
        print()

        if trend <= 0 and trend_test > 0:

          print("        POSSIBLE OVERTRAINING")
          print()
        print("      Lowest reconstuction error:")
        print("     ", str(rms_corr_error * 100), "% at epoch", corr_error_min)
        print()
        prev_last_test_5 = prev_last_test_4
        prev_last_test_4 = prev_last_test_3
        prev_last_test_3 = prev_last_test_2
        prev_last_test_2 = prev_last_test_1
        prev_last_test_1 = last_test
        last_test = 0

        if epoch > 0: #For sanity and time
          corrupted_error = 0
          print("Testing reconstruction error:")
          for n in range(test.shape[1]): # Unsupervised error check
            corrupted = test.copy()
            corrupted[:, n] = 0
            y_corr = sess.run(y_pred, feed_dict= {X: corrupted,
                                                  na_idx: test_na,
                                                  input_noise: 1,
                                                  drop_rate: 1})
            y_corr[np.invert(test_na)] = 0
            corrupted_error += mean_squared_error(test[:,n], y_corr[:,n])
          rms_corr_error = np.sqrt(corrupted_error / test.shape[1])
          print("Reconstruction loss: ", str(rms_corr_error * 100), "%")
          print()
          if rms_corr_error < best_rms_corr_error:
            print("    New minimum logged.")
            print()
            print("Updating imputed values")
            best_rms_corr_error = rms_corr_error
            corr_error_min = epoch
            y_imp, c_imp = sess.run([y_pred, cost], feed_dict= {X: data,
                                                                na_idx: na_array,
                                                                input_noise: 1,
                                                                drop_rate: 1})
            print("Imputed loss: ", str(np.sqrt(c_imp) * 100), "%")
            print()
            if save_checkpoint and epoch > checkpoint_delay:
              print("Checkpoint!")
              print()
              save_path = saver.save(sess, filepath)
              print("Model saved in file: %s" % save_path)
              print()

      if epochs_since_new_best and (epoch - corr_error_min) > patience:
        print("Consider my patience tested")
        break

  if save_checkpoint:
    return y_imp, saver
  else:
    return y_imp

"""The following functions require a saved model to load. Fortunately, the train
op is configured to save at the point where the unsupervised error is lowest."""

def bayesMI_point_estimate(samples= 50):
  with tf.Session as sess:
    tf.reset_default_graph()
    saver.restore(sess, filepath)
    print("Model restored.")

    for n in range(samples):
      y_out, c_out = sess.run([y_pred, cost], feed_dict= {X: data,
                                          na_idx: na_array,
                                          input_noise: 1,
                                          drop_rate: model_dropout})
      if n == 0:
        output = y_out
      else:
        output += y_out
      print("Sample", n, "error: ", str(np.sqrt(c_out) * 100), "%")

  output = output / samples
  error = np.sqrt(mean_squared_error(data[na_array], output[na_array]))
  print("Root mean squared error: ", error)
  output[na_array] = data[na_array]

  return output

"""For multiple (over) imputation, call the following function as many times
as desired. As of yet, there is no function to re-unify the outputs. Simply run
any analysis on the output of a single call, then average the results."""


def generate_sample(overimpute = False):
  with tf.Session as sess:
    tf.reset_default_graph()
    saver.restore(sess, filepath)
    print("Model restored.")

    y_out = sess.run(y_pred, feed_dict= {X: data,
                                         na_idx: na_array,
                                         input_noise: 1,
                                         drop_rate: model_dropout})
  if overimpute:
    return y_out
  else:
    y_out[na_array] = data[na_array]
    return y_out

def rescale(x, original_dataframe= raw, logged= False):
  values = scaler.inverse_transform(x)
  if logged:
    values = unlog_transform(x, unlogged_mins)
  out = original_dataframe.copy()
  out.iloc[:,:] = values

  return out

y_imp, saver = train_model(save_checkpoint= True)
