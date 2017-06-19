# -*- coding: utf-8 -*-
"""
This file contains the various major functions to be deployed in the main code.

Order of operations should be as follows:
  -Import data, remove superfluous columns
  -Mark and collect categorical variables
  -Prep the data and note NA locations

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def categorical_handler(df,
                        continuous_column_indices,
                        class_column_indices,
                        iloc = True):
  """This moves the categorical columns to one side for ease of handling during
  model training. Ordinal categories can be treated as continuous variables...
  this is mainly for columns with no inherent ordering. All columns have been
  made explicit to prevent issues later in the program"""
  if iloc:
    cont_data = df.iloc[:, continuous_column_indices]
    cat_data = df.iloc[:, class_column_indices]
  elif iloc == False:
    cont_data = df.loc[continuous_column_indices]
    cat_data = df.loc[class_column_indices]
  else:
    raise ValueError

  out = pd.concat([cont_data, cat_data], axis = 1)
  cat_indices = [int(cont_data.shape[1]), int(out.shape[1])]
  return out, cat_indices


def dataprep(df,
             geocol,
             timecol,
             categorical_data = False,
             cat_indices = None):
  """This is to generate our NA index, as well as being ready for later modules.
  Reverse onehot isn't implemented, as its assumed you'll run your analysis direct
  on the output. It may be implemented in future"""
  data = df.set_index([geocol, timecol])

  if categorical_data:
    if cat_indices == None:
      raise ValueError
    categories = data.iloc[:, cat_indices[0]: cat_indices[1]]
    onehot = pd.get_dummies(categories)
    continuous = data.iloc[:, 0:cat_indices[0]]
    na_index_continuous = continuous.notnull().as_matrix()
    na_index_onehot = onehot.notnull().as_matrix()
    continuous = continuous.fillna(continuous.mean())

    return continuous, onehot, na_index_continuous, na_index_onehot

  else:
    na_index = data.notnull()
    data = data.fillna(data.median())

    return data, na_index

def na_locator(df):
  """This is for if you have fully cleaned data ready to impute"""
  return df.notnull()


def log_transform(df):
  """If using the log transform, maximise input data to minimise issues regarding
  negative values being fed to the exponent function on the reverse transform"""
  unlogged_mins = df.min()

  for n in range(df.shape[1]): # Log scale between 0 and log(max)
    df.iloc[:,n] = np.log1p(df.iloc[:,n] - unlogged_mins[n])

  return df, unlogged_mins

def trn_test_split(data, na_index, test_proportion):
  test_split = int(data.shape[0] - (data.shape[0] * test_proportion))
  train = data.copy()
  train_na = na_index.copy()

  seed = np.random.get_state()
  np.random.shuffle(train)
  np.random.set_state(seed)
  np.random.shuffle(train_na)
  test = train[test_split:, :]
  test_na = train_na[test_split:, :]
  train = train[0:test_split, :]
  train_na = train_na[0:test_split, :]

  return train, train_na, test, test_na

def batch_iter(inputs, targets, batch_size = 128):
  assert inputs.shape[0] == targets.shape[0]
  indices = np.arange(inputs.shape[0])
  np.random.shuffle(indices)

  for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
    excerpt = indices[start_idx:start_idx + batch_size]

    yield inputs[excerpt], targets[excerpt]

def unlog_transform(df, unlogged_mins):
  """Used for sanity-checking data, or for any reason you need to recover the
  original values"""
  for n in range(df.shape[1]):
    df.iloc[:,n] = np.expm1(df.iloc[:,n]) + unlogged_mins[n]
  return df



