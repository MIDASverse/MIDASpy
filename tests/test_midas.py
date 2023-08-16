import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import csv
import MIDASpy as md

def test_some_functionality():
    # Load the data
    np.random.seed(441)
    data_path = os.path.join(os.path.dirname(__file__), "test_data", "adult_data.csv")
    data_0 = pd.read_csv(data_path)
    data_0.columns.str.strip()

    def spike_in_generation(data):
        spike_in = pd.DataFrame(np.zeros_like(data), columns= data.columns)
        for column in data.columns:
            subset = np.random.choice(data[column].index[data[column].notnull()], 5000, replace= False)
            spike_in.loc[subset, column] = 1
        return spike_in

    spike_in = spike_in_generation(data_0)
    original_value = data_0.loc[4, 'hours_per_week']
    data_0[spike_in == 1] = np.nan

    categorical = ['workclass','marital_status','relationship','race','class_labels','sex','education','occupation','native_country']
    data_cat, cat_cols_list = md.cat_conv(data_0[categorical])

    data_0.drop(categorical, axis = 1, inplace = True)
    constructor_list = [data_0]
    constructor_list.append(data_cat)
    data_in = pd.concat(constructor_list, axis=1)

    na_loc = data_in.isnull()
    data_in[na_loc] = np.nan

    imputer = md.Midas(layer_structure = [256,256], vae_layer = False, seed = 89, input_drop = 0.75)
    imputer.build_model(data_in, softmax_columns = cat_cols_list)
    imputer.train_model(training_epochs = 2)

    imputations = imputer.generate_samples(m=2).output_list
    model = md.combine(y_var = "capital_gain", X_vars = ["education_num","age"], df_list = imputations)
