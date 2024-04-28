import MIDASpy.midas_base as md
import numpy as np
import pandas as pd

df_c = pd.DataFrame({
    'xn': np.random.randn(100),
    'xp': np.random.randint(low = 0, high = 100, size = 100),
    'xb': np.random.randint(low = 0, high = 1, size = 100),
    'xc': np.random.choice(['A', 'B', 'C'], size = 100, replace = True)
    }
)

df_cat, cat_col_list = md.cat_conv(df_c[['xc']])

df_c.drop(columns = ['xc'], inplace = True)
df_c = pd.concat([df_c, df_cat], axis = 1)

def spike_in_generation(data):
    spike_in = pd.DataFrame(np.zeros_like(data), columns= data.columns)
    for column in data.columns:
        subset = np.random.choice(data[column].index[data[column].notnull()], int(data.shape[0]*0.3), replace= False)
        spike_in.loc[subset, column] = 1
    return spike_in

spike = spike_in_generation(df_c)
df_m = df_c.copy()

df_m[spike == 1] = np.nan


mid = md.Midas()
mid.build_model(df_m, binary_columns=['xb'], softmax_columns=cat_col_list, positive_columns=['xp'])
mid.train_model(training_epochs = 10)



df_m2 = df_m.drop('xp', axis = 1, inplace = False)
mid = md.Midas()
mid.build_model(df_m2, binary_columns=['xb'], softmax_columns=cat_col_list)
mid.train_model(training_epochs = 10)


df_m3 = df_m.drop('xb', axis = 1, inplace = False)
mid = md.Midas()
mid.build_model(df_m3, positive_columns=['xp'], softmax_columns=cat_col_list)
mid.train_model(training_epochs = 10)

df_m4 = df_m.drop(['A','B','C'], axis = 1, inplace = False)
mid = md.Midas()
mid.build_model(df_m4, binary_columns=['xb'], positive_columns=['xp'])
mid.train_model(training_epochs = 10)


df_m5 = df_m.drop(['A','B','C','xn'], axis = 1, inplace = False)
mid = md.Midas()
mid.build_model(df_m5, binary_columns=['xb'], positive_columns=['xp'])
mid.train_model(training_epochs = 10)

df_m5 = df_m.drop(['A','B','C','xn'], axis = 1, inplace = False)
mid = mdp.Midas()
mid.build_model(df_m5, binary_columns=['xb'])
mid.train_model(training_epochs = 10)


import MIDASpy as mdp



# 

