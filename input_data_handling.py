"""
"Adult" dataset from the following source:
Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml/adult].
Irvine, CA: University of California, School of Information and Computer Science.

"News" dataset from the following source:
 K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support
 System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA
 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.
 Available athttps://archive.ics.uci.edu/ml/datasets/Online+News+Popularity

"""
import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
import requests
import os

directory = 'data/'
if not os.path.exists(directory):
    os.makedirs(directory)

n_samples = 25
cat_wipe = 2 # 1 in x
corruption_levels = [0.3, 0.5, 0.7, 0.9]

adult_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'sex',
               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
               'class_labels']
categorical = ['workclass', 'education', 'marital_status', 'occupation', 'relationship',
               'race', 'sex', 'native_country']

adult_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                          names= adult_names, header= None,
                          na_values= '?', skipinitialspace=True)
adult_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                          names= adult_names, header= None, skiprows = 1,
                          na_values= '?', skipinitialspace=True)
for n in categorical:
  adult_train[n] = adult_train[n].astype('category')
  adult_test[n] = adult_test[n].astype('category')
  adult_test['class_labels'] = adult_test['class_labels'].str.rstrip('.')
adult_data = pd.concat([adult_train, adult_test], axis=0, ignore_index= True)
adult_labels = adult_data['class_labels']
adult_data.drop('class_labels', axis= 1, inplace = True)



# MAR Corruption mask generator
np.random.seed(851)
nrow = adult_data.shape[0]
ncol = adult_data.shape[1]
corrupted = pd.DataFrame(np.zeros_like(adult_data),
                         index= adult_data.index,
                         columns= adult_data.columns)
for i in corruption_levels:
  columns = int(ncol * i)
  output_name = int(i * 10)
  if i == 1.0:
    columns = ncol
  for j in range(n_samples):
    filepath = "data/adult_MARtest_" + str(output_name) + "_sample_" + str(j) + ".csv"
    col_index = np.random.choice(np.arange(0, ncol),
                                 size= columns,
                                 replace= False)
    for n in col_index:
      if adult_data.columns[n] in categorical:
        rows = np.random.choice(np.arange(0, nrow),
                                size= int(nrow/cat_wipe),
                                replace= False)
      else:
        rows = np.random.choice(np.arange(0, nrow),
                                size= int(nrow/2),
                                replace= False)
      corrupted.iloc[rows, n] = 1
    print("Corruption:", str((corrupted.sum().sum() * 100)/(nrow*ncol)), "%")
    corrupted.to_csv(filepath)
    corrupted[:] = 0

#MNAR Corruption mask generator
#Some values removed due to extreme skew#
mnar_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
             'marital_status', 'occupation', 'relationship', 'race', 'sex',
             'hours_per_week', 'native_country']

for i in corruption_levels:
  columns = int(ncol * i)
  output_name = int(i * 10)
  if i == 1.0:
    columns = ncol

  for j in range(n_samples):
    filepath = "data/adult_MNARtest_" + str(output_name) + "_sample_" + str(j) + ".csv"
    col_index = np.random.choice(mnar_cols,
                                 size= columns,
                                 replace= False)
    corrupted = pd.DataFrame(np.zeros_like(adult_data),
                             index= adult_data.index,
                             columns= adult_data.columns)
    for colname in col_index:
      gating = str(adult_data[colname].dtypes)

      if colname in categorical:
        rows = np.random.choice(np.arange(0, nrow),
                                size= int(nrow/cat_wipe),
                                replace= False)
        bias = np.random.choice(['>50K', '<=50K'])
        mask = adult_labels == bias
        mask = mask.astype('int', inplace= True)
        mask.loc[rows] = 0
        corrupted[colname] = mask

      else:
        bias = np.random.choice(['high', 'low'])
        rows = np.random.choice(np.arange(0, nrow),
                                size= int(nrow/2),
                                replace= False)
        if bias == 'high':

          cutoff = adult_data[colname].quantile(0.75)
          mask = adult_data[colname] >= cutoff
          mask = mask.astype('int', inplace= True)
          mask.loc[rows] = 0
          corrupted[colname] = mask
        else:
          cutoff = adult_data[colname].quantile(0.25)
          mask = adult_data[colname] <= cutoff
          mask = mask.astype('int', inplace= True)
          mask.loc[rows] = 0
          corrupted[colname] = mask

    print("Corruption:", str((corrupted.sum().sum() * 100)/(nrow*ncol)), "%")
    corrupted.to_csv(filepath)
    corrupted[:] = 0

adult_data = pd.concat([adult_data, adult_labels], axis= 1, copy= False)
adult_data.to_csv("data/adult_data.csv")

r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip")
with ZipFile(BytesIO(r.content)) as z:
   with z.open("OnlineNewsPopularity/OnlineNewsPopularity.csv") as f:
      news_data = pd.read_csv(f, skipinitialspace=True)

news_data.drop(news_data.index[31037], axis= 0, inplace= True)
news_data = news_data.sample(frac= 1).reset_index(drop = True) #Extreme value, almost certainly erroneous
news_labels = news_data.shares
news_data.drop(['url', 'shares'], axis= 1, inplace= True)
nrow = news_data.shape[0]
ncol = news_data.shape[1]
corrupted = pd.DataFrame(np.zeros_like(news_data),
                         index= news_data.index,
                         columns= news_data.columns)
for i in corruption_levels:
  columns = int(ncol * i)
  output_name = int(i * 10)
  if i == 1.0:
    columns = ncol
  for j in range(n_samples):
    filepath = "data/news_MARtest_" + str(output_name) + "_sample_" + str(j) + ".csv"
    col_index = np.random.choice(np.arange(0, ncol),
                                 size= columns,
                                 replace= False)
    for n in col_index:
      rows = np.random.choice(np.arange(0, nrow),
                              size= int(nrow/2),
                              replace= False)
      corrupted.iloc[rows, n] = 1
    print("Corruption:", str((corrupted.sum().sum() * 100)/(nrow*ncol)), "%")
    corrupted.to_csv(filepath)
    corrupted[:] = 0


# Problematic values, where quantiles are all one value, have been excluded:
mnar_cols = ['timedelta', 'n_tokens_title', 'n_tokens_content', 'n_unique_tokens',
             'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs',
             'num_videos', 'average_token_length', 'num_keywords',
             'data_channel_is_lifestyle', 'data_channel_is_entertainment',
             'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech',
             'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min',
             'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg',
             'kw_avg_avg', 'self_reference_min_shares', 'self_reference_max_shares',
             'self_reference_avg_sharess', 'weekday_is_monday', 'weekday_is_tuesday',
             'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
             'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'LDA_00',
             'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
             'global_sentiment_polarity', 'global_rate_positive_words',
             'global_rate_negative_words', 'rate_positive_words', 'rate_negative_words',
             'avg_positive_polarity', 'min_positive_polarity', 'max_positive_polarity',
             'avg_negative_polarity', 'min_negative_polarity', 'max_negative_polarity',
             'title_subjectivity', 'title_sentiment_polarity', 'abs_title_subjectivity',
             'abs_title_sentiment_polarity']

binary = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus',
          'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world',
          'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday',
          'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday',
          'weekday_is_sunday', 'is_weekend']

for i in corruption_levels:
  columns = int(ncol * i)
  output_name = int(i * 10)
  if i == 1.0:
    columns = ncol
  for j in range(n_samples):
    filepath = "data/news_MNARtest_" + str(output_name) + "_sample_" + str(j) + ".csv"
    col_index = np.random.choice(mnar_cols,
                                 size= columns,
                                 replace= False)
    corrupted = pd.DataFrame(np.zeros_like(news_data),
                             index= news_data.index,
                             columns= news_data.columns)
    for colname in col_index:
      rows = np.random.choice(np.arange(0, nrow),
                              size= int(nrow/2),
                              replace= False)
      bias = np.random.choice(['high', 'low'])

      if colname in binary:
        if bias == 'high':
          cutoff = news_labels.quantile(0.66)
          mask = news_labels >= cutoff
        else:
          cutoff = news_labels.quantile(0.33)
          mask = news_labels <= cutoff
        mask = mask.astype('int', inplace= True)
        mask.loc[rows] = 0
        corrupted[colname] = mask

      else:
        if bias == 'high':
          cutoff = news_data[colname].quantile(0.66)
          mask = news_data[colname] >= cutoff
        else:
          cutoff = news_data[colname].quantile(0.33)
          mask = news_data[colname] <= cutoff
        mask = mask.astype('int', inplace= True)
        mask.loc[rows] = 0
        corrupted[colname] = mask

    print("Corruption:", str((corrupted.sum().sum() * 100)/(nrow*ncol)), "%")
    corrupted.to_csv(filepath)
    corrupted[:] = 0

news_data = pd.concat([news_data, news_labels], axis= 1, copy= False)
news_data.to_csv("data/news_data.csv")