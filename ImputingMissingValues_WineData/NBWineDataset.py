#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Read in & munge wine information dataset.  US wines only
wine_df=pd.read_csv('/Users/bonythomas/new.csv')
wine_df=wine_df.loc[wine_df.country=='US',['points','price','region_1', 'variety2','winery2']]
wine_df=wine_df.dropna(axis=0,how='any')

# Map point values to categories
bin_map={
    100:'90+',
    99:'90+',
    98:'90+',
    97:'90+',
    96:'90+',
    95:'90+',
    94:'90+',
    93:'90+',
    92:'90+',
    91:'90+',
    90:'90+',
    89:'<90',
    88:'<90',
    87:'<90',
    86:'<90',
    85:'<90',
    84:'<90',
    83:'<90',
    82:'<90',
    81:'<90',
    80:'<90',
    79:'<90',
    78:'<90',
    77:'<90',
    76:'<90'}
wine_df['point_bins'] = wine_df.points.map (bin_map)
wine_df.point_bins.unique ()  # Ensure no records are un-binned
wine_df = wine_df.drop ('points', axis=1)
obj_df = wine_df.select_dtypes(include=['object']).copy()
wine_df['point_bins']=pd.factorize(obj_df['point_bins'])[0]
# Split into train/test datasets
wine_train, wine_test = train_test_split (wine_df, test_size=0.15)

# Prepare train data for classification tree
regn_lab = LabelEncoder ().fit (np.unique (wine_df.region_1.values))
var_lab = LabelEncoder ().fit (np.unique (wine_df.variety2.values))
wnry_lab = LabelEncoder ().fit (np.unique (wine_df.winery2.values))
wine_train['regn_enc'] = regn_lab.transform (wine_train.region_1)
wine_train['var_enc'] = var_lab.transform (wine_train.variety2)
wine_train['wnry_enc'] = wnry_lab.transform (wine_train.winery2)
wine_train = wine_train.drop (['region_1', 'variety2', 'winery2'], axis=1)

# Prepare test data for classification tree
wine_test['regn_enc'] = regn_lab.transform (wine_test.region_1)
wine_test['var_enc'] = var_lab.transform (wine_test.variety2)
wine_test['wnry_enc'] = wnry_lab.transform (wine_test.winery2)
wine_test = wine_test.drop (['region_1', 'variety2', 'winery2'], axis=1)


# Train classification tree
x = wine_train.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']]
y = wine_train['point_bins']
x_test = wine_test.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']]
y_test = wine_test['point_bins']
clf = GaussianNB()
clf.fit(x, y)
clf.predict(x_test)
# Report classification results.  training dataset first, then test.
train_error = y == clf.predict (x)
test_error = wine_test['point_bins'] == clf.predict (wine_test.loc[:, ['price', \
                                                                       'regn_enc', 'var_enc', 'wnry_enc']])
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('   training accuracy: ', '{:.1%}'.format (sum (train_error) / len (
    train_error)))
print ('   test accuracy: ', '{:.1%}'.format (sum (test_error) / len (test_error)))

