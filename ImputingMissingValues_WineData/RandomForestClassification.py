#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn.tree import export_graphviz
import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pydotplus
import os
from sklearn.preprocessing import LabelEncoder, label_binarize

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
class_names=['price','region', 'variety','winery']
#print(class_names.head)
obj_df = wine_df.select_dtypes(include=['object']).copy()
wine_df['point_bins']=pd.factorize(obj_df['point_bins'])[0]
#print("Points factorized:",wine_df['point_bins'].head)

# Split into 50/50 train/test datasets
wine_train, wine_test = train_test_split (wine_df, test_size=0.15)

# Prepare train data for classification tree
regn_lab = LabelEncoder ().fit (np.unique (wine_df.region_1.values))
var_lab = LabelEncoder ().fit (np.unique (wine_df.variety2.values))
wnry_lab = LabelEncoder ().fit (np.unique (wine_df.winery2.values))
wine_train['regn_enc'] = regn_lab.transform (wine_train.region_1)
wine_train['var_enc'] = var_lab.transform (wine_train.variety2)
wine_train['wnry_enc'] = wnry_lab.transform (wine_train.winery2)
wine_train = wine_train.drop (['region_1', 'variety2', 'winery2'], axis=1)
target=['points']
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
min_samp_split = 2
clf = RandomForestClassifier (min_samples_split=min_samp_split, max_features=None)
clf = clf.fit (x, y)
train_error = y == clf.predict (x)
test_error = wine_test['point_bins'] == clf.predict (wine_test.loc[:, ['price', \
                                                                       'regn_enc', 'var_enc', 'wnry_enc']])
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('   ', clf.n_features_, ' features out of: 4 features')
print ('   training accuracy: ', '{:.1%}'.format (sum (train_error) / len (
    train_error)))
print ('   test accuracy: ', '{:.1%}'.format (sum (test_error) / len (test_error)))

# Report feature importance
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('Feature importance for max_leaves model')
print (pd.DataFrame ([clf.feature_importances_], columns=x.columns.values))

# Train classification tree - remove varietal information
x = wine_train.loc[:, ['price', 'regn_enc', 'wnry_enc']]
y = wine_train['point_bins']
min_samp_split = 2
clf = RandomForestClassifier (min_samples_split=min_samp_split, max_features=None)
clf = clf.fit (x, y)

# Report classification results.  train first, then test.
# Varietal information removed - no tree termination criteria
train_error = y == clf.predict (x)
test_error = wine_test['point_bins'] == clf.predict (wine_test.loc[:, ['price', \
                                                                       'regn_enc', 'wnry_enc']])
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('   Varietal information removed')
print ('   training accuracy: ', '{:.1%}'.format (sum (train_error) / len (
    train_error)))
print ('   test accuracy: ', '{:.1%}'.format (sum (test_error) / len (test_error)))

# Train classification tree - remove region information
x = wine_train.loc[:, ['price', 'var_enc', 'wnry_enc']]
y = wine_train['point_bins']
min_samp_split = 2
clf = RandomForestClassifier (min_samples_split=min_samp_split, max_features=None)
clf = clf.fit (x, y)

# Report classification results.  train first, then test.
# Region information removed - no tree termination criteria
train_error = y == clf.predict (x)
test_error = wine_test['point_bins'] == clf.predict (wine_test.loc[:, ['price', \
                                                                       'var_enc', 'wnry_enc']])
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('   Region information removed')
print ('   training accuracy: ', '{:.1%}'.format (sum (train_error) / len (
    train_error)))
print ('   test accuracy: ', '{:.1%}'.format (sum (test_error) / len (test_error)))

# Train classification tree - remove varietal & region information
x = wine_train.loc[:, ['price', 'wnry_enc']]
y = wine_train['point_bins']
min_samp_split = 2
clf = RandomForestClassifier (min_samples_split=min_samp_split, max_features=None)
clf = clf.fit (x, y)

# Report classification results.  train first, then test.
# Varietal & region information removed - no tree termination criteria
train_error = y == clf.predict (x)
test_error = wine_test['point_bins'] == clf.predict (wine_test.loc[:, ['price', \
                                                                       'wnry_enc']])
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('   Varietal & Region information removed')
print ('   training accuracy: ', '{:.1%}'.format (sum (train_error) / len (
    train_error)))
print ('   test accuracy: ', '{:.1%}'.format (sum (test_error) / len (test_error)))

# Re-train with 4 features
x = wine_train.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']]
y = wine_train['point_bins']
min_samp_split = 2
clf = RandomForestClassifier (min_samples_split=min_samp_split, max_features=None)
clf = clf.fit (x, y)


