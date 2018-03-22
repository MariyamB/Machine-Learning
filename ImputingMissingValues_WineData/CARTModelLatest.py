#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, tree, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pydotplus
import itertools
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize

# Read in & munge wine information dataset.  US wines only
from sklearn.utils import resample

wine_df=pd.read_csv('/Users/bonythomas/new.csv')
wine_df=wine_df.loc[wine_df.country=='US',['points','price','region_1', 'variety2','winery2']]
wine_df=wine_df.dropna(axis=0,how='any')
print(wine_df.columns)
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
class_names=['price','winery']
obj_df = wine_df.select_dtypes(include=['object']).copy()
wine_df['point_bins']=pd.factorize(obj_df['point_bins'])[0]

NintyAbove=wine_df.loc[wine_df.point_bins==0]
NintyBelow=wine_df.loc[wine_df.point_bins==1]
print("NintyAbove",NintyAbove.shape)
print("NintyBelow",NintyBelow.shape)
# Split into 50/50 train/test datasets
wine_train, wine_test = train_test_split (wine_df, test_size=0.15)
print("Wine Training data shape:",wine_train.shape)
print("Counts of <90 wines",wine_train.loc[wine_train.point_bins==0].shape)
print("Counts of 90+ wines",wine_train.loc[wine_train.point_bins==1].shape)
print("Wine Testing data shape:",wine_test.shape)

#Upsample minority class <90
df_majority = wine_train.loc[wine_train.point_bins==1]
df_minority = wine_train.loc[wine_train.point_bins==0]
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=22483,    # to match majority class
                                 random_state=123) # reproducible results

wine_train = pd.concat([df_majority, df_minority_upsampled])

# Prepare train data for classification tree
regn_lab = LabelEncoder ().fit (np.unique (wine_df.region_1.values))
var_lab = LabelEncoder ().fit (np.unique (wine_df.variety2.values))
wnry_lab = LabelEncoder ().fit (np.unique (wine_df.winery2.values))
wine_train['regn_enc'] = regn_lab.transform (wine_train.region_1)
wine_train['var_enc'] = var_lab.transform (wine_train.variety2)
wine_train['wnry_enc'] = wnry_lab.transform (wine_train.winery2)
wine_train = wine_train.drop (['region_1', 'variety2', 'winery2'], axis=1)
target=['points']

print("Counts of <90 wines after upsampling",wine_train.loc[wine_train.point_bins==0].shape)
print("Counts of 90+ wines after upsampling",wine_train.loc[wine_train.point_bins==1].shape)

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
clf = DecisionTreeClassifier (min_samples_split=min_samp_split, max_features=None)
clf1 = clf.fit (x, y)
# BASELINE - all features, no tree termination criteria
train_error = y == clf.predict (x)
test_error1 = wine_test['point_bins'] == clf.predict (wine_test.loc[:, ['price', \
                                                                       'regn_enc', 'var_enc', 'wnry_enc']])
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('CART w/ #leaf nodes = ', clf.tree_.node_count)
print ('   ', clf.n_features_, ' features out of: 4 features')
print ('   training accuracy: ', '{:.1%}'.format (sum (train_error) / len (
    train_error)))
print ('   test accuracy: ', '{:.1%}'.format (sum (test_error1) / len (test_error1)))

# Report feature importance
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('Feature importance for max_leaves model')
print (pd.DataFrame ([clf.feature_importances_], columns=x.columns.values))

# Train classification tree removing variety cos ot has lowest importance
x = wine_train.loc[:, ['price', 'wnry_enc']]
y = wine_train['point_bins']
x_test = wine_test.loc[:, ['price', 'wnry_enc']]
y_test = wine_test['point_bins']
min_samp_split = 2
clf = DecisionTreeClassifier (min_samples_split=min_samp_split, max_features=None)
clf2 = clf.fit (x, y)
# BASELINE - all features, no tree termination criteria
train_error = y == clf.predict (x)
test_error1 = wine_test['point_bins'] == clf.predict (wine_test.loc[:, ['price','wnry_enc']])
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('CART w/ #leaf nodes after remving variety= ', clf.tree_.node_count)
print ('   ', clf.n_features_, ' features out of: 4 features')
print ('   training accuracy: ', '{:.1%}'.format (sum (train_error) / len (
    train_error)))
print ('   test accuracy: ', '{:.1%}'.format (sum (test_error1) / len (test_error1)))

# Report feature importance
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('Feature importance for max_leaves model')
print (pd.DataFrame ([clf.feature_importances_], columns=x.columns.values))


# We will rebuild a new tree by using above data and see how it works by tweeking the parameteres
x = wine_train.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']]
y = wine_train['point_bins']

dtree = tree.DecisionTreeClassifier (criterion="gini", splitter='random', max_leaf_nodes=20, min_samples_leaf=5,
                                     max_depth=5)
dtree.fit (x, y)
prunetrain_error = y == dtree.predict (x)
dot_data = StringIO ()
export_graphviz (clf2, out_file=dot_data, feature_names=class_names,filled=True, rounded=True,special_characters=True)
export_graphviz (clf2, out_file='wineDT1Pruned.dot')
graph = pydotplus.graph_from_dot_data (dot_data.getvalue ())
graph.write_png ("WineDTPruned.png")
'''
print("Feature importance after pruning:")
print (pd.DataFrame ([clf2.feature_importances_], columns=x.columns.values))
Prunetest_error1 = wine_test['point_bins'] == dtree.predict (
    wine_test.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']])
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')
print ('CART after pruning w/ #leaf nodes = ', dtree.tree_.node_count)
from sklearn.metrics import precision_recall_fscore_support as score
predicted = test_error1
y_test = y_test
precision, recall, fscore, support = score (y_test, predicted)
print('precision before pruning: {}'.format (precision))
print('recall  before pruning: {}'.format (recall))
print('fscore  before pruning: {}'.format (fscore))
print('support  before pruning: {}'.format (support))
from sklearn.metrics import precision_recall_fscore_support as score
precision1, recall1, fscore1, support1 = score (y_test, Prunetest_error1)
print('precision after pruning: {}'.format (precision1))
print('recall  after pruning: {}'.format (recall1))
print('fscore  after pruning: {}'.format (fscore1))
print('support  after pruning: {}'.format (support1))


# Control the number of n_estimators in ensemble functions
max_n_ests = 25

# Create dataframe to record results of ensembles.
results = pd.DataFrame ([], columns=list (['type', 'n_leaf', 'n_est', \
                                           'train_acc', 'test_acc']))

# Train bagging ensemble on iterations of n_estimators=i
# and iterations of stump max_leaf_nodes=j
for j in [500, 8000, 9999]:
    clf_stump = DecisionTreeClassifier (criterion="gini", splitter='random', max_leaf_nodes=j, min_samples_leaf=5,
                                        max_depth=5)
    for i in np.arange (1, max_n_ests):
        baglfy = BaggingClassifier (base_estimator=clf_stump, n_estimators=i,max_samples=1.0)
        baglfy = baglfy.fit (x, y)
        bag_tr_err = y == baglfy.predict (x)
        bag_tst_err = wine_test['point_bins'] == baglfy.predict ( \
            wine_test.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']])
        run_rslt = pd.DataFrame ([['bag', j, i, sum (bag_tr_err) / len (bag_tr_err),sum (bag_tst_err) / len (bag_tst_err)]],columns=list (['type', 'n_leaf', 'n_est', 'train_acc', 'test_acc']))
        results = results.append (run_rslt)

# Train bagging ensemble on iterations of n_estimators=i
# and iterations of stump max_leaf_nodes=j
# Train boosting ensemble on iterations of n_estimators=i
# and iterations of stump max_leaf_nodes=j

for j in [500, 8000, 9999]:
    clf_stump = DecisionTreeClassifier (criterion="gini", splitter='random', max_leaf_nodes=j, min_samples_leaf=5,
                                        max_depth=5)
    for i in np.arange (1, max_n_ests):
        print(i)
        bstlfy = AdaBoostClassifier (base_estimator=clf_stump, n_estimators=i)
        bstlfy = bstlfy.fit (x, y)
        bst_tr_err = y == bstlfy.predict (x)
        bst_tst_err = wine_test['point_bins'] == bstlfy.predict ( \
 \
            wine_test.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']])
        run_rslt = pd.DataFrame ([['bst', j, i, sum (bst_tr_err) / len (bst_tr_err),sum (bst_tst_err) / len (bst_tst_err)]],columns=list (['type', 'n_leaf', 'n_est', 'train_acc', 'test_acc']))
        results = results.append (run_rslt)
 # ROC curve for baseline classification tree
clf_probs = clf1.predict_proba (wine_test.loc[:, ['price', 'regn_enc', 'var_enc','wnry_enc']])
fpr1, tpr1, thr1 = roc_curve (np.where (wine_test['point_bins'] == 0, 1., 0.),clf_probs[:, 0])
clf_probs = dtree.predict_proba (wine_test.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']])
fpr2, tpr2, thr2 = roc_curve (np.where (wine_test['point_bins'] == 0, 1., 0.), clf_probs[:, 0])

# ROC curve for bagging ensemble using full classification trees

bag_probs = baglfy.predict_proba (wine_test.loc[:, ['price', 'regn_enc','var_enc', 'wnry_enc']])
fpr3, tpr3, thr3 = roc_curve (np.where (wine_test['point_bins'] == 0, 1., 0.),bag_probs[:, 0])

# ROC curve for boosting ensemble using full classification trees

boost_probs = bstlfy.predict_proba (wine_test.loc[:, ['price', 'regn_enc','var_enc', 'wnry_enc']])
fpr4, tpr4, thr4 = roc_curve (np.where (wine_test['point_bins'] == 0, 1., 0.), boost_probs[:, 0])

# Plot ROC Curves

plt.plot (fpr1, tpr1, color='#4d4d33', label='Baseline CART')
plt.plot (fpr2, tpr2, color='#0080ff', label='Pruned CART')
plt.plot (fpr3, tpr3, color='#b32400', label='Bagging Ensemble')
plt.plot (fpr4, tpr4, color='#661400', label='Boosting Ensemble')
plt.plot ([0., 1.], [0., 1.], color='k', linestyle='--')
plt.title ('ROC Curves for 90+ Point Wine Classification')
plt.xlabel ('false positive rate')
plt.ylabel ('true positive rate')
plt.legend (fontsize=8)
plt.show ()

# Baseline Classification Pruned Tree Perfromance

from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score (wine_test['point_bins'], Prunetest_error1)
print('baseline precision: {}'.format (precision))
print('baseline recall: {}'.format (recall))
print('baseline fscore: {}'.format (fscore))
print('baseline support: {}'.format (support))

# Baseline Classification Pruned Tree Perfromance after bagging
precision2, recall2, fscore2, support2 = score (wine_test['point_bins'], bst_tst_err)
print('precision after boosting: {}'.format (precision2))
print('recall  after boosting: {}'.format (recall2))
print('fscore  after boosting: {}'.format (fscore2))
print('support  after boosting: {}'.format (support2))
precision1, recall1, fscore1, support1 = score (wine_test['point_bins'], bag_tst_err)
print('precision after bagging: {}'.format (precision1))
print('recall  after bagging: {}'.format (recall1))
print('fscore  after bagging: {}'.format (fscore1))
print('support  after bagging: {}'.format (support1))
cm = confusion_matrix (y_test.values, test_error1)
cm1 = confusion_matrix (y_test.values, Prunetest_error1)
cm2 = confusion_matrix (y_test.values, bag_tst_err)
cm3 = confusion_matrix (y_test.values, bst_tst_err)


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:
        cm = cm.astype ('float') / cm.sum (axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')
        print(cm)
        plt.imshow (cm, interpolation='nearest', cmap=cmap)
        plt.title (title)
        plt.colorbar ()
        tick_marks = np.arange (len (classes))
        plt.xticks (tick_marks, classes, rotation=45)
        plt.yticks (tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max () / 2.
    for i, j in itertools.product (range (cm.shape[0]), range (cm.shape[1])):
        plt.text (j, i, format (cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout ()
    plt.ylabel ('True label')
    plt.xlabel ('Predicted label')
plt.figure ()
plot_confusion_matrix (cm, ['a', 'b'])
plt.figure ()
plot_confusion_matrix (cm1, ['a', 'b'])
plt.figure ()
plot_confusion_matrix (cm2, ['a', 'b'])
plt.figure ()
plot_confusion_matrix (cm3, ['a', 'b'])
plt.show ()
test_labels = ['90+ Wine', '<90 Wine']

# Calculate Accuracy Rate by using accuracy_score()

print("The test accuracy score of baseline CART is :", accuracy_score (wine_test['point_bins'], test_error1))
print("test Accuracy of pruned CART:", accuracy_score (wine_test['point_bins'], Prunetest_error1))
print(" testAccuracy of bagged CART:", accuracy_score (wine_test['point_bins'], bag_tst_err))
print(" testAccuracy of boosted CART:", accuracy_score (wine_test['point_bins'], bst_tst_err))


print("The train accuracy score of baseline CART is :", accuracy_score (wine_train['point_bins'], train_error))
print("train Accuracy of pruned CART:", accuracy_score (wine_train['point_bins'],prunetrain_error ))
print("train Accuracy of bagged CART:", accuracy_score (wine_train['point_bins'], bag_tr_err))
print("train Accuracy of boosted CART:", accuracy_score (wine_train['point_bins'], bst_tr_err))
'''