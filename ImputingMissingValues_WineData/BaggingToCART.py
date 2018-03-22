##########################################
#
# Bagging to improve accuracy

##########################################

import pandas as pd
import numpy as np
from scipy import interp
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score, average_precision_score, \
    precision_recall_curve, accuracy_score, auc
import matplotlib.pyplot as plt

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
wine_df['point_bins']=wine_df.points.map(bin_map)
wine_df.point_bins.unique() # Ensure no records are un-binned
wine_df=wine_df.drop('points',axis=1)
x= wine_df.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']]
y= wine_df['point_bins']
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
x_train = wine_train.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']]
y_train = wine_train['point_bins']

# Prepare test data for classification tree
wine_test['regn_enc'] = regn_lab.transform (wine_test.region_1)
wine_test['var_enc'] = var_lab.transform (wine_test.variety2)
wine_test['wnry_enc'] = wnry_lab.transform (wine_test.winery2)
wine_test = wine_test.drop (['region_1', 'variety2', 'winery2'], axis=1)
x_test = wine_test.loc[:, ['price', 'regn_enc', 'var_enc', 'wnry_enc']]
y_test = wine_test['point_bins']

# Learn to predict classes for a cross fold of 8
cv = StratifiedKFold(n_splits=10)
classifier = svm.SVC(kernel='rbf', probability=True, class_weight='balanced')
y_score = classifier.fit(x_train, y_train).decision_function(x_test)
proba = classifier.predict(x_test)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
average_precision = average_precision_score(y_test, y_score)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.step(recall, precision, color='b', alpha=0.2,where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)


print("The accuracy score is:",accuracy_score(y_test, proba))
print("The precision score is:",precision_score(y_test, proba))
print("The recall score is:",recall_score(y_test, proba))


#Plotting the ROC curve
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()