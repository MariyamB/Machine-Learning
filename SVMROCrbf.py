print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy import interp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Import some data to play with
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
file_loc = "/Users/bonythomas/PycharmProjects/CodeLearnings/CellDNA.xls"
df = pd.read_excel(file_loc, header=None, names=names)
cols=[u'A',u'B',u'C',u'D',u'E',u'F',u'G',u'H',u'I',u'J',u'K',u'L',u'M']
x = df[cols]

#Scaling the data for faster processing
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x)
x = scaling.transform(df[cols])
y = df[u'N'].map({ 0 : 0, 1 : 1,2 : 1,3 : 1,4 : 1,5 : 1,6 : 1,7 : 1,8 : 1,9 : 1,10 : 1})
n_classes = y.shape

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.5,random_state=0)

# Learn to predict classes for a cross fold of 8
cv = StratifiedKFold(n_splits=10)
classifier = svm.SVC(kernel='rbf', probability=True, class_weight='balanced')
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
proba = classifier.predict(X_test)
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









