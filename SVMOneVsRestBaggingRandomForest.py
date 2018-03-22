
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
file_loc = "/Users/bonythomas/PycharmProjects/CodeLearnings/CellDNA.xls"
df = pd.read_excel(file_loc, header=None, names=names)

cols=[u'A',u'B',u'C',u'D',u'E',u'F',u'G',u'H',u'I',u'J',u'K',u'L',u'M']
x = df[cols]
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x)
x = scaling.transform(df[cols])
y = df[u'N'].map({ 0 : 0, 1 : 1,2 : 1,3 : 1,4 : 1,5 : 1,6 : 1,7 : 1,8 : 1,9 : 1,10 : 1})
start = time.time()
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0 # SVM regularization parameter


start = time.time()
clf = SVC(kernel='linear', probability=True, class_weight='balanced')
clf.fit(x, y)
end = time.time()
print ("Bagging SVC", end - start,'s', clf.score(x,y))
proba = clf.predict_proba(x)
w = clf.coef_[0]
a = -w[0] / w[1]
print("Support Vectors of record 135:",clf.support_vectors_[135].tolist())
print("Support Vectors of record 162:",clf.support_vectors_[162].tolist())
#print("Support Vectors of record 892:",clf.support_vectors_[892].tolist())
#print("Support Vectors of record 1000:",clf.support_vectors_[1000].tolist())
#print("Support Vector Indices:",clf.support_)
#print("Number of Support Vectors:",clf.n_support_)
#print("Predicted Support Vectors:",proba)
#print("Coefficients",w)

plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()