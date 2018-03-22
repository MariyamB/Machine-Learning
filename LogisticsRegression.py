import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import matplotlib.pyplot as plt
import warnings
sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
from statsmodels.compat import scipy
warnings.filterwarnings("ignore")
from sklearn.datasets import load_svmlight_files
# figure number
fignum = 1
# Reading the Patients CSv file

#file = 'battledeath.xlsx' Sample xls file assignment
# Assigning predictors to x dataframe
#xl = pd.ExcelFile(file)
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
file_loc = "/Users/bonythomas/PycharmProjects/CodeLearnings/CellDNA.xls"
df = pd.read_excel(file_loc, header=None, names=names)
cols=[u'A',u'B',u'C',u'D',u'E',u'F',u'G',u'H',u'I',u'J',u'K',u'L',u'M']
x = df[cols]
y = df[u'N'].map({ 0 : 0, 1 : 1,2 : 1,3 : 1,4 : 1,5 : 1,6 : 1,7 : 1,8 : 1,9 : 1,10 : 1})


X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.30,)
# Apply Scaling to X_train and X_test
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.fit_transform(X_train)
X_test_std = std_scale.transform(X_test)
# Create SVM classification object
svc = svm.SVC(kernel='linear', C=1, gamma=1)
svc.fit(X_train_std,y_train)
print("Score:",svc.score(X_train_std,y_train))
#Predict Output
predicted= svc.predict(X_test_std)

# get the separating hyperplane
w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svc.intercept_[0]) / w[1]

#plot the parallels to the separating hyperplane that pass through the
# support vectors (margin away from hyperplane in direction
# perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
# 2-d.
margin = 1 / np.sqrt(np.sum(svc.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin
print("Support Vectors:",svc.support_vectors_[131].tolist())
print("Support Vector Indices:",svc.support_)
print("Number of Support Vectors:",svc.n_support_)
print("Predicted Support Vectors:",predicted)
print("Coefficients",w)

h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = x.iloc[:, 0].min() - 1, x.iloc[:, 0].max() + 1
y_min, y_max = x.iloc[:, 1].min() - 1, x.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
'''plt.figure(fignum, figsize=(4, 3))
plt.clf()
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(svc.support_vectors_.iloc[:, 0], svc.support_vectors_.iloc[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

plt.axis('tight')
x_min = -4.8
x_max = 4.2
y_min = -6
y_max = 6
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = svc.predict(np.c_[XX.ravel(), YY.ravel()])
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)
# Plot also the training points
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Interesting')
plt.ylabel('Not Interesting')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('Data')
plt.show()

# plot the line, the points, and the nearest vectors to the plane
plt.figure(fignum, figsize=(4, 3))
plt.clf()
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(svc.support_vectors_.iloc[:, 0], svc.support_vectors_.iloc[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

plt.axis('tight')
x_min = -4.8
x_max = 4.2
y_min = -6
y_max = 6

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = svc.predict(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.figure(fignum, figsize=(4, 3))
plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
fignum = fignum + 1
plt.show()'''