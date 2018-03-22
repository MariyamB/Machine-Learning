# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.cross_validation as cv
from matplotlib import pyplot
from sklearn import preprocessing, linear_model
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LassoLarsCV
# Linear regression package
from sklearn.linear_model import LinearRegression
import warnings

from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

lr = LinearRegression()
# Reading the Patients CSv file
df = pd.read_csv('/Users/bonythomas/patients.csv')
# Assigning predictors to x dataframe
x = pd.read_csv('/Users/bonythomas/patients.csv',
                usecols=["Age", "Gender", "Smoker", "Weight", "Height", "SelfAssessedHealthStatus", "Location"])
# Factoring the categorical variable by coding
obj_df = x.select_dtypes(include=['object']).copy()
converted_data = pd.factorize(obj_df['Gender'])[0]
converted_data1 = pd.factorize(obj_df['Location'])[0]
converted_data2 = pd.factorize(obj_df['SelfAssessedHealthStatus'])[0]
# Assign the converted variables to input x
x['Gender'] = converted_data
x['Location'] = converted_data1
x['SelfAssessedHealthStatus'] = converted_data2
# Assigning the dependent variable to y
y = pd.read_csv('/Users/bonythomas/patients.csv', usecols=["Systolic"])

# scaling every variable
predictors=x.copy()
predictors["Age"]=preprocessing.scale(predictors["Age"].astype("float64"))
predictors["Gender"]=preprocessing.scale(predictors["Gender"].astype("float64"))
predictors["Smoker"]=preprocessing.scale(predictors["Smoker"]. astype("float64"))
predictors["Weight"]=preprocessing.scale(predictors['Weight'].astype("float64"))
predictors["Height"]=preprocessing.scale(predictors["Height"].astype("float64"))
predictors["SelfAssessedHealthStatus"]=preprocessing.scale(predictors["SelfAssessedHealthStatus"].astype("float64"))
predictors["Location"]=preprocessing.scale(predictors["Location"].astype("float64"))
predictors_columns=["Age", "Gender", "Smoker", "Weight", "Height", "SelfAssessedHealthStatus", "Location"]

'''#Standardization and Min-Max scaling
std_scale = preprocessing.StandardScaler().fit(x[['Age', 'Smoker', 'Weight', 'Height', 'Gender', 'SelfAssessedHealthStatus']])
x_std = std_scale.transform(x[['Age', 'Smoker', 'Weight', 'Height', 'Gender', 'SelfAssessedHealthStatus']])
minmax_scale = preprocessing.MinMaxScaler().fit(x[['Age', 'Smoker', 'Weight', 'Height', 'Gender', 'SelfAssessedHealthStatus']])
x_minmax = minmax_scale.transform(x[['Age', 'Smoker', 'Weight', 'Height', 'Gender', 'SelfAssessedHealthStatus']])

print('Mean after standardization:\nAge={:.2f}, Smoker={:.2f},  Weight={:.2f},   Height={:.2f},   Sex={:.2f},    SelfAssessedHealthStatus={:.2f}  '
      .format(x_std[:,0].mean(), x_std[:,1].mean(), x_std[:,2].mean(), x_std[:,3].mean(), x_std[:,4].mean(), x_std[:,5].mean()))
print('\nStandard deviation after standardization:\nAge={:.2f}, Smoker={:.2f},  Weight={:.2f},   Height={:.2f},   Sex={:.2f},    SelfAssessedHealthStatus={:.2f}'
      .format(x_std[:,0].std(), x_std[:,1].std(), x_std[:,2].mean(), x_std[:,3].mean(),  x_std[:,4].mean(), x_std[:,5].mean()))

print('Min-value after min-max scaling:\nAge={:.2f}, Smoker={:.2f},  Weight={:.2f},   Height={:.2f},   Sex={:.2f},    SelfAssessedHealthStatus={:.2f}'
      .format(x_minmax[:,0].min(), x_minmax[:,1].min(), x_minmax[:,2].mean(), x_minmax[:,3].mean(),  x_minmax[:,4].mean(), x_minmax[:,5].mean()))
print('\nMax-value after min-max scaling:\nAge={:.2f}, Smoker={:.2f},  Weight={:.2f},   Height={:.2f},   Sex={:.2f},    SelfAssessedHealthStatus={:.2f}'
      .format(x_minmax[:,0].max(), x_minmax[:,1].max(), x_minmax[:,2].mean(), x_minmax[:,3].mean(),  x_minmax[:,4].mean(), x_minmax[:,5].mean()))

#plotting Age and Smoker predictors
def plot():
    pyplot.figure(figsize=(5,10))
    pyplot.scatter(df['Age'], df['Smoker'], color='red', label='input scale', alpha=0.3)
    pyplot.scatter(x_std[:,0], x_std[:,1], x_std[:,2],  color='black', label='Standardized', alpha=0.5)
    pyplot.scatter(x_minmax[:,0], x_minmax[:,1], color='purple', label='min-max scaled', alpha=0.3)
    pyplot.title('Alcohol and Malic Acid content of the wine dataset')
    pyplot.xlabel('Age')
    pyplot.ylabel('Smoker')
    pyplot.legend(loc='upper left')
    pyplot.grid()
    pyplot.tight_layout()
plot()
pyplot.show()

#Zoomnig into the three diff plota
fig, ax = pyplot.subplots(3, figsize=(6,50))
for a,d,l in zip(range(len(ax)),
               (df[['Age', 'Smoker']].values, df_std, df_minmax),
               ('Input scale', 'Standardized', 'min-max scaled')):
    for i,c in zip(range(1,4), ('red', 'blue', 'green')):
        ax[a].scatter(d[df['Systolic'].values == i, 0], d[df['Systolic'].values == i, 1], alpha=0.5, color=c, label='Class %s' %i)
    ax[a].set_title(l)
    ax[a].set_xlabel('Alcohol')
    ax[a].set_ylabel('Malic Acid')
    ax[a].legend(loc='upper left')
    ax[a].grid()

pyplot.tight_layout()

pyplot.show()'''


# Splitting the test and training data for building better prediction model
X_train, X_test, y_train, y_test = cv.train_test_split(predictors, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
#Fitting the model
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
lasso = Lasso(alpha=1)
res = lasso.fit(X_train,y_train)
print("Coefficients lasso training fit of", res.coef_.tolist())
print('Lasso:',lasso)


# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(X_train, y_train) #K fold is yen and not to use precomputed matrix.Here first fold is the validation set and the remaining 9 folds estimate the model

# print variable names and regression coefficients
print ('Coefficients from lasso lars',dict(zip(predictors.columns, model.coef_)) )#dic object creates dictionary and zip object creates lists

# Fit the regressor to the data
las=lasso.fit(x, y)

'''# Compute and print the coefficients
pyplot.figure()
pyplot.plot(las)
pyplot.xlabel('Lambda')
pyplot.ylabel('Coeff')
pyplot.title('Lasso Fit')
lasso_coef = lasso.coef_
print(lasso_coef)'''

#plot mean square error for each fold
print("Computing regularization path using the Lars lasso...")
m_log_alphascv = -np.log10(model.cv_alphas_)
print("Log alphas:",m_log_alphascv,"MSE:",model.cv_mse_path_)
pyplot.figure()
pyplot.plot(m_log_alphascv, model.cv_mse_path_, ':')
pyplot.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
pyplot.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
pyplot.legend()
pyplot.xlabel('-log(alpha)')
pyplot.ylabel('Mean squared error')
pyplot.title('Mean squared error on each fold')
pyplot.show()

'''#Plottin coef with xticks as column names
pyplot.plot(range(len(predictors_columns)), lasso_coef)
pyplot.xticks(range(len(predictors_columns)), predictors_columns, rotation=60)
pyplot.margins(0.02)
pyplot.title("Coeff for the predictors")
pyplot.show()'''

#Finding alphas and plotting the same on diabetics dataset
alphas = np.logspace(-4, -1, 10)
scores = np.empty_like(alphas)
for i,a in enumerate(alphas):
    lasso = linear_model.Lasso()
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    scores[i] = lasso.score(X_test, y_test)
    print("Alpha Values:", a, "Coefficients for fold: ",i, lasso.coef_)
lassocv = linear_model.LassoCV()
lassocv.fit(X_train, y_train)
lassocv_score = lassocv.score(X_train, y_train)
lassocv_alpha = lassocv.alpha_
print("Alpha Values:", a, "Coefficients for fold: ",i, lassocv.coef_)
pyplot.plot(alphas, scores, '-ko')
pyplot.axhline(lassocv_score, color='b', ls='--')
pyplot.axvline(lassocv_alpha, color='b', ls='--')
pyplot.xlabel(r'$\alpha$')
pyplot.ylabel('Score')
pyplot.xscale('log')
#sns.despine(offset=15)
pyplot.title('Alphas plots for all folds')
pyplot.show()

'''#Cross Validated MSE for Lasso Fit
foldid = scipy.random.choice(10, size = y.shape[0], replace = True)
cv1=cvglmnet(x = x.copy(),y = y.copy(),foldid=foldid,alpha=1)
cv0p5=cvglmnet(x = x.copy(),y = y.copy(),foldid=foldid,alpha=0.5)
cv0=cvglmnet(x = x.copy(),y = y.copy(),foldid=foldid,alpha=0)
f = pyplot.figure()
f.add_subplot(2,2,1)
cvglmnetPlot(cv1)
f.add_subplot(2,2,2)
cvglmnetPlot(cv0p5)
f.add_subplot(2,2,3)
cvglmnetPlot(cv0)
f.add_subplot(2,2,4)
pyplot.plot( scipy.log(cv1['lambdau']), cv1['cvm'], 'r.')
pyplot.hold(True)
pyplot.plot( scipy.log(cv0p5['lambdau']), cv0p5['cvm'], 'g.')
pyplot.plot( scipy.log(cv0['lambdau']), cv0['cvm'], 'b.')
pyplot.xlabel('log(Lambda)')
pyplot.ylabel(cv1['name'])
pyplot.xlim(-6, 4)
pyplot.ylim(0, 9)
pyplot.legend( ('alpha = 1', 'alpha = 0.5', 'alpha = 0'), loc = 'upper left', prop={'size':6});'''


#plot coefficient progrssion
m_log_alphascv = -np.log10(model.alphas_)
ax = pyplot.gca()
pyplot.plot(m_log_alphascv, model.coef_path_.T) #.T is to transpose the coeff_path_attri matrix to match the first dim of array of alpha values
pyplot.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha_CV')
print("Alpha Value:",m_log_alphascv, "Coefficients:",model.cv_mse_path_)
pyplot.ylabel('Regression Coefficients')
pyplot.xlabel('-log(alpha)')
pyplot.title('Regression coefficients for lasso plots')
pyplot.show()

# Indicate the lasso parameter that minimizes the average MSE acrossfolds.
lasso_fit = model.fit(predictors, y)
lasso_path = model.score(predictors, y)
pyplot.axvline(lasso_fit.alpha_, color = 'red')
pyplot.title("Lasso parameter")
print('Deg. Coefficient')
print(pd.Series(np.r_[lasso_fit.intercept_, lasso_fit.coef_]))
print("Lasso parameter:",lasso_fit.alpha_)
pyplot.show()

#MSE for training and testing data
train_error=mean_squared_error(y_train, model.predict(X_train))
test_error=mean_squared_error(y_test, model.predict(X_test))
print ("Traiing data MSE")
print (train_error)
print ("Testing data MSE")
print (test_error)

#R square from training and testing data
rsquared_train=model.score(X_train, y_train)
rsquared_test=model.score(X_test, y_test)
print ("Training data rSqaured")
print (rsquared_train)
print ("Testing data rSqaured")
print (rsquared_test)
pyplot.show()


'''#Cross Validation
k_fold = KFold(3)
cv_outer = KFold(len(x), n_folds=10)
lasso = LassoCV(cv=10)
scores = cross_val_score(lasso, x, y, cv=cv_outer)
kf = KFold(len(x), n_folds=5)
print (scores)
coefs_lasso = np.array([model.coef_ for model in model])
pyplot.figure(1)
ax = pyplot.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = pyplot.semilogx(alphas,coefs_lasso)
pyplot.gca().invert_xaxis()
lasso_cv = LassoCV(alphas=alphas, random_state=0)
kfoldsplit=kf.split(X_train,y)
#kf.get_n_splits(x)
print(kf)
print('kfoldsplit',kfoldsplit)
for k in cv_outer:
    lasso_cv.fit(X_train, y_train)
    pyplot.plot(lasso_cv.alpha_, lasso_cv.score(X_test, y_test))
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".format(k, lasso_cv.alpha_, lasso_cv.score(X_test, y_test)))
pyplot.xlabel('alpha')
pyplot.title('Scores plot')
pyplot.title('Cross validation for actual Data')
pyplot.show()'''

