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
df = pd.read_csv('/Users/bonythomas/default of credit card clients.csv')
# Assigning predictors to x dataframe
x = pd.read_csv('/Users/bonythomas/default of credit card clients.csv',
                usecols=["LIMIT_BAL","SEX","EDUCATION", "MARRIAGE", "AGE", "SEPT REPAY STS", "AUG REPAY STS", "JULY REPAY STS", "JUNE REPAY STS", "MAY REPAY STS", "APR REPAY STS","SEPT STMT", "AUG STMT"])

# Assign the converted variables to input x
x['country'] = converted_data
x['province'] = converted_data1
#x['region_1'] = converted_data2
x['variety'] = converted_data3
x['winery'] = converted_data4
x['description'] = converted_data5
x['designation'] = converted_data6'''
# Assigning the dependent variable to y
y = pd.read_csv('/Users/bonythomas/default of credit card clients.csv', usecols=["default"])

# scaling every variable
predictors=x.copy()
predictors["LIMIT_BAL"]=preprocessing.scale(predictors["LIMIT_BAL"].astype("float64"))
predictors["SEX"]=preprocessing.scale(predictors["SEX"].astype("float64"))
#predictors["EDUCATION"]=preprocessing.scale(predictors["EDUCATION"]. astype("float64"))
predictors["MARRIAGE"]=preprocessing.scale(predictors['MARRIAGE'].astype("float64"))
predictors["AGE"]=preprocessing.scale(predictors["AGE"].astype("float64"))
predictors["SEPT REPAY STS"]=preprocessing.scale(predictors["SEPT REPAY STS"].astype("float64"))
predictors["AUG REPAY STS"]=preprocessing.scale(predictors["AUG REPAY STS"].astype("float64"))
predictors["JULY REPAY STS"]=preprocessing.scale(predictors["JULY REPAY STS"].astype("float64"))
predictors["JUNE REPAY STS"]=preprocessing.scale(predictors["JUNE REPAY STS"].astype("float64"))
predictors["MAY REPAY STS"]=preprocessing.scale(predictors["MAY REPAY STS"].astype("float64"))
predictors["APR REPAY STS"]=preprocessing.scale(predictors["APR REPAY STS"].astype("float64"))
predictors["SEPT STMT"]=preprocessing.scale(predictors["SEPT STMT"].astype("float64"))
predictors["AUG STMT"]=preprocessing.scale(predictors["AUG STMT"].astype("float64"))
'''predictors["JULY STMT"]=preprocessing.scale(predictors["JULY STMT"].astype("float64"))
predictors["JUNE STMT"]=preprocessing.scale(predictors["JUNE STMT"].astype("float64"))
predictors["MAY STMT"]=preprocessing.scale(predictors["MAY STMT"].astype("float64"))
predictors["APR STMT"]=preprocessing.scale(predictors["APR STMT"].astype("float64"))'''
predictors_columns=["LIMIT_BAL","SEX","EDUCATION", "MARRIAGE", "AGE", "SEPT REPAY STS", "AUG REPAY STS", "JULY REPAY STS", "JUNE REPAY STS", "MAY REPAY STS", "APR REPAY STS","SEPT STMT","AUG STMT","JULY STMT",]


# Splitting the test and training data for building better prediction model
X_train, X_test, y_train, y_test = cv.train_test_split(predictors, y, test_size=0.2)
#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)
#Fitting the model
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
lasso = Lasso(alpha=1)
res = lasso.fit(X_train,y_train)
#print("Coefficients lasso training fit of", res.coef_.tolist())
print('Lasso:',lasso)


# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(X_train, y_train) #K fold is yen and not to use precomputed matrix.Here first fold is the validation set and the remaining 9 folds estimate the model

# print variable names and regression coefficients
print ('Coefficients from lasso lars',dict(zip(X_train.columns, model.coef_)) )#dic object creates dictionary and zip object creates lists

# Fit the regressor to the data
#las=lasso.fit(predictors, y)


#plot mean square error for each fold
print("Computing regularization path using the Lars lasso...")
m_log_alphascv = -np.log10(model.cv_alphas_)
#print("Log alphas:",m_log_alphascv,"MSE:",model.cv_mse_path_)
pyplot.figure()
pyplot.plot(m_log_alphascv, model.cv_mse_path_, ':')
pyplot.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
pyplot.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
pyplot.legend()
pyplot.xlabel('-log(alpha)')
pyplot.ylabel('Mean squared error')
pyplot.title('Mean squared error on each fold')
pyplot.show()


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
#print("Alpha Values:", a, "Coefficients for fold: ",i, lassocv.coef_)
pyplot.plot(alphas, scores, '-ko')
pyplot.axhline(lassocv_score, color='b', ls='--')
pyplot.axvline(lassocv_alpha, color='b', ls='--')
pyplot.xlabel(r'$\alpha$')
pyplot.ylabel('Score')
pyplot.xscale('log')
#sns.despine(offset=15)
pyplot.title('Alphas plots for all folds')
pyplot.show()


#plot coefficient progrssion
m_log_alphascv = -np.log10(model.alphas_)
ax = pyplot.gca()
pyplot.plot(m_log_alphascv, model.coef_path_.T) #.T is to transpose the coeff_path_attri matrix to match the first dim of array of alpha values
pyplot.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha_CV')
#print("Alpha Value:",m_log_alphascv, "Coefficients:",model.cv_mse_path_)
pyplot.ylabel('Regression Coefficients')
pyplot.xlabel('-log(alpha)')
pyplot.title('Regression coefficients for lasso plots')
pyplot.show()

# Indicate the lasso parameter that minimizes the average MSE acrossfolds.
lasso_fit = model.fit(x, y)
lasso_path = model.score(x, y)
pyplot.axvline(lasso_fit.alpha_, color = 'red')
pyplot.title("Lasso parameter")
print('Deg. Coefficient')
print(lasso_fit.intercept_)
print(dict(zip(X_train.columns, lasso_fit.coef_)))
#print("Lasso parameter:",lasso_fit.alpha_)
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


