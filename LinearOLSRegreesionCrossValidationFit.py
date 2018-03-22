#plotting the linear regression with OLS(Least square method) and training the model with cross validted data
#Predicting outliers with residuals plot,density plot and QQ[normal probability plot]

import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from pandas import DataFrame
from matplotlib import pyplot
import numpy as np
import statsmodels.formula.api as sm
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import train_test_split
from pandas.plotting import autocorrelation_plot
#from mpl_toolkits.mplot3d import axes3d(Getting Error while import Error Code:ImportError: No module named mpl_toolkits.mplot3d)

lr = LinearRegression()

#Reading the Patients CSv file
df=pd.read_csv('/Users/bonythomas/patients.csv')

#Assigning predictors to x dataframe
x=pd.read_csv('/Users/bonythomas/patients.csv', usecols=["Age", "Gender","Smoker", "Weight", "Height", "SelfAssessedHealthStatus", "Location"])

#Factoring the categorical variable by coding
obj_df = x.select_dtypes(include=['object']).copy()
converted_data=pd.factorize(obj_df['Gender'])[0]
converted_data1=pd.factorize(obj_df['Location'])[0]
converted_data2=pd.factorize(obj_df['SelfAssessedHealthStatus'])[0]

#Assign the converted variables to input x
x['Gender'] = converted_data
x['Location'] = converted_data1
x['SelfAssessedHealthStatus'] = converted_data2

#Assigning the dependent variable to y
y=pd.read_csv('/Users/bonythomas/patients.csv', usecols=["Systolic"])

#Splitting the test and training data for building better prediction model
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Linear Regression fit of the predictors and dependent variables
lr.fit(X_train, y_train)

#Predicted values from the fit model
predictions = lr.predict(x[["Age"]+["Gender"]+["Smoker"]+["Weight"]+["Height"]+["SelfAssessedHealthStatus"]+["Location"]])

#Linear regression model for the predictors and the dependent variable
model = sm.ols(formula='Systolic ~ Age+Gender+Smoker+Weight+Height+SelfAssessedHealthStatus+Location', data=df)

#Trying to plot measured vs predicted values of the model
#predicted=cross_val_predict(model, df, y, cv=10)
fig, ax = pyplot.subplots()
ax.scatter(y, predictions, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
pyplot.show()

#Fitting the model
fitted = model.fit()

#model Summary
print(fitted.summary())

#Calculating residuals
yData = df.as_matrix(columns = ['Systolic'])
res = yData - predictions
residuals = DataFrame(res)

# histogram plot
fig, ax = pyplot.subplots()
residuals.hist()
pyplot.title('Histogram Plot')

# density plot
ax.scatter(yData, predictions, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
residuals.plot(kind='kde')
pyplot.title('Density plot')

#normal probability plot
ax.scatter(yData, predictions, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
qqplot(residuals)
pyplot.title('QQ plot')

#Predicted Vs Actual(Plotting Cross-Validated Predictions)
fig, ax = pyplot.subplots()
ax.scatter(yData, predictions, edgecolors=(0, 0, 0))
ax.plot([yData.min(), yData.max()], [yData.min(), yData.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
pyplot.title('Predicted Vs Actual for cross validation')
pyplot.show()


