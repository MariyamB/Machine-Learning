import pandas as pd
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.pyplot import plot
from sklearn import linear_model as lm
df=pd.read_csv('/Users/bonythomas/patients.csv')
x=pd.read_csv('/Users/bonythomas/patients.csv', usecols=["Age", "Gender","Smoker", "Weight", "Height", "SelfAssessedHealthStatus", "Location"])
y=pd.read_csv('/Users/bonythomas/patients.csv', usecols=["Systolic"])
obj_df = df.select_dtypes(include=['object']).copy()
converted_data=pd.factorize(obj_df['Gender'])[0]
converted_data1=pd.factorize(obj_df['Location'])[0]
converted_data2=pd.factorize(obj_df['SelfAssessedHealthStatus'])[0]
x['Gender'] = converted_data
x['Location'] = converted_data1
x['SelfAssessedHealthStatus'] = converted_data2
print('Predictors',x)
print('Predicted Value',y)
regr = lm.LinearRegression()
regr.fit(x,y)
model1=smf.OLS(y,x)
result=model1.fit()
predictions=y-result
print('Intercept',regr.intercept_)
print('Coefficients',regr.coef_)
plot(result,predictions)
plot.show()
print(result.summary())