import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
rng = np.random.RandomState(0)
df = pd.read_csv('/Users/bonythomas/winemag-data_first150k.csv')
icols=['points']
jcols=['country']
X = pd.DataFrame(df.drop(df.columns[[1,2,3,4,5,6,7]], axis=1))
Y = pd.DataFrame(df.drop(df.columns[[0,1,2,3,5,6,7]], axis=1))
#X = X.as_matrix().astype(np.float)
#y = Y.as_matrix().astype(np.float)
#np.concatenate((X, Y), axis=0)
df = pd.concat([pd.DataFrame(X, columns=jcols),pd.DataFrame(Y, columns=icols)], axis=1)
notnans = df[jcols].notnull().all(axis=1)
df_notnans = df[notnans]
obj_df = df_notnans.select_dtypes(include=['object']).copy()
#converted_data=pd.factorize(obj_df['country'])[0]
converted_data=preprocessing.scale(obj_df['country'].astype("float64"))
df_notnans['country'] = converted_data
# Split into 75% train and 25% test
X_train, X_test, y_train, y_test = train_test_split(df_notnans[jcols], df_notnans[icols],
                                                       train_size=0.75,
                                                       random_state=4)
regr_multirf = RandomForestRegressor(max_depth=30,random_state=0)
# Fit on the train data
regr_multirf.fit(X_train, np.asarray(y_train).ravel())
df_nans = df.loc[~notnans].copy()
#print("Shape of df_nans",df_nans.shape)
df_nans[jcols]=regr_multirf.predict(df_nans[icols])

print(df_nans.head)

df=df_nans
print(df)