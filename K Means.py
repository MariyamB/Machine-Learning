import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

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
class_names=['price','region', 'variety','winery']
obj_df = wine_df.select_dtypes(include=['object']).copy()
wine_df['point_bins']=pd.factorize(obj_df['point_bins'])[0]

NintyAbove=wine_df.loc[wine_df.point_bins==0]
NintyBelow=wine_df.loc[wine_df.point_bins==1]
print("NintyAbove",NintyAbove.shape)
print("NintyBelow",NintyBelow.shape)
# Split into 50/50 train/test datasets
wine_train, wine_test = train_test_split (wine_df, test_size=0.15)
print("Wine Training data shape:",wine_train.shape)
print("Wine Testing data shape:",wine_test.shape)

# Prepare train data for classification tree
regn_lab = LabelEncoder ().fit (np.unique (wine_df.region_1.values))
var_lab = LabelEncoder ().fit (np.unique (wine_df.variety2.values))
wnry_lab = LabelEncoder ().fit (np.unique (wine_df.winery2.values))
wine_train['regn_enc'] = regn_lab.transform (wine_train.region_1)
wine_train['var_enc'] = var_lab.transform (wine_train.variety2)
wine_train['wnry_enc'] = wnry_lab.transform (wine_train.winery2)
wine_train = wine_train.drop (['region_1', 'variety2', 'winery2'], axis=1)
target=['points']
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
# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(wine_train)
# Getting the cluster labels
labels = kmeans.predict(wine_test)
# Centroid values
centroids = kmeans.cluster_centers_

# Comparing with scikit-learn centroids
print(centroids) # From sci-kit learn

