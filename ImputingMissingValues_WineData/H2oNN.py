#Importing h2o package to perfrom deep learning
import h2o
import np as np
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import pandas as pd
from matplotlib.pyplot import plot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
h2o.init(nthreads = -1, max_mem_size = 8)

# Assigning predictors to x dataframe
#Reading training CSV file as a dataframe using Pandas
data_training = pd.read_csv('/Users/bonythomas/RtrainData.csv')
X=data_training[0:49]

y=data_training[0:-1]
#Reading testing CSV file as a dataframe using Pandas
data_testing = pd.read_csv('/Users/bonythomas/RtestData.csv')
y_test = data_testing[:51]
X_test=data_testing[0:49]
#Converting the dataframe to H2o dataframe
data_training=h2o.H2OFrame(data_training.values.tolist())
data_testing=h2o.H2OFrame(data_testing.values.tolist())
#Factorizing the target value for classification
data_training['C51'] = data_training['C51'].asfactor()
data_training['C51'].levels()
data_testing['C51'] = data_testing['C51'].asfactor()
data_testing['C51'].levels()
#Assigning the target column header to y
y = 'C51'
#Assigning the predictors header to x
x = list(data_training.columns)
#remove the response from x
x.remove(y)

# Initialize and train the DL estimator:
#One Epoch With hidden  layer
dl_1 = H2ODeepLearningEstimator(hidden=[200,200],epochs=1, activation="rectifier",loss = "crossentropy")
dl_1.train(x=x, y=y,training_frame=data_training)

#250 epochs with hidden layers
dl_2 = dl_250 = H2ODeepLearningEstimator(hidden=[11,13,17,19],checkpoint=dl_1, epochs=250,activation="rectifier",loss = "crossentropy")
dl_2.train(x=x, y=y, training_frame=data_training)

dl_3 = dl_250 = H2ODeepLearningEstimator(hidden=[11,13,17,19],checkpoint=dl_1, epochs=250,activation="rectifier",loss = "crossentropy")
dl_3.train(x=x, y=y, training_frame=data_training,validation_frame=data_testing)
target_names = ['class 0', 'class 1']
pred = dl_2.predict(data_testing[0:-1]).as_data_frame(use_pandas=True)

#rf_perf1 = dl_1.model_performance(data_testing)
#rf_perf2 = dl_2.model_performance(data_testing)
#rf_perf3 = dl_3.model_performance(data_testing)
#print("Predictions:",pred)
#print("Perfromance on test",rf_perf1)
#print("Perfromance on test",rf_perf2)
#print("Perfromance on test",rf_perf3)
#pred=dl_2.predict(data_testing).as_data_frame(use_pandas=True)
print("Pred",pred)
np.savetxt("/Users/bonythomas/test.csv", pred, delimiter=",")

plt(type='roc', train=False,show=True)

#print ("AUC on Test:",rf_perf1.auc())
#print ("AUC on Test:",rf_perf2.auc())
#print ("AUC on Test:",rf_perf3.auc())
