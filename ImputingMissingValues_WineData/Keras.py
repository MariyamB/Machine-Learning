import pandas as pd
# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))

# Add an output layer
model.add(Dense(1, activation='sigmoid'))


# Assigning predictors to x dataframe
data_validation = pd.read_csv('/Users/bonythomas/Balancedvalidation.csv')
data_training = pd.read_csv('/Users/bonythomas/Balancedtraining.csv')
data_testing = pd.read_csv('/Users/bonythomas/Balancedtesting.csv')

X_train=data_training.iloc[:,1:80].values.reshape(-1)
y_train=data_training.iloc[:,:81]
X_test=data_testing.iloc[:,1:80]
y_test=data_testing.iloc[:,:81]


# Model config
model.get_config()

# List all weight tensors
model.get_weights()

model.compile (loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

model.fit (X_train, y_train, epochs=20, batch_size=1, verbose=1)
y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test,verbose=1)

print(score)

# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(y_test, y_pred)

# Precision
precision_score(y_test, y_pred)


# Recall
recall_score(y_test, y_pred)


# F1 score
f1_score(y_test,y_pred)



