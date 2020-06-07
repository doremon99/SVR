#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'Latin-1')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Splitting the set 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = sc_y.inverse_transform(y_test)
#Training the set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

#Predicting the Result
y_pred = sc_y.inverse_transform(regressor.predict(X_test))
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

