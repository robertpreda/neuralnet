import keras
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout,Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

np.random.seed(123)

'''
data preprocessing
'''
# load in data
df = pd.read_csv('challenge_train.csv', low_memory = False)
y = df['verdict']
X = df.drop(labels = ['md5','verdict'],axis = 'columns')
X.astype(np.float)
y.replace(to_replace={'trojan' : 1,'clean' : 0},inplace=True)
y.astype(np.float)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#constructing model

model = Sequential()
model.add(Dense(units = X_train.shape[1],activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=2, shuffle=True)
(loss, accuracy) = model.evaluate(x_test, y_test, batch_size=64)
model.save_weights('network.h5')
print("Test accuracy: ", accuracy)



