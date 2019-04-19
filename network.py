import keras
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout,Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

depth = int(input("Enter desired depth: "))
num_of_hidden = int(input("Enter number of hidden nodes per layer: "))
dropout = float(input("Enter dropout ratio: "))

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

# splitting data in train and test
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#constructing, training and evaluation of model
model = Sequential()
model.add(Dense(units = X_train.shape[1],activation = 'sigmoid', input_dim = X_train.shape[1]))
for _ in range(depth):
	model.add(Dense(num_of_hidden,activation = 'sigmoid'))
	model.add(Dropout(dropout))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=2, shuffle=True)

(loss, accuracy) = model.evaluate(x_test, y_test, batch_size=64)
model.save_weights('network.h5')
print("Test accuracy: ", accuracy)
accuracy = history.history['acc']
epochs = range(len(accuracy))
plt.plot(epochs,accuracy,'r',label='accuracy')
plt.grid()
plt.legend()
plt.show()



