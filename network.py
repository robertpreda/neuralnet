import keras
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout,Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt

pca = PCA(n_components=3)


#hyperparameters
depth = 5
num_of_hidden = 16
dropout = 0.35

# np.random.seed(123)

'''
data preprocessing
'''
# load in data
df = pd.read_csv('challenge_train.csv', low_memory = False)

#for i in range(0, df.shape[0],10000):
#	print("mere ",i)
#	df.iloc[i],df.iloc[-i] = df.iloc[-i],df.iloc[i]


df = shuffle(df)
y = df['verdict']
X = df.drop(labels = ['md5','verdict'],axis = 'columns')
X.astype(np.float)

y.replace(to_replace={'trojan' : 1,'clean' : 0},inplace=True)
y.astype(np.float)


'''
	renaming the columns

'''
cols = []
vals = []
count = 0
for label,content in X.items():
    cols.append(label)
    vals.append('col'+str(count))
    count += 1
rename_dict = dict(zip(cols,vals))
X = X.rename(index = str, columns = rename_dict) 

X = X[['col0','col242','col72','col75','col76']]
X = preprocessing.scale(X)
# splitting data in train and test
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

#constructing, training and evaluation of model
model = Sequential()
# input layer
model.add(Dense(units = X_train.shape[1],activation = 'relu', input_dim = X_train.shape[1]))

# hidden layers
model.add(Dense(units = 16,activation = 'relu'))
model.add(Dense(units = 16, activation = 'relu'))

# output layer
model.add(Dense(1,activation = 'sigmoid'))

# compilation, fitting and evaluating model
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=16, epochs=5,validation_data = (x_valid, y_valid),shuffle = True)
(loss, accuracy) = model.evaluate(x_test, y_test, batch_size=16)
model.save('network.h5')
print("Test accuracy: ", accuracy)
'''accuracy = history.history['acc']
epochs = range(len(accuracy))
plt.plot(epochs,accuracy,'ro',label='accuracy')
plt.grid()
plt.legend()
plt.show()'''
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test,y_pred.round())
print(cm)



