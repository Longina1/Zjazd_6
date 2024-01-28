import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential   #szkilelt sieci
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.random import set_seed
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.dataset.mnist import load_data

mnist = fetch_openml('mnist_784')



X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=.2, random_state=0)

model = DecisionTreeClassifier(max_depth=999, min_samples_split=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred))

model.score(X_test, y_test)

#KNN
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred))
model.score(X_test, y_test)


##NN
(X_train, y_train), (X_test, y_test) = load_data()
X_train.shape, X_test.shape

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))

model.compile(loss='sparse_categorical_crossentropy')

result = model.fit(X_train, y_train, epochs=10)
y_pred = model.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred))
model.score(X_test, y_test)

#lub
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy')

result = model.fit(X_train, y_train, epochs=10)
y_pred[0]
y_pred_class = np.argmax(y_pred, axis=-1)
y_pred_class[0]
y_pred = model.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred_class))