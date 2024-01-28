import numpy as np
import pandas as pd


from tensorflow.keras.models import Sequential #szkielet
from tensorflow.keras.layers import Dense #wszystkie połączenia
from tensorflow.random import set_seed #losowość

set_seed(0) #te same wyniki

model = Sequential()
model.add(Dense(4, input_shape=[1], activation='linear'))
#dane powinny być jednowymiarowe - zwykłe dane, nie macierz 2 X 2; 4 neurony, potem 2 i 1, funkacja aktywacji: liniowa

model.add(Dense(2, activation='linear'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) # jak mają się zmieniać wagi

df = pd.read_csv('f-c 1.csv', usecols=[1,2])
print(df.head())

import matplotlib.pyplot as plt
plt.scatter(df.F, df.C)
plt.show()

#result = model.fit(df.F, df.C, verbose=0)
result = model.fit(df.F, df.C, epochs=100, verbose=0)
print(result.history.keys())

df1 = pd.DataFrame(result.history) # z result zrobi raamkę pandasową
print(df1.head())
df1.plot()
plt.show()

y_pred = model.predict(df.F) #y_pred to celsujesze przewidzieane przez model
plt.scatter(df.F, df.C)
plt.plot(df.F, y_pred, c='r')
plt.show()