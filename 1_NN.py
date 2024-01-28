import numpy as np
import pandas as pd

# komponenty do budowy sieci
from tensorflow.keras.models import Sequential    # szkielet
from tensorflow.keras.layers import Dense   # wszystkie połączenia
from tensorflow.random import set_seed  # losowosc

set_seed(0)  #te same wyniki

model = Sequential()
model.add(Dense(1, input_shape=[1], activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1))

# kompilacja
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
# mse - mean square error
# mae - mean average error

df = pd.read_csv('f-c.csv', usecols=[1, 2])
print(df.head())

import matplotlib.pyplot as plt


result = model.fit(df.F, df.C, epochs=20000, verbose=0)
print(result.history.keys())

df1 = pd.DataFrame(result.history)
print(df1.head())
df1.plot()
plt.show()

y_pred = model.predict(df.F)   # policz stopnie C
plt.scatter(df.F, df.C)
plt.plot(df.F, y_pred, c='r')
plt.show()