import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')
print(type(penguins))
print(penguins.head().to_string())

#sns.pairplot(penguins, hue='species')
#plt.show()

penguins_filtered = penguins.drop(columns=['island', 'sex']).dropna() # wyrzuca wszystkie wiersze, w których nie ma wartości
print(penguins_filtered)

penguins_features = penguins_filtered.drop(columns='species')
target = pd.get_dummies(penguins['species'])
print(target)

from sklearn.model_selection import train_test_split
from tensorflow import keras
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)


X_train, X_test, y_train, y_test = train_test_split(penguins_features, target, test_size=0.2, random_state=0)
#x = 4 kolumny features, random state - jeśli ten sam wyniki będą identyczne

print(X_train.shape)

inputs = keras.Input(4) #struktura danych wejściowych, 4 cechy - 4 kolumny
hidden_layer1 = keras.layers.Dense(4, activation='relu')(inputs)
hidden_layer2 = keras.layers.Dense(4, activation='relu')(hidden_layer1)
output_layer = keras.layers.Dense(3, activation='softmax')(hidden_layer2)


model = keras.Model(inputs=inputs, outputs=output_layer)
print(model.summary())

model.compile(optimizer='rmsprop', loss=keras.losses.categorical_crossentropy)
result = model.fit(X_train, y_train, epochs=100)

sns.lineplot(x=result.epoch, y=result.history['loss'])
plt.show()

y_pred = model.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=target.columns)
predicted_species = prediction.idxmax(axis='columns') # zzwaraca tylko to, gdzie jest największa wartość
print(predicted_species)

from sklearn.metrics import confusion_matrix
actual_species = y_test.idxmax(Axis='columns')
matrix = confusion_matrix(actual_species, predicted_species)
print(matrix)

