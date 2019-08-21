from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# generate 2d classification dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)

# scale input data to [-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

# define model
model = keras.Sequential()
model.add(keras.layers.Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# compile model
opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=1)

# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# plot training history
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
