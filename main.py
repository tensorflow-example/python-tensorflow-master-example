from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# https://www.tensorflow.org/beta/tutorials/keras/basic_classification

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale these values to a range of 0 to 1 before feeding them to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

# model: build
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# model: compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model: train
history = model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    epochs=10,
    verbose=1,
    batch_size=None)

# model: accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\ntest accuracy:', test_acc)

# model: predictions
predictions = model.predict(test_images)

print('\nprediction: ', np.argmax(predictions[0]))
print('actual:     ', test_labels[0], '\n')

# plot training history
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
