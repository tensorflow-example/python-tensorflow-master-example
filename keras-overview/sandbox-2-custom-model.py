from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import numpy as np

# get training and test data with Numpy arrays
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # Define your layers here.
        self.layer_1 = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.layer_2 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_3 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_4 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.layer_4(x)


model = MyModel(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, batch_size=32, epochs=5)

# test model
# print("\nEvaluation")
# model.evaluate(x_test,  y_test, verbose=2)
