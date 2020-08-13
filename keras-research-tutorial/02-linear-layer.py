from tensorflow import keras
import tensorflow as tf


class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True, )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


linear_layer = Linear(4)  # Instantiate our lazy layer
y = linear_layer(tf.ones((3, 3)))  # This will also call `build(input_shape)` then `call(inputs)`
