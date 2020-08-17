from tensorflow import keras
import tensorflow as tf


class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True, )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


linear = Linear(4)  # Instantiate our lazy layer
y = linear(tf.ones((3, 2)))  # This will also call `build(input_shape)` then `call(inputs)`
y = linear(tf.convert_to_tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
