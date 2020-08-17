import tensorflow as tf
from tensorflow import keras


class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units, input_dim):
        super(Linear, self).__init__()

        w_init = tf.ones_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32"), trainable=True)
        print('\nW\n', self.w)

        b_init = tf.ones_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype="float32"), trainable=True)
        print('\nB\n', self.b)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


inputs = tf.ones((6, 2))
print('\nINPUTS\n', inputs)

linear_layer = Linear(units=4, input_dim=2)  # Instantiate our layer
y = linear_layer(inputs)  # this would execute the `call` function
print('\nY\n', y)

assert y.shape == (6, 4)
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
