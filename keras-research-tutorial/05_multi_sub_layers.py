import tensorflow as tf
from tensorflow import keras


class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MLP(keras.layers.Layer):
    """Simple stack of Linear layers."""

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        print('\nafter linear 1:', x)
        x = tf.nn.relu(x)
        print('\nafter linear 1 (relu):', x)
        x = self.linear_2(x)
        print('\nafter linear 2:', x)
        x = tf.nn.relu(x)
        print('\nafter linear 2 (relu):', x)
        x = self.linear_3(x)
        print('\nafter linear 3:', x)
        return x


# manually built mlp
mlp = MLP()
# same as
# mlp = keras.Sequential([
#     keras.layers.Dense(32, activation=tf.nn.relu),
#     keras.layers.Dense(32, activation=tf.nn.relu),
#     keras.layers.Dense(10)])

# The first call to the `mlp` object will create the weights
y = mlp(tf.ones(shape=(3, 64)))

assert y.shape == (3, 10)
# Weights are recursively tracked.
assert len(mlp.weights) == 6
