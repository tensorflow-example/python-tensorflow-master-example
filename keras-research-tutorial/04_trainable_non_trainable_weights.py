from tensorflow import keras
import tensorflow as tf


class ComputeSum(keras.layers.Layer):
    """Returns the sum of the inputs."""

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # Create a non-trainable weight
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        i = tf.reduce_sum(inputs, axis=0)  # sum of columns
        self.total.assign_add(i)
        return self.total


my_sum = ComputeSum(2)
print('\nself.total: ', my_sum.total.numpy())

x = tf.convert_to_tensor([[1., 2.], [3., 4.]])

y = my_sum(x)
print('\ny.numpy(): ', y.numpy())  # [4. 6.]
print('\nself.total: ', my_sum.total.numpy())

y = my_sum(x)
print('\ny.numpy(): ', y.numpy())  # [4. 4.]
print('\nself.total: ', my_sum.total.numpy())

assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []
