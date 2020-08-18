import tensorflow as tf
from tensorflow import keras

# Prepare our model, loss, and optimizer.
model = keras.Sequential([keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(10)])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


# Create a training step function.
@tf.function  # Make it fast.
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)
        gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


# Prepare a dataset.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(64)

for step, (x, y) in enumerate(dataset):
    loss = train_on_batch(x, y)
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
