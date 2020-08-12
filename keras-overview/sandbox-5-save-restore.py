from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

# get training and test data with Numpy arrays
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, batch_size=32, epochs=5)

########################
# SAVE RESTORE WEIGHTS #
########################

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./save/weights/my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./save/weights/my_model')

####################################
# SAVE RESTORE MODEL CONFIGURATION #
####################################

# json serialize deserialize
json_string = model.to_json()
fresh_model = tf.keras.models.model_from_json(json_string)

# yaml serialize deserialize
yaml_string = model.to_yaml()
fresh_model = tf.keras.models.model_from_yaml(yaml_string)

#############################
# SAVE RESTORE ENTIRE MODEL #
#############################

# Save entire model to a HDF5 file
model.save('./save/entire-model/my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('./save/entire-model/my_model.h5')
