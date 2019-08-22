from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dataframe = pd.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# baseline
def create_baseline():
    # create model
    model = keras.Sequential()
    # Dropout can be applied to input neurons called the visible layer
    # a new Dropout layer between the input (or visible layer) and the first hidden layer.
    # The dropout rate is set to 20%, meaning one in 5 inputs will be randomly excluded from each update cycle.
    # Additionally, as recommended in the original paper on Dropout, a constraint is imposed on the weights for
    # each hidden layer, ensuring that the maximum norm of the weights does not exceed a value of 3. This is done
    # by setting the kernel_constraint argument on the Dense class when constructing the layers
    model.add(keras.layers.Dropout(0.2, input_shape=(60,)))
    model.add(keras.layers.Dense(60, kernel_initializer='normal', activation='relu',
                                 kernel_constraint=keras.constraints.max_norm(3)))
    model.add(keras.layers.Dense(30, kernel_initializer='normal', activation='relu',
                                 kernel_constraint=keras.constraints.max_norm(3)))
    model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


np.random.seed(seed)
estimators = [
    ('standardize', StandardScaler()),
    ('mlp', keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_baseline, epochs=300, batch_size=16, verbose=1))
]
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
