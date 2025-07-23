import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import sys
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import time
import keras_tuner

# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ensure stability across runs
keras.backend.clear_session()
tf.random.set_seed(42)

os.chdir(r'C:\Users\selin\5318_project2\Assignment2Data')

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_train_full = X_train / 255.
X_test = X_test / 255.

def plot_examples(data, n_rows=4, n_cols=10):
    """Plot a grid of MNIST examples of a specified size."""

    # Size figure depending on the size of the grid
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))

    for row in range(n_rows):
        for col in range(n_cols):

            # Get next index of image
            index = n_cols * row + col

            # Plot the image at appropriate place in grid
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(data[index], cmap="binary")
            plt.axis('off')

    plt.show()
# plot_examples(X_train)

# Check the format of the label by looking at the first five examples
print(y_train[0:5])
# List all unique labels in the training set
print(np.unique(y_train))

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train, train_size=0.9)


# Define our MLP layer by layer
def build_model(hp):
    model = keras.models.Sequential()

    # Add the input layer
    model.add(keras.layers.Input(shape=[28,28,3]))
    model.add(keras.layers.Flatten())

    # Add 2 hidden layers, treating the number of hidden neurons
    # and the activation function in each as hyperparameters to tune over
    for i in range(1, 3):
        model.add(
            keras.layers.Dense(
                units=hp.Choice(f"units_{i}", values=[100, 200]),
                activation=hp.Choice("activation", values=["relu", "sigmoid", "tanh"])
            )
        )


    # Add the output layer for 9 class classification
    model.add(keras.layers.Dense(9, activation="softmax"))

    # Set up the learning rate values to be tuned over and define the model
    learning_rate = hp.Choice('learning_rate', values=[0.1, 0.01, 0.001])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=1,
    overwrite=True,
    directory="keras_tuning_results",
    project_name="assignment2_mlp"
)

tuner.search_space_summary()
tuner.search(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid))
tuner.results_summary()
# Retrieve the best model from tuning
best_model = tuner.get_best_models()[0]

# Retrain a model on the entire training set
best_hps = tuner.get_best_hyperparameters()[0]
model_nn = build_model(best_hps)

# Train the classifier.
history = model_nn.fit(X_train, y_train, epochs=128,
                    validation_data=(X_valid, y_valid))

# Convert the history dictionary to a Pandas dataframe and extract the accuracies
accuracies = pd.DataFrame(history.history)[['accuracy', 'val_accuracy']]
print(accuracies)

# Plot the accuracies
accuracies.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0.1, 1)
plt.xlabel('Epoch')
plt.show()

# Evaluate the classifier on the test data.
loss, accuracy = model_nn.evaluate(X_test, y_test)
print(f"Accuracy on test data: {accuracy:.4f}")
