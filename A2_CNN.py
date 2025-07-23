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

# Define our CNN layer by layer with hyperparameter tuning
def build_model(hp):
    model = keras.models.Sequential()
    # Add the input layer for 28x28 grayscale images
    model.add(keras.layers.Input(shape=[28, 28, 3]))  # Assuming 28x28 grayscale input
    # Add the first convolutional layer, with number of filters and kernel size as hyperparameters
    kernel_size_1 = (hp.Choice("kernel_size_1", values=[3, 5]), hp.Choice("kernel_size_1", values=[3, 5]))
    model.add(
        keras.layers.Conv2D(
            filters=hp.Choice("filters_1", values=[32, 64, 128]),  # Hyperparameter for number of filters
            kernel_size=kernel_size_1,  # Hyperparameter for kernel size
            activation=hp.Choice("activation_1", values=["relu", "tanh"]),  # Hyperparameter for activation function
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Max pooling after the convolutional layer
    # Add the second convolutional layer, similarly as a hyperparameter
    kernel_size_2 = (hp.Choice("kernel_size_2", values=[3, 5]), hp.Choice("kernel_size_2", values=[3, 5]))
    model.add(
        keras.layers.Conv2D(
            filters=hp.Choice("filters_2", values=[64, 128, 256]),  # Hyperparameter for number of filters
            kernel_size=kernel_size_2,  # Hyperparameter for kernel size
            activation=hp.Choice("activation_2", values=["relu", "tanh"]),  # Hyperparameter for activation function
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Max pooling after the second convolutional layer
    # Flatten the feature maps from the convolutional layers
    model.add(keras.layers.Flatten())
    # Dropout layer for regularization (dropout rate as a hyperparameter)
    model.add(keras.layers.Dropout(hp.Float("dropout_rate", min_value=0.3, max_value=0.5, step=0.1)))

    # Output layer with 9 classes (softmax for classification)
    model.add(keras.layers.Dense(9, activation="softmax"))

    # Set up the learning rate as a hyperparameter
    learning_rate = hp.Choice('learning_rate', values=[0.1, 0.01, 0.001])

    # Compile the model with SGD optimizer and sparse categorical crossentropy loss
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
    project_name="assignment2_cnn"
)

tuner.search_space_summary()
tuner.search(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid))
tuner.results_summary()
# Retrieve the best model from tuning
best_model = tuner.get_best_models()[0]

# Retrain a model on the entire training set
best_hps = tuner.get_best_hyperparameters()[0]
model_cnn = build_model(best_hps)

# Train the classifier.
history = model_cnn.fit(X_train, y_train, epochs=128,
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
loss, accuracy = model_cnn.evaluate(X_test, y_test)
print(f"Accuracy on test data: {accuracy:.4f}")
