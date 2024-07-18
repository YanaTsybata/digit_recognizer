import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST download
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Conclusion info abot data
print("Size training set: ", x_train.shape)
print("Size testing set: ", x_test.shape)

#model creation
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
#Visualisation example img
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Tag: {y_train[0]}")
plt.show()