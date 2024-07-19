import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Print the shapes of the training and testing sets
print("Training set size:", x_train.shape)
print("Test set size:", x_test.shape)

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nAccuracy on test data: {test_acc:.4f}')

# Function to display image and prediction
def display_prediction(ax, image, true_label, prediction):
    ax.imshow(image.reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(prediction)
    color = 'green' if predicted_label == true_label else 'red'
    ax.set_title(f"Prediction: {predicted_label}\nTrue value: {true_label}", color=color)
    ax.axis('off')

# Create a figure with 3 rows and 3 columns
fig = plt.figure(figsize=(15, 15))

# Plot training & validation accuracy
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(history.history['accuracy'], label='Training accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation accuracy')
ax1.set_title('Model accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Plot training & validation loss
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(history.history['loss'], label='Training loss')
ax2.plot(history.history['val_loss'], label='Validation loss')
ax2.set_title('Model loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

# Select random images from the test set
num_samples = 5
sample_indices = np.random.randint(0, len(x_test), num_samples)

# Get predictions for the selected images
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]
predictions = model.predict(sample_images)

# Display the results
for i in range(num_samples):
    ax = fig.add_subplot(3, 3, i+4)  # Start from the 4th position in the grid
    display_prediction(ax, sample_images[i], sample_labels[i], predictions[i])

plt.tight_layout()
plt.show()