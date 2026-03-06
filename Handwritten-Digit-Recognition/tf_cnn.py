# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.models import Sequential

# -------- Prepare Dataset --------
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255.0
test_images  = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255.0

print("TRAIN IMAGES: ", train_images.shape)
print("TEST IMAGES: ", test_images.shape)

# -------- Create Model --------
model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # For multiclass classification, use softmax (not sigmoid) for probabilities
    layers.Dense(10, activation='softmax')
])

# -------- Compile Model --------
# softmax -> NOT logits
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.summary()

# -------- Train Model --------
epochs = 10
history = model.fit(train_images, train_labels, epochs=epochs)

# -------- Visualize Training Results --------
acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, loss, label='Loss')
plt.legend(loc='lower right')
plt.title('Training Accuracy and Loss')
plt.show()

# -------- Helper: predict class ids --------
def predict_classes(x):
    """Return class indices from model.predict probabilities."""
    probs = model.predict(x, verbose=0)           # shape: (N, 10)
    return np.argmax(probs, axis=1)               # shape: (N,)

# -------- Test Image --------
image = train_images[1:2]                          # keeps shape (1,28,28,1)
pred = predict_classes(image)
plt.imshow(image[0].squeeze(), cmap='gray')
plt.title(f'Prediction: {pred[0]}')
plt.axis('off')
plt.show()
print('Prediction of model:', int(pred[0]))

image = train_images[2:2+1]
pred = predict_classes(image)
plt.imshow(image[0].squeeze(), cmap='gray')
plt.title(f'Prediction: {pred[0]}')
plt.axis('off')
plt.show()
print('Prediction of model:', int(pred[0]))

# -------- Test Multiple Images --------
images = test_images[1:5]                          # shape (4,28,28,1)
preds = predict_classes(images)                    # vector of 4 ints

print("Test images array shape:", images.shape)
plt.figure(figsize=(6,6))
for i, (img, p) in enumerate(zip(images, preds), start=1):
    plt.subplot(2,2,i)
    plt.axis('off')
    plt.title(f"Predicted: {int(p)}")
    plt.imshow(img.squeeze(), cmap='gray')
plt.tight_layout()
plt.show()

# -------- Save Model --------
# You can use Keras v3 format (recommended) or .h5 if you prefer.
model.save("tf-cnn-model.keras")   # new format
model.save("tf-cnn-model.h5")      # legacy HDF5 format

# -------- Load Model --------
loaded_model = models.load_model("tf-cnn-model.h5")
probs = loaded_model.predict(train_images[2:3], verbose=0)
loaded_pred = int(np.argmax(probs, axis=1)[0])
plt.imshow(train_images[2].squeeze(), cmap='gray')
plt.title(f'Loaded model prediction: {loaded_pred}')
plt.axis('off')
plt.show()
print('Prediction of loaded model:', loaded_pred)
