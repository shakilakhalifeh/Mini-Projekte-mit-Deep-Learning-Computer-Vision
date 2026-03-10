"""
Model: CNN Architekture

Conv2D(filters, kernel) --> z.B.: Conv2D(32, 3x3)

Bedeutung:

32 Feature Maps, Kernel = 3x3

Das Modell lernt: edges, textures, shapes

"""

# Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models

# Create a function for our CNN model
def build_cnn_model(input_shape=(64,64,3), num_classes=10):

  model = models.Sequential()

  # Conv Layer 1
  model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))

  # Conv Layer 2
  model.add(layers.Conv2D(64, (3,3), activation='relu'))

  # Max Pooling
  model.add(layers.MayPooling2D((2,2)))

  # Flatten
  model.add(layers.Flatten())

  # Dense Layer
  model.add(layers.Dense(128, activation='relu'))

  # Output Layer
  model.add(layers.Dense(num_classes, activation='softmax'))

  return model
