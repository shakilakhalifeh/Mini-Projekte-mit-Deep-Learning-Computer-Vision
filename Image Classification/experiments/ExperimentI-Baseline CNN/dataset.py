"""
Wir benutzen CIFAR10 Dataset:
Klassen sind:
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
"""

# Import Libraries
import tensorflow as tf

# Create a function that its purpose is, loading CIFAR10 Dataset
def load_dataset():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  x_train = x_train / 255.0
  x_test = x_test / 255.0

  return x_train, y_train, x_test, y_test
