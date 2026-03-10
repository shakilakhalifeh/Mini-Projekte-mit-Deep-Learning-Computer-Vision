"""
Bash Terminal: 
python train.py
"""
# Importieren unsere Frameworks
from model import build_cnn_model
from dataset import load_dataset

# Importieren Bibliothek
import tensorflow as tf

# Create a function for the purpose of training the dataset with our CNN model
def train():

  # Load data
  x_train, y_train, x_test, y_test = load_dataset()

  # Build model
  model = build_cnn_model()

  # Compile moodel
  model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
  )

  # Train
  model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
  )

  # Save model
  model.save("cnn_baseline_model.h5")


if __name__ == "__main__":
  train()
