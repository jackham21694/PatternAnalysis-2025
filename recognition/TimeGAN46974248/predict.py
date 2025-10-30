"""
Shows example usage of the trained model. Prints out results and provides visualisations.

"""

import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import data_loader
from modules import Autoencoder

#-------------------------------------------------------------PREDICT.PY-----------------

#Evaluating our trained autoencoder (uses checkpoint weights from colab code)

num_layers = 3 # Almost every paper
hidden_dim = 16 # Since we only have 20 data features
num_features = 20 # See dataset.py

model = Autoencoder(hidden_dim, num_layers, num_features)


checkpoint_path = "/content/drive/MyDrive/TimeGANWork/checkpoints/autoencoder"
ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print("Restored trained Autoencoder from }")
else:
    raise ValueError("No checkpoint found. Check the path!")


# Load Evaluation Dataset
file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv'
X_train, X_eval, X_test = data_loader(file_path)
X_train = X_train.numpy()
X_eval = X_eval.numpy()
X_test = X_test.numpy()

# Use our trained model to reconstruct our evaluation dataset
X_reconstructed = model(X_eval)
mse_loss = tf.keras.losses.MeanSquaredError()
loss = mse_loss(X_eval, X_reconstructed)
print("Reconstruction loss on evaluation set:", loss.numpy())

# Use our trained model to reconstruct our training dataset
X_train_reconstructed = model(X_train)
mse_loss_train = tf.keras.losses.MeanSquaredError()
loss_train = mse_loss_train(X_train, X_train_reconstructed)
print("Reconstruction loss on training set:", loss_train.numpy())
