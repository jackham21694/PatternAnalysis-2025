"""
Shows example usage of the trained model. Prints out results and provides visualisations.

"""

import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import data_loader
from modules import Autoencoder
from dataset import data_visualisation

#-------------------------------------------------------------PREDICT.PY-----------------

#Evaluating our trained autoencoder (uses checkpoint weights from colab code)

num_layers = 3 # Almost every paper
hidden_dim = 64 # Since we only have 20 data features
num_features = 20 # See dataset.py

model = Autoencoder(hidden_dim, num_layers, num_features)

checkpoint_path = "/content/drive/MyDrive/TimeGANWork/checkpoints/autoencoder"
ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print("Restored trained Autoencoder from")
else:
    raise ValueError("No checkpoint found. Check the path!")


# Load Evaluation Dataset
file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv'
X_train, X_eval, X_test, price_mean, price_std, volume_mean, volume_std = data_loader(file_path)
X_train = X_train.numpy()
X_eval = X_eval.numpy()
X_test = X_test.numpy()


# De-normalise both our original data, and our reconstructed data for visualisation

def denormalise(X_norm, price_mean, price_std, volume_mean, volume_std):
    X_flat = X_norm.reshape(-1, X_norm.shape[-1])
    
    price_idx  = [i*4 for i in range(5)] + [i*4 + 2 for i in range(5)]
    volume_idx = [i*4 + 1 for i in range(5)] + [i*4 + 3 for i in range(5)]
    
    X_flat[:, price_idx]  = X_flat[:, price_idx] * price_std + price_mean
    X_flat[:, volume_idx] = X_flat[:, volume_idx] * volume_std + volume_mean
    
    return X_flat.reshape(X_norm.shape)

mse_loss = tf.keras.losses.MeanSquaredError()

# Use our trained model to reconstruct our evaluation dataset
X_eval_reconstructed = model(X_eval).numpy()
X_eval_denorm = denormalise(X_eval, price_mean, price_std, volume_mean, volume_std)
X_eval_recon_denorm = denormalise(X_eval_reconstructed, price_mean, price_std, volume_mean, volume_std)

mid_orig, mid_recon, spread_orig, spread_recon, return_orig, return_recon = data_visualisation(
    X_eval_denorm, X_eval_recon_denorm
)

mid_loss = mse_loss(mid_orig, mid_recon)
spread_loss = mse_loss(spread_orig, spread_recon)
return_loss = mse_loss(return_orig, return_recon)
print("MidPrice loss on training set:", mid_loss.numpy())
print("Spread loss on training set:", spread_loss.numpy())
print("Return loss on training set:", return_loss.numpy())


# Use our trained model to reconstruct our training dataset
X_train_reconstructed = model(X_train).numpy()
X_train_denorm = denormalise(X_train, price_mean, price_std, volume_mean, volume_std)
X_train_recon_denorm = denormalise(X_train_reconstructed, price_mean, price_std, volume_mean, volume_std)

mid_orig, mid_recon, spread_orig, spread_recon, return_orig, return_recon = data_visualisation(
    X_train_denorm, X_train_recon_denorm
)

mid_loss = mse_loss(mid_orig, mid_recon)
spread_loss = mse_loss(spread_orig, spread_recon)
return_loss = mse_loss(return_orig, return_recon)
print("MidPrice loss on training set:", mid_loss.numpy())
print("Spread loss on training set:", spread_loss.numpy())
print("Return loss on training set:", return_loss.numpy())
