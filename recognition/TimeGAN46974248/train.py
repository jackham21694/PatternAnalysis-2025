"""
Contains the source code for training, validating, testing, and saving your model. The model is
imported from 'modules.py' and the data loader imported from 'dataset.py'. Make sure to plot the
losses and metrics during training. 

"""
from dataset import data_loader
from modules import Autoencoder, batch_generator
import numpy as np
import tensorflow as tf
import os

# Saving weights in Colab
checkpoint_path = "/content/drive/MyDrive/TimeGANWork/checkpoints/autoencoder"

# Create the directory if it doesn't exist
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
    print(f"Created directory: {checkpoint_path}")



file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv'
X_train, X_eval, X_test = data_loader(file_path)
X_train = X_train.numpy()
X_eval = X_eval.numpy()
X_test = X_test.numpy()


# Define our hyperparamters
num_layers = 3 # Almost every paper
hidden_dim = 64 # Changed for experimental purposes, was not capturing midprice well
num_features = 20 # See dataset.py
seq_len = 50 # See dataset.py

training_iterations = 8000
batch_size = 64


initial_lr = 0.0003
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,      
    decay_rate=0.96,
    staircase=True
)


#---------------------------AUTOENCODER TRAINING--------------------------------
model = Autoencoder(hidden_dim, num_layers, num_features)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
mse_loss = tf.keras.losses.MeanSquaredError()
print(mse_loss)

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

print("Start Embedding Network Training")

# Checking if a checkpoint exists
start_fresh = True
if not start_fresh and manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("Initializing from scratch")



for itt in range(training_iterations):
    X_mb, _ = batch_generator(X_train, [seq_len]*len(X_train), batch_size)
    X_mb = np.array(X_mb, dtype=np.float32)

    with tf.GradientTape() as tape:

        # Convert batch to tensor
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)

        # Forward pass
        X_tilde = model(X_mb)

        # Flatten sequences for loss computation: [timesteps, features]
        X_mb_flat = tf.reshape(X_mb, (-1, X_mb.shape[-1]))
        X_tilde_flat = tf.reshape(X_tilde, (-1, X_tilde.shape[-1]))

        price_idx  = [0,2,4,6,8,10,12,14,16,18]  # ask + bid prices
        volume_idx = [1,3,5,7,9,11,13,15,17,19]  # ask + bid volumes

        X_mb_price  = tf.gather(X_mb_flat, price_idx, axis=1)
        X_mb_volume = tf.gather(X_mb_flat, volume_idx, axis=1)
        X_tilde_price  = tf.gather(X_tilde_flat, price_idx, axis=1)
        X_tilde_volume = tf.gather(X_tilde_flat, volume_idx, axis=1)

        price_loss  = mse_loss(X_mb_price, X_tilde_price)
        volume_loss = mse_loss(X_mb_volume, X_tilde_volume)

        # Weight for the midprice
        loss = price_loss + volume_loss

    # ------------------- Backpropagation -------------------
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if itt % 100 == 0:
        print(f"step: {itt}/{training_iterations}, e_loss: {loss.numpy():.6f}")
        manager.save()  # Save checkpoint every 100 steps or as desired

print("Finish Embedding Network Training")








# Train the Generator and Discriminator
#
# Generator     - Produce Latent Sequences from Random Noise
#               - Adversarial Loss (for now)
#
# Discriminator - Classify real and fake latent sequences
#               - Cross-Entropy Loss between real/fake labels

