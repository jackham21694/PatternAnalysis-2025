"""
Contains the source code for training, validating, testing, and saving your model. The model is
imported from 'modules.py' and the data loader imported from 'dataset.py'. Make sure to plot the
losses and metrics during training. 

"""
from dataset import data_loader
from modules import Autoencoder, batch_generator
import numpy as np

file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv'
data = data_loader(file_path).numpy()

# Define our hyperparamters
num_layers = 3 # Almost every paper
hidden_dim = 8 # Since we only have 20 data features
num_features = 20 # See dataset.py
seq_len = 50 # See dataset.py

training_iterations = 5000
batch_size = 124

#---------------------------AUTOENCODER TRAINING--------------------------------
model = Autoencoder(hidden_dim, num_layers, num_features)
optimizer = tf.keras.optimizers.Adam()
mse_loss = tf.keras.losses.MeanSquaredError()
print(mse_loss)

# Saving weights in Colab
checkpoint_path = "/content/drive/MyDrive/TimeGANWork/checkpoints/autoencoder"
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

print("Start Embedding Network Training")

# Checking if a checkpoint exists
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("Initializing from scratch")



for itt in range(training_iterations):
    X_mb, _ = batch_generator(data, [seq_len]*len(data), batch_size)
    X_mb = np.array(X_mb, dtype=np.float32)

    with tf.GradientTape() as tape:
        # Ensure the model call and loss calculation are within the tape context
        X_tilde = model(X_mb)
        loss = 10.0 * tf.sqrt(mse_loss(X_mb, X_tilde))  # same as original scaling

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if itt % 500 == 0:
        print(f"step: {itt}/{training_iterations}, e_loss: {loss.numpy():.4f}")
        manager.save()  # Save checkpoint every 1000 steps

print("Finish Embedding Network Training")








# Train the Generator and Discriminator
#
# Generator     - Produce Latent Sequences from Random Noise
#               - Adversarial Loss (for now)
#
# Discriminator - Classify real and fake latent sequences
#               - Cross-Entropy Loss between real/fake labels

