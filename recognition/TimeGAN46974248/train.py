"""
Contains the source code for training, validating, testing, and saving your model. The model is
imported from 'modules.py' and the data loader imported from 'dataset.py'. Make sure to plot the
losses and metrics during training. 

"""
from dataset import data_loader
from modules import Autoencoder, Supervisor, Generator, Discriminator
from utils import random_generator, batch_generator
import numpy as np
import tensorflow as tf
import os


file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
X_train, X_eval, X_test, price_min, price_max = data_loader(file_path, 50, 10)
X_train = X_train.numpy()
X_eval = X_eval.numpy()
X_test = X_test.numpy()


# Define our hyperparamters
num_layers = 3
hidden_dim = 64
num_features = 20
seq_len = 50

training_iterations = 8000
batch_size = 128

initial_lr = 0.0005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=2000,
    decay_rate=0.95,
    staircase=True
)


#---------------------------AUTOENCODER TRAINING--------------------------------
model = Autoencoder(hidden_dim, num_layers, num_features)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
mse_loss = tf.keras.losses.MeanSquaredError()
print(mse_loss)

print("Start Embedding Network Training")


for itt in range(training_iterations):
    X_mb = batch_generator(X_train, [seq_len]*len(X_train), batch_size)
    X_mb = np.array(X_mb, dtype=np.float32)

    with tf.GradientTape() as tape:

        # Convert batch to tensor
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)

        # Forward pass
        X_tilde = model(X_mb)

        # Flatten sequences for loss computation: [timesteps, features]
        X_mb_flat = tf.reshape(X_mb, (-1, X_mb.shape[-1]))
        X_tilde_flat = tf.reshape(X_tilde, (-1, X_tilde.shape[-1]))
        price_loss  = mse_loss(X_mb_flat, X_tilde_flat)

        # Variance preservation - PER FEATURE TYPE
        var_price_real = tf.math.reduce_variance(X_mb_flat, axis=1)
        var_price_recon = tf.math.reduce_variance(X_tilde_flat, axis=1)
        var_price_loss = tf.reduce_mean(tf.abs(tf.sqrt(var_price_real + 1e-6) - tf.sqrt(var_price_recon + 1e-6)))

        # Weighted combination of losses
        loss = 10*tf.sqrt(price_loss) + 100*var_price_loss

    # ------------------- Backpropagation -------------------
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if itt % 100 == 0:
        print(f"step: {itt}/{training_iterations}, e_loss: {loss.numpy():.6f}")

model.save_weights('/content/drive/MyDrive/autoencoder.weights.h5')
print("Finish Embedding Network Training")


#---------------------------------------------------------------------------SUPERVISOR/GENERATOR/DISCRIMINATOR TRAINING -------------------------------------------------------------

num_layers = 3
hidden_dim = 32
num_features = 20
seq_len = 50

training_iterations = 1000
batch_size = 64

# Model Definitions
autoencoder = Autoencoder(hidden_dim, num_layers, num_features)
autoencoder.load_weights('autoencoder.weights.h5')

supervisor = Supervisor(hidden_dim, num_layers)
generator  = Generator(hidden_dim, num_layers)
discriminator = Discriminator(hidden_dim, num_layers)


# Optimisers
supervisor_optimiser    = tf.keras.optimizers.Adam()
generator_optimiser     = tf.keras.optimizers.Adam()
autoencoder_optimiser   = tf.keras.optimizers.Adam()
discriminator_optimiser = tf.keras.optimizers.Adam()

# Data Loading
file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv'
X_train, X_eval, X_test, price_min, price_max, volume_mean, volume_std = data_loader(file_path)
X_train = X_train.numpy()
X_eval = X_eval.numpy()
X_test = X_test.numpy()

"""
Supervised Loss Training

The original paper trains the supervisior and generator together initially,
purely so the generator is intialised with temporal knowledge before adversarial
training. We want the generator to produce sequences that follow the latent
dynamics learned by the supervisor. This will help stabilise training later.

"""
for itt in range(training_iterations):
  # Generate our mini batch, and convert to numpy array
  X_mb = batch_generator(X_train, [seq_len]*len(X_train), batch_size)
  X_mb = np.array(X_mb, dtype=np.float32)

  with tf.GradientTape() as tape:
    # Send batch through our pretrained embedder
    H = autoencoder.embed(X_mb)

    # Send new latent representation into supervisor to predict the next time
    # steps (apart from last).
    H_hat_supervise = supervisor(H)

    # Compute our supervised loss
    g_loss_s = tf.reduce_mean(tf.keras.losses.mse(H[:,1:,:], H_hat_supervise[:,:-1,:]))

  # Adjust the trainable parameters for both the generator and supervisor
  gradients_supervisor = tape.gradient(g_loss_s,  supervisor.trainable_variables)
  gradients_generator  = tape.gradient(g_loss_s,  generator.trainable_variables)

  supervisor_optimiser.apply_gradients(zip(gradients_supervisor, supervisor.trainable_variables))
  generator_optimiser.apply_gradients(zip(gradients_generator, generator.trainable_variables))


  if itt % 100 == 0:
        print(f"Step {itt}, Supervised Loss: {g_loss_s.numpy():.4f}")

"""
Joint Training

The original paper for every training step, they train the generator twice
then the discriminator, as to prevent the discriminator overpowering the 
generator early (common for GAN training stabilisation). 

"""

for itt in range(training_iterations):
  for i in range(2):
    X_mb = batch_generator(X_train, [seq_len]*len(X_train), batch_size)
    X_mb = tf.convert_to_tensor(np.array(X_mb, dtype=np.float32))

    # Random noise for our generator
    Z_mb = random_generator(batch_size, hidden_dim, [50]*batch_size, 50)

    with tf.GradientTape() as tape:
      # Use generator to create synthetic embeddings from noise vector
      H_hat = generator(Z_mb, training=True)

      # Use the supervisor to predict time steps using synthetic data
      S_hat = supervisor(H_hat, training=True)

      # Use the recovery to convert synthetic data in the original feature space
      X_hat = autoencoder.recovery(S_hat, training=True)

      # Supervised Loss for our Generator
      H_real = autoencoder.embed(X_mb)
      H_supervised = supervisor(H_real)
      supervised_generator_loss = tf.reduce_mean(tf.keras.losses.mse(H_real[:,1:,:], H_supervised[:,:-1,:]))


      # Universal Loss for our Generator (Adversarial)
      Y_fake   = discriminator(S_hat, training=True)
      Y_fake_e = discriminator(H_hat, training=True) 

      bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
      G_loss_U = bce(tf.ones_like(Y_fake), Y_fake)
      G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)

      # Additional Moment Loss for Generator (Helps stabilise Training)
      mean_real, var_real = tf.nn.moments(X_mb, axes=[0, 1])
      mean_hat, var_hat   = tf.nn.moments(X_hat, axes=[0, 1])
      mean_loss = tf.reduce_mean(tf.abs(mean_real - mean_hat))
      var_loss  = tf.reduce_mean(tf.abs(tf.sqrt(var_real + 1e-6) - tf.sqrt(var_hat + 1e-6)))


      # Total Generator Loss (constants here taken from OG Paper)
      G_loss = G_loss_U + G_loss_U_e + 100 * tf.sqrt(supervised_generator_loss) + 100 * (mean_loss + var_loss)
    
    # Apply gradients to both generator and supervisor
    gradients_supervisor = tape.gradient(G_loss, supervisor.trainable_variables)
    gradients_generator  = tape.gradient(G_loss, generator.trainable_variables)

    supervisor_optimiser.apply_gradients(zip(gradients_supervisor, supervisor.trainable_variables))
    generator_optimiser.apply_gradients(zip(gradients_generator, generator.trainable_variables))

    # Embedder and Recovery Training
    with tf.GradientTape() as tape_e:
      X_tilde = autoencoder(X_mb, training=True)
      E_loss_T0 = tf.reduce_mean(tf.keras.losses.mse(X_mb, X_tilde))
      E_loss = 10 * tf.sqrt(E_loss_T0) + 0.1 * supervised_generator_loss

    # Apply gradients to both embedder, and recovery
    trainable_vars_e = autoencoder.trainable_variables
    gradients_e = tape_e.gradient(E_loss, trainable_vars_e)
    autoencoder_optimiser.apply_gradients(zip(gradients_e, trainable_vars_e))
  
  """
  Discriminator Training

  Before training we check the current performance of the discriminator to
  ensure it is not overperforming/overpowering our generator. The constant 0.15
  is used inside the original paper.
  """

  X_mb = batch_generator(X_train, [seq_len]*len(X_train), batch_size)
  X_mb = tf.convert_to_tensor(np.array(X_mb, dtype=np.float32))

  # Random noise for our generator
  Z_mb = random_generator(batch_size, hidden_dim, [50]*batch_size, 50)

  with tf.GradientTape() as tape_d_test:
    # Fake Data
    H_hat = generator(Z_mb, training=False)
    S_hat = supervisor(H_hat, training=False)

    # Real Data
    H_real = autoencoder.embed(X_mb, training=False)

    # Discriminator Results
    Y_real = discriminator(H_real, training=False)
    Y_fake = discriminator(S_hat, training=False)

    # Calculate loss
    D_loss = bce(tf.ones_like(Y_real), Y_real) + bce(tf.zeros_like(Y_fake), Y_fake)

  if D_loss.numpy() > 0.15:
    with tf.GradientTape() as tape_d:
        H_hat = generator(Z_mb, training=True)
        S_hat = supervisor(H_hat, training=True)
        H_real = autoencoder.embed(X_mb, training=True)

        Y_real = discriminator(H_real, training=True)
        Y_fake = discriminator(S_hat, training=True)

        D_loss = bce(tf.ones_like(Y_real), Y_real) + bce(tf.zeros_like(Y_fake), Y_fake)
    
    gradients_d = tape_d.gradient(D_loss, discriminator.trainable_variables)
    discriminator_optimiser.apply_gradients(zip(gradients_d, discriminator.trainable_variables))
    

  if itt % 1000 == 0:
      print(
          f"step: {itt}/{training_iterations}, "
          f"d_loss: {D_loss:.4f}, "
          f"g_loss_u: {G_loss_U:.4f}, "
          f"g_loss_s: {np.sqrt(supervised_generator_loss):.4f}, "
          f"g_loss_v: {(mean_loss + var_loss):.4f}, "
          f"e_loss_t0: {np.sqrt(E_loss_T0):.4f}"
      )