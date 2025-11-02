"""
Module: train.py
Author: Jack Ham, 46974248

Description
-----------

I apologise for the contents of this file, the top of the file is an attempt 
at a nice functionaly implementation, but it does not seem to want to work for some
reason, scroll all the way down there is my full sequential colab file that works for the
training.

"""

from dataset import data_loader
from modules import Autoencoder, Supervisor, Generator, Discriminator
from utils import random_generator, batch_generator
from predict import reconstructed_indicators, evaluate_autoencoder, generate_lob_heatmap_report
import numpy as np
import tensorflow as tf
import os
from google.colab import drive


# ------------------------------------- DRIVER SCRIPT --------------------------------------------------
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
X_train, X_eval, X_test, price_min, price_max = data_loader(file_path, 100, 10)
X_train = X_train.numpy()
X_eval = X_eval.numpy()
X_test = X_test.numpy()


# Define our hyperparamters
num_layers = 3
hidden_dim = 64
num_features = 20
seq_len = 100

training_iterations = 4000
batch_size = 64

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

train_history = {
    'iteration': [],
    'total_loss': [],
    'price_loss': [],
    'var_loss': [],
    'mid_loss': [],
    'spread_loss': []
}

print(mse_loss)

print("Start Embedding Network Training")


for itt in range(training_iterations):
    X_mb = batch_generator(X_train, batch_size)
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


        # Midprice Loss
        mid_real = (X_mb[:, :, 0] + X_mb[:, :, 1]) / 2.0
        mid_recon = (X_tilde[:, :, 0] + X_tilde[:, :, 1]) / 2.0
        mid_loss = tf.reduce_mean(tf.square(mid_real - mid_recon))

        # 3. Spread loss
        spread_real = X_mb[:, :, 0] - X_mb[:, :, 1]
        spread_recon = X_tilde[:, :, 0] - X_tilde[:, :, 1]
        spread_loss = tf.reduce_mean(tf.square(spread_real - spread_recon))

        # Weighted combination of losses
        loss = 10*tf.sqrt(price_loss) + 25*var_price_loss + 50.0*mid_loss + 250*spread_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    train_history['iteration'].append(itt)
    train_history['total_loss'].append(loss.numpy())
    train_history['price_loss'].append(price_loss.numpy())
    train_history['var_loss'].append(var_price_loss.numpy())
    train_history['mid_loss'].append(mid_loss.numpy())
    train_history['spread_loss'].append(spread_loss.numpy())



    if itt % 100 == 0:
        print(f"Step: {itt}, "
          f"price_loss: {price_loss.numpy():.6f}, var_price_loss: {var_price_loss.numpy():.6f}, "
          f"mid_loss: {mid_loss.numpy():.6f}, spread_loss: {spread_loss.numpy():.6f}")

model.save_weights('/content/drive/MyDrive/autoencoder.weights.h5')
print("Finish Embedding Network Training")


reconstructed_indicators(X_eval, model, price_min, price_max)

reconstructed_indicators(X_train, model, price_min, price_max)

evaluate_autoencoder(model, X_eval)




num_layers = 3
hidden_dim = 64
num_features = 20
seq_len = 100

training_iterations = 501
batch_size = 128

# Model Definitions
autoencoder = Autoencoder(hidden_dim, num_layers, num_features)

# "Build" the model by calling it on dummy input
dummy_input = tf.zeros((1, seq_len, num_features))   # (batch, seq_len, features)
_ = autoencoder(dummy_input)


autoencoder.load_weights('/content/drive/MyDrive/autoencoder.weights.h5')

supervisor = Supervisor(hidden_dim, num_layers)
generator  = Generator(hidden_dim, num_layers)
discriminator = Discriminator(hidden_dim, num_layers)

# Dummy calls to empty models

dummy_z = tf.zeros((batch_size, seq_len, hidden_dim))
_ = generator(dummy_z)
_ = discriminator(dummy_z)
_ = supervisor(dummy_z)

generator.load_weights('/content/drive/MyDrive/generator.weights.h5')
supervisor.load_weights('/content/drive/MyDrive/supervisor.weights.h5')
discriminator.load_weights('/content/drive/MyDrive/discriminator.weights.h5')

# Optimisers (supervisor is optimised with the generator)
generator_optimiser     = tf.keras.optimizers.Adam(learning_rate=0.0001)
autoencoder_optimiser   = tf.keras.optimizers.Adam()
discriminator_optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Data Loading
file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
X_train, X_eval, X_test, price_min, price_max = data_loader(file_path, seq_len=seq_len, depth=10)
X_train = X_train.numpy()
X_eval = X_eval.numpy()
X_test = X_test.numpy()


timegan_history = {
    'iteration': [],
    'd_loss': [],
    'g_loss_u': [],
    'g_loss_u_e': [],
    'g_loss_s': [],
    'g_loss_v': [],
    'g_loss_total': [],
    'e_loss_t0': [],
    'e_loss_total': [],
    'mean_loss': [],
    'var_loss': [],
    'ssim': [],
    'kl_mid': [],
    'kl_returns': [],
    'kl_spread': []
}



"""
Supervised Loss Training

The original paper trains the supervisior and generator together initially,
purely so the generator is intialised with temporal knowledge before adversarial
training. We want the generator to produce sequences that follow the latent
dynamics learned by the supervisor. This will help stabilise training later.

"""
for itt in range(0):
  # Generate our mini batch, and convert to numpy array
  X_mb = batch_generator(X_train,  batch_size)
  X_mb = np.array(X_mb, dtype=np.float32)

  with tf.GradientTape() as tape:
    # Embed real data sample
    H_real = autoencoder.embed(X_mb)

    # Send latent representation into supervisor to predict the next time
    # steps (apart from last).
    H_hat_supervise = supervisor(H_real, training=True)

    # Compute our supervised loss
    g_loss_s = tf.reduce_mean(tf.keras.losses.mse(H_real[:,1:,:], H_hat_supervise[:,:-1,:]))

  # Adjust the trainable parameters for both the generator and supervisor
  trainable_vars = generator.trainable_variables + supervisor.trainable_variables
  gradients = tape.gradient(g_loss_s, trainable_vars)

  generator_optimiser.apply_gradients(zip(gradients, trainable_vars))


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
    X_mb = batch_generator(X_train, batch_size)
    X_mb = tf.convert_to_tensor(np.array(X_mb, dtype=np.float32))

    # Random noise for our generator
    Z_mb = random_generator(batch_size, hidden_dim, [seq_len]*batch_size, seq_len)

    with tf.GradientTape() as tape:
      # Use generator to create synthetic embeddings from noise vector
      H_hat = generator(Z_mb, training=True)

      # Use the supervisor to predict time steps using synthetic data
      S_hat = supervisor(H_hat, training=True)

      # Use the recovery to convert synthetic data in the original feature space
      X_hat = autoencoder.recovery(S_hat)

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
      G_loss = G_loss_U + G_loss_U_e + 100 * tf.sqrt(supervised_generator_loss) + 200 * (mean_loss + var_loss)

    # Apply gradients to both generator and supervisor
    trainable_vars = generator.trainable_variables + supervisor.trainable_variables
    gradients = tape.gradient(G_loss, trainable_vars)
    generator_optimiser.apply_gradients(zip(gradients, trainable_vars))

    # Embedder and Recovery Training
    with tf.GradientTape() as tape_e:
      X_tilde = autoencoder(X_mb)
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

  X_mb = batch_generator(X_train, batch_size)
  X_mb = tf.convert_to_tensor(np.array(X_mb, dtype=np.float32))

  # Random noise for our generator
  Z_mb = random_generator(batch_size, hidden_dim, [seq_len]*batch_size, seq_len)

  with tf.GradientTape() as tape_d_test:
    # Fake Data
    H_hat = generator(Z_mb, training=False)
    S_hat = supervisor(H_hat, training=False)

    # Real Data
    H_real = autoencoder.embed(X_mb)

    # Discriminator Results
    Y_real = discriminator(H_real, training=False)
    Y_fake = discriminator(S_hat, training=False)

    # Calculate loss
    D_loss = bce(tf.ones_like(Y_real), Y_real) + bce(tf.zeros_like(Y_fake), Y_fake)

  if D_loss.numpy() > 0.15:
    with tf.GradientTape() as tape_d:
        H_hat = generator(Z_mb, training=True)
        S_hat = supervisor(H_hat, training=True)
        H_real = autoencoder.embed(X_mb)

        Y_real = discriminator(H_real, training=True)
        Y_fake = discriminator(S_hat, training=True)

        D_loss = bce(tf.ones_like(Y_real), Y_real) + bce(tf.zeros_like(Y_fake), Y_fake)

    gradients_d = tape_d.gradient(D_loss, discriminator.trainable_variables)
    discriminator_optimiser.apply_gradients(zip(gradients_d, discriminator.trainable_variables))





  if itt % 10 == 0:

      # Generate synthetic data for evaluation
      Z_eval = random_generator(batch_size, hidden_dim, [seq_len]*batch_size, seq_len)
      H_hat_eval = generator(Z_eval, training=False)
      S_hat_eval = supervisor(H_hat_eval, training=False)
      X_hat_eval = autoencoder.recovery(S_hat_eval)

      # Get real data batch for comparison
      X_real_eval = batch_generator(X_train, batch_size)
      X_real_eval = np.array(X_real_eval)



      # Compute SSIM and KL metrics
      # feature_ask=0 (Ask Price L1), feature_bid=10 (Bid Price L1)
      ssim_val, kl_mid_val, kl_returns_val, kl_spread_val = visualize_lob(
          X_real_eval,
          X_hat_eval.numpy(),
          feature_ask=0,
          feature_bid=1
      )


      timegan_history['iteration'].append(itt)
      timegan_history['d_loss'].append(D_loss.numpy())
      timegan_history['g_loss_u'].append(G_loss_U.numpy())
      timegan_history['g_loss_u_e'].append(G_loss_U_e.numpy())
      timegan_history['g_loss_s'].append(np.sqrt(supervised_generator_loss.numpy()))
      timegan_history['g_loss_v'].append((mean_loss + var_loss).numpy())
      timegan_history['g_loss_total'].append(G_loss.numpy())
      timegan_history['e_loss_t0'].append(np.sqrt(E_loss_T0.numpy()))
      timegan_history['e_loss_total'].append(E_loss.numpy())
      timegan_history['mean_loss'].append(mean_loss.numpy())
      timegan_history['var_loss'].append(var_loss.numpy())
      timegan_history['ssim'].append(ssim_val)
      timegan_history['kl_mid'].append(kl_mid_val)
      timegan_history['kl_returns'].append(kl_returns_val)
      timegan_history['kl_spread'].append(kl_spread_val)


      print(
          f"step: {itt}/{training_iterations}, "
          f"d_loss: {D_loss:.4f}, "
          f"g_loss_u: {G_loss_U:.4f}, "
          f"g_loss_s: {np.sqrt(supervised_generator_loss):.4f}, "
          f"g_loss_v: {(mean_loss + var_loss):.4f}, "
          f"e_loss_t0: {np.sqrt(E_loss_T0):.4f}"
      )

generator.save_weights('/content/drive/MyDrive/generator.weights.h5')
supervisor.save_weights('/content/drive/MyDrive/supervisor.weights.h5')
autoencoder.save_weights('/content/drive/MyDrive/autoencoder.weights.h5')
discriminator.save_weights('/content/drive/MyDrive/discriminator.weights.h5')

generate_lob_heatmap_report(generator, supervisor, autoencoder, X_test)





#---------------IN CASE THE ABOVE SCRIPT IS ERRORING HERE IS MY SEQUENTIAL GOOGLE COLAB FILE ----------------------#
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#------------------------------------DATASET.PY-----------------------

def data_loader(path, seq_len, depth):
    # Load raw CSV data
    raw_data = np.loadtxt(path, delimiter=',')

    # Remove volumes values
    volume_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    raw_data = np.delete(raw_data, volume_idx, axis=1)

    # Define sequence length
    num_sequences = raw_data.shape[0] // seq_len
    total_timesteps = num_sequences * seq_len

    # Trim leftover timesteps
    trimmed_data = raw_data[:total_timesteps]
    shaped_sample_data = trimmed_data.reshape((num_sequences, seq_len, (depth*2)))

    # Shuffle and prepare train, eval, test datasets
    np.random.seed(46974248)
    np.random.shuffle(shaped_sample_data)

    # Split 70/10/20
    train_end = int(0.7 * num_sequences)
    val_end   = int(0.8 * num_sequences)

    train = shaped_sample_data[:train_end]
    val   = shaped_sample_data[train_end:val_end]
    test  = shaped_sample_data[val_end:]

    # Combine train + val for statistics
    train_val = np.vstack([train.reshape(-1,  (depth*2)),
                           val.reshape(-1,  (depth*2))])

    # --- Compute min/max for prices and volumes ---
    price_min  = train_val.min(axis=0, keepdims=True)
    price_max  = train_val.max(axis=0, keepdims=True)

    # --- Normalisation function ---
    def normalise(data):
        data_flat = data.reshape(-1,  (depth*2))

        # Min-max scale prices to [0, 1]
        data_flat = (
            (data_flat - price_min) /
            (price_max - price_min + 1e-8)
        )

        return data_flat.reshape(data.shape)

    # Apply normalization
    train_norm = normalise(train)
    val_norm   = normalise(val)
    test_norm  = normalise(test)

    # Return tensors and scaling stats for reconstruction
    return (
        tf.convert_to_tensor(train_norm, dtype=tf.float32),
        tf.convert_to_tensor(val_norm, dtype=tf.float32),
        tf.convert_to_tensor(test_norm, dtype=tf.float32),
        price_min, price_max
    )


"""
LOBSTER Data has three key indicators, being midprice, spread, and return.

Here we will visualise

"""
def data_visualisation(X_orig, X_recon):

    # Flatten/Reshape data to get rid of sequences
    X_orig_flat = tf.reshape(X_orig, (-1, X_orig.shape[-1]))
    X_recon_flat = tf.reshape(X_recon, (-1, X_recon.shape[-1]))

    # Calculate the mid price for these samples
    best_ask_orig = X_orig_flat[:,0] # Entire First Column
    best_bid_orig = X_orig_flat[:,1] # Entire Second Column
    best_ask_recon = X_recon_flat[:,0]
    best_bid_recon = X_recon_flat[:,1]

    midprice_orig = (best_ask_orig + best_bid_orig) / 2.0
    midprice_recon = (best_ask_recon + best_bid_recon) / 2.0

    # Calculate the spread for these samples
    spread_orig = (best_ask_orig- best_bid_orig)
    spread_recon = (best_ask_recon - best_bid_recon)


    # Calculate the return (MidPrice Returns), no value last
    # timestamp, since it is geometric
    return_orig  = midprice_orig[1:] - midprice_orig[:-1]
    return_recon = midprice_recon[1:] - midprice_recon[:-1]

    plt.figure(figsize=(16,4))
    plt.plot(midprice_orig.numpy(), color='blue')
    plt.plot(midprice_recon.numpy(), label="Reconstructed", color='orange', linestyle='--')
    plt.title("Midprice")
    plt.xlabel("Time Steps")
    plt.ylabel("Midprice")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(16,4))
    plt.plot(spread_orig.numpy(), label="Original", color='green', linewidth=0.8)
    plt.plot(spread_recon.numpy(), label="Reconstructed", color='red', linestyle='--', linewidth=0.8)
    plt.title("Spread")
    plt.xlabel("Time Steps")
    plt.ylabel("Spread")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(16,4))
    plt.plot(return_orig.numpy(), label="Original", color='blue', marker='o', markersize=2)
    plt.plot(return_orig.numpy(), label="Reconstructed", color='orange', marker='x', markersize=2, linestyle='--')
    plt.title("Return Comparison")
    plt.title("Return")
    plt.xlabel("Time Steps")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()
    return midprice_orig, midprice_recon, spread_orig, spread_recon, return_orig, return_recon


# --------------------- MODULES.PY-------------------------------

def batch_generator(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)[:batch_size]
    X_mb = np.array([data[i] for i in idx], dtype=np.float32)
    T_mb = np.array([time[i] for i in idx], dtype=np.int32)
    return X_mb, T_mb

def random_generator (batch_size, z_dim, T_mb, max_seq_len):
  Z_mb = np.zeros((batch_size, max_seq_len, z_dim), dtype=np.float32)
  for i in range(batch_size):
    temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    Z_mb[i, :T_mb[i], :] = temp_Z
  return tf.convert_to_tensor(Z_mb)


class Autoencoder(tf.keras.Model):
    def __init__(self, hidden_dim, num_layers, num_features):
        super(Autoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_features = num_features

        # Build embedder layers
        self.embedder_grus = []
        for i in range(num_layers):
            self.embedder_grus.append(
                tf.keras.layers.GRU(hidden_dim, activation='tanh',
                                    return_sequences=True,
                                    name=f"embedder_gru_{i}")
            )
        self.embedder_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_dim, activation='sigmoid')
        )

        # Build recovery layers
        self.recovery_grus = []
        for i in range(num_layers):
            self.recovery_grus.append(
                tf.keras.layers.GRU(hidden_dim, activation='tanh',
                                    return_sequences=True,
                                    name=f"recovery_gru_{i}")
            )
        self.recovery_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_features, activation='sigmoid')
        )

    def call(self, X):
        # Embedder
        H = X
        for gru in self.embedder_grus:
            H = gru(H)
        H = self.embedder_dense(H)
        # Recovery
        R = H
        for gru in self.recovery_grus:
            R = gru(R)
        X_tilde = self.recovery_dense(R)
        return X_tilde

    def embed(self, X):
        H = X
        for gru in self.embedder_grus:
            H = gru(H)
        H = self.embedder_dense(H)
        return H

    def recovery(self, H):
        R = H
        for gru in self.recovery_grus:
            R = gru(R)
        X_tilde = self.recovery_dense(R)
        return X_tilde





# Usually has one less layer then all the other components to keep it simple
class Supervisor(tf.keras.Model):
  def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Building GRU Layers
        self.grus = []
        for i in range(num_layers-1):
            self.grus.append(
                tf.keras.layers.GRU(
                    hidden_dim,
                    activation='tanh',
                    return_sequences=True,
                    name=f"supervisor_gru_{i}"
                )
            )

        # Output layer maps back to hidden_dim (next-step latent)
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_dim, activation=None)
        )

  def call(self, H):
        S = H
        for gru in self.grus:
            S = gru(S)
        S = self.dense(S)
        return S



class Generator(tf.keras.Model):
    def __init__(self, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        # Build GRU layers
        self.grus = []
        for i in range(num_layers):
            self.grus.append(
                tf.keras.layers.GRU(
                    hidden_dim,
                    activation='tanh',
                    return_sequences=True,
                    name=f"generator_gru_{i}"
                )
            )

        # Fully connected output layer to map back into latent dimension
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_dim, activation=None)
        )

    def call(self, Z):
        G = Z
        for gru in self.grus:
            G = gru(G)
        E = self.dense(G)
        return E


class Discriminator(tf.keras.Model):
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build GRU layers
        self.grus = []
        for i in range(num_layers):
            self.grus.append(
                tf.keras.layers.GRU(
                    hidden_dim,
                    activation='tanh',
                    return_sequences=True,
                    name=f"discriminator_gru_{i}"
                )
            )

        # Output layer: maps GRU outputs to scalar logits for real/fake classification
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation=None)
        )

    def call(self, H):
        D = H
        for gru in self.grus:
            D = gru(D)
        Y_hat = self.dense(D)
        return Y_hat
    

#-------------------------------------------------------------TRAIN.PY-----------------
file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
X_train, X_eval, X_test, price_min, price_max = data_loader(file_path, 100, 10)
X_train = X_train.numpy()
X_eval = X_eval.numpy()
X_test = X_test.numpy()


# Define our hyperparamters
num_layers = 3
hidden_dim = 64
num_features = 20
seq_len = 100

training_iterations = 4000
batch_size = 64

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

train_history = {
    'iteration': [],
    'total_loss': [],
    'price_loss': [],
    'var_loss': [],
    'mid_loss': [],
    'spread_loss': []
}

print(mse_loss)

print("Start Embedding Network Training")


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
        price_loss  = mse_loss(X_mb_flat, X_tilde_flat)

        # Variance preservation - PER FEATURE TYPE
        var_price_real = tf.math.reduce_variance(X_mb_flat, axis=1)
        var_price_recon = tf.math.reduce_variance(X_tilde_flat, axis=1)
        var_price_loss = tf.reduce_mean(tf.abs(tf.sqrt(var_price_real + 1e-6) - tf.sqrt(var_price_recon + 1e-6)))


        # Midprice Loss
        mid_real = (X_mb[:, :, 0] + X_mb[:, :, 1]) / 2.0
        mid_recon = (X_tilde[:, :, 0] + X_tilde[:, :, 1]) / 2.0
        mid_loss = tf.reduce_mean(tf.square(mid_real - mid_recon))

        # 3. Spread loss
        spread_real = X_mb[:, :, 0] - X_mb[:, :, 1]
        spread_recon = X_tilde[:, :, 0] - X_tilde[:, :, 1]
        spread_loss = tf.reduce_mean(tf.square(spread_real - spread_recon))

        # Weighted combination of losses
        loss = 10*tf.sqrt(price_loss) + 25*var_price_loss + 50.0*mid_loss + 250*spread_loss

    # ------------------- Backpropagation -------------------
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    # ------------------- Recording Losses ------------------
    train_history['iteration'].append(itt)
    train_history['total_loss'].append(loss.numpy())
    train_history['price_loss'].append(price_loss.numpy())
    train_history['var_loss'].append(var_price_loss.numpy())
    train_history['mid_loss'].append(mid_loss.numpy())
    train_history['spread_loss'].append(spread_loss.numpy())



    if itt % 100 == 0:
        print(f"Step: {itt}, "
          f"price_loss: {price_loss.numpy():.6f}, var_price_loss: {var_price_loss.numpy():.6f}, "
          f"mid_loss: {mid_loss.numpy():.6f}, spread_loss: {spread_loss.numpy():.6f}")

model.save_weights('/content/drive/MyDrive/autoencoder.weights.h5')
print("Finish Embedding Network Training")


#---------------------------------------------------------------------------SUPERVISOR/GENERATOR/DISCRIMINATOR TRAINING -------------------------------------------------------------
num_layers = 3
hidden_dim = 64
num_features = 20
seq_len = 100

training_iterations = 501
batch_size = 128

# Model Definitions
autoencoder = Autoencoder(hidden_dim, num_layers, num_features)

# "Build" the model by calling it on dummy input
dummy_input = tf.zeros((1, seq_len, num_features))   # (batch, seq_len, features)
_ = autoencoder(dummy_input)


autoencoder.load_weights('/content/drive/MyDrive/autoencoder.weights.h5')

supervisor = Supervisor(hidden_dim, num_layers)
generator  = Generator(hidden_dim, num_layers)
discriminator = Discriminator(hidden_dim, num_layers)


# Dummy calls to empty models

dummy_z = tf.zeros((batch_size, seq_len, hidden_dim))
_ = generator(dummy_z)
_ = discriminator(dummy_z)
_ = supervisor(dummy_z)

generator.load_weights('/content/drive/MyDrive/generator.weights.h5')
supervisor.load_weights('/content/drive/MyDrive/supervisor.weights.h5')
discriminator.load_weights('/content/drive/MyDrive/discriminator.weights.h5')

# Optimisers (supervisor is optimised with the generator)
generator_optimiser     = tf.keras.optimizers.Adam(learning_rate=0.0001)
autoencoder_optimiser   = tf.keras.optimizers.Adam()
discriminator_optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Data Loading
file_path = '/content/drive/MyDrive/TimeGANWork/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
X_train, X_eval, X_test, price_min, price_max = data_loader(file_path, seq_len=seq_len, depth=10)
X_train = X_train.numpy()
X_eval = X_eval.numpy()
X_test = X_test.numpy()


timegan_history = {
    'iteration': [],
    'd_loss': [],
    'g_loss_u': [],
    'g_loss_u_e': [],
    'g_loss_s': [],
    'g_loss_v': [],
    'g_loss_total': [],
    'e_loss_t0': [],
    'e_loss_total': [],
    'mean_loss': [],
    'var_loss': [],
    'ssim': [],
    'kl_mid': [],
    'kl_returns': [],
    'kl_spread': []
}



"""
Supervised Loss Training

The original paper trains the supervisior and generator together initially,
purely so the generator is intialised with temporal knowledge before adversarial
training. We want the generator to produce sequences that follow the latent
dynamics learned by the supervisor. This will help stabilise training later.

"""
for itt in range(0):
  # Generate our mini batch, and convert to numpy array
  X_mb, _ = batch_generator(X_train, [seq_len]*len(X_train), batch_size)
  X_mb = np.array(X_mb, dtype=np.float32)

  with tf.GradientTape() as tape:
    # Embed real data sample
    H_real = autoencoder.embed(X_mb)

    # Send latent representation into supervisor to predict the next time
    # steps (apart from last).
    H_hat_supervise = supervisor(H_real, training=True)

    # Compute our supervised loss
    g_loss_s = tf.reduce_mean(tf.keras.losses.mse(H_real[:,1:,:], H_hat_supervise[:,:-1,:]))

  # Adjust the trainable parameters for both the generator and supervisor
  trainable_vars = generator.trainable_variables + supervisor.trainable_variables
  gradients = tape.gradient(g_loss_s, trainable_vars)

  generator_optimiser.apply_gradients(zip(gradients, trainable_vars))


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
    X_mb, _ = batch_generator(X_train, [seq_len]*len(X_train), batch_size)
    X_mb = tf.convert_to_tensor(np.array(X_mb, dtype=np.float32))

    # Random noise for our generator
    Z_mb = random_generator(batch_size, hidden_dim, [seq_len]*batch_size, seq_len)

    with tf.GradientTape() as tape:
      # Use generator to create synthetic embeddings from noise vector
      H_hat = generator(Z_mb, training=True)

      # Use the supervisor to predict time steps using synthetic data
      S_hat = supervisor(H_hat, training=True)

      # Use the recovery to convert synthetic data in the original feature space
      X_hat = autoencoder.recovery(S_hat)

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
      G_loss = G_loss_U + G_loss_U_e + 100 * tf.sqrt(supervised_generator_loss) + 200 * (mean_loss + var_loss)

    # Apply gradients to both generator and supervisor
    trainable_vars = generator.trainable_variables + supervisor.trainable_variables
    gradients = tape.gradient(G_loss, trainable_vars)
    generator_optimiser.apply_gradients(zip(gradients, trainable_vars))

    # Embedder and Recovery Training
    with tf.GradientTape() as tape_e:
      X_tilde = autoencoder(X_mb)
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

  X_mb, _ = batch_generator(X_train, [seq_len]*len(X_train), batch_size)
  X_mb = tf.convert_to_tensor(np.array(X_mb, dtype=np.float32))

  # Random noise for our generator
  Z_mb = random_generator(batch_size, hidden_dim, [seq_len]*batch_size, seq_len)

  with tf.GradientTape() as tape_d_test:
    # Fake Data
    H_hat = generator(Z_mb, training=False)
    S_hat = supervisor(H_hat, training=False)

    # Real Data
    H_real = autoencoder.embed(X_mb)

    # Discriminator Results
    Y_real = discriminator(H_real, training=False)
    Y_fake = discriminator(S_hat, training=False)

    # Calculate loss
    D_loss = bce(tf.ones_like(Y_real), Y_real) + bce(tf.zeros_like(Y_fake), Y_fake)

  if D_loss.numpy() > 0.15:
    with tf.GradientTape() as tape_d:
        H_hat = generator(Z_mb, training=True)
        S_hat = supervisor(H_hat, training=True)
        H_real = autoencoder.embed(X_mb)

        Y_real = discriminator(H_real, training=True)
        Y_fake = discriminator(S_hat, training=True)

        D_loss = bce(tf.ones_like(Y_real), Y_real) + bce(tf.zeros_like(Y_fake), Y_fake)

    gradients_d = tape_d.gradient(D_loss, discriminator.trainable_variables)
    discriminator_optimiser.apply_gradients(zip(gradients_d, discriminator.trainable_variables))





  if itt % 10 == 0:

      # Generate synthetic data for evaluation
      Z_eval = random_generator(batch_size, hidden_dim, [seq_len]*batch_size, seq_len)
      H_hat_eval = generator(Z_eval, training=False)
      S_hat_eval = supervisor(H_hat_eval, training=False)
      X_hat_eval = autoencoder.recovery(S_hat_eval)

      # Get real data batch for comparison
      X_real_eval, _ = batch_generator(X_train, [seq_len]*len(X_train), batch_size)
      X_real_eval = np.array(X_real_eval)



      # Compute SSIM and KL metrics
      # feature_ask=0 (Ask Price L1), feature_bid=10 (Bid Price L1)
      ssim_val, kl_mid_val, kl_returns_val, kl_spread_val = visualize_lob(
          X_real_eval,
          X_hat_eval.numpy(),
          feature_ask=0,
          feature_bid=1
      )


      timegan_history['iteration'].append(itt)
      timegan_history['d_loss'].append(D_loss.numpy())
      timegan_history['g_loss_u'].append(G_loss_U.numpy())
      timegan_history['g_loss_u_e'].append(G_loss_U_e.numpy())
      timegan_history['g_loss_s'].append(np.sqrt(supervised_generator_loss.numpy()))
      timegan_history['g_loss_v'].append((mean_loss + var_loss).numpy())
      timegan_history['g_loss_total'].append(G_loss.numpy())
      timegan_history['e_loss_t0'].append(np.sqrt(E_loss_T0.numpy()))
      timegan_history['e_loss_total'].append(E_loss.numpy())
      timegan_history['mean_loss'].append(mean_loss.numpy())
      timegan_history['var_loss'].append(var_loss.numpy())
      timegan_history['ssim'].append(ssim_val)
      timegan_history['kl_mid'].append(kl_mid_val)
      timegan_history['kl_returns'].append(kl_returns_val)
      timegan_history['kl_spread'].append(kl_spread_val)


      print(
          f"step: {itt}/{training_iterations}, "
          f"d_loss: {D_loss:.4f}, "
          f"g_loss_u: {G_loss_U:.4f}, "
          f"g_loss_s: {np.sqrt(supervised_generator_loss):.4f}, "
          f"g_loss_v: {(mean_loss + var_loss):.4f}, "
          f"e_loss_t0: {np.sqrt(E_loss_T0):.4f}"
      )

generator.save_weights('/content/drive/MyDrive/generator.weights.h5')
supervisor.save_weights('/content/drive/MyDrive/supervisor.weights.h5')
autoencoder.save_weights('/content/drive/MyDrive/autoencoder.weights.h5')
discriminator.save_weights('/content/drive/MyDrive/discriminator.weights.h5')


