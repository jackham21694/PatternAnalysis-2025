"""
Contains the source code of the components of your model. Each component will be implemented as a class or a function.

"""

import tensorflow as tf
import numpy as np

def batch_generator(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)[:batch_size]
    X_mb = np.array([data[i] for i in idx], dtype=np.float32)
    T_mb = np.array([time[i] for i in idx], dtype=np.int32)
    return X_mb, T_mb

def random_generator (batch_size, z_dim, T_mb, max_seq_len):
  Z_mb = list()
  for i in range(batch_size):
    temp = np.zeros([max_seq_len, z_dim])
    temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    temp[:T_mb[i],:] = temp_Z
    Z_mb.append(temp_Z)
  return Z_mb


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
            tf.keras.layers.Dense(hidden_dim, activation=None)
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
            tf.keras.layers.Dense(num_features, activation=None)
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
