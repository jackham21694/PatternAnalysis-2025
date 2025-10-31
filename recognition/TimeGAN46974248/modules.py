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


# Usually has one less layer then all the other components to keep it simple
class Supervisor(tf.keras.Model):
  def __init__(self, hidden_dim, num_layers, num_features):
        super(Supervisor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Building GRU Layers
        self.supervisor_grus = []
        for i in range(num_layers-1):
            self.supervisor_grus.append(
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