"""
Module: modules.py
Author: Jack Ham, 46974248

Description
-----------

This modules contains the classes for our TimeGAN model, including
the autoencoder, supervisor, generator, and discriminator. They are all
built similarly with minor differences.

"""

import tensorflow as tf

class Autoencoder(tf.keras.Model):
    """
    Autoencoder class that has an embedder and recovery component that can be called
    individually. The embedder stacks GRU recurrent networks to map the original data to a
    latent space with dimension 'hidden_dim'. A fully connected layer with a sigmoid activation
    function is also included as per the original TimeGAN paper.
    """
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





class Supervisor(tf.keras.Model):
  """
  Supervisor class that stacks GRU recurrent networks along with a fully connected layer.
  Note that the number of layers is one less as per the original paper.
  """
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
    """
    Generator class that stacks GRU recurrent networks along with a fully connected layer.
    """
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
    """
    Discriminator class that stacks GRU recurrent networks along with a fully connected layer.
    """
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
