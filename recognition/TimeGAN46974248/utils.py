"""
Module: utils.py
Author: Jack Ham, 46974248

Description
-----------

This module includes a batch generator that randomly pulls a batch from the data, and 
a random_generator which is used for the generator as random noise input.

"""

import numpy as np

def batch_generator(data, batch_size):
    """
    Creates a shuffled array of indices ranging from 0 to the length of the
    given data, and takes the first 'batch_size' samples, and returns the
    result as a numpy array.

    Args:
        data: dataset to pull from
        batch_size: number of samples to be pulled from dataset.
    
    Returns:
        A numpy array with 'batch_size' number of samples from the dataset.
    """
    no = len(data)
    idx = np.random.permutation(no)[:batch_size]
    X_mb = np.array([data[i] for i in idx], dtype=np.float32)
    return X_mb

def random_generator (batch_size, hidden_dim, T_mb, max_seq_len):
  """
  Creates a random noise vector of the shape [batch_size, seq_len, hidden_dim].
  Also provides support with max_seq_len if all sequences have a different
  number of values in a timestep.
  """
  Z_mb = list()
  for i in range(batch_size):
    temp = np.zeros([max_seq_len, hidden_dim])
    temp_Z = np.random.uniform(0., 1, [T_mb[i], hidden_dim])
    temp[:T_mb[i],:] = temp_Z
    Z_mb.append(temp_Z)
  return Z_mb



def denormalise(X_norm, price_min, price_max):
    # Flatten to 2D for easy scaling
    X_flat = X_norm.reshape(-1, X_norm.shape[-1])

    # Reverse min–max scaling for all features
    X_flat = X_flat * (price_max - price_min) + price_min

    # Reshape back to original
    return X_flat.reshape(X_norm.shape)