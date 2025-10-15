"""
Contains the data loader for loading and preprocessing your data

"""



"""
Information regarding the LOBSTER dataset. The structure of the LOBSTER dataset is as follows:

A given sample or time step is of the structure [Ask Price 1, Ask Size 1, Bid Price 1, Bid Size 1, ...]
and the length of one sample is depth x 4.

The stucture of our data, expected by the TimeGAN is [num_sequences, sequence_length, num_features]
where the whole trading day is split into sequeunces of a certain length.

"""





import numpy as np
import tensorflow as tf

def data_loader(path):
    raw_data = np.loadtxt(path, delimiter=',') # shape will be (entries) x (depth x 4)

    # Initially we only want to consider a small amount of the dataset, Here we are looking for 
    # 1000 sequences of length 50
    sample_data = raw_data[:50000]
    shaped_sample_data = sample_data.reshape((1000, 50, 20))
    normalised = normalise_min_max(shaped_sample_data)

    # Convert to 3D tensor of the form [num_sequences, sequence_length, num_features]
    return tf.convert_to_tensor(normalised, dtype=tf.float32)


def normalise_min_max(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


data = data_loader("C:/Users/Jack Ham/OneDrive/2025 - Sem 2/COMP3710/lobsterDepth5/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv")
