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
import matplotlib.pyplot as plt

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

"""
LOBSTER Data has three key indicators, being midprice, spread, and return.

Here we will visualise 

"""
def data_visualisation(data):
    
    # Our dataset has 1000 sequences with 50x20, meaning we have 50,000 timestamps in total
    # First we reshape data so that it is 50,0000 x 20
    flat = tf.reshape(data, (50000, 20))
    
    # Calculate the mid price for these samples
    best_ask = flat[:,0] # Entire First Column
    best_bid = flat[:,2] # Entire Third Column
    midprices = (best_ask + best_bid) / 2.0
    
    # Calculate the spread for these samples
    spreads = (best_ask- best_bid)

    # Calculate the return (MidPrice Returns), no value last timestamp, since it is geometric
    returns = midprices[1:] - midprices[:-1]
    
    plt.figure(figsize=(16,4))
    plt.plot(midprices.numpy(), color='blue')
    plt.title("Midprice")
    plt.xlabel("Time Steps")
    plt.ylabel("Midprice")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(16,4))
    plt.plot(spreads, color='green', linewidth=0.8)
    plt.title("Spread")
    plt.xlabel("Time Steps")
    plt.ylabel("Spread")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(16,4))
    plt.plot(returns.numpy(), color='red', marker='o', markersize=2)
    plt.title("Return")
    plt.xlabel("Time Steps")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()
    return midprices, spreads, returns