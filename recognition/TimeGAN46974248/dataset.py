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

def data_loader(path, depth=5, num_features=20):
    # shape will be (entries) x (depth x 4)
    raw_data = np.loadtxt(path, delimiter=',') 

    # We choose a sequence length of 50, leave out 35 timesteps at the end
    num_sequences = raw_data.shape[0] // 50 
    total_timesteps = num_sequences * 50

    # Discard the 35 letfover timesteps (35 for our lobster depth 5 data)
    trimmed_data = raw_data[:total_timesteps]
    
    # Custom Normalisation Process (MidPrice Averaging and Volume Z-Score)

    # Need to sort data for each timestep into the four categories
    ask_prices_idx = [i*4 for i in range(depth)]
    ask_volumes_idx = [i*4 + 1 for i in range(depth)]
    bid_prices_idx = [i*4 + 2 for i in range(depth)]
    bid_volumes_idx = [i*4 + 3 for i in range(depth)]

    # Midprice for each time step
    midprice = (trimmed_data[:, ask_prices_idx[0]] + trimmed_data[:, bid_prices_idx[0]]) / 2

    # Normalising all the prices for each time step using the respective midprice
    for idx in ask_prices_idx + bid_prices_idx:
        trimmed_data[:, idx] = (trimmed_data[:, idx] - midprice) / (midprice + 1e-8)

    # Normalising the volumes (small addition in case sigma is 0)
    for idx in ask_volumes_idx + bid_volumes_idx:
            mu = trimmed_data[:, idx].mean()
            sigma = trimmed_data[:, idx].std()
            trimmed_data[:, idx] = (trimmed_data[:, idx] - mu) / (sigma + 1e-8)


    shaped_sample_data = trimmed_data.reshape((num_sequences, 50, num_features))

    # Setting random seed and shuffling data
    np.random.seed(46974248)
    np.random.shuffle(shaped_sample_data)

    # Splits dataset so first 70% is train, 10% is eval, and last 10% is test.
    train_end = int(0.7 * num_sequences)
    eval_end  = int(0.8 * num_sequences)
    train = shaped_sample_data[:train_end]
    eval  = shaped_sample_data[train_end:eval_end]
    test  = shaped_sample_data[eval_end:]



    # Convert to 3D tensor of the form [num_sequences, sequence_length, num_features]
    return (tf.convert_to_tensor(train, dtype=tf.float32),
            tf.convert_to_tensor(eval, dtype=tf.float32),
            tf.convert_to_tensor(test, dtype=tf.float32))









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