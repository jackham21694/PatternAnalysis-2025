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

def data_loader(path, num_features=20):
    # shape will be (entries) x (depth x 4)
    raw_data = np.loadtxt(path, delimiter=',')

    # We choose a sequence length of 50, leave out 35 timesteps at the end
    num_sequences = raw_data.shape[0] // 50
    total_timesteps = num_sequences * 50

    # Discard the 35 letfover timesteps (35 for our lobster depth 5 data)
    trimmed_data = raw_data[:total_timesteps]
    shaped_sample_data = trimmed_data.reshape((num_sequences, 50, num_features))

    # Shuffle and prepare train, eval, test datasets
    np.random.seed(46974248)
    np.random.shuffle(shaped_sample_data)

    # Splits dataset so first 70% is train, 10% is eval, and last 10% is test.
    train_end = int(0.7*num_sequences)
    val_end  = int(0.8*num_sequences)
    train = shaped_sample_data[:train_end]
    val  = shaped_sample_data[train_end:val_end]
    test  = shaped_sample_data[val_end:]

    # Indexes for prices and volumes
    ask_prices_idx  = [i*4 for i in range(5)]
    ask_volumes_idx = [i*4 + 1 for i in range(5)]
    bid_prices_idx  = [i*4 + 2 for i in range(5)]
    bid_volumes_idx = [i*4 + 3 for i in range(5)]

    # Group prices and volume indices
    price_idx = ask_prices_idx + bid_prices_idx
    volume_idx = ask_volumes_idx + bid_volumes_idx

    # Concatenate our train and validation sets (we use statistics for the
    # joint set)
    train_val = np.vstack([train.reshape(-1, num_features),
                           val.reshape(-1, num_features)])


    # Price (we use min-max scaling here)
    price_min = train_val[:, price_idx].min(axis=0, keepdims=True)
    price_max = train_val[:, price_idx].max(axis=0, keepdims=True)


    # Volume (small offset to avoid zero division)
    volume_mean = train_val[:, volume_idx].mean(axis=0, keepdims=True)
    volume_std  = train_val[:, volume_idx].std(axis=0, keepdims=True) + 1e-8

    # Applied z-score normalisation for volume, and min-max for prices
    def normalise(data):
        # Flatten sequences (get rid of) so we can deal with each timestep
        data_flat = data.reshape(-1, num_features)

        #Normalise prices, volumes seperately

        data_flat[:, price_idx] = 2.0 * ((data_flat[:, price_idx] - price_min) /
                                         (price_max - price_min + 1e-8)) - 1.0
        data_flat[:, volume_idx] = (data_flat[:, volume_idx] - volume_mean) / volume_std
        return data_flat.reshape(data.shape)

    train_norm = normalise(train)
    val_norm   = normalise(val)
    test_norm  = normalise(test)

    # We also return the statistics for reconstruction for our evaluation
    return (tf.convert_to_tensor(train_norm, dtype=tf.float32),
            tf.convert_to_tensor(val_norm, dtype=tf.float32),
            tf.convert_to_tensor(test_norm, dtype=tf.float32),
            price_min, price_max,
            volume_mean, volume_std)









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
    best_bid_orig = X_orig_flat[:,2] # Entire Third Column
    best_ask_recon = X_recon_flat[:,0] 
    best_bid_recon = X_recon_flat[:,2] 

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