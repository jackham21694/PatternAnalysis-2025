"""
Module: dataset.py
Author: Jack Ham, 46974248

Description
-----------

This module handles the data loading and preprocessing for a LOBSTER
formatted dataset. It also include basic visualisation of common
financial indicators midprice, spread, and return.

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def data_loader(path, seq_len, depth):
    """
    This function takes a file path assuming to hold the traditional LOBSTER data format of
    [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ask_price_2, ask_size_2, bid_price_2, bid_size_2, ...]

    Given a sequence length, it will shape the data into [number_sequences, sequence_length, num_features]
    where it will best fit the given sequence length and discard the remaining timesteps. The number of 
    features depends on the given depth (e.g. depth of 5: num_features = 4 * 5).

    It splits the newly shaped data into training, validation and test sets, all normalised
    using min-max scaling.

    Args:
        path: filepath of csv file
        sequence_length: desired length of sequence
        depth: the depth of the lobster data
    
    Returns:
        A tensor for each training, validation, and testing datasets, as well as
        meaningful statistics so the normalisation can be reverted if required.

    """
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

    # Apply normalization
    train_norm = normalise(train, depth, price_max, price_min)
    val_norm   = normalise(val, depth, price_max, price_min)
    test_norm  = normalise(test, depth, price_max, price_min)

    # Return tensors and scaling stats for reconstruction
    return (
        tf.convert_to_tensor(train_norm, dtype=tf.float32),
        tf.convert_to_tensor(val_norm, dtype=tf.float32),
        tf.convert_to_tensor(test_norm, dtype=tf.float32),
        price_min, price_max
    )

def normalise(data, depth, price_max, price_min):
    """
    Normalises the given lobster data using min-max scaling, seperately for volume and price.

    Args:
        data: lobster data to be normalised
        depth: depth of lobster data being used
        price_max, price_min: maximum and minimum price of 'data'.

    Returns:
        Data scaled between 0 and 1.
    """
    data_flat = data.reshape(-1,  (depth*2))

    # Min-max scale prices to [0, 1]
    data_flat = (
        (data_flat - price_min) /
        (price_max - price_min + 1e-8)
    )
    return data_flat.reshape(data.shape)

def data_visualisation(X_orig, X_recon):
    """
    This function will plot the three financial indicators of midprice, spread, and return
    for the original and reconstructed LOBSTER datasets.

    Args:
        X_orig:  lobster data shaped as [no_sequences, sequence_length, num_features]
        X_recon: lobster data shaped as [no_sequences, sequence_length, num_features]
    
    Returns:
        midprice, spread, and return for both original and reconstructed datasets.
        
    """

    # Flatten/Reshape data to get rid of sequences
    X_orig_flat = tf.reshape(X_orig, (-1, X_orig.shape[-1]))
    X_recon_flat = tf.reshape(X_recon, (-1, X_recon.shape[-1]))

    # Obtain the best ask and bid prices 
    best_ask_orig = X_orig_flat[:,0] # Entire First Column
    best_bid_orig = X_orig_flat[:,1] # Entire Second Column
    best_ask_recon = X_recon_flat[:,0] 
    best_bid_recon = X_recon_flat[:,1] 

    # Calculat the midprice
    midprice_orig = (best_ask_orig + best_bid_orig) / 2.0
    midprice_recon = (best_ask_recon + best_bid_recon) / 2.0

    # Calculate the spread 
    spread_orig = (best_ask_orig- best_bid_orig)
    spread_recon = (best_ask_recon - best_bid_recon)


    # Calculate the return no value last timestamp, since it is geometric
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