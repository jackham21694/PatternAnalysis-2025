"""
Contains the source code of the components of your model. Each component will be implemented as a class or a function.

"""

import tensorflow as tf
from utils import rnn_cell





def embedder (X, T):
    # Groups all the GRU and Fully Connected Layer Weights and Biases with the prefix of 'embedder'
    with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):

        # Defining a GRU cell with hidden_dim amount of units, we have 3 layers of GRU, which is then combined into
        # one layer using MultiRNNCell. So the hidden vectors in the first layer will be used as input to the next layer
        # so we can better map the more complex features of the data. The way GRU works is that it will start with a vector
        # of zeroes of dimension [1 x hidden_dim], it will take the first input/timestep which is [1 x num_features] and
        # the GRU will use it gates and update this vector, and update the vector. This is repeated for all 50 sequences and
        # we will have [50x8]. This process is repeated two more times since we are asking for 3 layers.
        e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell('gru', hidden_dim) for _ in range(num_layers)])
        
        # Uses the multi-cell GRU we just created to output the hidden state/vector at each timestep (timestep = 1x8), the second output we do not care about
        # So our final output will be [50x8] each sequence represented by a vector of length 8 instead of 20.
        e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)

        # Takes each hidden vector and maps it to latent space. Note that the input and output dimension are actually the same, so 
        # we are not really reducing the input_dimensions at all. The GRU vectors only considered the previous timestep (along with the feedback loop)
        # and so the fully connected layer aims to combine all these features linearly, to give a more realistic latent space.
        H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
        return H

"""
Takes the latent representation and maps it back to time series format
"""
def recovery (H, T):      

    # Groups al the GRU and Fully Connceted Layer Weights and Biases with the prefix 'recovery'
    with tf.variable_scope("recovery", reuse = tf.AUTO_REUSE):       
      
      # Creating the exact same 3-layer GRU network as the embedder this time not changing dimensionality.
      r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell('gru', hidden_dim) for _ in range(num_layers)])

      # So our final output will still be [50x8]
      r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
      
      # This time we do our dimensionality reduction in the fully connected-layer where we go from 
      # [50x8] back to [50x20]
      X_tilde = tf.contrib.layers.fully_connected(r_outputs, num_features, activation_fn=tf.nn.sigmoid) 
    return X_tilde
