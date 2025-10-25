"""
Contains the source code for training, validating, testing, and saving your model. The model is
imported from 'modules.py' and the data loader imported from 'dataset.py'. Make sure to plot the
losses and metrics during training. 

"""

from dataset import data_loader
from modules import embedder
import tensorflow as tf

data = data_loader("C:/Users/Jack Ham/OneDrive/2025 - Sem 2/COMP3710/lobsterDepth5/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv")



# Training Embedding and Recovery Networks
#
# Loss: Reconstruction Loss

# The original paper uses hidden layer of 24 inside github walkthrough, so we will start with this value

# Training autoencoder
H = embedder(X)
X_tilde = recovery(H)

autoencoder = Model(inputs=X,
                    outputs=X_tilde,
                    name='Autoencoder')

autoencoder.summary() # Total params for our setup is 22, 412


# Defining Losses for Autoencoder (just reconstruction_loss)
recovery_loss = tf.losses.mean_squared_error(X, X_tilde)
recovery_loss_adjusted = 10*tf.sqrt(recovery_loss) # original paper made this numerical adjustment

# Defining our optimiser

autoencoder_optimiser = tf.train.AdamOptimizer().minimize(recovery_loss_adjusted, var_list = embedder.trainable_variables + recovery.trainable_variables)







# Train the Generator and Discriminator
#
# Generator     - Produce Latent Sequences from Random Noise
#               - Adversarial Loss (for now)
#
# Discriminator - Classify real and fake latent sequences
#               - Cross-Entropy Loss between real/fake labels

