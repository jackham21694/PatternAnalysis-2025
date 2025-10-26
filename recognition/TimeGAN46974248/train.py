"""
Contains the source code for training, validating, testing, and saving your model. The model is
imported from 'modules.py' and the data loader imported from 'dataset.py'. Make sure to plot the
losses and metrics during training. 

"""
from utils import batch_generator
from dataset import data_loader
from modules import embedder, recovery
import tensorflow as tf
import numpy as np

data = data_loader("C:/Users/Jack Ham/OneDrive/2025 - Sem 2/COMP3710/lobsterDepth5/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv")

# Training autoencoder
num_layers = 3 # Almost every paper
hidden_dim = 8 # Since we only have 20 data features
num_features = 20 # See dataset.py
seq_len = 50 # See dataset.py

training_iterations = 50000
batch_size = 124

# These placeholders essentially help tensorflow structure the computational graph, and knows what
# input to expect
# The NONE's are the batch size that will be inputted layer during the training processes
X = tf.placeholder(tf.float32, [None, seq_len, num_features], name="RealData")
T = tf.placeholder(tf.int32, [None], name = "myinput_t")



H = embedder(X)
X_tilde = recovery(H)

# Embedder and Recovery Variables that we saved earlier
e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]


# Autoencoder Network Loss, note first we are not considering supervised loss
# Original paper makes this additional numerical adjustment
E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
E_loss = 10*tf.sqrt(E_loss_T0)


# Defining our optimiser
E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)

## TimeGAN training   
sess = tf.Session()
sess.run(tf.global_variables_initializer())
    
# 1. Embedding network training
print('Start Embedding Network Training')
    
for itt in range(training_iterations):
    # All of our sequences are of equal length, so the second variable we just parse a list of [50, 50, 50, ...]
    # Also note they way we are batching here means we will get duplicate values and not see everything probably
    # depending on the number of iterations, which kind of represents epochs but more random.
    X_mb, T_mb = batch_generator(data, [seq_len]*1000, batch_size)           
    
    # Singular training step, using the optimiser and loss function with our mini-batch, and we get our step loss       
    _, step_e_loss = sess.run([E_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})        
    # Check loss every thousand interations
    if itt % 1000 == 0:
        print('step: '+ str(itt) + '/' + str(training_iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) ) 

print('Finish Embedding Network Training')








# Train the Generator and Discriminator
#
# Generator     - Produce Latent Sequences from Random Noise
#               - Adversarial Loss (for now)
#
# Discriminator - Classify real and fake latent sequences
#               - Cross-Entropy Loss between real/fake labels

