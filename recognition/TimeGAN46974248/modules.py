"""
Contains the source code of the components of your model. Each component will be implemented as a class or a function.

"""

from dataset import data_loader

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam





def make_rnn(model, n_layers, hidden_units, output_units):
    for i in range(n_layers): 
        model.add(GRU(units=hidden_units,
                    return_sequences=True,
                    name=f'GRU_{i + 1}'))
    model.add(Dense(units=output_units,
                    activation = 'sigmoid',
                    name='OUT'))
    return model


class Supervisor(Model):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        model = Sequential(name='Supervisor')
        model.add(Input(shape=input_shape))
        model = make_rnn(model, 
                         n_layers=2, 
                         hidden_units = self.hidden_dim, 
                         output_units = self.hidden_dim)

        return model


class Generator(Model):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        model = Sequential(name='Generator')
        model.add(Input(shape=input_shape))
        model = make_rnn(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=self.hidden_dim)
        return model

class Discriminator(Model):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        model = Sequential(name='Discriminator')
        model = make_rnn(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=1)
        return model

class Recovery(Model):
    def __init__(self, hidden_dim, n_seq):
        self.hidden_dim=hidden_dim
        self.n_seq=n_seq
        return

    def build(self, input_shape):
        recovery = Sequential(name='Recovery')
        recovery.add(Input(shape=input_shape, name='EmbeddedData'))
        recovery = make_rnn(recovery,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.n_seq)
        return recovery

class Embedder(Model):

    def __init__(self, hidden_dim):
        self.hidden_dim=hidden_dim
        return

    def build(self, input_shape):
        embedder = Sequential(name='Embedder')
        embedder.add(Input(shape=input_shape, name='Data'))
        embedder = make_rnn(embedder,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.hidden_dim)
        return embedder