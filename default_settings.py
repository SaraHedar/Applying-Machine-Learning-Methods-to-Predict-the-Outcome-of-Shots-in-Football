import numpy as np
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, LSTM, Flatten, Reshape
from tensorflow import keras

def lstm_default():
    """Default layers for LSTM network. 
    """
    net1 = [
        LSTM(10, return_sequences = True), 
        LSTM(12), 
        Flatten(), 
    ]
    net2 = [
        LSTM(10, return_sequences = True), 
        Flatten(), 
        Dense(12, activation = 'relu'), 
    ]
    return [net1, net2]

def conv_lstm_default():
    """Default layers for combined conv and LSTM network. 
    """
    net1 = [
        Conv1D(10, 3, activation='relu'), 
        Conv1D(12, 5, activation='relu'), 
        Flatten(), 
    ]
    net2 = [
        Conv1D(10, 3, activation='relu'), 
        Flatten(), 
    ]
    return [net1, net2]

def conv1_default():
    """Default layers for conv 1D network. 
    """
    net1 = [
        Conv1D(10, 3, activation='relu'), 
        Conv1D(12, 5, activation='relu'), 
        Flatten(), 
    ]
    net2 = [
        Conv1D(10, 3, activation='relu'), 
        Flatten(), 
    ]
    return [net1, net2]

def conv2_default():
    """Default layers for conv 2D network. 
    """
    net1 = [
        Conv2D(10, (2, 1), activation='relu'), 
        Conv2D(12, (3, 1), activation='relu'), 
        Flatten(), 
    ]
    net2 = [
        Conv2D(10, (2, 1), activation='relu'), 
        Flatten(), 
    ]
    return [net1, net2]

def mlp_default():
    """Default layers for MLP network. 
    """
    net1 = [
        Dense(100, activation = 'relu'),
        Dense(200, activation = 'relu'),  
        Flatten(), 
    ]
    net2 = [
        Dense(10, activation='relu'), 
        Dense(20, activation = 'relu'), 
        Dense(10, activation = 'relu'), 
    ]
    return [net1, net2]


def get_epochs():
    """Return default number of epochs. 
    """ 
    return [200]

def get_batch_size():
    """Return default batch sizes. 
    """ 
    return [2048, 4096]


def get_optimizers():
    """Return default optimizers. 
    """ 
    learning_rates = np.logspace(-1, -5, 10)
    opts = []
    for learning_rate in learning_rates: 
        opts.append(keras.optimizers.Adam(learning_rate = learning_rate))
    return opts
