import numpy as np
import tensorflow as tf
import time
import warnings
import pandas as pd
import utils
import sklearn
from tensorflow import keras
from tensorflow.keras import Sequential
from plotting import *
from default_settings import *
from datasets import *

VERBOSE = 1



def get_models_from_layers(dataset, model_layers, network_type):
    """Return the configuration of a keras.Sequential model. 

    Parameters:
        dataset (Dataset): Dataset object represnting the data to train the network on. 
        model_layers (list): List of keras.layers objects.  
        network_type (str): Network name.  
    Returns:
        dict: The model configuration. 
    """
    model = keras.Sequential()
    model.add(Reshape(get_input_shape(dataset.input_shape, network_type), input_shape = dataset.input_shape))
    for layer in model_layers:
        model.add(layer)
    model.add(Dense(dataset.output_shape, activation = dataset.output_activation))
    return model.get_config()


def get_models(dataset, network_type):
    """Return list of the configurations (dicts) of a keras.Sequential() model. 

    Parameters:
        dataset (Dataset): Dataset object represnting the data to train the network on. 
        network_type (str): Network name.  
    Returns:
        dict: The model configuration. 
    """
    if network_type == 'log_reg':
        return [logreg_model(dataset)]
    
    deafult_layers = get_network_func(network_type)()
    result = []
    for model_layers in deafult_layers:
        result.append(get_models_from_layers(dataset, model_layers, network_type))
    return result

def logreg_model(dataset):
    """Return the configuration of a keras.Sequential() logistic regression model. 

    Parameters:
        dataset (Dataset): Dataset object represnting the data to train the network on. 
    Returns:
        dict: Dict with the TP, FP, FN, TP and the threshold. 
    """
    model = keras.Sequential()
    model.add(Reshape(get_input_shape(dataset.input_shape, 'log_reg'), input_shape = dataset.input_shape, ))
    model.add(Dense(dataset.output_shape, activation = dataset.output_activation))
    return model.get_config()

def restore_model_from_configuration(config):
    """"Takes a keras.Sequential model configuration as imput and return the corresponing Sequential object. 

    Parameters:
        config (dict): The model configuration. 
    Returns:
        keras.Sequential: The Sequential model. 
    """
    return tf.keras.Sequential.from_config(config)


def restore_set_configuration(configuration):
    """"Restore untrained model, optimizer, epochs and batch size from confuíguration

    Parameters:
        configuration (dict): Configuration of training session. 
    Returns:
        keras.Sequential: Untrained model. 
        keras.optimzer: Optimizer.
        int: Epochs
        int: Batch size

    """
    model =  restore_model_from_configuration(configuration['model'])
    optimizer = keras.optimizers.get(configuration['optimizer'])
    return model, optimizer, configuration['epochs_max'], configuration['batch_size']


def get_network_func(network_type):
    """Return the function name to cal to get the default layers. 
    """
    mapping = {
        'mlp': mlp_default, 
        'conv1': conv1_default,
        'conv2': conv2_default, 
        'lstm': lstm_default, 
        'conv_lstm': conv_lstm_default, 
    }
    return mapping[network_type]


def retrain_from_conf(config, dataset, X_train, y_train, X_val, y_val, goal_metric):
    """"Retrains a model from the model configuration

    Parameters:
        config (dict): Model configuration. 
        dataset (Dataset): Dataset object.
        X_train (ndarray): Input training data.
        y_train (ndarray): Output training data.
        X_val (ndarray): Input validation data.
        X_val (ndarray): Output validation data.
        goal_metric (dict): Dict with a goal value for the validation metric and metric to evalaute. 
    Returns:
        Sequential: Train model.
        dict: Training history. 
    """
    metric = 'val_' + dataset.main_metric
    callbacks = [keras.callbacks.EarlyStopping(
        monitor = metric, 
        verbose = VERBOSE,
        patience = 5,
        mode = 'max',
        restore_best_weights=True)]
    final_metric = -1
    i = 0 # number of training attemmpts performed
    print("Goal metric", goal_metric)
    while final_metric < goal_metric:
        print("Retrain ", str(i))
        model, optimizer, epochs, batch_size = restore_set_configuration(config)
        model.compile(optimizer=optimizer, loss=dataset.loss, metrics=dataset.metrics)
        t0 = time.time()
        training_hist = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = epochs, batch_size=batch_size, callbacks=callbacks)
        t1 = time.time()
        his = training_hist.history
        final_metric = max(his[metric])
        i += 1
        if i >= 10:
            print("Trained 10 times, not finding good fit in retrain")
            break
        if final_metric > 0.95:
            print("Val score over 0.95, keeping model")
            break
    print("Metrics names:", model.metrics_names)
    print("Final metric retrain: ", str(final_metric))
    his['training_time'] = t1-t0
    return model, his

def train_set(dataset, model_config, optimizer, epochs, batch_size, X_train, y_train, X_val, y_val, network_type, num_train_per_comb):
    """"Trains a set up NUM_TRAIN_PER_COMB number of times
    Parameters:
        dataset (Dataset): Dataset
        model_config (dict): Model configuration
        optimizer (keras.optimizer): Optimizer
        epochs (int): Maximum number of epochs. 
        batch_size (int): Batch size. 
        X_train (ndarray): Train input. 
        y_train (ndarray): Train output. 
        X_val (ndarray): Validation input. 
        y_val (ndarray): Validaion output. 
        network_type (str): Network type 
    Returns:
        dict: Summary of training of set. 
    """
    history = []
    callbacks = [keras.callbacks.EarlyStopping(
        monitor='val_' + dataset.main_metric, 
        verbose=VERBOSE,
        patience=5,
        mode='max',
        restore_best_weights=False)]
    for i in range(num_train_per_comb):
        print("\n")
        print("Training run " + str(i+1) + " out of " + str(num_train_per_comb) + " for set")
        keras_model = restore_model_from_configuration(model_config)
        keras_model.compile(optimizer = optimizer, loss = dataset.loss, metrics = dataset.metrics)
        history.append(train_model(keras_model, epochs, batch_size, X_train, y_train, X_val, y_val, callbacks))
    summary = gather_training_measures(model_config, get_optimizer_conf(optimizer), epochs, batch_size, history, dataset, network_type)
    return summary



def evaluate_model(model, X_test, y_test, dataset_name):
    """"Evalaute trained Sequential model on test dataset. 

    Parameters:
        model (keras.Sequential): Trained model. 
        X_test (ndarray): Input data. 
        y_test (ndarray): Output data. 
        dataset_name (str): Name of dataset. 
    Returns:
        dict: Stats from evaluation on test dataset. 
        
    """
    test_metrices = model.evaluate(X_test, y_test, verbose=0)

    print("test_metrices:",  test_metrices)
    t0 = time.time()
    predicted = model.predict(X_test)
    t1 = time.time()
    out = {
        'test_time': t1-t0
    }
    
    if dataset_name == 'fotball': # if binary classification
        cm = plot_cm(y_test, predicted)
        fpr, tpr = plot_roc("Test", y_test, predicted)
        precission, recall = plot_prc("Test", y_test, predicted)
        out['cm'] = cm
        out['roc-curve'] = {
            'fpr': fpr, 
            'tpr': tpr, 
        }
        out['pr-curve'] = {
            'precission' : precission, 
            'recall': recall, 
        }
        
    
    metrics = model.metrics_names
    
    for metric, value in zip(metrics, test_metrices):
        out[metric] = value

    return out


def train_model(model, epochs, batch_size, X_train, y_train, X_val, y_val, callbacks):
    """"Train a Keras Sequential model. 

    Parameters:
        model (keras.Seuqential): Model to train. 
        epochs (int): Maximum number of epochs. 
        batch_size (int): Batch size. 
        X_train (ndarray): Train input. 
        y_train (ndarray): Train output. 
        X_val (ndarray): Validation input. 
        y_val (ndarray): Validaion output. 
        callbacks (list): List of keras.callbacks objects. 
    Returns:
        dict: Training history.  
        
    """
    t0 = time.time()
    history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=VERBOSE)
    t1 = time.time()
    his = history.history
    his['training_time'] = t1-t0
    return his

def run_training(dataset, network_type, n_combinations, X_train, y_train, X_val, y_val, num_train_per_comb):
    """"Run a training session. Trains n_combinations configurations networks num_train_per_comb. 

    Parameters:
        dataset (Dataset): Dataset object.
        network_type (str): Detrmins which network type to run. 
        n_combinations (int): How many random configurations to run in the session. 
        X_train (ndarray): Train input. 
        y_train (ndarray): Train output. 
        X_val (ndarray): Validation input. 
        y_val (ndarray): Validaion output. 
        num_train_per_comb (int): How many times to run each configuration. 
    Returns:
        dict: Training history.  
        
    """
    models = get_models(dataset, network_type)
    optimizers = get_optimizers()
    epochs = get_epochs()
    batch_sizes = get_batch_size()
    combinations = pick_settings([models, optimizers, epochs, batch_sizes], n_combinations)
    hists = []
    print("Training " +  str(len(combinations)) +" combinations")
    for i, setting in enumerate(combinations):
        print("\n")
        print("Training combination " + str(i+1))
        model = models[setting[0]]
        optimizer = optimizers[setting[1]] 
        epoch = epochs[setting[2]]
        batch_size = batch_sizes[setting[3]]
        print("Epochs", epoch, "batch size: ", batch_size, ", \noptimizer", optimizer.get_config(), ", \nnetwork type: ", network_type)
        hists.append(train_set(dataset, model, optimizer, epoch, batch_size, X_train, y_train, X_val, y_val, network_type, num_train_per_comb))
    return hists


def run_retraining(dataset, network_type, n_combinations, X_train, y_train, X_val, y_val, num_train_per_comb):
    """"Run a training session. Trains n_combinations configurations networks num_train_per_comb. 

    Parameters:
        dataset (Dataset): Dataset object.
        network_type (str): Detrmins which network type to run. 
        n_combinations (int): How many random configurations to run in the session. 
        X_train (ndarray): Train input. 
        y_train (ndarray): Train output. 
        X_val (ndarray): Validation input. 
        y_val (ndarray): Validaion output. 
        num_train_per_comb (int): How many times to run each configuration. 
    Returns:
        dict: Training history.  
        
    """
    models = get_models(dataset, network_type)
    optimizers = get_optimizers()
    epochs = get_epochs()
    batch_sizes = get_batch_size()
    combinations = pick_settings([models, optimizers, epochs, batch_sizes], n_combinations)
    hists = []
    print("Training " +  str(len(combinations)) +" combinations")
    for i, setting in enumerate(combinations):
        print("Training combination " + str(i+1))
        model = models[setting[0]]
        optimizer = optimizers[setting[1]] 
        epoch = epochs[setting[2]]
        batch_size = batch_sizes[setting[3]]
        print("Epochs", epoch, "batch size: ", batch_size, ", \noptimizer", optimizer.get_config(), ", \nnetwork type: ", network_type)
        hists.append(train_set(dataset, model, optimizer, epoch, batch_size, X_train, y_train, X_val, y_val, network_type, num_train_per_comb))
    return hists
