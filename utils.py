import numpy as np
import warnings
import os
import pickle
import itertools
import random

def drop_col(X, all_col, col_to_keep):
    """"Picks requested features in input data X. 

    Parameters:
        X (ndarray): Input data. 
        all_col (list): List of strings with column names is correct locations corresponding to X. 
        col_to_keep (list): List of strings with columns to keep in X. 
    Returns:
        ndarray: Subset of X with the features requested. 
        list: Columns of new input data, same as col_to_keep. 
        
    """
    if len(col_to_keep) == 0: raise ValueError("no columns requested")
    if X.shape[-1] != len(all_col): raise ValueError("Wrong dimensions in drop_col")
    ind = []
    for col in col_to_keep:
        ind.append(all_col.index(col))  
    X = X[..., ind]
    return X, col_to_keep

def get_optimizer_conf(optimizer):
    """"Reuturn dict of optimizer konfiguration which can be restord by calling keras.optimizers.get

    Parameters:
        optimizer (keras.optimizers): The keras optimizer object
    Returns:
        dict: Dict representing a Keras optimizer. 
    """
    conf = optimizer.get_config()
    name = conf.pop('name')
    get_input = {
        'class_name': name, 
        'config': conf, 
    }
    return get_input



def summarize_set(hists):
    """Prints summary of a training set. 
    """
    sorted_hists = sorted(hists, key=lambda k: k['training']['best_member']['final_val_metric'], reverse = True)
    n = min(len(hists), 3)
    print("Summary of training set: \n")
    print("Top " + str(n) + " configurations: ")
    metric = hists[0]['performance_measure_of_best_member']
    for i in range(n):
        print("Configuration " + str(i) + ":")
        
        print("Highest final", str(metric), sorted_hists[i]['training']['best_member']['final_val_metric'])
        print("Highest final val all training runs", str(metric), sorted_hists[i]['training']['final_val_metric'])
        print("Highest final training all training runs", str(metric), sorted_hists[i]['training']['final_training_metric'])
        
        print("Mean val accuracy:", sorted_hists[i]['training']['mean_val_acc'])
        print("Mean val ", str(metric), sorted_hists[i]['training']['mean_val_metric'])
        print("Mean val loss: ", sorted_hists[i]['training']['mean_val_loss'])
        
        
        print("Mean training accuracy: ", sorted_hists[i]['training']['mean_training_acc'])
        print("Mean training", str(metric), sorted_hists[i]['training']['mean_training_metric'])
        print("Mean training loss: ", sorted_hists[i]['training']['mean_training_loss'])
        
        print("Mean training time: ", sorted_hists[i]['training']['mean_traing_time'])
        print("Medain trained epochs: ", sorted_hists[i]['training']['median_epochs'])
        print("\n")
        
    print("Best configuration: ")
    print_set_configuration(sorted_hists[0]['configuration'])
    return sorted_hists, sorted_hists[0]['training']['best_member']['final_val_metric']


def save_session(config, dataset, training_hist, test_hist, save_file, filename = None):
    """"Save stats from training/test, model config.  

    Parameters:
        config (dict): Model configuration. 
        dataset (Dataset): Dataset object.  
        training_hist (dict): Batch size. 
        save_file (Bolean): True if want to save file. 
        y_train (ndarray): Train output. 
        X_val (ndarray): Validation input. 
        y_val (ndarray): Validaion output. 
        callbacks (list): List of keras.callbacks objects. 
        filename (str): File name. 
    Returns:
        dict: Training history.  
        
    """
    result = {
            'session_config': config, 
            'dataset': dataset.as_dict(), 
            'test_hist': test_hist, 
            'training_hist': training_hist, 
    }
    if save_file:
        folder = './training_out'
        if not os.path.exists(folder):
          os.makedirs(folder)
        if filename is None: filename = dataset.version + '_' + config['network_type']
        path = folder + '/' + filename + '_' +str(dataset.num_timesteps) + '.p'

        if filename is None:
            raise ValueError('Provide file name to save session')
        elif os.path.isfile(path):
            warnings.warn('Fileame already exist, did not save new file')

        else: 
            print("Print output to file " + path)
            with open(path, 'wb') as fp:
                pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    return result

def gather_training_measures(model_config, optimizer_config, epochs, batch_size, history, dataset, network_type):
        """Gather stats from training
        """
        final_val_loss = [ member['val_loss'][-1] for member in history ] 
        final_val_acc = [ member['val_accuracy'][-1] for member in history ] 
        final_training_loss = [ member['loss'][-1] for member in history ] 
        final_training_acc = [ member['accuracy'][-1] for member in history ] 
        
        mean_train_time = np.mean([ member['training_time'] for member in history ])
        trained_epochs = [ len(member['accuracy']) for member in history ] 
        result = {
            'training': {
                'mean_traing_time': mean_train_time, 
                'mean_val_acc': np.mean(final_val_acc), 
                'mean_val_loss': np.mean(final_val_loss), 
                'mean_training_acc': np.mean(final_training_acc), 
                'mean_training_loss': np.mean(final_training_loss), 
                'median_epochs': np.median(trained_epochs),
                'training_history': history, 

            },
            'configuration': {
                'model': model_config, 
                'optimizer': optimizer_config, 
                'epochs_max': epochs, 
                'epochs_per': trained_epochs, 
                'batch_size': batch_size, 
                'dataset_name': dataset.name, 
                'dataset_version': dataset.version, 
                'num_timesteps': dataset.num_timesteps, 
                'network_type': network_type, 
                
            }, 
        }

        perf_metric = dataset.main_metric
    
        pref_metric_final  = [ member['val_' + perf_metric][-1] for member in history ] 
        best_member = np.argmax(pref_metric_final) # highest acc or pr-auc at last step
        best_member_stats = {
            'index': best_member, 
            'final_val_metric': history[best_member]['val_' + perf_metric][-1],
            'final_val_loss': history[best_member]['val_loss'][-1],
            'epochs': len(history[best_member]['val_accuracy']),
        }
        final_val_perf_metric = [ member['val_' + perf_metric][-1] for member in history ]
        final_training_perf_metric = [ member[perf_metric][-1] for member in history ]
        
        result['training']['final_val_metric'] = final_val_perf_metric
        result['training']['final_training_metric'] = final_training_perf_metric
        
        result['training']['mean_val_metric'] = np.mean(final_val_perf_metric)
        result['training']['mean_training_metric'] = np.mean(final_training_perf_metric)

        result['performance_measure_of_best_member'] = perf_metric 
        result['training']['best_member'] = best_member_stats 
        return result



def print_set_configuration(set_configuration):
    """"Prints the configureation of a training session: Model, optimizer, batch size and epochs. 

    Parameters:
        set_configuration (dist): Configuration of a training session. 
    """
    optimizer = set_configuration['optimizer']
    print("Optimizer", optimizer['class_name'], optimizer['config'])
    print("Batch size", set_configuration['batch_size'])
    print("Epochs max", set_configuration['epochs_max'])
    print("Epochs per training", set_configuration['epochs_per'])
    print("Model Configuration: ")

    conf_to_print = {
        'Dense': ['units', 'activation'], 
        'Flatten': [], 
        'Reshape': ['target_shape'], 
        'Conv1D': ['filters', 'kernel_size', 'strides'],
        'Conv2D': ['filters', 'kernel_size', 'strides'], 
        'LSTM': ['units'], 
        
    }
    
    layers = set_configuration['model']['layers']
    for i, layer in enumerate(layers):
        layer_name = layer['class_name']
        layer_arg = [layer['config'][key] for key in conf_to_print[layer_name]]
        print("Layer " + str(i+1))
        print("\t", layer_name, layer_arg)



def pick_settings(variables, n):
    """"Picks combinations of training settings. 

    Parameters:
        variables (list): List of list. Inner list correpond to each variable. 
        n (int): Number of combinations to pick. 
    Returns:
        list: List of list. Lenght of outer list is n. 
                Each inner list is a integer of length 4, 
                corresponding to the index of each feature. 
    """
    if len(variables) != 4: raise ValueError("pick_settings not called correctly, expected 4 inner lists in input 'variables'")
    # calc possible variations in each list in varaibles
    max_index = [len(val) for val in variables]
    ranges = [list(range(ran)) for ran in max_index]
    # calc all possible combinations
    result = list(itertools.product(ranges[0], ranges[1], ranges[2], ranges[3]))
    num = min(n, len(result))
    if num > 20: warnings.warn('Large number ofcombinations requested, ' + str(num) + " reqested")
    # sample combinations
    sampled = random.sample(result, num)
    out = [list(samp) for samp in sampled]
    return out

def print_retrain_summary(training_hist, test_hist, dataset_col, main_metric, goal_metric):
    """"Print retrain summary 

    Parameters:
        training_hist (dict): 
        n (int): Number of combinations to pick.
    """
    print('Training time: ' , training_hist['training_time'], 'Predict time: ' , test_hist['test_time'])
    print('Test', main_metric, test_hist[main_metric])
    print("Goal metric:", goal_metric)
    print('Final val ', main_metric, training_hist['val_' + main_metric][-1])
    print('Features:', dataset_col)