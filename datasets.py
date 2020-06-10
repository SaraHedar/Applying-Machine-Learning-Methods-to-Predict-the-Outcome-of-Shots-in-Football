import numpy as np
import copy
from tensorflow import keras
from imblearn.over_sampling import SMOTE 
from sklearn import model_selection
from utils import *
from sklearn.utils import shuffle


class Dataset:
    '''
    Class representing a Dataset

    Args:
        name (str): Name of Dataset
        input_shape (tuple): Input shape if a sample, tuple of integers. len(input_shape) == 1 if num_timesteps == 1, 2 otherwise. 
        col (list): Ordered list of string names of the columns at teach time step. len(col) should be same as input_shape[-1].
        resample (Boolean): True of Dataset was resampled.
        num_timesteps (int): Number of timesteps in dataset. 
        version (str): String which helps pick the features. 
    '''
    def __init__(self, name, input_shape, col, resample, num_timesteps, version = None):
        self.name = name
        self.input_shape = input_shape
        self.col = col
        self.resample = resample
        self.num_timesteps = num_timesteps
        self.version = version
        if name == 'mnist':
            self.output_shape = 10
            self.output_activation = 'softmax'
            self.loss = 'categorical_crossentropy'
            self.metrics = ['accuracy']
            self.main_metric = 'accuracy'

        elif name == 'fotball': 
            if version is None: raise ValueError("version = None note allowed for fotball dataset")
            self.output_shape = 1
            self.output_activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
            self.metrics = [keras.metrics.BinaryAccuracy(name='accuracy'),
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.AUC(name='pr-auc', curve='PR'), 
                            keras.metrics.AUC(name='roc-auc')]
            self.main_metric = 'pr-auc'
        
        # add addition additional elif if you want to use your own datsset
                
        else: raise ValueError("Not suported dataset")

    def __repr__(self):
        return "Dataset: " + self.name + " col: " + str(self.col) + "\n" + "input_shape: " + str(self.input_shape) + ", output_shape: " + str(self.output_shape) + ", " + str(self.output_activation) + ", " + str(self.loss)  + ", " + str(self.get_metrics()) + ', resmaple:' + str(self.resample) + ", number of time steps: "+ str(self.num_timesteps) + ",  version: "+ str(self.version)
    
    """
    Return list of string names of the metrics of the network
    """
    def get_metrics(self):
        if self.name == 'fotball':
            return [metric.get_config()['name'] for metric in self.metrics]
        else: return self.metrics

    """
    Return dict representation of the Dataset. 
    """    
    def as_dict(self):
        met = self.get_metrics()
        result = {
            'name': self.name, 
            'input_shape': self.input_shape, 
            'col': self.col, 
            'metrics': met.copy(), 
            'num_timesteps': self.num_timesteps, 
            'version': self.version,
            'resample': self.resample, 
            'output_shape': self.output_shape, 
            'output_activation': self.output_activation,
            'loss': self.loss,
            'main_metric': self.main_metric, 
            
        }
        return copy.deepcopy(result)


def norm_img(X_test, X_train):
    """"Scales inout data by larges value in train set. 

    Parameters:
        X_test (ndarray): Test data. 
        X_train (ndarray): Train data.  
    Returns:
        ndarray: Scaled test data.
            ndarray: Scaled train data. 
    """
    max_val = np.max(X_train, axis = None)
    X_test /= max_val
    X_train /= max_val
    return X_test, X_train


def read_mnist(n):
    """"Reads the MNIST dataset

    Parameters:
        n (int): Number of samples to draw. If larger than dataset the entire dataset will be read. 
    Returns:
        Datasets, sample shape. 
    """
    (X, y), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    if n is not None and n < X.shape[0]:
        X, _, y, _ =  model_selection.train_test_split(X, y, train_size = n, stratify = y, random_state = 42)

    X = X.astype('float32')
    X_test = X_test.astype('float32')

    # change to range [0, 1]
    X_test, X = norm_img(X_test = X_test, X_train = X)
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 45)
    # one hot encoding
    y_train = keras.utils.to_categorical(y_train, dtype='float32')
    y_val = keras.utils.to_categorical(y_val, dtype='float32')
    y_test = keras.utils.to_categorical(y_test, dtype='float32')
    input_shape = X_test.shape[1:] # tuple with input shape
    output_shape = y_test.shape[-1] # output shape, number of classes for classification
    col = None
    return X_train, y_train, X_val, y_val, X_test, y_test, input_shape, output_shape, col


def read_data(dataset_name, num_timesteps = 1, version = None, resample = False, requested_col = []):
    """"Reads data. 

    Parameters:
        dataset_name (str): Name of dataset.  
        num_timesteps (int): Number of time steps. 
        version (str): Version. 
        resample (boolean): True if use resampling. 
        requested_col (list): List of columns reqested by the user. 
    Returns:
        Dataset: Dataset object. 
        Training, validaiton and test data. 
        
    """
    if dataset_name == 'mnist':
        X_train, y_train, X_val, y_val, X_test, y_test, input_shape, output_shape, col  = read_mnist(10000)
    
    elif dataset_name == 'fotball':
        if num_timesteps == 1:
            data = np.load('./Datasets/fotball_1_event.npz', allow_pickle=True)
            X = data['X']
            assert len(X.shape) == 2
            y = data['y']
            col = data['col'].tolist()
            col_to_pick = get_columns(col, version, requested_col)

            if len(col_to_pick) > 0:
                X, col = drop_col(X, col, col_to_pick)
        

            X, y = shuffle(X, y)
            X_train, X_test, y_train, y_test  = model_selection.train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 42)
            
            if resample:
                X_train, y_train = smote(X_train, y_train)
            
            X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size = 0.20, stratify = y_train, random_state = 420)
        
        else:
            data = np.load('./Datasets/fotball_m_event.npz', allow_pickle=True)
            X = data['X']
            assert len(X.shape) == 3
            max_time_steps = X.shape[1]
            if max_time_steps < num_timesteps: raise ValueError("More time steps requested than avaliable")
            
            y = data['y']
            col = data['col'].tolist()

            col_to_pick = get_columns(col, version, requested_col)
            
            if len(col_to_pick) > 0:
                X, col = drop_col(X, col, col_to_pick)
            
            X, y = shuffle(X, y)
            
            if num_timesteps > max_time_steps:
                X = X[:,-num_timesteps:,:]
                assert X.shape[1] == num_timesteps
                
            X_train, X_test, y_train, y_test  = model_selection.train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 42)
            
            
            if resample:
                # first flatten
                old_shape = X_train.shape[1:]
                X_train = X_train.reshape((X_train.shape[0],) + (np.prod(old_shape),))
                X_train, y_train = smote(X_train, y_train)
                X_train = X_train.reshape((X_train.shape[0],) + old_shape)
            
            X_train, X_val, y_train, y_val  = model_selection.train_test_split(X_train, y_train, test_size = 0.20, stratify = y_train, random_state = 420)

    else: raise ValueError("Unsuported dataset")
        
    dataset = Dataset(dataset_name, X_train.shape[1:], col, resample, num_timesteps, version)
    return dataset, (X_train, y_train), (X_val, y_val), (X_test, y_test)

def smote(X_train, y_train):
    """"Apply SMOTE to data. 

    Parameters:
        X_train (ndarray): Input data. 
        y_train (ndarray): Output data. 
    Returns:
        ndarray: Resampled input data. 
        ndarray: Resampled output data. 
        
    """
    sm = SMOTE(random_state = 2) 
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 
    return X_train_res, y_train_res



def get_columns(orginial_col, version, column):
    """"Return the columns to pick. 

    Parameters:
        orginial_col (list): Original columns of dataset. 
        version (str): Determins columns to pick. 
        column (list): List of column to pick or remove. 
    Returns:
        ndarray: Resampled input data. 
        ndarray: Resampled output data. 
        
    """
    if version == 'start_pos':
        col_to_pick = ['x_start', 'y_start']
    elif version == 'best':
        to_remove = ['tag_301', 'tag_102', 'matchId', 'id', 'teamId', 'playerId', 'league_0', 'league_1', 'league_2', 'league_3', 'league_4', 'league_5', 'league_6', 'matchPeriod']
        col_to_pick = list(set(orginial_col)-set(to_remove))
    elif version == 'pick_col':
        col_to_pick = column
    elif version == 'remove_col':
        col_to_pick = list(set(orginial_col)-set(column))
    else: raise NotImplementedError("version not implemented in version_mapping, version: ", version)
    return col_to_pick


def get_input_shape(input_shape, network_type):
    """Return the required input shape of differnt network archtectures. 

    Parameters:
        input_shape (tuple): Original input shape of dataset shape. 
        network_type (str): Name of the network type. 
    Returns:
        tuple: Input shape required by the network type. 
    """
    if network_type in ['mlp', 'log_reg']:
        input_shape = (np.prod(input_shape), )

    elif network_type == 'lstm':
        if len(input_shape) == 1:
            input_shape = (1,) + input_shape
        elif len(input_shape) == 2:
            input_shape = input_shape
        else: raise ValueError("Unexpected input shape in get_input_shape")

    elif network_type in ['conv1', 'conv_lstm']:
        if len(input_shape) == 1:
            input_shape = (1,) + input_shape
        elif len(input_shape) == 2:
            input_shape = input_shape
        else: raise ValueError("Unexpected input shape in get_input_shape")
    
    elif network_type == 'conv2':
        if len(input_shape) == 2:
            input_shape = input_shape + (1,), 
        elif len(input_shape) == 3:
            input_shape = input_shape
        else: raise ValueError("Unexpected input shape in get_input_shape")

    else: raise ValueError("Unexpected network type in get_input_shape")
    return input_shape
