import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from pickle import dump, load

def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="path", metavar="PATH TO DATASETS", default=None,
                        help="Path to the dataset files")
    parser.add_argument("--number_time_steps", dest="m", metavar="NUMBER OF TIME STEPS", default=10,
                        help="Specify the maximum number of time steps for the time series predictions", type = int)  
 
    args = parser.parse_args()
    assert os.path.exists(args.path), "Path not found."
    assert args.m > 1, "Requested too few events for series."

    return args

def get_envents_to_keep(events, m):
    """Extracts all shot events and m-1 events prior to the shot events. 

    Parameters:
        events (DateFrame): DataFrame of evets. 
        m (int): The maximum number of events to in include in a time series sample.

    Returns:
        list: 1D list of booleans. True elements are to be kept. 
    """
    events_to_keep = (events['eventId'] == 10).to_list() # the shot events, boolean vector of length number of samples
    count = 0
    for i, event in reversed(list(enumerate(events_to_keep))):
        if event: # if the event is a shot
            count = 0
        elif count < m:
            events_to_keep[i] = True
            count += 1
        else: count += 1
    return events_to_keep 



def read_data(filename, index, m):
    """Returns DataFrame of one League.

    Reads a .json file, load to dataframe and and adds the 'league' 
    as a ordinally encoded column. 

    Parameters:
        events (str): File name
        index (int): Index to encode the league as. 

    Returns:
        DataFrame: DataFrame of events of one league. 
    """
    with open(filename) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['league'] = index # add a leauge column
    df = df[get_envents_to_keep(df, m)] # only keep m-1 events prior to a shot event, m = 10 by default
    print("\t\t\tAdded " + str(df.shape[0]) + " events ")
    return df


def read_data_run(filenames, path, m):
    """Returns DataFrame of envents from all files in filenames.

    Parameters:
        filenames (list): List of string file names with path
        index (int): Index to encode the league as. 

    Returns:
        DataFrame: DataFrame of events of all leagues
    """
    frames = []
    for index, filename in enumerate(filenames):
        print("\t\tReading data from file " + filename)
        frames.append(read_data(path + filename, index, m))
    result = pd.concat(frames, ignore_index = True)
    print("\tExtracted " + str(result.shape[0]) + " events and " + str((result[result['eventId'] == 10]).shape[0]) + " shots")
    return result



def dropNames(events):
    """Removes the eventName and subEventName. 
    
    The information is avaliable through the eventId and subEventId. 

    Parameters:
        events (DataFrame): DataFrame of events. 

    """
    to_drop = ['subEventName', 'eventName']
    print("\tDropping subEventName and eventName columns")
    events.drop(columns=to_drop, inplace=True)


def split_tags(events):
    """Split the tags into separate one-hot encoded features.

    Parameters:
        events (DataFrame): DataFrame of events. 

    Returns:
        DataFrame: DataFrame of events of all leagues
    """
    print("\tSplitting tags into separate features")
    tag_series = events.pop('tags') # extract tags
    all_tags = [] # all tags in dataset with duplicates
    all_tags_per_row = [] # list of list, inner list is tag per sample, outer list is samples 
    for index, ids in tag_series.items(): # iterate over samples
        per_row = []
        for idet in ids: # iterate over tags per sample
            all_tags.append(idet['id'])
            per_row.append(idet['id'])
        all_tags_per_row.append(per_row)
    unique_tags_sorted = sorted(set(all_tags)) # all the unique tags in the dataset
    num_samples = tag_series.size
    
    encoded_tags = np.zeros((num_samples, len(unique_tags_sorted)))
    for i, tag_of_row in enumerate(all_tags_per_row): # loop over samples
        for j, tag in enumerate(unique_tags_sorted): # lopp over all tags
            encoded_tags[i][j] = int(tag in tag_of_row) 
    df = pd.DataFrame(data=encoded_tags, columns=['tag_'+str(tag) for tag in unique_tags_sorted])
    df = df.rename(columns = {'tag_101':'goal'}) # the encoded tags
    df = pd.concat([events, df], axis = 1) # merge events and encoded tags
    return df

def extract_postion(events):
    """Split positon column into 4 separate features and scale to rage [0, 1]

    Parameters:
        events (DataFrame): DataFrame of events. 

    Returns:
        DataFrame: DataFrame of events with split field positions. 
    """
    print("\tSplitting position into separate features")
    pos = events.pop('positions')
    nd_pos = np.zeros((events.shape[0], 4), dtype='float32')
    added_zer = []
    for i, row in pos.items(): # loop over samples
        if len(row) == 1: # handle samples with on missing position
            row.append({'y': 0, 'x': 0})
            added_zer.append([i, events.iloc[i]['id']])
        nd_pos[i] = [row[0]['x'], row[0]['y'], row[1]['x'], row[1]['y']]
    nd_pos = nd_pos / 100.0 # change scale form [0, 100] --> [0, 1]
    columns = ['x_start', 'y_start', 'x_end', 'y_end']
    events = pd.concat([events, pd.DataFrame(nd_pos, columns=columns)], axis = 1)
    print('\t\tAdded end positon (0, 0) to ' + str(len(added_zer)) + " sampels")
    return events

def get_combine_columns():
    """Help function for combine_goal_position

    Returns:
        list: list of list, represent which column to combin and how to combine them.
    """
    tag_to_combine = [
        [[1201], 'low_center'],
        [[1202, 1210, 1217], 'low_right'],
        [[1203], 'center'], 
        [[1204, 1211, 1218], 'center_left'], 
        [[1205, 1212, 1219], 'pos_low_left'], 
        [[1206, 1213, 1220], 'center_right'], 
        [[1207, 1214, 1221], 'high_center'], 
        [[1208, 1215, 1222], 'high_left'], 
        [[1209, 1216, 1223], 'high_right'], 
    ]
    for row in tag_to_combine:
        row[0] = ['tag_'+str(tag) for tag in row[0]]
    return tag_to_combine

def combine_goal_position(events):
    """Combines features 

    Parameters:
        events (DataFrame): DataFrame of events. 

    Returns:
        DataFrame: DataFrame of events with combined goal positions.
    """
    print("\tCombining goal position tags")
    col_to_combine =  get_combine_columns() # get mapping
    combined_tags = np.zeros((events.shape[0], len(col_to_combine))) # encoded tags
    col_names = [row[-1] for row in col_to_combine]
    for i, tags_to_combine in enumerate(col_to_combine): # loop over the new columns
        to_comb = (events[tags_to_combine[0]]).to_numpy()
        combined = to_comb.any(axis=1)
        combined_tags[:, i] = combined # updatate combined feature
    col_to_drop = list(pd.core.common.flatten([row[0] for row in col_to_combine]))
    events.drop(columns=col_to_drop, inplace=True)
    events = pd.concat([events, pd.DataFrame(combined_tags, columns=col_names)], axis = 1) # merge
    return events

def scale_encode(events):
    """Encodes columns

    Parameters:
        events (DataFrame): DataFrame of events. 

    Returns:
        DataFrame: DataFrame of events with encoded columns.
    """
    print("\tEncodes matchPeriod, playerId, teamId, league")
    # ordinal encoder
    print("\t\tEncodes matchPeriod, playerId and teamId with ordinal encoder. Save encoder to Encoder/ord_encoder.pkl")
    ord_cols = ['matchPeriod', 'playerId', 'teamId',]
    ordinal_enc = OrdinalEncoder()
    events[ord_cols] = ordinal_enc.fit_transform(events[ord_cols])
    folder = './Encoder'
    if not os.path.exists(folder):
          os.makedirs(folder)
    
    filename = folder + '/ord_encoder.pkl'
    dump(ordinal_enc, open(filename, 'wb'))
    
    # one-hot encoder
    print("\t\tOne-hot encodes league")
    one_hot_cols = ['league']
    events = pd.get_dummies(events, columns = one_hot_cols)
    return events

def drop_same_entry_columns(events):
    """Remoes columns with only one value. 

    Parameters:
        events (DataFrame): DataFrame of events. 

    Returns:
        DataFrame: DataFrame of events without columns of only one value. 
    """
    print("\t\tRemoving columns with only one value")
    events = events.copy()
    col_to_drop = [] 
    for key, value in events.iteritems(): 
        if len(value.unique()) == 1:
            col_to_drop.append(key)
    events.drop(columns=col_to_drop, inplace=True)
    return events


def extract_shots_and_valid_columns(events):
    """Extracts shots for shot dataset

    Parameters:
        events (DataFrame): DataFrame of events. 

    Returns:
        ndarray: ndarray of input X of the shots. Shape (num_shots, num_features)
        ndarray: ndarray of output y of the shots.  Shape (num_shots)
        list: list of string columns names of input X
    """
    print("\tExtracting shots for shot dataset")
    shots = events[events['eventId'] == 10].copy()
    col_to_drop = [] 
    for key, value in shots.iteritems(): 
        if len(value.unique()) == 1:
            col_to_drop.append(key)
    shots.drop(columns=col_to_drop, inplace=True)

    
    y = shots.pop('goal').to_numpy(dtype='float32')
    X = shots.to_numpy(dtype='float32')
    print("\t\tExtracted " + str(X.shape[0]) + " shots")
    return X, y, shots.columns

def extract_shot_series(events, m = 10):
    """Extracts shots series

    Parameters:
        events (DataFrame): DataFrame of events. 

    Returns:
        ndarray: ndarray of input X of the shots. Shape (num_shots, num_events, num_features)
        ndarray: ndarray of output y of the shots.  Shape (num_shots,)
        list: list of string columns names of input X
    """
    print("\tExtracting shot series, maximum number of events is " + str(m))
    y = events.pop('goal') 
    y = y[events['eventId']==10].to_numpy(dtype='float32')
    # wanted shape = (samples, time_steps, features) = (shots.shape[0], m, shots.shape[1])
    events = drop_same_entry_columns(events) # drop any columns which have only one value
    column_names = events.columns
    org_shape = events.shape  # orginal shape (samples, features)
    num_features = org_shape[-1]
    shots = events['eventId'] == 10 # series with True if event is shot
    shots_index = events.index[shots].tolist()
    X = np.zeros((sum(shots), m, num_features)) 

    indexMatchId = events.columns.get_loc('matchId')
    indexMatchPeriod = events.columns.get_loc('matchPeriod')
    inexEventSec = events.columns.get_loc('eventSec')
    i = 0 # index over shots
    count = 0
    for shot in X: # loop over shots
        shot_i = shots_index[i] # index of the shot in the events DataFrame
        sample = events.iloc[range(shot_i-m+1, shot_i+1), :].to_numpy() # The shot sample
        lastMatchId = sample[-1, indexMatchId]  # matchId of shot
        lastMatchPeriod = sample[-1, indexMatchPeriod] # matchPeriod of shot
        lastEventSec = sample[-1, inexEventSec] # time the shot took place
        j = m-1 # index of rows in sample, start at m-1, loops revered order
        
        for k in reversed(range(m-1)): # Determins if zero-padding is needed for the shot sample
            if lastMatchId != sample[k, indexMatchId] or lastMatchPeriod != sample[k, indexMatchPeriod] or np.abs(lastEventSec-sample[k, inexEventSec] > 30): 
                count += 1
                break
            j -= 1
        
        if j >= 1: # Zero padd if j >= 1
            sample[:][:j][:] = np.zeros((1, j, num_features))

        X[i] = sample 
        i += 1
    print("\t\tZero padded " + str(count) + " samples")
    return X, y, column_names

def normalize_time(X_1, col_1, X_m, col_m):
    """Normalizes time

    Parameters:
        X_1 (ndarray): Shots input. 
        col_1 (ndarray): Columns of shot dataset. 
        X_m (ndarray): Shot series input. 
        col_m (ndarray): Columns of shot series dataset.

    Returns:
        ndarray: ndarray of input X of the shots. Shape (num_shots, num_events, num_features)
        ndarray: ndarray of input X of the shot series.  Shape (num_shots, num_features)
    """
    print("\tNormalize time, save encoder to Encoder/min_max_encoder.pkl")
    X_1 = X_1.copy()
    X_m = X_m.copy()
    
    column = 'eventSec'
    index_1 = list(col_1).index(column)
    index_m = list(col_m).index(column)
    
    min_max_enc = MinMaxScaler()
    X_m_time_shape = X_m[:, :, index_m].shape
    X_m[:, :, index_m]  = min_max_enc.fit_transform(X_m[:, :, index_m].reshape(-1, 1)).reshape(X_m_time_shape) 
    X_1_time_shape = X_1[:, index_1].shape
    X_1[:, index_1]= min_max_enc.transform(X_1[:, index_1].reshape(-1, 1)).reshape(X_1_time_shape)
    filename = 'Encoder/min_max_encoder.pkl'
    dump(min_max_enc, open(filename, 'wb'))
    return X_1, X_m

def save_datasets(X_1, y_1, col_1, X_m, y_m, col_m):
    """Save dataset with one and multiple events. 
    Parameters:
        X_1 (ndarray): Shots input. 
        y_1 (ndarray): Shots output. 
        col_1 (ndarray): Columns of shot dataset. 
        X_m (ndarray): Shot series input. 
        y_m (ndarray): Shots series output. 
        col_m (ndarray): Columns of shot series dataset.
    """
    print("\tSave datasets to folder Datasets")
    folder = './Datasets'
    if not os.path.exists(folder):
          os.makedirs(folder)
    np.savez(folder + '/fotball_1_event.npz', X=X_1, y=y_1, col = col_1)
    np.savez(folder + '/fotball_m_event.npz', X=X_m, y=y_m, col = col_m)


def main():
    """Runs the preprocessing
    """
    args = parse_arguments()

    filenames = [
        '/events_England.json',
        '/events_France.json', 
        '/events_Germany.json', 
        '/events_Italy.json', 
        '/events_Spain.json',
        '/events_World_Cup.json',
        '/events_European_Championship.json', 
    ]
    print("Start preprocessing of " + str(len(filenames)) + " files")
    events = read_data_run(filenames, args.path, args.m)
    dropNames(events)
    events = split_tags(events)
    events = extract_postion(events)
    events = combine_goal_position(events)
    events = scale_encode(events)
    print("\tDrop Event ID")
    events.drop(columns = ['id'], inplace = True)
    events.loc[events.eventId == 6, 'subEventId'] = 60
    X_shots, y_shots, col_shots = extract_shots_and_valid_columns(events)
    X_series, y_series, col_series = extract_shot_series(events)
    X_shots, X_series = normalize_time(X_shots, col_shots, X_series, col_series)
    save_datasets(X_shots, y_shots, col_shots,X_series, y_series, col_series)
    print("Preprocessing finished")

if __name__ == "__main__":
    main()
