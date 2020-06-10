# Applying-Machine-Learning-Methods-to-Predict-the-Outcome-of-Shots-in-Football

The public repository for the master thesis Applying Machine Learning Methods to Predict the Outcome of Shots in Football.

The tesis explores a publicly avalaible dataset which covers match events in football. The dataset is avaliable [here](https://figshare.com/collections/Soccer_match_event_dataset/4415000/2), is described in [this](https://www.nature.com/articles/s41597-019-0247-7) paper. The content of the event dataset is further described [here](https://apidocs.wyscout.com/matches-wyid-events). 

## Preprocessing

To run the preprocessing call the file preprocessing.py. The data_path argument is the path to the dataset files downlowded from [here](https://figshare.com/articles/Events/7770599). The second argument is the maximum number of events to include in time series, that is the maximum number of events to include in a sample representing a shot. The dataset .json files are stored in a folder named original_datasets in the current directory and 10 events (including the shot event) is requested. 

```bash
python preprocessing.py --data_path original_datasets --number_time_steps 10
```
The preprocessed dataset will be saved into a folder called Dataset into two .npz files. 

## Run Training

To run training run the notebook called run.ipynb.

The output files will be stored in the folder training_out. 




