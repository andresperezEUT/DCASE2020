"""
data_preparation_event_number_model_training.py

This script is used to load diffuseness files and create a consolidated array for event_number_model
It takes significant time since it is not paralelized...several improvements could be made

The output arrays (x , y) are stored in /models/event_number/input_data

"""
from baseline import parameter
import os
import numpy as np
import pandas as pd

params = parameter.get_params()
data_rootfolder_path = os.path.join(params['dataset_dir'], 'num_sources/') # path to folders
data_output_path =  os.path.join(params['dataset_dir'], 'models/event_number/input_data') # path to arrays

# Import data and parse in pandas dataframes

for subdir, dirs, files in os.walk(data_rootfolder_path):
    for file in files:
        os.chdir(subdir)
        if file == 'diffuseness.npy':
            x = np.load(file).T
            if 'framedata_x' in locals():
                framedata_x = np.vstack((framedata_x, x))
            else:
                framedata_x = x
        if file == 'num_sources.npy':
            y = np.load(file)
            if 'framedata_y' in locals():
                framedata_y = np.concatenate((framedata_y, y))
            else:
                framedata_y = y
i = 0
columns = []
for value in x[0]:
    i += 1
    column = 'v' + str(i)
    columns.append(column)

dff_x = pd.DataFrame(data=framedata_x, columns=columns)
dff_y = pd.DataFrame(data=framedata_y, columns=['target'])

print("Features dataset size : ",dff_x.shape)
print("Targets dataset size : ",dff_y.shape)
dff_x.to_pickle(os.path.join(data_output_path, 'training_x_event_number.pkl'))
dff_y.to_pickle(os.path.join(data_output_path, 'training_y_event_number.pkl'))



