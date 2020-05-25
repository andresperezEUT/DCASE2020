'''
Due to the large amount of features in diffuseness files (1200), this script develops several dimensionality reduction
techniques to override computation problems in further model training.
As a first approach, we calculate the average value for each N consequtive cells, resulting an array of (original_len/N)
A PCA analysis is also implemented to validate advanced DR techniques
'''

from baseline import parameter
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

params = parameter.get_params()
data_input_path = os.path.join(params['dataset_dir'],'models/event_number/input_data' ) # path to folders
data_output_path =  os.path.join(params['dataset_dir'], 'models/event_number/input_data') # path to arrays

# Import data and parse in pandas dataframes
dff_x=pd.read_pickle(os.path.join(data_input_path,'training_x_event_number.pkl'))
dff_y=pd.read_pickle(os.path.join(data_input_path,'training_y_event_number.pkl'))

n=120


i=0
print(dff_x.shape)
for index,row in dff_x.iterrows():
    i+=1
    aux=0
    cont=0
    for j in range(len(row)):
        cont+=1
        aux=aux+row[j]
        if cont==n:
            aux=aux/n
            if j==n-1:
                reduced_array=np.array(aux)
                aux=0
                cont=0
            else:
                reduced_array=np.append(reduced_array,np.array(aux))
                aux=0
                cont=0
    if i==1:
        reduced_df=reduced_array
    else:
        reduced_df=np.vstack((reduced_df,reduced_array))
    if i%1000==0:
        print(reduced_df.shape)
print(i)


dff_x.to_pickle(os.path.join(data_output_path, 'training_x_reduced_event_number.pkl'))

