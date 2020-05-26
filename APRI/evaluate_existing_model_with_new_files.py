
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
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# List of classes
def get_class_name_dict():
    return {
        0: 'alarm',
        1: 'crying_baby',
        2: 'crash',
        3: 'barking_dog',
        4: 'running_engine',
        5: 'female_scream',
        6: 'female_speech',
        7: 'burning_fire',
        8: 'footsteps',
        9: 'knocking_on_door',
        10:'male_scream',
        11:'male_speech',
        12:'ringing_phone',
        13:'piano'
    }


params = parameter.get_params()
event_type= get_class_name_dict().values()
data_folder_path = os.path.join(params['dataset_dir'], 'oracle_mono_signals/audio_features/') # path to arrays

model_input_path = os.path.dirname(os.path.realpath(__file__))+'/models/event_class_beam_all/model.joblib'
# Import data and parse in pandas dataframes
rows=[]
for event in event_type:
    array_path= os.path.join(data_folder_path,event) #path to file
    for array in os.scandir(array_path):
        row=event+os.path.splitext(array.name)[0]
        rows.append(row)
        x=np.load(array)
        y=str(event)
        if 'data_x' in locals():
            data_x=np.vstack((data_x,x))
            data_y = np.vstack((data_y, y))
        else:
            data_x=x
            data_y=y
columns=[]
i=0
for value in x:
    i+=1
    column='v'+str(i)
    columns.append(column)
df_x=pd.DataFrame(data=data_x,index=rows,columns=columns)
df_y=pd.DataFrame(data=data_y,index=rows,columns=['target'])
print(df_x.shape)
model = joblib.load(model_input_path)
events=[]
for index,row in df_x.iterrows():
    event_class = model.predict(np.array([row]))
    events.append(event_class)
accuracy=accuracy_score(df_y['target'],events)
confusion_matrix=confusion_matrix(df_y['target'],events)
print(accuracy)
print(confusion_matrix)