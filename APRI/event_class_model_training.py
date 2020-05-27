"""
event_class_model_training.py

This script is used to train a clasiffier aimed to identify audio events from a set of classes
Several models are evaluated using a simple pipeline and unique gridsearch for each algorithm

The trained model is stored as a joblib file in the folder ....

"""

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
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from APRI.utils import get_class_name_dict

# List of classes

params = parameter.get_params()
event_type= get_class_name_dict().values()
data_folder_path = os.path.join(params['dataset_dir'], 'oracle_mono_signals_beam_all/audio_features_beam_all/') # path to arrays
model_output_path =  os.path.join(params['dataset_dir'], 'models/event_class_xgb/') # path to arrays

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
# Defining some pipelines. GB, RF and SVC

pipe_rf = Pipeline([('scl', StandardScaler()),('reg', RandomForestClassifier(random_state=42))])

pipe_gb = Pipeline([('scl', StandardScaler()),('reg', GradientBoostingClassifier(random_state=42))])

pipe_svr = Pipeline([('scl', StandardScaler()),('reg', SVC())])

pipe_XGB = Pipeline([('scl',StandardScaler()),('reg',xgb.XGBClassifier(objective="multi:softprob", random_state=42))])

# Defining some Grids

grid_params_rf = [{'reg__n_estimators': [500,1000],
                   'reg__max_depth': [8,16,32],
                   'reg__max_features': ["auto"],
                   'reg__min_samples_split': [2,4]}]

grid_params_gb = [{'reg__learning_rate': [0.01,0.02,0.03],
                   'reg__n_estimators' : [100,500,1000],
                   'reg__max_depth'    : [4,6,8]}]


grid_params_svr = [{'reg__kernel': ['rbf'],
                    'reg__gamma': [1e-10,1e-8,1e-6,1e-5,1e-4,0.01, 0.1],
                    'reg__C': [1, 10, 100, 1000, 10000,100000,1000000,1e8,1e10,1e12]}]

grid_params_svr = [{'reg__kernel': ['rbf'],
                    'reg__gamma': [0.01],
                    'reg__C': [100]}]

grid_params_XGB = [{'reg__colsample_bytree': [0.1,0.8],
                    "reg__learning_rate": [0.01,0.1], # default 0.1
                    "reg__max_depth": [6], # default 3
                    "reg__n_estimators": [500]}]


# Defining some grid searches

gs_rf = GridSearchCV(estimator=pipe_rf,param_grid=grid_params_rf,scoring='accuracy',cv=2,verbose=10,n_jobs=-1)

gs_gb = GridSearchCV(estimator=pipe_gb,param_grid=grid_params_gb,scoring='accuracy',cv=2,verbose=10,n_jobs=-1)

gs_svr = GridSearchCV(estimator=pipe_svr,param_grid=grid_params_svr,scoring='accuracy',cv=5,verbose=10,n_jobs=-1)

gs_XGB = GridSearchCV(estimator=pipe_XGB,param_grid=grid_params_XGB,scoring='accuracy',cv=4,verbose=10,n_jobs=-1)

grids = [gs_rf, gs_gb, gs_svr]
grid_dict = {0: 'random_forest',
             1: 'gradient_boosting',
             2: 'svc'}
grids = [gs_XGB]
grid_dict = {0: 'xgb'}

# Split train and test
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y['target'], test_size=0.15, random_state=42)
best_acc = 0
best_cls = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
     # Fit grid search

    gs.fit(train_x,train_y)
    # Best params
    print('Best params: %s' % gs.best_params_)
    # Best training data r2
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Prediction using best model
    y_pred = gs.predict(test_x)
    print('Prediction accuracy for best params: %.3f ' % accuracy_score(test_y, y_pred))
    print('Prediction accuracy for best params:')
    print(confusion_matrix(test_y,y_pred))
    # Track best (highest test accuracy) model
    if accuracy_score(test_y, y_pred) > best_acc:
        best_acc = accuracy_score(test_y, y_pred)
        best_gs = gs
        best_cls = idx
print('\n Classifier with best score: %s' % grid_dict[best_cls])
joblib.dump(best_gs.best_estimator_, model_output_path+'provisional_train/model.joblib')
joblib.dump(best_gs.best_params_, model_output_path+'provisional_train/params.joblib')







