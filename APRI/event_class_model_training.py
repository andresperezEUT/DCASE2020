"""
event_class_model_training.py

This script is used to train a classifier aimed to identify audio events from a set of classes
Several models are evaluated using a simple pipeline and unique gridsearch for each algorithm.


"""

from baseline import parameter
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from APRI.utils import get_class_name_dict

# List of classes
params = parameter.get_params()

if params['user'] == 'FAIK':
    import xgboost as xgb

event_type= get_class_name_dict().values()
data_folder_path = # path to arrays
data_aug_folder_path = # path to arrays
model_output_path =  # path to arrays

# Import data and parse in pandas dataframes
rows=[]
for event in event_type:
    print("Event ",str(event))
    i=0
    array_path= os.path.join(data_folder_path,event) #path to file
    for array in os.scandir(array_path):
        i+=1
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

number_sample=1300
for event in event_type:
    print("Completing event ",str(event))
    i=0
    array_path= os.path.join(data_aug_folder_path,event) #path to file
    for array in os.scandir(array_path):
        if np.count_nonzero(data_y == str(event))<number_sample:
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
print(data_x.shape)
columns = np.load(os.path.join(data_folder_path, 'column_labels.npy')).tolist()
df_x=pd.DataFrame(data=data_x,index=rows,columns=columns)
df_y=pd.DataFrame(data=data_y,index=rows,columns=['target'])

# Defining some pipelines. GB, RF, XGB and SVC
pipe_rf = Pipeline([('reg', RandomForestClassifier(random_state=42))])
pipe_gb = Pipeline([(('reg', GradientBoostingClassifier(random_state=42)))])
pipe_svr = Pipeline([('scl', StandardScaler()),('reg', SVC())])
pipe_XGB = Pipeline([('reg',xgb.XGBClassifier(booster = "gbtree", objective = "multi:softmax",num_class=14,random_state=42))])

# Defining some Grids
grid_params_rf = [{'reg__n_estimators': [200],
                   'reg__max_depth': [16],
                   'reg__max_features': ["auto"],
                   'reg__min_samples_split': [2]}]
grid_params_gb = [{'reg__learning_rate': [0.01,0.02,0.03],
                   'reg__n_estimators' : [100,500,1000],
                   'reg__max_depth'    : [4,6,8]}]
grid_params_svr = [{'reg__kernel': ['rbf'],
                    'reg__gamma': [0.001],
                    'reg__C': [10]}]
grid_params_XGB = [{'reg__colsample_bytree': [0.3],
                   "reg__learning_rate": [0.15,0.2], # default 0.1
                    "reg__max_depth": [3], # default 3
                    "reg__n_estimators": [500]}]

# Defining some grid searches
gs_rf = GridSearchCV(estimator=pipe_rf,param_grid=grid_params_rf,scoring='accuracy',cv=5,verbose=10,n_jobs=-1)
gs_gb = GridSearchCV(estimator=pipe_gb,param_grid=grid_params_gb,scoring='accuracy',cv=5,verbose=10,n_jobs=-1)
gs_svr = GridSearchCV(estimator=pipe_svr,param_grid=grid_params_svr,scoring='accuracy',cv=5,verbose=10,n_jobs=-1)
gs_XGB = GridSearchCV(estimator=pipe_XGB,param_grid=grid_params_XGB,scoring='f1_weighted', cv=5,verbose=10,n_jobs=-1)

grids = [gs_rf, gs_svr,gs_XGB]
grid_dict = {0: 'random_forest',
             1: 'svc',
             2: 'XGB'}

# Split train and test
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y['target'], test_size=0.10, random_state=42)
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







