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
from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import joblib
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from APRI.utils import get_class_name_dict
from sklearn.metrics import classification_report
from baseline.cls_feature_class import create_folder
from APRI.get_audio_features import *
# List of classes

print(sorted(sklearn.metrics.SCORERS.keys()))
params = parameter.get_params()
event_type= get_class_name_dict().values()
data_folder_path = os.path.join(params['dataset_dir'], 'oracle_mono_signals_beam_all/audio_features_beam_all/') # path to arrays
#data_aug_folder_path = os.path.join(params['dataset_dir'], 'oracle_mono_signals_beam_all_aug/audio_features_optimized_aug/') # path to arrays
model_output_path =  os.path.join(params['dataset_dir'], 'models/event_class_xgb/') # path to arrays
augmentation=False

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
if augmentation:
    number_sample=2000
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
# Defining some pipelines. GB, RF and SVC



pipe_rf = Pipeline([('reg', RandomForestClassifier(random_state=42))])

pipe_gb = Pipeline([(('reg', GradientBoostingClassifier(random_state=42)))])

pipe_svr = Pipeline([('scl', StandardScaler()),('reg', SVC())])

pipe_XGB = Pipeline([('reg',xgb.XGBClassifier(booster = "gbtree", objective = "multi:softprob",num_class=14,random_state=42))])

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
                   "reg__learning_rate": [0.3], # default 0.1
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

grids = [gs_XGB]
grid_dict = {0: 'XGB'}

# Split train and test
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y['target'], test_size=0.20, random_state=42)
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
    print('Prediction f1-score for best params: %.3f ' % f1_score(test_y, y_pred,average='weighted'))
    print('Prediction accuracy for best params: %.3f ' % accuracy_score(test_y, y_pred))
    print('Prediction accuracy for best params:')
    print(confusion_matrix(test_y,y_pred))
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print(classification_report(test_y,y_pred))
    # Track best (highest test accuracy) model
    if accuracy_score(test_y, y_pred) > best_acc:
        best_acc = accuracy_score(test_y, y_pred)
        best_gs = gs
        best_cls = idx
    #joblib.dump(best_gs.best_estimator_, model_output_path + 'provisional_train/'+ grid_dict[idx]+ '/model.joblib')
    #joblib.dump(best_gs.best_params_, model_output_path + 'provisional_train/'+ grid_dict[idx]+ '/params.joblib')
print('\n Classifier with best score: %s' % grid_dict[best_cls])

joblib.dump(best_gs.best_estimator_, model_output_path+'provisional_train/model.joblib')
joblib.dump(best_gs.best_params_, model_output_path+'provisional_train/params.joblib')































#Check
rows=[]
'''

for event in event_type:
    print("Event ",str(event))
    i=0
    data_path= os.path.join('/home/ribanez/Escritorio/audios/',event) #path to file
    data_path_o= os.path.join('/home/ribanez/Escritorio/audios/audio_features',event) #path to file
    create_folder(os.path.join(data_path_o, event))
    for audio in os.scandir(data_path):
        print("Extracting features from ", event + ' ' + audio.name)
        audio_features, column_labels = compute_audio_features(audio, options)
        file_name = os.path.splitext(audio.name)[0]
        if i == 0:
            np.save(os.path.join(data_path_o, 'column_labels.npy'), column_labels)
            i += 1
        np.save(os.path.join(data_path_o, event, file_name + '.npy'), audio_features)
'''

'''
for event in event_type:
    i = 0
    array_path = os.path.join('/home/ribanez/Escritorio/audios/audio_features',event,event)  # path to file
    array_path = os.path.join(data_folder_path, event)  # path to file
    for array in os.scandir(array_path):
        i+=1
        row=event+os.path.splitext(array.name)[0]
        rows.append(row)
        x=np.load(array)
        y=str(event)
        if 'datacheck_x' in locals():
            datacheck_x=np.vstack((datacheck_x,x))
            datacheck_y = np.vstack((datacheck_y, y))
        else:
            datacheck_x=x
            datacheck_y=y

dfcheck_x=pd.DataFrame(data=datacheck_x,index=rows,columns=columns)
test_y=pd.DataFrame(data=datacheck_y,index=rows,columns=['target'])
joblib.dump(best_gs.best_estimator_, model_output_path+'provisional_train/model.joblib')
y_pred = gs.predict(dfcheck_x)

print('Prediction f1-score for best params: %.3f ' % f1_score(test_y, y_pred, average='weighted'))
print('Prediction accuracy for best params: %.3f ' % accuracy_score(test_y, y_pred))
print('Prediction accuracy for best params:')
print(confusion_matrix(test_y, y_pred))
'''
'''
model= joblib.load( model_output_path + 'provisional_train/random_forest/model.joblib')
importances=model.steps[0][1].feature_importances_
df_importance=pd.DataFrame(data=importances,index=columns,columns=['importance'])
df_importance = df_importance.sort_values(by='importance',ascending=False)
df_importance.to_pickle(os.path.join(model_output_path + 'provisional_train/random_forest/important_columns.pkl'))
'''






