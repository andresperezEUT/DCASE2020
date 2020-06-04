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
import pickle
import csv


# List of classes
'''
print(sorted(sklearn.metrics.SCORERS.keys()))
params = parameter.get_params()
event_type= get_class_name_dict().values()
data_folder_path = os.path.join(params['dataset_dir'], 'oracle_mono_signals_beam_all/audio_features_beam_all/') # path to arrays
data_aug_folder_path = os.path.join(params['dataset_dir'], 'oracle_mono_signals_beam_all_aug/audio_features_optimized_aug/') # path to arrays
model_output_path =  os.path.join(params['dataset_dir'], 'models/event_class_xgb/') # path to arrays
augmentation=True
'''

def get_key(val):
    classes=get_class_name_dict()
    for key, value in classes.items():
         if val == value:
             return key

'''
# Import data and parse in pandas dataframes
rows=[]
rows_test=[]
for event in event_type:
    n=0
    print("Event ",str(event))
    i=0
    array_path= os.path.join(data_folder_path,event) #path to file
    for array in os.scandir(array_path):
        i+=1
        x = np.load(array)
        y = get_key(str(event))
        if n>200:
            row=event+os.path.splitext(array.name)[0]
            rows.append(row)
            if 'data_x' in locals():
                data_x=np.vstack((data_x,x))
                data_y = np.vstack((data_y, y))
            else:
                data_x=x
                data_y=y

        else:
            row_test=event+os.path.splitext(array.name)[0]
            rows_test.append(row_test)
            if 'data_x_test' in locals():
                data_x_test=np.vstack((data_x_test,x))
                data_y_test = np.vstack((data_y_test, y))
            else:
                data_x_test=x
                data_y_test=y
            n+=1
#Add data_augmentations to dataset
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
                y = get_key(str(event))
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

df_x_test=pd.DataFrame(data=data_x_test,index=rows_test,columns=columns)
df_y_test=pd.DataFrame(data=data_y_test,index=rows_test,columns=['target'])

df_x.to_pickle('/home/ribanez/Escritorio/audios/df_x.pkl')
df_y.to_pickle('/home/ribanez/Escritorio/audios/df_y.pkl')
df_x_test.to_pickle('/home/ribanez/Escritorio/audios/df_x_test.pkl')
df_y_test.to_pickle('/home/ribanez/Escritorio/audios/df_y_test.pkl')
'''


#Datasets
'''
print(df_x.shape)
print(df_y.shape)
print(df_x_test.shape)
print(df_y_test.shape)
#Split dataset train/test
#train_x, test_x, train_y, test_y = train_test_split(df_x, df_y['target'], test_size=0.20, random_state=42)

#DMatrices:XGB format
#dtrain = xgb.DMatrix(train_x, label=train_y)
#dtest = xgb.DMatrix(test_x, label=test_y)

dtrain = xgb.DMatrix(df_x, label=df_y)
dtest = xgb.DMatrix(df_x_test, label=df_y_test)
'''


#Params
params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'num_class':14,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'multi:softmax',
}
params['eval_metric'] = "mlogloss"

num_boost_round = 5000



#Parameters max_depth and min_child_weight
'''


gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(3,6)
    for min_child_weight in range(1,2)
]
min_metric = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=2,
        metrics={'mlogloss'},
        early_stopping_rounds=5,
        verbose_eval=5
    )    # Update best mlogloss
    mean_metric = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_metric, boost_rounds))
    if mean_metric < min_metric:
        min_metric = mean_metric
        best_params = (max_depth,min_child_weight)
        print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_metric))
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_metric))
params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]
'''
params['max_depth'] = 3
params['min_child_weight'] = 1

# Parameters sumbsample and colsample
'''
gridsearch_params = [
    (subsample, colsample)
    for subsample in [0.1,0.5,1]
    for colsample in [0.1,0.5,1]
]


min_mae = float("Inf")
best_params = None# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=2,
        metrics={'mlogloss'},
        early_stopping_rounds=5,
        verbose_eval=5
    )    # Update best score
    mean_mae = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
        print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]
'''
params['subsample'] =1
params['colsample_bytree'] =0.1

# Parameter eta
'''
min_mae = float("Inf")
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))    # We update our parameters
    params['eta'] = eta    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=2,
            metrics=['mlogloss'],
            early_stopping_rounds=5,
            verbose_eval=5
          )    # Update best score
    mean_mae = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
        print("Best params: {}, MAE: {}".format(best_params, min_mae))

params['eta'] =best_params
'''
params['eta'] =0.05


#Dataset load

df_x=pd.read_pickle('/home/ribanez/Escritorio/audios/df_x.pkl')
df_y=pd.read_pickle('/home/ribanez/Escritorio/audios/df_y.pkl')
df_x_test=pd.read_pickle('/home/ribanez/Escritorio/audios/df_x_test.pkl')
df_y_test=pd.read_pickle('/home/ribanez/Escritorio/audios/df_y_test.pkl')


dtrain = xgb.DMatrix(df_x, label=df_y)
dtest = xgb.DMatrix(df_x_test, label=df_y_test)

#Training
'''
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain,"Train"),(dtest, "Test")],
    early_stopping_rounds=20
)
model.save_model('/home/ribanez/Escritorio/audios/model.bin')

'''


loaded_model = xgb.Booster()
loaded_model.load_model('/home/ribanez/Escritorio/audios/model.bin')

print('Test predictions with trained mode...')
y_pred = loaded_model.predict(dtest)

print('Train predictions with trained mode...')
y_pred_t = loaded_model.predict(dtrain)
print('Confussion matrix test:')
print(confusion_matrix(np.array(df_y_test['target']), y_pred))
print('Prediction accuracy for test: %.3f ' % accuracy_score(np.array(df_y_test['target']), y_pred))
print('Prediction accuracy for train: %.3f ' % accuracy_score(np.array(df_y['target']), y_pred_t))


# Checking performance with sample
print('Checking performance for quick sample...')
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

for event in event_type:
    i = 0
    array_path = os.path.join('/home/ribanez/Escritorio/audios/audio_features',event,event)  # path to file
    #array_path = os.path.join(data_folder_path, event)  # path to file
    for array in os.scandir(array_path):
        print(array_path)
        print(array.name)
        i+=1
        row=event+os.path.splitext(array.name)[0]
        rows.append(row)
        x=np.load(array,allow_pickle=True)
        y = get_key(str(event))
        if 'datacheck_x' in locals():
            datacheck_x=np.vstack((datacheck_x,x))
            datacheck_y = np.vstack((datacheck_y, y))
        else:
            datacheck_x=x
            datacheck_y=y

columns = np.load('/home/ribanez/Escritorio/audios/audio_features/alarm/column_labels.npy').tolist()
dfcheck_x=pd.DataFrame(data=datacheck_x,index=rows,columns=columns)
test_y=pd.DataFrame(data=datacheck_y,index=rows,columns=['target'])
#test_y=datacheck_y
dcheck = xgb.DMatrix(dfcheck_x, label=test_y)

y_pred = loaded_model.predict(dcheck)
print('Confussion matrix for sample...')
print(confusion_matrix(np.array(test_y['target']), y_pred))
print('Prediction accuracy for sample: %.3f ' % accuracy_score(np.array(test_y['target']), y_pred))

with open('/home/ribanez/Escritorio/audios/prediction.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(y_pred)