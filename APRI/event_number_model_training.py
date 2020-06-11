"""
event_number_model_training.py

This script is used to train a clasiffier aimed to identify the number of audio events in an audio frame level
Training features values are DirAC diffuseness calculated by generate_diffuseness_and_source_count.by
The output of the model is 0, 1 or 2 events
Several models are evaluated using a simple pipeline and unique gridsearch for each algorithm
Sklearn modules are used as first aproach: Random Forest, Gradient Boosting and Support Vector Classifiers
If GB is chosen, XGBoost or Light XGB will be eventually implemented since they optimize computation (paralelization)
The trained model is stored as a joblib file in the folder ....

"""

from baseline import parameter
import os
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
import joblib
# import xgboost as xgb
import numpy as np
import scipy

if os.environ.get('USER') == 'ribanez':
    user = 'FAIK'
    import xgboost as xgb


params = parameter.get_params()
data_input_path = os.path.join(params['dataset_dir'],'models/event_number/input_data' ) # path to folders
model_output_path =  os.path.join(params['dataset_dir'], 'models/event_number/') # path to arrays

# Import data and parse in pandas dataframes
dff_x=pd.read_pickle(os.path.join(data_input_path,'training_x_event_number.pkl'))
dff_y=pd.read_pickle(os.path.join(data_input_path,'training_y_event_number.pkl'))

dff_y_binary=dff_y.astype(int)
dff_y_binary=dff_y_binary.astype(str)
dff_y_binary['target'] = pd.Categorical(dff_y_binary['target'])
#dff_y_binary['target'].replace({'2': '1'}, inplace=True)
print(dff_y_binary.describe())
print(dff_y['target'].value_counts())
#df=pd.DataFrame(columns=)
array_dff_x=dff_x.values
dff_x['mean']=np.mean(array_dff_x, axis=1)
dff_x['min']=np.min(array_dff_x, axis=1)
dff_x['max']=np.max(array_dff_x, axis=1)
dff_x['mean']=np.mean(array_dff_x, axis=1)
dff_x['median']=np.median(array_dff_x, axis=1)
dff_x['mode']=np.percentile(array_dff_x,50,axis=1)
dff_x['range']=np.ptp(array_dff_x, axis=1)
dff_x['p10']=np.percentile(array_dff_x,10,axis=1)
dff_x['p20']=np.percentile(array_dff_x,20,axis=1)
dff_x['p25']=np.percentile(array_dff_x,25,axis=1)
dff_x['p30']=np.percentile(array_dff_x,30,axis=1)
dff_x['p40']=np.percentile(array_dff_x,40,axis=1)
dff_x['p60']=np.percentile(array_dff_x,60,axis=1)
dff_x['p70']=np.percentile(array_dff_x,70,axis=1)
dff_x['p75']=np.percentile(array_dff_x,75,axis=1)
dff_x['p80']=np.percentile(array_dff_x,80,axis=1)
dff_x['p90']=np.percentile(array_dff_x,90,axis=1)
dff_x['iqr']=scipy.stats.iqr(array_dff_x, axis=1,rng=(25,75),interpolation='lower')
dff_x['std']=np.std(array_dff_x, axis=1)
dff_x['var']=np.var(array_dff_x, axis=1)
dff_x['skew']=scipy.stats.skew(array_dff_x, axis=1)
dff_x['kurt']=scipy.stats.kurtosis(array_dff_x, axis=1)

columns=['mean','min','max','median','mode','range','p10','p20','p25','p30','p40','p60','p70','p75','p80','p90','iqr','std','var','skew','kurt']
dff_x=dff_x[columns]

print("Dimensiones dataset  ")
print(dff_x.shape)

'''
n=20
i=0
j=0
k=0
dfaux=dff_x
#dfaux = dff_x
lista=[]
for column in range(len(dff_x.columns)):
    i+=1
    k+=1
    if i==n:
        j+=1
        lista.append('v' + str(j))
        dfaux['v'+str(j)]=dff_x['v'+str(k)]
        i=0
dfaux=dfaux[lista]
print(len(dfaux.columns))
dff_x=dfaux
'''
# Defining some pipelines. GB, RF and SVC

pipe_rf = Pipeline([('reg', RandomForestClassifier(random_state=42))])

pipe_gb = Pipeline([('reg', GradientBoostingClassifier(random_state=42))])

pipe_svr = Pipeline([('reg', SVC())])

pipe_XGB = Pipeline([('reg',xgb.XGBClassifier(random_state=42))])

# Defining some Grids

grid_params_rf = [{'reg__n_estimators': [100],
                   'reg__max_depth': [8],
                   'reg__max_features': ["sqrt"],
                   'reg__min_samples_split': [4]}]

grid_params_gb = [{'reg__learning_rate': [0.01,0.02,0.03],
                   'reg__n_estimators' : [100,500,1000],
                   'reg__max_depth'    : [4,6,8]}]

grid_params_svr = [{'reg__kernel': ['rbf'],
                    'reg__gamma': [1e-8,0.9],
                    'reg__C': [1, 10000000]}]

grid_params_XGB = [{'reg__colsample_bytree': [0.1,0.9],
                    "reg__learning_rate": [0.01,0.5], # default 0.1
                    "reg__max_depth": [3], # default 3
                    "reg__n_estimators": [100,200]}]

# Defining some grid searches
jobs=-1

gs_rf = GridSearchCV(estimator=pipe_rf,param_grid=grid_params_rf,scoring='accuracy',cv=2,verbose=10,n_jobs=-1)

gs_gb = GridSearchCV(estimator=pipe_gb,param_grid=grid_params_gb,scoring='accuracy',cv=5,verbose=10,n_jobs=1)

gs_svr = GridSearchCV(estimator=pipe_svr,param_grid=grid_params_svr,scoring='accuracy',cv=2,verbose=10,n_jobs=-1)

gs_XGB = GridSearchCV(estimator=pipe_XGB,param_grid=grid_params_XGB,scoring='accuracy',cv=4,verbose=10,n_jobs=-1)

grids = [gs_rf]

grid_dict = {0: 'xgb'}

# Split train and test

train_x, test_x, train_y, test_y = train_test_split(dff_x, dff_y_binary['target'], test_size=0.10, random_state=42)

# Train

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

# Save model (local)
joblib.dump(best_gs.best_estimator_, model_output_path+'/model.joblib')
joblib.dump(best_gs.best_params_, model_output_path+'/params.joblib')
























