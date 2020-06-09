'''
This script executes the pipeline from dataframes (pandas) to trained models.
The steps defined in the pipeline are the following:
- Parameters
- Dataframe splits
- (optional) Feature selection
- optional (gridsearch)
- modelling
- results
- dump

'''

# Dependencies
import os, datetime
from baseline import parameter
import pandas as pd
from APRI.get_dataframes import *
from APRI.get_model_utils import *
import pickle
import csv
import joblib

# Import general parametes
params = parameter.get_params()
dataset_dir= os.path.join(params['dataset_dir'])
this_file_path = os.path.dirname(os.path.abspath(__file__))

# Parameters
mode='new' # new or modify
pipeline_modelling='' #if mode is 'modify'
pipeline_feature_engineering='Datasets_foa_dev_2020-06-09'

build_dataframes=True
data_augmentation=False
feature_selection=False
random_forest_model=False
svc_model=False
xgb_model=False
gb_model=True
ensemble=True
## Dataframe split:
mode_split='challenge_1'
split_options_balanced=[100,10,3000] #number of events per class in test, validation and train splits
split_options_all=[0.2,0.01] #number of events per class in test, validation and train splits
## Feature selection
fs_gridsearch=False
fs_threshold=0.0025
## xgb
xgb_gridsearch=False
#rf
rf_gridsearch=False
## svc
svc_gridsearch=False
## gb
gb_gridsearch=True

#Feature selection
ft_mode='manual'


# Create root folder for the execution
if not os.path.exists(os.path.join(dataset_dir, 'models')):
    os.makedirs(os.path.join(dataset_dir, 'models'))
if mode=='new':
    if feature_selection:
        if ft_mode=='automatic':
            fs_label='fs'+str(fs_threshold).replace('.','')
        else:
            fs_label='fsmanual'
    else:
        fs_label=''
    print("Creating folder for the pipeline")
    root_folder = os.path.join(dataset_dir,'models','Modelling_'+mode_split+'_'+fs_label+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(root_folder)
elif mode=='modify':
    root_folder=os.path.join(dataset_dir,'models',pipeline_modelling)
    print("Modifying pipeline ",pipeline_modelling)


# Dataframe splits
## Get source dataframes:
if mode_split=='balanced':
    split_options=split_options_balanced
elif mode_split=='all':
    split_options=split_options_all
else:
    split_options=[]
if build_dataframes:
    print('Step: Building dataframes')
    df_real=pd.read_pickle(os.path.join(params['dataset_dir'],pipeline_feature_engineering,'source_dataframes/dataframe_source_real.pkl'))
    #print(df_real.shape)

    #df_aux = df_real[df_real.index.str.contains("ov1", regex=False)]
    #print(df_aux.shape)
    #df_real=df_aux

    if data_augmentation:
        df_aug=pd.read_pickle(os.path.join(params['dataset_dir'],pipeline_feature_engineering,'source_dataframes/dataframe_source_aug.pkl'))
    else:
        df_aug=df_real.iloc[0:0]
    if not os.path.exists(os.path.join(root_folder, 'datasets')):
        os.makedirs(os.path.join(root_folder, 'datasets'))
    if mode_split=='balanced':
        dataset_path = os.path.join(root_folder, 'datasets')
        df_test,df_val,df_train=get_dataframe_balanced_split(df_real, df_aug, split_options[0], split_options[1], split_options[2])
        df_test.to_pickle(os.path.join(dataset_path,'df_test.pkl'))
        df_val.to_pickle(os.path.join(dataset_path,'df_val.pkl'))
        df_train.to_pickle(os.path.join(dataset_path,'df_train.pkl'))
    elif mode_split=='all':
        df_test,df_val,df_train=get_dataframe_split(df_real, df_aug, split_options[0], split_options[1])
        dataset_path = os.path.join(root_folder, 'datasets')
        df_test.to_pickle(os.path.join(dataset_path,'df_test.pkl'))
        df_val.to_pickle(os.path.join(dataset_path,'df_val.pkl'))
        df_train.to_pickle(os.path.join(dataset_path,'df_train.pkl'))
    elif mode_split=='challenge_1':
        df_test,df_val,df_train=get_dataframe_split_challenge1(df_real, df_aug)
        dataset_path = os.path.join(root_folder, 'datasets')
        df_test.to_pickle(os.path.join(dataset_path, 'df_test.pkl'))
        df_val.to_pickle(os.path.join(dataset_path, 'df_val.pkl'))
        df_train.to_pickle(os.path.join(dataset_path, 'df_train.pkl'))
else:
    dataset_path = os.path.join(root_folder, 'datasets')



#load datasets:
df_test=pd.read_pickle(os.path.join(dataset_path, 'df_test.pkl'))
df_val=pd.read_pickle(os.path.join(dataset_path, 'df_val.pkl'))
df_train=pd.read_pickle(os.path.join(dataset_path, 'df_train.pkl'))
y_train=df_train['target']
x_train=df_train.drop(['target'],axis=1)
y_test=df_test['target']
x_test=df_test.drop(['target'],axis=1)
y_val=df_val['target']
x_val=df_val.drop(['target'],axis=1)
print(df_train.columns)
print(df_train.shape)
'''
#testing
df=pd.read_pickle(os.path.join('/home/ribanez/movidas/dcase20/dcase20_dataset/Datasets_oracle_mono_testing_2020-06-07_18-02/source_dataframes/dataframe_source_real.pkl'))
df_test = df.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=0.1))
df_train = df.drop(df_test.index)
y_test=df_test['target']
x_test=df_test.drop(['target'],axis=1)
y_train=df_train['target']
x_train=df_train.drop(['target'],axis=1)
y_val=y_test
x_val=x_test

print(df.columns)
print(df.shape)
'''
# Feature selection
if feature_selection:
    if not os.path.exists(os.path.join(root_folder, 'features_selection')):
        os.makedirs(os.path.join(root_folder, 'features_selection'))
    if ft_mode=='automatic':

        x_train,x_test,x_val,columns,rf_params=get_feature_selection(y_train,x_train,y_test,x_test,y_val,x_val,fs_gridsearch,fs_threshold,df_test)
    elif ft_mode=='manual':
        x_train,x_test,x_val,columns=get_feature_selection_manual(x_train,x_test,x_val)
    np.save(os.path.join(root_folder, 'features_selection/x_train.npy'),x_train)
    np.save(os.path.join(root_folder, 'features_selection/x_test.npy'), x_test)
    np.save(os.path.join(root_folder, 'features_selection/x_val.npy'), x_val)
    np.save(os.path.join(root_folder, 'features_selection/columns.npy'), columns)

#Models
if os.path.exists(os.path.join(root_folder, 'features_selection')):
    x_train=np.load(os.path.join(root_folder, 'features_selection/x_train.npy'),allow_pickle=True)
    x_test=np.load(os.path.join(root_folder, 'features_selection/x_test.npy'),allow_pickle=True)
    x_val=np.load(os.path.join(root_folder, 'features_selection/x_val.npy'),allow_pickle=True)
    columns=np.load(os.path.join(root_folder, 'features_selection/columns.npy'),allow_pickle=True)
if not os.path.exists(os.path.join(root_folder, 'models')):
    os.makedirs(os.path.join(root_folder, 'models'))
models_path=os.path.join(root_folder, 'models')

#xgb
if xgb_model:
    if not os.path.exists(os.path.join(models_path, 'xgb')):
        os.makedirs(os.path.join(models_path, 'xgb'))
    model_xgb=train_xgb(x_train,y_train,x_test,y_test,x_val,y_val,xgb_gridsearch)
    model_xgb.save_model(os.path.join(models_path, 'xgb/model_xgb.bin'))

#rf
if random_forest_model:
    if not os.path.exists(os.path.join(models_path, 'rf')):
        os.makedirs(os.path.join(models_path, 'rf'))
    model_rf = train_rf(x_train, y_train, x_test, y_test, x_val, y_val, rf_gridsearch)
    joblib.dump(model_rf, os.path.join(models_path, 'rf/model_rf.bin'))
#svc
if svc_model:
    if not os.path.exists(os.path.join(models_path, 'svc')):
        os.makedirs(os.path.join(models_path, 'svc'))
    model_svc = train_svc(x_train, y_train, x_test, y_test, x_val, y_val, svc_gridsearch)
    joblib.dump(model_svc, os.path.join(models_path, 'svc/model_svc.bin'))

#svc
if gb_model:
    if not os.path.exists(os.path.join(models_path, 'gb')):
        os.makedirs(os.path.join(models_path, 'gb'))
    model_gb = train_gb(x_train, y_train, x_test, y_test, x_val, y_val, gb_gridsearch)
    joblib.dump(model_gb, os.path.join(models_path, 'gb/model_gb.bin'))











# Save settings:
settings=dict()
settings['name']=root_folder
settings['mode']=mode
settings['pipeline_feature_modelling']=pipeline_modelling
settings['build_dataframes']=True
if build_dataframes:
    settings['split_mode']=mode_split
    settings['split_options']=split_options
settings['data_augmentation']=data_augmentation
settings['features_selection']=feature_selection
if feature_selection:
    settings['fs_gridserach']=fs_gridsearch
    settings['fs_threshold']=fs_threshold
settings['random_forest_model']=random_forest_model
settings['svc_model']=svc_model
settings['xgb_model']=xgb_model
settings['ensemble']=ensemble


w = csv.writer(open(os.path.join(root_folder,'settings.csv'), "w"))
for key, val in settings.items():
    w.writerow([key, val])


