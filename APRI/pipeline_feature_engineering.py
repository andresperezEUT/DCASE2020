'''
This script executes the pipeline from mono events to dataframes for modelling.
The pipeline can be tuned by modifying the parameters in the top of the script.
Steps:
- Parameters
- (Optional) Data Augmentation
- (Optional) Extra events: events obtained from previous executions
- Audio features extraction from each event dataset (real, augmented, extra)
- Creation of dataframes (real, augmented, extra)

'''

import os, datetime
from baseline import parameter
from APRI.training_batch_data_augmentation import *
from APRI.training_batch_generate_audio_features import *
from APRI.get_dataframes import *
import pickle

# Import general parametes
params = parameter.get_params()
dataset_dir= os.path.join(params['dataset_dir'])

# Parameters
mode='new' # new or modify
pipeline='Datasets_2020-06-05_22-15' #if mode is 'modify'
original_event_dataset='oracle_mono_testing'

extra_events=False
data_augmentation=False
audio_parameters_real=True
audio_parameters_aug=False
audio_parameters_extra=False
creating_dataframe_real=True
creating_dataframe_aug=False
creating_dataframe_extra=False

## Data augmentation parameters:
def get_data_augmentation_parameters():
    aug_options=dict()
    ### White noise
    aug_options['white_noise']=True
    aug_options['noise_rate']=0.01
    ### Time stretching
    aug_options['time_stretching']=True
    aug_options['rates']=[0.8,1.2]
    ### Pitch shifting
    aug_options['pitch_shifting']=True
    aug_options['steps']=[-1,1]
    ### Time shifting
    aug_options['time_shifting']=True
    return aug_options
## Audio features parameters
def get_audio_features_options():
    options = dict()
    options['sampleRate'] = 24000
    options['frameSize'] = 2048
    options['hopSize'] = 1024
    options['skipSilence'] = True
    return options

# Create root folder for the execution
if mode=='new':
    print("Creating folder for the pipeline")
    root_folder = os.path.join(dataset_dir, 'Datasets_'+original_event_dataset+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(root_folder)
elif mode=='modify':
    root_folder=os.path.join(dataset_dir,pipeline)
    print("Modifying pipeline "+pipeline)

# Data Augmentation
if data_augmentation:
    print("Step: data augmentation")
    aug_folder=os.path.join(root_folder,'data_augmentation')
    aug_options=get_data_augmentation_parameters()
    training_batch_data_augmentation(original_event_dataset, aug_folder, aug_options,params)

# Audio features
if audio_parameters_real:
    print("Step: extracting features for real events")
    afr_folder=os.path.join(root_folder,'audio_features_real')
    af_options=get_audio_features_options()
    training_batch_generate_audio_features(os.path.join(dataset_dir,original_event_dataset),afr_folder,af_options)
if audio_parameters_aug:
    print("Step: extracting features for augmented events")
    afa_folder=os.path.join(root_folder,'audio_features_aug')
    data_augmentation_folder=os.path.join(root_folder,'data_augmentation')
    af_options=get_audio_features_options()
    training_batch_generate_audio_features(data_augmentation_folder,afa_folder,af_options)
if extra_events:
    if audio_parameters_extra:
        afe_folder=os.path.join(root_folder,'audio_features_extra')
        if os.path.isdir(os.path.join(dataset_dir,original_event_dataset+'_extra')):
            print("Step:extracting features for extra events")
            extra_events_folder=os.path.join(dataset_dir,original_event_dataset+'_extra')
            af_options = get_audio_features_options()
            training_batch_generate_audio_features(afe_folder, extra_events_folder, af_options,params,True)
        else:
            print('INFO: The folder for extra events does not exist. The folder should be in the same path that the dataset folder, with the smae name +"_extra"')

# Creating datasets
if creating_dataframe_real:
    print('Step: generating source dataframe real')
    input_path=os.path.join(root_folder,'audio_features_real')
    output_path=os.path.join(root_folder,'source_dataframes')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_real=get_source_dataframes(input_path)
    df_real.to_pickle(os.path.join(output_path,'dataframe_source_real.pkl'))


if creating_dataframe_aug:
    print('Step: generating source dataframe ')
    input_path=os.path.join(root_folder,'audio_features_aug')
    output_path=os.path.join(root_folder,'source_dataframes')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_aug=get_source_dataframes(input_path)
    df_aug.to_pickle(os.path.join(output_path,'dataframe_source_aug.pkl'))
if creating_dataframe_extra:
    print('Step: generating source dataframe real')
    input_path=os.path.join(root_folder,'audio_features_extra')
    output_path=os.path.join(root_folder,'source_dataframes')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_extra=get_source_dataframes(input_path,True)
    df_extra.to_pickle(os.path.join(output_path,'dataframe_source_extra.pkl'))
