"""
pipeline_feature_engineering.py

This script executes the pipeline from mono events to dataframes for modelling.
The pipeline can be tuned by modifying the parameters in the top of the script.
Steps:
- Parameters
- (Optional) Data Augmentation
- (Optional) Extra events: events obtained from previous executions
- Audio features extraction from each event dataset (real, augmented, extra)
- Creation of dataframes (real, augmented, extra)
"""

import os, datetime
from baseline import parameter
from APRI.training_batch_data_augmentation import *
from APRI.training_batch_generate_audio_features import *
from APRI.get_dataframes import *
from APRI.localization_detection import *
import pickle
import soundfile as sf
from APRI.utils import plot_metadata, get_class_name_dict, get_mono_audio_from_event, Event
from baseline.cls_feature_class import create_folder
import time

# Import general parametes
params = parameter.get_params()
dataset_dir= os.path.join(params['dataset_dir'])

# Parameters
mode='new' # new or modify
pipeline='Datasets_2020-06-05_22-15' #if mode is 'modify'
#original_event_dataset='oracle_mono_testing'

extra_events=False
data_augmentation=True
audio_parameters_real=True
audio_parameters_aug=True
audio_parameters_extra=False
creating_dataframe_real=True
creating_dataframe_aug=True
creating_dataframe_extra=False

## Data augmentation parameters:
def get_data_augmentation_parameters():
    aug_options=dict()
    ### White noise
    aug_options['white_noise']=True
    aug_options['noise_rate']=[0.02]
    ### Time stretching
    aug_options['time_stretching']=True
    aug_options['rates']=[0.7,1.3]
    ### Pitch shifting
    aug_options['pitch_shifting']=True
    aug_options['steps']=[-2,2]
    ### Time shifting
    aug_options['time_shifting']=True
    ### Ir
    aug_options['ir_merge']=True
    aug_options['ir_set']= os.path.dirname(os.path.realpath(__file__)) + '/IR'
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
    root_folder = os.path.join(dataset_dir, 'Datasets_'+'foa_dev'+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(root_folder)
elif mode=='modify':
    root_folder=os.path.join(dataset_dir,pipeline)
    print("Modifying pipeline "+pipeline)



# Events oracle
data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev')  # path to audios
gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev')  # path to annotations
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]
events_folder=os.path.join(root_folder,'mono_events')
class_name_dict=get_class_name_dict()
fs = params['fs']
window = params['window']
window_size = params['window_size']
window_overlap = params['window_overlap']
nfft = params['nfft']
D = params['D'] # decimate factor
frame_length = params['label_hop_len_s']
debug=False
beamforming_mode='omni'
occurrences_per_class = np.zeros(params['num_classes'], dtype=int)
if not os.path.exists(events_folder):
    os.makedirs(events_folder)
for class_name in class_name_dict.values():
    folder = os.path.join(events_folder, class_name)
    create_folder(folder)
for audio_file_idx, audio_file_name in enumerate(audio_files):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print("{}: {}, {}".format(audio_file_idx, st, audio_file_name))
    ############################################
    # Open file
    audio_file_path = os.path.join(data_folder_path, audio_file_name)
    b_format, sr = sf.read(audio_file_path)
    # Get spectrogram
    stft = compute_spectrogram(b_format, sr, window, window_size, window_overlap, nfft, D)
    ############################################
    metadata_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
    metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)
    csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")
    event_list = parse_annotations(csv, debug)
    num_events = len(event_list)
    for event_idx in range(num_events):
        event = event_list[event_idx]
        mono_event = get_mono_audio_from_event(b_format, event, beamforming_mode, fs, frame_length)
        event_occurrence_idx = occurrences_per_class[event.get_classID()]
        mono_file_name = str(event_occurrence_idx) + '.wav'
        class_name = class_name_dict[event.get_classID()]
        sf.write(os.path.join(events_folder, class_name, os.path.splitext(audio_file_name)[0] + '_' + mono_file_name), mono_event, sr)
        occurrences_per_class[event.get_classID()] += 1
# Data Augmentation
if data_augmentation:
    print("Step: data augmentation")
    aug_folder=os.path.join(root_folder,'data_augmentation')
    aug_options=get_data_augmentation_parameters()
    training_batch_data_augmentation(os.path.join(root_folder,'mono_events'), aug_folder, aug_options,params)



# Audio features
if audio_parameters_real:
    print("Step: extracting features for real events")
    afr_folder=os.path.join(root_folder,'audio_features_real')
    af_options=get_audio_features_options()
    training_batch_generate_audio_features(os.path.join(root_folder,'mono_events'),afr_folder,af_options)
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
    print('Step: generating source dataframe augmented ')
    input_path=os.path.join(root_folder,'audio_features_aug')
    output_path=os.path.join(root_folder,'source_dataframes')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_aug=get_source_dataframes(input_path)
    df_aug.to_pickle(os.path.join(output_path,'dataframe_source_aug.pkl'))
if creating_dataframe_extra:
    print('Step: generating source dataframe extra')
    input_path=os.path.join(root_folder,'audio_features_extra')
    output_path=os.path.join(root_folder,'source_dataframes')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_extra=get_source_dataframes(input_path,True)
    df_extra.to_pickle(os.path.join(output_path,'dataframe_source_extra.pkl'))
