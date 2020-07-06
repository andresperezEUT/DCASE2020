'''
This file contains methods to get a prediction for event_class according to the model trained in event_class_model_training.py
and stored in 'params['dataset_dir']/models/...'
The input is an audio file  extracted from original dataset
The method get_features music_extractor:
- calculates the audio features for the audio file that are used as input in the model framework
The method event_class_prediction:
- executes the model and return the event class for a given audio file

'''

import joblib
import random
from APRI.utils import get_class_name_dict
from APRI.get_audio_features import *
import pandas as pd
import os



# get event key from event descriptor
def get_key(val):
    classes=get_class_name_dict()
    for key, value in classes.items():
         if val == value:
              return key

# get audio features for a given audio file
def get_features_music_extractor(audio):
    options = dict()
    options['sampleRate'] = 24000
    options['frameSize'] = 2048
    options['hopSize'] = 1024
    options['skipSilence'] = True
    audio=np.float32(audio)
    audio_features, column_labels = compute_audio_features(audio, options)
    audio_features=audio_features.reshape(-1,1)
    audio_features=pd.DataFrame(data=audio_features.T,index=['pred'],columns=[column_labels])
    return audio_features

# load trained model and features used by the model
def get_event_class_model(model_name):
    if 'xgb' in model_name:
        model_input_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name + '/model.bin'
        model = xgb.Booster()
        model.load_model(model_input_path)
        try:
            columns=np.load(os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name + '/columns.npy',allow_pickle=True)
            feat_sel=True
        except:
            columns=()
            feat_sel=False
    else:
        model_input_path = os.path.dirname(os.path.realpath(__file__)) + '/models/'+model_name+'/model.bin'
        model = joblib.load(model_input_path)
        try:
            columns=np.load(os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name + '/columns.npy',allow_pickle=True)
            feat_sel=True
        except:
            columns=()
            feat_sel=False
    return model,columns,feat_sel

# returns event class key for a given audio file
def event_class_prediction(audio,model_name):
    model,columns,feat_sel = get_event_class_model(model_name)
    if 'xgb' in model_name:
        variables=get_features_music_extractor(audio)
        if feat_sel:
            variables=variables[[columns]]
        variables=xgb.DMatrix(variables)
        event_class = model.predict(variables)
        class_idx=int(event_class)
    else:
        variables=get_features_music_extractor(audio)
        if feat_sel:
            variables=variables[[columns]]
        event_class = model.predict(variables)
        class_idx=int(event_class)
    return class_idx

# random prediction of event class
def event_class_prediction_random(audio_path):
    return random.randint(0,13)

