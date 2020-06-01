'''
This file contains methods to get a prediction for event_class according to the model trained in event_class_model_training.py
and stored in 'params['dataset_dir']/models/...'
The input is an audio file  extracted from original dataset
The function:
- calculates the audio features for the audio file that are used as input in the model framework
- executed the model and return the event class

'''

import numpy as np
import joblib
import os
import essentia as es
import essentia.standard as ess
import random
from essentia.standard import MusicExtractor as ms
from APRI.utils import get_class_name_dict
from APRI.get_audio_features import *
import sys
import pandas as pd

### FUNCIONES PARA PREDICCIÓN DE EVENT_CLASS ###

def get_key(val):
    classes=get_class_name_dict()
    for key, value in classes.items():
         if val == value:
             return key

def get_features_music_extractor(audio):
    options = dict()
    options['sampleRate'] = 24000
    options['frameSize'] = 2048
    options['hopSize'] = 1024
    options['skipSilence'] = True
    audio_features, column_labels = compute_audio_features_from_audio(audio, options)
    audio_features=audio_features.reshape(-1,1)
    audio_features=pd.DataFrame(data=audio_features.T,index=['pred'],columns=[column_labels])
    return audio_features

def get_event_class_model(model_name):
    model_input_path = os.path.dirname(os.path.realpath(__file__)) + '/models/'+model_name+'/model.joblib'
    model = joblib.load(model_input_path)
    return model


def event_class_prediction(audio,model_name):
    variables=get_features_music_extractor(audio)
    model=get_event_class_model(model_name)
    names=model.steps[0][1].get_booster().feature_names
    variables=variables[names]
    if len(variables)==0:
        event_idx=event_class_prediction_random(audio)
    else:
        event_class=model.predict(variables)
        event_idx = get_key(event_class)
    return event_idx

def event_class_prediction_random(audio_path):
    return random.randint(0,13)
