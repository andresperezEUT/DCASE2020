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
import sys

### FUNCIONES PARA PREDICCIÓN DE EVENT_CLASS ###

def get_feature_list():
    return [
         'lowlevel.barkbands.dmean',
         'lowlevel.barkbands.mean',
         'lowlevel.barkbands.var',
         'lowlevel.erbbands.dmean',
         'lowlevel.erbbands.mean',
         'lowlevel.erbbands.var',
         'lowlevel.gfcc.mean',
         'lowlevel.melbands.dmean',
         'lowlevel.melbands.mean',
         'lowlevel.melbands.var',
         'lowlevel.mfcc.mean',
         'lowlevel.spectral_contrast_coeffs.dmean',
         'lowlevel.spectral_contrast_coeffs.mean',
         'lowlevel.spectral_contrast_coeffs.var',
         'lowlevel.spectral_contrast_valleys.dmean',
         'lowlevel.spectral_contrast_valleys.mean',
         'lowlevel.spectral_contrast_valleys.var',
         'rhythm.beats_loudness_band_ratio.dmean',
         'rhythm.beats_loudness_band_ratio.mean',
         'rhythm.beats_loudness_band_ratio.var',
         'tonal.hpcp.dmean',
         'tonal.hpcp.mean',
         'tonal.hpcp.var',
         'tonal.chords_histogram',
         'tonal.thpcp',
         'lowlevel.pitch_salience.dmean',
         'lowlevel.pitch_salience.mean',
         'lowlevel.pitch_salience.var',
         'lowlevel.silence_rate_20dB.dmean',
         'lowlevel.silence_rate_20dB.mean',
         'lowlevel.silence_rate_20dB.var',
         'lowlevel.silence_rate_30dB.dmean',
         'lowlevel.silence_rate_30dB.mean',
         'lowlevel.silence_rate_30dB.var',
         'lowlevel.silence_rate_60dB.dmean',
         'lowlevel.silence_rate_60dB.mean',
         'lowlevel.silence_rate_60dB.var'
    ]

def get_key(val):
    classes=get_class_name_dict()
    for key, value in classes.items():
         if val == value:
             return key

def get_features_music_extractor(audio):
    #TODO: set verbose parameter to false
    #TODO: add parameter 'analysisSampleRate=24000' in order to speed up process
    # In MusicExtractor.compute: Parameter sampleRate = 24000 is not within specified range: {32000,44100,48000}
    try:
        features, features_frames = ms(
                                                  analysisSampleRate=44100,
                                                  chromaprintCompute=True,
                                                  lowlevelFrameSize=4096,
                                                  lowlevelHopSize=2048,
                                                  tonalFrameSize=4096,
                                                  tonalHopSize=2048,
                                                  rhythmStats=["mean", "var", "dmean"],
                                                  lowlevelStats=["mean", "var", "dmean"],
                                                  )(audio)
        feature_list = get_feature_list()
        audio_features = []
        for feature in feature_list:
            x = features[feature]
            if type(x) is float:
                x = np.array(x)
                y = [x]
            else:
                y = x.tolist()
            audio_features = audio_features + y
        audio_features = np.array(audio_features)
        audio_features.tolist()
    except:
        audio_features=[]
        print('MusicExtractor returns an error. Setting random class...')
    return audio_features

def get_event_class_model(model_name):
    #TODO: avoid load the model for each audio file
    model_input_path = os.path.dirname(os.path.realpath(__file__)) + '/models/'+model_name+'/model.joblib'
    model = joblib.load(model_input_path)
    return model


def event_class_prediction(audio,model_name):

    variables=get_features_music_extractor(audio)
    model=get_event_class_model(model_name)
    if len(variables)==0:
        event_idx=event_class_prediction_random(audio)
    else:
        event_class=model.predict(np.array([variables]))
        event_idx = get_key(event_class)
    return event_idx

def event_class_prediction_random(audio_path):
    return random.randint(0,13)
