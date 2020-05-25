'''
This file contains functions to get a prediction for event_class according to the model trained in event_class_model_training.py
and stored in 'params['dataset_dir']/models/event_class/'
The input is an audio file  extracted from original dataset
The function:
- calculates the audio features for the audio file that are used as input in the model framework
- executed the model and return the event class

'''

from baseline import parameter
import os
import numpy as np
import pandas as pd
import joblib
import os
model_input_path = os.path.dirname(os.path.realpath(__file__))+'/models/event_class/model.joblib'
import essentia.standard as es



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

def get_features_music_extractor(audio_path):
    print("Extracting features...")
    features, features_frames = es.MusicExtractor(lowlevelFrameSize=4096,
                                                  lowlevelHopSize=2048,
                                                  tonalFrameSize=4096,
                                                  tonalHopSize=2048,
                                                  rhythmStats=["mean", "var", "dmean"],
                                                  lowlevelStats=["mean", "var", "dmean"],
                                                  )(audio_path)
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
    return audio_features.tolist()


def event_class_prediction(audio_path):
    variables=get_features_music_extractor(audio_path)
    model = joblib.load(model_input_path)
    event_class=model.predict(np.array([variables]))
    return event_class


### EJEMPLO DE PREDICCIÓN DE UN AUDIO ###

audio_path='/home/ribanez/movidas/dcase20/dcase20_dataset/oracle_mono_signals/male_scream/4.wav'
print(event_class_prediction(audio_path))





