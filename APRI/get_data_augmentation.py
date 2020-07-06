"""
get_audio_features.py

This script contains methods for data augmentation. Thus, this script is used just for training purpose.
The modified audios are generated applying different transformations. Default parameters:
aug_options:
    'white_noise'=True    'noise_rate'=0.01
    'time_stretching'=True    'rates'=[0.8,1.2]
    'pitch_shifting']=True    'steps'=[-1,1]
    'time_shifting'=True

compute_data_augmentation() is used in train circuit and takes as inputs:
- audio: original file in .wav format
- aug_options: parameters for data augmentation
and outputs:
- modified files according to the input parameters

"""
# Dependencies
import librosa
import numpy as np
import random
import soundfile as sf
import scipy
import os

# Compute data augmentation
## Add white noise
def adding_white_noise(data,noise_rate):
    wn = np.random.randn(len(data))
    data = data + noise_rate * wn
    return data
## Stretching time
def stretch_audio(data, rate=1):
    data = librosa.effects.time_stretch(data, rate)
    return data
## pitch shifting
def pitch_shifting(data,steps=1):
    data=librosa.effects.pitch_shift(data, 24000, steps, bins_per_octave=12, res_type='kaiser_best')
    return data
## time shifting
def time_shifting(data,shift):
    data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        data[:shift] = 0
    else:
        data[shift:] = 0
    return data
## ir merging
def ir_merge(data,ir_path):
    file_name=str(random.randint(0, 9))+'.wav'
    h, sr_h = sf.read(os.path.join(ir_path,file_name))
    assert sr_h == 24000  # sample rates should match
    data = scipy.signal.fftconvolve(data, h)
    return data

# get augmented audios
def compute_data_augmentation(audio,aug_options):
    modified_audios=[]
    modified_audios_names=[]
    if aug_options['white_noise']:
        data_wn1 = adding_white_noise(audio,aug_options['noise_rate'][0])
        modified_audios.append(data_wn1)
        modified_audios_names.append('_wn1')
    if aug_options['time_stretching']:
        data_sadown = stretch_audio(audio,aug_options['rates'][0])
        data_saup = stretch_audio(audio,aug_options['rates'][1])
        modified_audios.append(data_sadown)
        modified_audios_names.append('_sa'+str(aug_options['rates'][0]))
        modified_audios.append(data_saup)
        modified_audios_names.append('_sa'+str(aug_options['rates'][1]))
    if aug_options['pitch_shifting']:
        data_psd = pitch_shifting(audio, aug_options['steps'][0])
        data_psu = pitch_shifting(audio, aug_options['steps'][1])
        modified_audios.append(data_psd)
        modified_audios_names.append('_ps(' + str(aug_options['steps'][0])+')')
        modified_audios.append(data_psu)
        modified_audios_names.append('_ps(' + str(aug_options['steps'][1])+')')
    if aug_options['time_shifting']:
        data_ts = time_shifting(audio,round(len(audio) / 2))
        modified_audios.append(data_ts)
        modified_audios_names.append('_ts')
    if aug_options['ir_merge']:
        data_ir = ir_merge(audio,aug_options['ir_set'])
        modified_audios.append(data_ir)
        modified_audios_names.append('_ir')
    return modified_audios,modified_audios_names

