'''
Generates extra audio files using data augmentation techniques
'''



import librosa
import numpy as np
import os
from APRI.get_audio_features import *
from APRI.utils import get_class_name_dict


params = parameter.get_params()
event_type= get_class_name_dict().values()
data_folder_path = os.path.join(params['dataset_dir'], 'oracle_mono_signals_beam_all/') # path to audios
audio_augmented_output_path= os.path.join(params['dataset_dir'], 'oracle_mono_signals_beam_all_aug/') # path to audios

def save_audio_file(output_path, original_name,event, suffix, data):
    path=os.path.join(output_path,event,original_name+suffix+'.wav') #path to file
    librosa.output.write_wav(path, data, 24000, norm=False)
def load_audio_file(file_path):
    data = librosa.load(file_path,sr=None)[0]
    return data

#adding white noise
def adding_white_noise(data):
    wn = np.random.randn(len(data))
    data = data + 0.01 * wn
    return data

#time stretching
def stretch_audio(data, rate=1):
    data = librosa.effects.time_stretch(data, rate)
    return data

#pitch shifting
def pitch_shifting(data,steps=1):
    data=librosa.effects.pitch_shift(data, 24000, steps, bins_per_octave=12, res_type='kaiser_best')
    return data

#time shifting
def time_shifting(data,shift):
    data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        data[:shift] = 0
    else:
        data[shift:] = 0
    return data


for event in event_type:
    create_folder(os.path.join(audio_augmented_output_path, event))
    audio_path = os.path.join(data_folder_path, event)  # path to file
    i=0
    for audio in os.scandir(audio_path):
        original_name = os.path.splitext(audio.name)[0]
        data=load_audio_file(audio.path)
        data_wn=adding_white_noise(data)
        data_sa09=stretch_audio(data,0.8)
        data_sa11 = stretch_audio(data, 1.2)
        data_psd=pitch_shifting(data,-1)
        data_psu=pitch_shifting(data,1)
        data_ts=time_shifting(data,round(len(data)/2))
        save_audio_file(audio_augmented_output_path,original_name,event,'_wn',data_wn)
        save_audio_file(audio_augmented_output_path, original_name, event, '_sa08', data_sa09)
        save_audio_file(audio_augmented_output_path, original_name, event, '_sa12', data_sa11)
        save_audio_file(audio_augmented_output_path, original_name, event, '_psd', data_psd)
        save_audio_file(audio_augmented_output_path, original_name, event, '_psu', data_psu)
        save_audio_file(audio_augmented_output_path, original_name, event, '_ts', data_ts)
        if i%100==0:
            print('DA. Processing: ',str(event),' ',i,' completed.')
        i+=1