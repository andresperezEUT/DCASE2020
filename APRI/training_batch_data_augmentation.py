"""
training_batch_data_augmentation.py

Given a event dataset, this script generates augmented data.
The number and parameters of the transformed files can be modified by using the parameters on the top of this script.
Output files are stored for model training purpose in a folder (input) and are ordered by event class.
"""


# Dependencies
import os
from APRI.get_data_augmentation import *
from APRI.utils import get_class_name_dict


# Auxiliar
def save_audio_file(output_path, original_name, event, suffix, data):
    path = os.path.join(output_path, event, original_name + suffix + '.wav')  # path to file
    librosa.output.write_wav(path, data, 24000, norm=False)


def load_audio_file(file_path):
    data = librosa.load(file_path, sr=24000)[0]
    return data


def training_batch_data_augmentation(input_path,output_folder,aug_options,params):
    event_type= get_class_name_dict().values()
    #input_path = os.path.join(params['dataset_dir'],original_dataset)
    output_path= output_folder

    # Create modified audio files
    for event in event_type:
        if not os.path.exists(os.path.join(output_path, event)):
            os.makedirs(os.path.join(output_path, event))
        audio_path = os.path.join(input_path, event)
        i=0
        for audio in os.scandir(audio_path):
            original_name = os.path.splitext(audio.name)[0]
            data=load_audio_file(audio.path)
            modified_audios,modified_audios_names=compute_data_augmentation(data,aug_options)
            for i in range(len(modified_audios)):
                save_audio_file(output_path, original_name, event, modified_audios_names[i], modified_audios[i])
            if i%100==0:
                print('DA. Processing: ',str(event),' ',i,' completed.')
            i+=1
