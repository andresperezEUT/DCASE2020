from baseline import parameter
import os
import soundfile as sf
import numpy as np

from baseline.cls_feature_class import create_folder
# from .utils import Event, get_class_name_dict


###################################

class Event:
    def __init__(self, classID, frames, azis, eles):
        self._classID = classID
        self._frames = frames
        self._azis = azis
        self._eles = eles

    def get_classID(self):
        return self._classID

    def get_frames(self):
        return self._frames

    def get_azis(self):
        return self._azis

    def get_eles(self):
        return self._eles



def get_class_name_dict():
    return {
        0: 'alarm',
        1: 'crying_baby',
        2: 'crash',
        3: 'barking_dog',
        4: 'running_engine',
        5: 'female_scream',
        6: 'female_speech',
        7: 'burning_fire',
        8: 'footsteps',
        9: 'knocking_on_door',
        10:'male_scream',
        11:'male_speech',
        12:'ringing_phone',
        13:'piano'
    }

###################################



params = parameter.get_params()
data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev') # path to audios
gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev') # path to annotations
fs = params['fs']
class_name_dict = get_class_name_dict()

occurrences_per_class = np.zeros(params['num_classes'], dtype=int)

# Iterate over all audio files
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]
for audio_file_name in audio_files:


    # TODO: for the moment, only check the overlap 1 cases
    if 'ov1' in audio_file_name:

        print('------------------------')
        print(audio_file_name)

        # Open audio file
        b_format, sr = sf.read(os.path.join(data_folder_path, audio_file_name))

        # Get associated metadata file and load content into memory
        metadata_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
        metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)
        csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")

        # TODO: Just assume for the moment that each new class is a new event instance
        # Delimite events by new class indices
        event_list = []
        num_rows = csv.shape[0]
        frames = csv[:,0]
        classIDs = csv[:,1]
        azis = csv[:,3]
        eles = csv[:,4]

        class_change_rows =  np.append(0, (np.argwhere(classIDs[1:]-classIDs[:-1] != 0) +1 ).flatten())
        num_events = len(class_change_rows)
        for r in range(num_events):
            start_row = class_change_rows[r]
            end_row = class_change_rows[r+1] if r+1 < num_events else num_rows
            event_list.append(Event(int(classIDs[start_row]),
                                    frames[start_row:end_row],
                                    azis[start_row:end_row],
                                    eles[start_row:end_row]))

        # Prepare folders
        output_path = params['oracle_mono_signals']
        create_folder(output_path)
        for class_name in class_name_dict.values():
            folder = os.path.join(output_path, class_name)
            create_folder(folder)

        # Get monophonic estimates of the event, and save into files
        # TODO: just pick W channel for the moment
        for event in event_list:
            frames = event.get_frames()
            w = params['label_hop_len_s'] # frame length of the annotations
            start_time_samples = int(frames[0] * w * fs)
            end_time_samples = int((frames[-1]+1) * w * fs) # add 1 here so we extend the duration a bit
            mono_event = b_format[start_time_samples:end_time_samples, 0]

            event_occurrence_idx = occurrences_per_class[event.get_classID()]
            mono_file_name = str(event_occurrence_idx) + '.wav'
            class_name = class_name_dict[event.get_classID()]

            # write file
            sf.write(os.path.join(output_path, class_name, mono_file_name), mono_event, sr)
            # increment counter
            occurrences_per_class[event.get_classID()] += 1

            print(class_name, mono_file_name)
