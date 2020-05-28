'''
generate_audio_from_annotations.py

This file will iterate over the development dataset, and extract single-channel audio excerpts based on the annotations.
Options:
- write_file (bool): Actually generate output files and folders
- plot (bool): Plot the annotations for each file (not recommended on the whole dataset)
- debug (bool): Print complementary information on the extraction step (again, not recommended on the whole dataset)

- beamforming_mode ('beam' or 'omni')
-- 'beam' will steer a maxRE (aka maximum directivity) beam towards the source
-- 'omni' will just take the audio from the W channel

- overlap_mode ('ov1', 'ov2' or 'all'):
-- 'ov1' extracts only the audio files with ov1
-- 'ov2' extracts only the audio files with ov2
-- 'all' extracts all files regardless of the overlap amount

All files will be created in an output folder located in `params['dataset_dir']`.
The output folder will have a name according to the structure: 'oracle_mono_signals_[beamforming_mode]_[overlap_mode]
'''
from APRI.localization_detection import parse_annotations
from baseline import parameter
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from baseline.cls_feature_class import create_folder
from APRI.utils import plot_metadata, get_class_name_dict, get_mono_audio_from_event, Event
import warnings

# %% CONFIG

write_file = False
plot = False
debug = False

beamforming_mode = 'beam' # 'beam' or 'omni'
overlap_mode = 'all' # 'ov1', 'ov2' or 'all'


# %% DEFINITIONS

params = parameter.get_params()
data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev/') # path to audios
gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev') # path to annotations
fs = params['fs']
class_name_dict = get_class_name_dict()

# Main output folder for storing data
output_base_name = 'oracle_mono_signals_' + beamforming_mode + '_' + overlap_mode
output_path = os.path.join(params['dataset_dir'], output_base_name)

occurrences_per_class = np.zeros(params['num_classes'], dtype=int)



# %% ANALYSIS

# Iterate over all audio files
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]
# audio_files = ['fold1_room1_mix036_ov2.wav']# TODO REMOVE
# for audio_file_name in [audio_files[0]]:
for audio_file_name in audio_files:

    if (overlap_mode is 'all') or (overlap_mode in audio_file_name):

        print('------------------------')
        print(audio_file_name)

        # Open audio file
        b_format, sr = sf.read(os.path.join(data_folder_path, audio_file_name))

        # Get associated metadata file and load content into memory
        metadata_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
        metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)
        csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")

        # ############################################
        # # Delimite events
        event_list = parse_annotations(csv, debug)


        ############################################
        # Prepare folders
        if write_file:
            create_folder(output_path)
            for class_name in class_name_dict.values():
                folder = os.path.join(output_path, class_name)
                create_folder(folder)

        ############################################
        # Get monophonic estimates of the event, and save into files
        for event_idx, event in enumerate(event_list):

            mono_event = get_mono_audio_from_event(b_format, event, beamforming_mode, fs, params['label_hop_len_s'])

            ######################
            event_occurrence_idx = occurrences_per_class[event.get_classID()]
            mono_file_name = str(event_occurrence_idx) + '.wav'
            class_name = class_name_dict[event.get_classID()]

            ######################
            # write file
            if write_file:
                sf.write(os.path.join(output_path, class_name, mono_file_name), mono_event, sr)
            # increment counter
            occurrences_per_class[event.get_classID()] += 1


        if plot:
            plot_metadata(metadata_file_name)

        if debug:
            frames = []
            for e in event_list:
                frames.append(e.get_frames())
            plt.figure()
            plt.grid()
            for f in range(len(frames)):
                plt.plot(frames[f])
