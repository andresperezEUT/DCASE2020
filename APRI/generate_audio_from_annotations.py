from baseline import parameter
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from baseline.cls_feature_class import create_folder
from APRI.utils import plot_metadata, get_class_name_dict, mono_extractor, Event
import warnings

# %% CONFIG

write_file = False
plot = True
beamforming_mode = 'beam'


# %% DEFINITIONS

params = parameter.get_params()
data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev') # path to audios
gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev') # path to annotations
fs = params['fs']
class_name_dict = get_class_name_dict()

# Main output folder for storing data
output_path = os.path.join(params['dataset_dir'], 'oracle_mono_signals_'+beamforming_mode)

occurrences_per_class = np.zeros(params['num_classes'], dtype=int)



# %% ANALYSIS

# Iterate over all audio files
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]
# audio_files = ['fold1_room1_mix001_ov1.wav']# TODO REMOVE
audio_files = ['fold6_room1_mix100_ov2.wav']# TODO REMOVE
for audio_file_name in [audio_files[0]]:


    if 'ov1' in audio_file_name:
    # if True:    # TODO REMOVE

        print('------------------------')
        print(audio_file_name)

        # Open audio file
        b_format, sr = sf.read(os.path.join(data_folder_path, audio_file_name))

        # Get associated metadata file and load content into memory
        metadata_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
        metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)
        csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")


        ############################################
        # Delimite events
        #
        event_list = []
        num_rows = csv.shape[0]
        frames = csv[:,0]
        classIDs = csv[:,1]
        azis = csv[:,3]
        eles = csv[:,4]

        ######################
        # OLD: Just assume for the moment that each new class is a new event instance
        # class_change_rows =  np.append(0, (np.argwhere(classIDs[1:]-classIDs[:-1] != 0) +1 ).flatten())
        # num_events = len(class_change_rows)
        # for r in range(num_events):
        #     start_row = class_change_rows[r]
        #     end_row = class_change_rows[r+1] if r+1 < num_events else num_rows
        #     event_list.append(Event(int(classIDs[start_row]),
        #                             frames[start_row:end_row],
        #                             azis[start_row:end_row],
        #                             eles[start_row:end_row]))

        ######################
        # New version, still incomplete: each event is a continuous set of frames of the same class
        indices = (np.argwhere(frames[1:] - frames[:-1] != 1) + 1).flatten()
        start_frames = frames[indices]
        start_frames = np.insert(start_frames, 0, frames[0])

        indices = (np.argwhere(frames[1:] - frames[:-1] != 1)).flatten()
        end_frames = frames[indices]
        end_frames = np.append(end_frames, frames[-1]) # number of the last frame of the event

        ######################
        num_events = start_frames.size
        for r in range(num_events):
            print('----')
            print(r)
            start_idx = np.argwhere(frames==start_frames[r])[0,0] # index on the frames vector
            end_idx = np.argwhere(frames==end_frames[r])[0,0]

            event_classID = int(classIDs[start_idx])
            event_frames =  np.arange(int(start_frames[r]), int(end_frames[r])+1)
            event_azis =  azis[start_idx:end_idx+1]
            event_eles =  eles[start_idx:end_idx+1]
            event_list.append(Event(event_classID, event_frames, event_azis, event_eles))

            print(event_classID)
            print(event_frames)
            print(event_azis)
            print(event_eles)

        # Prepare folders
        if write_file:
            create_folder(output_path)
            for class_name in class_name_dict.values():
                folder = os.path.join(output_path, class_name)
                create_folder(folder)

        ############################################
        # Get monophonic estimates of the event, and save into files
        for event in event_list:
            frames = event.get_frames()
            w = params['label_hop_len_s'] # frame length of the annotations
            start_time_samples = int(frames[0] * w * fs)
            end_time_samples = int((frames[-1]+1) * w * fs) # add 1 here so we extend the duration a bit


            if beamforming_mode == 'omni':
                mono_event = mono_extractor(b_format[start_time_samples:end_time_samples],
                                            mode=beamforming_mode)

            elif beamforming_mode == 'beam':
                azi_frames = event.get_azis()
                ele_frames = event.get_eles()
                # frames to samples; TODO: interpolation would be cool
                num_frames = frames.size
                num_samples = int(num_frames * w * fs)
                samples_per_frame = int(w*fs)

                azi_samples = np.zeros(num_samples)
                ele_samples = np.zeros(num_samples)
                for idx in range(num_frames):
                    azi_samples[(idx*samples_per_frame):(idx+1)*samples_per_frame ] = azi_frames[idx]
                    ele_samples[(idx*samples_per_frame):(idx+1)*samples_per_frame ] = ele_frames[idx]

                mono_event = mono_extractor(b_format[start_time_samples:end_time_samples],
                                            azis=azi_samples*np.pi/180, # deg2rad
                                            eles=ele_samples*np.pi/180, # deg2rad
                                            mode=beamforming_mode)

            else:
                warnings.warn('MONO METHOD NOT KNOWN"', UserWarning)

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



#
# # %% GET NUMBER OF SOURCES PER FRAME
#
# L = 600
# num_sources = np.zeros(L)
# csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")
# frame_indices = csv[:,0]
# for frame_idx in frame_indices:
#     num_sources[int(frame_idx)] += 1
#
# plt.figure()
# plt.plot(num_sources)
#
# # %%
#
# # 1) locate region between zeros
# num_sources==0
# frames_with_zero_source = np.argwhere(num_sources==0).flatten()
# indices = (np.argwhere(frames_with_zero_source[1:]-frames_with_zero_source[:-1] != 1) +1).flatten()
# transitions_to_zero = frames_with_zero_source[indices]
# R = transitions_to_zero.size
# region_starts = np.zeros(R)
# region_ends = np.zeros(R)
# for idx in range(R-1):
#     region_starts[idx+1] = transitions_to_zero[idx]
#     region_ends[idx] = transitions_to_zero[idx]
# region_ends[-1] = transitions_to_zero[-1]
#
#
# # 2) check within regions
#
# for r in range(R):
#     # remove zeros at the beginning
#     num