from baseline import parameter
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from baseline.cls_feature_class import create_folder
from APRI.utils import plot_metadata, get_class_name_dict, mono_extractor, Event
import warnings

# %% CONFIG

write_file = True
plot = False
debug = False
beamforming_mode = 'beam' # or 'omni'


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
# audio_files = ['fold1_room1_mix036_ov2.wav']# TODO REMOVE
# for audio_file_name in [audio_files[0]]:
for audio_file_name in audio_files:

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
    instances = csv[:,2]
    azis = csv[:,3]
    eles = csv[:,4]


    # source count
    N = 600

    num_sources = np.zeros(N)
    for frame_idx in frames:
        num_sources[int(frame_idx)] += 1

    current_events = [] # maximum two allowed

    for frame_idx, frame in enumerate(frames[:-1]):
        if debug:
            print(frame_idx, frame)
        frame = int(frame)

        if frame_idx == 0:
            last_frame = -1 # avoid problem when starting with 2 sources
        else:
            last_frame = int(frames[frame_idx-1])

        if (frame - last_frame) > 1:
        # if num_sources[frame] == 0:
            # clear list of current events
            if debug:
                print('finish all')
            while (len(current_events)!=0):
                event_list.append(current_events[-1])
                current_events.remove(current_events[-1])

        if num_sources[frame] == 1:
            # if last was 0, first just started
            if num_sources[frame-1] == 0:
                classID = int(classIDs[frame_idx])
                instance = instances[frame_idx]
                frameNumber = [frame]
                azi = [azis[frame_idx]]
                ele = [eles[frame_idx]]
                e = Event(classID, instance, frameNumber, azi, ele)
                if debug:
                    print('new event---1')
                    e.print()
                current_events.append(e)

            # if last was 1, continue as before
            elif num_sources[frame-1] == 1:

                # ensure that last event was same as this one
                classID = int(classIDs[frame_idx])
                instance = instances[frame_idx]
                e = current_events[0]
                if classIDs[frame_idx] == e.get_classID() and instances[frame_idx] == e.get_instance():
                    # it is same: just proceed normal
                    e = current_events[0]
                    e.add_frame(frame)
                    e.add_azi(azis[frame_idx])
                    e.add_ele(eles[frame_idx])
                else:
                    # instantaneous change: remove last and add new
                    # print('instantaneous change!')
                    e = current_events[0]
                    event_list.append(e)
                    current_events.remove(e)
                    # add new
                    frameNumber = [frame]
                    azi = [azis[frame_idx]]
                    ele = [eles[frame_idx]]
                    e = Event(classID, instance, frameNumber, azi, ele)
                    if debug:
                        print('new event---1')
                        e.print()
                    current_events.append(e)


            # if last was 2, second source just finished
            elif num_sources[frame-1] == 2:
                if debug:
                    print('finish event')
                classID = int(classIDs[frame_idx])
                instance = instances[frame_idx]
                frameNumber = [frame]
                azi = [azis[frame_idx]]
                ele = [eles[frame_idx]]
                # find which of the current events are we: same classID and instance number
                class0 = current_events[0].get_classID()
                class1 = current_events[1].get_classID()
                instance0 = current_events[0].get_instance()
                instance1 = current_events[1].get_instance()
                if classID == class0 and instance == instance0:
                    event_idx = 0
                elif classID == class1 and instance == instance1:
                    event_idx = 1
                else:
                    warnings.warn('something wird happen!')
                    continue
                # first add current event data, as regular
                e = current_events[event_idx]
                e.add_frame(frame)
                e.add_azi(azis[frame_idx])
                e.add_ele(eles[frame_idx])
                # then remove other event and add it to the main list
                event_idx = np.mod(event_idx+1,2)
                e = current_events[event_idx]
                event_list.append(e)
                current_events.remove(e)

        elif num_sources[frame] == 2:

            # check cold start: 2 starting at the same time
            if last_frame < frame - 1:
                # just add it normal
                classID = int(classIDs[frame_idx])
                instance = instances[frame_idx]
                frameNumber = [frame]
                azi = [azis[frame_idx]]
                ele = [eles[frame_idx]]
                e = Event(classID, instance, frameNumber, azi, ele)
                if debug:
                    print('new event---1')
                    e.print()
                current_events.append(e)

            else:
                # if last was 1, second just started
                if num_sources[frame - 1] == 1:


                    if len(current_events) == 2:
                        # 1 to 2 instantaneous change!
                        last_classID = int(classIDs[frame_idx-1])
                        last_instance = instances[frame_idx-1]
                        event0 = current_events[0]
                        event1 = current_events[1]
                        if event0.get_classID() == last_classID and event0.get_instance() == last_instance:
                            # remove the other one
                            event_list.append(event1)
                            current_events.remove(event1)
                        else:
                            # remove this one
                            event_list.append(event0)
                            current_events.remove(event0)
                        # now add the new one normal
                        classID = int(classIDs[frame_idx])
                        instance = instances[frame_idx]
                        frameNumber = [frame]
                        azi = [azis[frame_idx]]
                        ele = [eles[frame_idx]]
                        e = Event(classID, instance, frameNumber, azi, ele)
                        if debug:
                            print('new event---2')
                            e.print()
                        current_events.append(e)


                    elif len(current_events) == 1:
                        e = current_events[0]
                        # if same class as existing event, it's continuation
                        # if classIDs[frame_idx] == e.get_classID() and instances[frame_idx] == 0:
                        if classIDs[frame_idx] == e.get_classID() and instances[frame_idx] == e.get_instance():
                            # add normally
                            e.add_frame(frame)
                            e.add_azi(azis[frame_idx])
                            e.add_ele(eles[frame_idx])
                        # if not same class and instance, it's a new event
                        else:
                            classID = int(classIDs[frame_idx])
                            instance = instances[frame_idx]
                            frameNumber = [frame]
                            azi = [azis[frame_idx]]
                            ele = [eles[frame_idx]]
                            e = Event(classID, instance, frameNumber, azi, ele)
                            if debug:
                                print('new event---2')
                                e.print()
                            current_events.append(e)

                    else:
                        warnings.warn('something wird happen, v2!')
                        continue

                # if last was 2, continue
                # elif num_sources[frame - 1] == 2:
                elif num_sources[last_frame] == 2:

                    if len(current_events) == 2:
                        # we have two active sources, so find the one that corresponds by class nummber
                        classID = int(classIDs[frame_idx])
                        instance = instances[frame_idx]
                        found = False
                        for e_idx, e in enumerate(current_events):
                            if e.get_classID() == classID and e.get_instance() == instance:
                                found = True
                                e.add_frame(frame)
                                e.add_azi(azis[frame_idx])
                                e.add_ele(eles[frame_idx])
                                last_used_current_event_idx = e_idx
                        if not found:
                            # instantaneous change: remove one current event and add another one
                            # which to remove? the one with different class that our last one, or different instance
                            # if current_events[0].get_classID() != current_events[1].get_classID():
                            e = current_events[np.mod(last_used_current_event_idx+1,2)]
                            event_list.append(e)
                            current_events.remove(e)

                            classID = int(classIDs[frame_idx])
                            instance = instances[frame_idx]
                            frameNumber = [frame]
                            azi = [azis[frame_idx]]
                            ele = [eles[frame_idx]]
                            e = Event(classID, instance, frameNumber, azi, ele)
                            if debug:
                                print('new event---2')
                                e.print()
                            current_events.append(e)

                    elif len(current_events) == 1:
                        # it might happen with a cold start from 0 to 2
                        # so add the current source
                        classID = int(classIDs[frame_idx])
                        instance = instances[frame_idx]
                        frameNumber = [frame]
                        azi = [azis[frame_idx]]
                        ele = [eles[frame_idx]]
                        e = Event(classID, instance, frameNumber, azi, ele)
                        if debug:
                            print('new event---2')
                            e.print()
                        current_events.append(e)

                    else:
                        warnings.warn('something wird happen, v2!')
                        continue


    # release last ongoing event
    e = current_events[0]
    event_list.append(e)
    current_events.remove(e)

    if debug:
        for e in event_list:
            e.print()
            print('---')


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

        frames = event.get_frames()
        w = params['label_hop_len_s'] # frame length of the annotations
        samples_per_frame = int(w * fs)
        start_time_samples = int(frames[0] * samples_per_frame)
        end_time_samples = int((frames[-1]+1) * samples_per_frame) # add 1 here so we push the duration to the end



        if beamforming_mode == 'omni':
            mono_event = mono_extractor(b_format[start_time_samples:end_time_samples],
                                        mode=beamforming_mode)

        elif beamforming_mode == 'beam':
            azi_frames = event.get_azis()
            ele_frames = event.get_eles()
            # frames to samples; TODO: interpolation would be cool
            num_frames = len(frames)
            num_samples = num_frames * samples_per_frame

            assert(end_time_samples - start_time_samples == num_samples)

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

    if debug:
        frames = []
        for e in event_list:
            frames.append(e.get_frames())
        plt.figure()
        plt.grid()
        for f in range(len(frames)):
            plt.plot(frames[f])
