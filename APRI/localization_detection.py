'''
localization_detection.py

Estimate DOAs and onsets/offsets from the given audio files.
Return results as a list of Events.
'''


from APRI.utils import *


def parse_annotations(annotation_file, debug=False):
    """
    parse annotation file and return event_list
    :param annotation_file: file instance
    :return: event_list
    """
    ############################################
    # Delimite events
    #
    event_list = []
    frames = annotation_file[:, 0]
    classIDs = annotation_file[:, 1]
    instances = annotation_file[:, 2]
    azis = annotation_file[:, 3] * np.pi / 180 # gt is in degrees, but Event likes rads
    eles = annotation_file[:, 4] * np.pi / 180

    # source count
    N = 600

    num_sources = np.zeros(N)
    for frame_idx in frames:
        num_sources[int(frame_idx)] += 1

    current_events = []  # maximum two allowed

    for frame_idx, frame in enumerate(frames[:-1]):
        if debug:
            print(frame_idx, frame)
        frame = int(frame)

        if frame_idx == 0:
            last_frame = -1  # avoid problem when starting with 2 sources
        else:
            last_frame = int(frames[frame_idx - 1])

        if (frame - last_frame) > 1:
            # if num_sources[frame] == 0:
            # clear list of current events
            if debug:
                print('finish all')
            while (len(current_events) != 0):
                event_list.append(current_events[-1])
                current_events.remove(current_events[-1])

        if num_sources[frame] == 1:
            # if last was 0, first just started
            if num_sources[frame - 1] == 0:
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
            elif num_sources[frame - 1] == 1:

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
            elif num_sources[frame - 1] == 2:
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
                event_idx = np.mod(event_idx + 1, 2)
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
                        last_classID = int(classIDs[frame_idx - 1])
                        last_instance = instances[frame_idx - 1]
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
                            e = current_events[np.mod(last_used_current_event_idx + 1, 2)]
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

    return event_list


def ld_oracle(stft, audio_file_name, gt_folder_path):
    """
    parse groundtruth
    # TODO: check memory leak on file open
    :param stft: this is in fact not used, but kept for compatibility when called from run.py
    :return:
    """
    metadata_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
    metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)

    csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")
    return parse_annotations(csv)


def ld_basic(stft, diff_th):
    """
    Diffuseness mask

    :param stft:
    :param diff_th:
    :return:
    """

    M, K, N = stft.shape
    DOA = doa(stft) # Direction of arrival
    diff = diffuseness(stft) # Diffuseness
    diff_mask = diff <= diff_th

    # segment audio based on diffuseness mask
    source_activity = np.empty(N)
    for n in range(N):
        source_activity[n] = np.any(diff_mask[:,n]) # change here discriminative function

    # compute statistics of relevant DOAs
    active_frames = np.argwhere(source_activity>0).squeeze()
    num_active_frames = active_frames.size
    estimated_doa_per_frame = np.empty((num_active_frames,2))

    for af_idx, af in enumerate(active_frames):
        active_bins = diff_mask[:,af]
        doas_active_bins = DOA[:,active_bins,af]
        for a in range(2): # angle
            estimated_doa_per_frame[af_idx,a] = circmedian(doas_active_bins[a])

    # segmentate active bins into "events"
    frame_changes = np.argwhere(active_frames[1:] - active_frames[:-1] != 1).flatten()
    frame_changes = np.insert(frame_changes, 0, -1)
    event_list = []
    for idx in range(len(frame_changes)-1):
        start_frame_idx = frame_changes[idx]+1
        end_frame_idx = frame_changes[idx+1]
        frames = active_frames[start_frame_idx:end_frame_idx+1]
        azis = estimated_doa_per_frame[start_frame_idx:end_frame_idx + 1, 0]
        eles = estimated_doa_per_frame[start_frame_idx:end_frame_idx + 1, 1]
        event_list.append(Event(-1, -1, frames, azis, eles))

    return event_list


def ld_basic_dereverb(stft, diff_th):

    L = 10
    tau = 1
    p = 0.25
    i_max = 20
    ita = 1e-4
    epsilon = 1e-8

    stft_dry, _, _ = estimate_MAR_sparse_parallel(stft, L, tau, p, i_max, ita, epsilon)

    return ld_basic(stft_dry, diff_th)
