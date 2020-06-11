'''
localization_detection.py

Estimate DOAs and onsets/offsets from the given audio files.
Return results as a list of Events.
'''


from APRI.utils import *
from baseline.cls_feature_class import create_folder
import tempfile
import matlab.engine
from scipy.io import loadmat


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

                # find which of the current events are we: same classID and instance number
                class0 = current_events[0].get_classID()
                class1 = current_events[1].get_classID()
                instance0 = current_events[0].get_instance()
                instance1 = current_events[1].get_instance()

                both_finished = False
                if classID == class0 and instance == instance0:
                    event_idx = 0
                elif classID == class1 and instance == instance1:
                    event_idx = 1
                else:
                    # This is a strange case happening in 'fold3_room2_mix044_ov2.wav',
                    # where two sources finish and a new one starts
                    both_finished = True
                if not both_finished:
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
                else:
                    # Terminate both, and start the new one
                    e1 = current_events[-1]
                    event_list.append(e1)
                    current_events.remove(e1)
                    e0 = current_events[0]
                    event_list.append(e0)
                    current_events.remove(e0)
                    # add new
                    frameNumber = [frame]
                    azi = [azis[frame_idx]]
                    ele = [eles[frame_idx]]
                    e = Event(classID, instance, frameNumber, azi, ele)
                    if debug:
                        print('new event---1')
                        e.print()
                    current_events.append(e)






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

    # ERROR CHECK
    # Check that all frames are monotonically increasing by one
    if debug:
        print('ERROR CHECK...')
    for e in event_list:
        f = np.asarray(e.get_frames())
        if f.size > 1:
            assert np.allclose(f[1:] - f[:-1], np.ones(f.size - 1))
    if debug:
        print('ERROR CHECK OK')

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


def ld_basic_dereverb_filter(stft, diff_th=0.3, L=5, event_minimum_length=4):
    """
    same as basic, but dereverb+filter out events shorter than a given parameter
    :param stft:
    :param diff_th:
    :return:
    """

    # dereverb
    # L = 5
    tau = 1
    p = 0.25
    i_max = 10
    ita = 1e-4
    epsilon = 1e-8
    stft_dry, _, _ = estimate_MAR_sparse_parallel(stft, L, tau, p, i_max, ita, epsilon)

    # l&d
    event_list_full = ld_basic(stft_dry, diff_th)

    # filter
    # todo: probably easy to optimize
    event_list_clean = []
    for e in event_list_full:
        if len(e.get_frames()) >= event_minimum_length:
            event_list_clean.append(e)

    return event_list_clean

def ld_particle(stft, diff_th, K_th, V_azi, V_ele, in_sd, in_sdn, init_birth, in_cp, num_particles, debug_plot=False):
    """
    find single-source tf-bins, and then feed them into the particle tracker
    :param stft:
    :param diff_th:
    :return:
    """

    # decimate in frequency
    M, K, N = stft.shape
    stft = stft[:, :K // 2, :]
    M, K, N = stft.shape

    # parametric analysis
    DOA = doa(stft)  # Direction of arrival
    diff = diffuseness(stft, dt=2)  # Diffuseness
    diff_mask = diff <= diff_th

    # create masked doa with nans
    doa_masked = np.empty((2, K, N))
    for k in range(K):
        for n in range(N):
            if diff_mask[k, n]:
                doa_masked[:, k, n] = DOA[:, k, n]
            else:
                doa_masked[:, k, n] = np.nan

    # decimate DOA in time
    DOA_decimated = np.empty((2, K, N // 2))  # todo fix number
    for n in range(N // 2):
        # todo fix numbers depending on decimation factor
        # todo: nanmean but circular!!!
        DOA_decimated[:, :, n] = np.nanmean([doa_masked[:, :, n * 2], doa_masked[:, :, n * 2 - 1]], axis=0)
    M, K, N = DOA_decimated.shape

    # Create lists of azis and eles for each output frame size
    # Filter out spureous candidates
    azis = [[] for n in range(N)]
    eles = [[] for n in range(N)]
    for n in range(N):
        a = DOA_decimated[0, :, n]
        e = DOA_decimated[1, :, n]
        azis_filtered = a[~np.isnan(a)]
        if len(azis_filtered) > K_th:
            azis[n] = azis_filtered
            eles[n] = e[~np.isnan(e)]

    if debug_plot:
        plt.figure()
        # All estimates
        for n in range(N):
            if len(azis[n]) > 0:
                a = np.mod(azis[n] * 180 / np.pi, 360)
                plt.scatter(np.ones(len(a)) * n, a, marker='x', edgecolors='b')
        # Circmedian
        for n in range(N):
            if len(azis[n]) > 0:
                a = np.mod(azis[n] * 180 / np.pi, 360)
                plt.scatter(n, np.mod(circmedian(a, 'deg'), 360), facecolors='none', edgecolors='k')

        # boxplot
        import seaborn as sns
        a = []
        for n in range(N):
            if len(azis[n]) > 0:
                a.append(np.mod(azis[n] * 180 / np.pi, 360))
            else:
                a.append([])
        plt.figure()
        sns.boxplot(data=a)

        # # number of single-source bins in frequency for each n
        # plt.figure()
        # plt.grid()
        # for n in range(N):
        #     if len(azis[n]) > 0:
        #         plt.scatter(n, len(azis[n]), marker='x',  edgecolors='b')

    # TODO: separate frames with two overlapping sources

    # Save into temp file
    fo = tempfile.NamedTemporaryFile()
    csv_file_path = fo.name + '.csv'
    output_file_path = (os.path.splitext(csv_file_path)[0]) + '.mat'

    with open(csv_file_path, 'a') as csvfile:
        writer = csv.writer(csvfile)
        for n in range(len(azis)):
            if len(azis[n]) > 0:  # if not empty, write
                # time = n * seconds_per_frame
                time = n * 0.1
                azi = np.mod(circmedian(azis[n]) * 180 / np.pi, 360)  # csv needs degrees, range 0..360
                ele = 90 - (np.median(eles[n]) * 180 / np.pi)  # csv needs degrees
                writer.writerow([time, azi, ele])


    # Call Matlab
    eng = matlab.engine.start_matlab()
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    matlab_path = this_file_path + '/../multiple-target-tracking-master'
    eng.addpath(matlab_path)
    eng.func_tracking(csv_file_path, float(V_azi), float(V_ele), float(in_sd),
                      float(in_sdn), init_birth, in_cp, float(num_particles), nargout=0)

    # Load output matlab file
    output = loadmat(output_file_path)
    output_data = output['tracks'][0]
    num_events = output_data.size
    # each element of output_data is a different event
    # order of stored data is [time][[azis][eles][std_azis][std_eles]]

    # convert output data into Events
    event_list = []
    for n in range(num_events):

        frames = (output_data[n][0][0] / 0.1).astype(int)  # frame numbers

        # sometimes there are repeated frames; clean them
        diff = frames[1:] - frames[:-1]
        frames = np.insert(frames[1:][diff != 0], 0, frames[0])

        if len(frames) > 1:
            # TODO: FILTER OUT SHORT EVENTS HERE

            azis = output_data[n][1][0] * np.pi / 180.  # in rads
            azis = [a - (2*np.pi) if a > np.pi else a for a in azis] # adjust range to [-pi, pi]
            eles = (90 - output_data[n][1][1]) * np.pi / 180.  # in rads, incl2ele
            event_list.append(Event(-1, -1, frames, azis, eles))

    def interpolate_event(e):

        # TODO: IT REMOVES LAST ELEMENT, PROBABLY NEED TO ADD IT MANUALLY

        frames = e.get_frames()
        azis = e.get_azis()
        eles = e.get_eles()

        new_frames = []
        new_azis = []
        new_eles = []

        frame_dist = frames[1:] - frames[:-1]
        for fd_idx, fd in enumerate(frame_dist):
            if fd == 1:
                # contiguous, set next
                new_frames.append(frames[fd_idx])
                new_azis.append(azis[fd_idx])
                new_eles.append(eles[fd_idx])
            else:
                start = frames[fd_idx]
                end = frames[fd_idx+1]
                new_frames.extend(np.arange(start, end, 1).tolist())
                new_azis.extend(np.linspace(azis[fd_idx], azis[fd_idx+1], fd).tolist())
                new_eles.extend(np.linspace(eles[fd_idx], eles[fd_idx+1], fd).tolist())

        return Event(-1, -1, np.asarray(new_frames), np.asarray(new_azis), np.asarray(new_eles))

    interpolated_event_list = []
    for e in event_list:
        interpolated_event_list.append(interpolate_event(e))
    event_list = interpolated_event_list



        # start_frame = frames[0]
        # end_frame = frames[-1]
        # new_frames = np.arange(start_frame, end_frame+1, 1)
        # new_azis = np.empty(new_frames.size)
        # new_eles = np.empty(new_frames.size)
        #
        # for nf_idx, nf in enumerate(new_frames):
        #     # find if present in original
        #     idx = np.argwhere(frames == nf)
        #     if idx.size == 0:
        #         # not found!
        #     elif idx.size == 1:
        #         # found!
        #         idx = idx[0][0]
        #         # add directly to new position lists
        #         new_azis[nf_idx] = azis[idx]
        #         new_eles[nf_idx] = eles[idx]
        #
        #
        #     else:
        #         warnings.warn('something strange happened!')



        # for n in range(len(f)-1):
        #     dist2next = f[n+1] -



    # # # uncomment for plot doa estimates and particle trajectories
    # plt.figure()
    # plt.grid()
    # # framewise estimates
    # est_csv = np.loadtxt(open(csv_file_path, "rb"), delimiter=",")
    # t = est_csv[:, 0] * 10
    # a = est_csv[:, 1]
    # e = est_csv[:, 2]
    # plt.scatter(t, a, marker='x', edgecolors='b')
    # # particle filter
    # for e_idx, e in enumerate(event_list):
    #     azis = np.asarray(e.get_azis()) * 180 / np.pi
    #     azis = [a + (360) if a < 0 else a for a in azis] # adjust range to [-pi, pi]
    #     plt.plot(e.get_frames(), azis, marker='.', color='chartreuse')

    return event_list
