'''
generate_audio_from_analysis.py

'''
import datetime
from APRI.localization_detection import *
from APRI.compute_metrics import compute_metrics
from APRI.event_class_prediction import *
import time

# %% Parameters


def most_frequent(List):
    dict = {}
    count, itm = 0, ''
    for item in reversed(List):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count:
            count, itm = dict[item], item
    return (itm)

def different_elements(list):
    res = []
    for l in list:
        if l not in res:
            res.append(l)
    return res

# preset = 'particle'
# preset = '4REPORT'
preset = '4EVALUATION2'
write = True
plot = False
quick = False

params = parameter.get_params(preset)
mode = params['mode']
data_folder_path = os.path.join(params['dataset_dir'], 'foa_'+mode) # path to audios
gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_'+mode) # path to annotations
this_file_path = os.path.dirname(os.path.abspath(__file__))
result_folder_path = os.path.join(this_file_path, params['results_dir'], preset + '_ORACLE_CLASS')
if quick:
    result_folder_path += '_Q!' # save quick results in separated folders, so that eval.py can benefit from it
create_folder(result_folder_path)


# Main output folder for storing data
output_base_name = 'extracted_events'
output_path = os.path.join(params['dataset_dir'], output_base_name)


# numbers
M = 4
N = 600
fs = params['fs']
window = params['window']
window_size = params['window_size']
window_overlap = params['window_overlap']
nfft = params['nfft']
D = params['D'] # decimate factor
frame_length = params['label_hop_len_s']

beamforming_mode = params['beamforming_mode']
class_name_dict = get_class_name_dict()


# Dataset
all_audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]
# quick_audio_files = ['fold1_room1_mix007_ov1.wav',
                     # 'fold2_room1_mix007_ov1.wav',
                     # 'fold3_room1_mix007_ov1.wav',
                     # 'fold4_room1_mix007_ov1.wav',
                     # 'fold5_room1_mix007_ov1.wav',
                     # 'fold6_room1_mix007_ov1.wav',
                     # 'fold1_room1_mix037_ov2.wav',
                     # 'fold2_room1_mix037_ov2.wav',
                     # 'fold3_room1_mix037_ov2.wav',
                     # 'fold4_room1_mix037_ov2.wav',
                     # 'fold5_room1_mix037_ov2.wav',
                     # ]
quick_audio_files = ['fold1_room1_mix003_ov1.wav']
# quick_audio_files = ['fold1_room1_mix001_ov1.wav']

occurrences_per_class = np.zeros(params['num_classes'], dtype=int)


# create forlders
############################################
# Prepare folders
# import shutil
if write:
#     # delete output path if exists
#     if os.path.exists(output_path):
#         shutil.rmtree(output_path)
#     # now create it
#     create_folder(output_path)
    for class_name in class_name_dict.values():
        folder = os.path.join(output_path, class_name)
        create_folder(folder)

# %% Analysis

start_time = time.time()

print('                                              ')
print('-------------- PROCESSING FILES --------------')
print('Folder path: ' + data_folder_path              )
print('Pipeline: ' + params['preset_descriptor']      )
if quick:
    print('Quick!')

if quick:
    audio_files = quick_audio_files
else:
    audio_files = all_audio_files
for audio_file_idx, audio_file_name in enumerate(audio_files):

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print("{}: {}, {}".format(audio_file_idx, st, audio_file_name))

    ############################################
    # # Preprocess: prepare file output in case
    if write:
        file_name = os.path.splitext(audio_file_name)[0]
        csv_file_name = file_name + '.csv'
        csv_file_path = os.path.join(result_folder_path, csv_file_name)
        if os.path.exists(csv_file_path):
            continue # SKIP EXISTING FILES!

    ############################################
    # Open file
    audio_file_path = os.path.join(data_folder_path, audio_file_name)
    b_format, sr = sf.read(audio_file_path)
    # TODO: CHECK PERFORMANCE AFTER THAT. DATA WAS ALREADY IN SN3D!!!!
    # b_format *= np.array([1, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])  # N3D to SN3D
    # Get spectrogram
    stft = compute_spectrogram(b_format, sr, window, window_size, window_overlap, nfft, D)


    ############################################
    # Localization and detection analysis: from stft to event_list
    ld_method_string = params['ld_method']
    ld_method = locals()[ld_method_string]
    if ld_method_string == 'ld_oracle':
        ld_method_args = [audio_file_name, gt_folder_path] # need to pass the current name to get the associated metadata file
    else:
        ld_method_args = params['ld_method_args']
    event_list = ld_method(stft, *ld_method_args)


    # Get associated metadata file and load content into memory
    # file_name = os.path.splitext(audio_file_name)[0]
    metadata_file_name = file_name + '.csv'
    metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)
    csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")
    gt_event_list = parse_annotations(csv, debug=False)


    #### ASSOCIATION

    # # Associate each event estimation to a class
    # gt_event_occurence_all = []
    # for e in event_list:
    #     frames = e.get_frames()
    #
    #     gt_event_occurence = []
    #     for frame in frames:
    #         # find all other events with same frame
    #         for gt_e_idx, gt_e in enumerate(gt_event_list):
    #             gt_frames = gt_e.get_frames()
    #             if frame in gt_frames:
    #                 gt_event_occurence.append(gt_e.get_classID())
    #     gt_event_occurence_all.append(gt_event_occurence)

    # association by most repeated
    # associated_class = []
    # for occ_list in gt_event_occurence_all:
    #     ass_class = most_frequent(occ_list)
    #     associated_class.append(ass_class)

    # BY AZIMUTH
    # Associate each event estimation to a class
    gt_event_occurence_all = []
    for e in event_list:
        frames = e.get_frames()

        gt_event_occurence = []
        for frame in frames:
            # find all other events with same frame
            for gt_e_idx, gt_e in enumerate(gt_event_list):
                gt_frames = gt_e.get_frames()
                frame_idx = np.argwhere(np.asarray(gt_frames)==frame).flatten() # index of frame in this gt_frame_list
                # print(frame_idx)
                if frame_idx.size == 0:
                    #nothing!
                    pass
                elif frame_idx.size == 1:
                    # just one source: append
                    idx = frame_idx[0]
                    info = [gt_e.get_classID(), gt_e.get_azis()[idx]]
                    # info = [gt_frames[idx], gt_e.get_azis()]
                    gt_event_occurence.append(info)
                else:
                    warnings.warn('!!!!')
            # print(gt_event_occurence)
        gt_event_occurence_all.append(gt_event_occurence)

    # each entry in gt_event_occurence_all corresponds to an event
    assert len(event_list) == len(gt_event_occurence_all)


    # classIDs = []
    for e_idx, e in enumerate(event_list):
        # print('----')
        # print('idx', e_idx)
        # print(e.get_frames())
        zzz = np.asarray(gt_event_occurence_all[e_idx])
        if len(zzz) == 0:
            # no idea, so give it the class of the last event (for example)
            if e_idx > 0:
                last_event_class = event_list[e_idx-1].get_classID()
            else:
                # just whatever
                last_event_class = 5
            e.set_classID(last_event_class)
        else:
            unique_classes = different_elements(zzz[:,0])

            mean = scipy.stats.circmean(e.get_azis(), high=np.pi, low=-np.pi)
            std = scipy.stats.circstd(e.get_azis(), high=np.pi, low=-np.pi)
            eeeee = np.asarray([mean, std])
            # print('E', mean, std)
            mse = []
            for uc in unique_classes:
                uc = int(uc)
                azis = zzz[zzz[:,0]==uc][:,1]
                mean = scipy.stats.circmean(azis, high=np.pi, low=-np.pi)
                std = scipy.stats.circstd(azis, high=np.pi, low=-np.pi)
                gggggg = np.asarray([mean, std])

                # dist = np.sqrt( np.abs(gggggg - eeeee)**2 )
                # dist = scipy.stats.circmean(np.asarray([gggggg, eeeee]), high=np.pi, low=-np.pi)

                # check std of our estimation
                if eeeee[1] > 1:
                    # compare stds
                    dist = np.abs(eeeee[1] -std)
                else:
                    # compare means
                    # dist = np.min( [np.abs(gggggg-eeeee), 2*np.pi - np.abs(gggggg-eeeee)] )
                    dist = np.min( [np.abs(gggggg[0]-eeeee[0]), 2*np.pi - np.abs(gggggg[0]-eeeee[0])] )
                # print(uc, mean, std, dist)
                mse.append(dist)

            classID = int(unique_classes[np.argmin(mse)])
            # print('class',classID)
            e.set_classID(classID)
            # classIDs.append(classID)

    if plot:
        cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
        plt.figure()
        for e in event_list:
            plt.plot(e.get_frames(), e.get_azis(), '.-', color=cmap[e.get_classID()])
        for e_gt in gt_event_list:
            plt.scatter(e_gt.get_frames(), e_gt.get_azis(), marker='x', color=cmap[e_gt.get_classID()])
        plt.grid()


    ############################################
    # Get monophonic estimates of the event, and predict the classes
    # TODO: modify so file writting is not needed
    num_events = len(event_list)
    for event_idx in range(num_events):
        event = event_list[event_idx]
        # event.print()

        mono_event = get_mono_audio_from_event(b_format, event, beamforming_mode, fs, frame_length)

        ######################
        event_occurrence_idx = occurrences_per_class[event.get_classID()]
        mono_file_name = str(event_occurrence_idx) + '.wav'
        class_name = class_name_dict[event.get_classID()]

        ######################
        # write file
        if write:
            output_name = file_name +'_'+ mono_file_name
            output_name = os.path.join(output_path, class_name, output_name)
            sf.write(output_name, mono_event, sr)
        # increment counter
        occurrences_per_class[event.get_classID()] += 1

        ######################
        # write csv
        # csv_file_name = file_name + '.csv'
        # csv_file_path = os.path.join(result_folder_path, csv_file_name)
        event.export_csv(csv_file_path)



print('-------------- PROCESSING FINISHED --------------')
print('                                                 ')


if mode == 'dev':
    print('-------------- COMPUTE DOA METRICS --------------')
    compute_metrics(gt_folder_path, result_folder_path, params)


# %%
end_time = time.time()

print('                                               ')
print('-------------- PROCESS COMPLETED --------------')
print('                                               ')
print('Elapsed time: ' + str(end_time-start_time)      )



