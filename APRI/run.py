'''
run.py

Main SELDT loop
Execute the full analysis, from audio to output file.
In dev mode, compute evaluation metrics too.
'''
import datetime
import tempfile

import matplotlib.pyplot as plt
from baseline.cls_feature_class import create_folder
from APRI.localization_detection import *
from APRI.compute_metrics import compute_metrics
from APRI.event_class_prediction import *
import time

# %% Parameters
preset = 'mi_primerito_dia'
params = parameter.get_params(preset)
write = True
plot = False


data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev') # path to audios
gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev') # path to annotations
this_file_path = os.path.dirname(os.path.abspath(__file__))
result_folder_path = os.path.join(this_file_path, params['results_dir'], preset) # todo change
create_folder(result_folder_path)

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

# %% Analysis

start_time = time.time()

print('                                              ')
print('-------------- PROCESSING FILES --------------')
print('Folder path: ' + data_folder_path              )
print('Pipeline: ' + params['preset_descriptor']      )

# Iterate over all audio files
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]

# Uncomment the following lines if you want a specific file
# audio_files = ['fold1_room1_mix007_ov1.wav',
#                'fold2_room1_mix007_ov1.wav',
#                'fold3_room1_mix007_ov1.wav',
#                'fold4_room1_mix007_ov1.wav',
#                'fold5_room1_mix007_ov1.wav',
#                'fold6_room1_mix007_ov1.wav']

for audio_file_idx, audio_file_name in enumerate(audio_files):

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print("{}: {}, {}".format(audio_file_idx, st, audio_file_name))

    ############################################
    # Preprocess: prepare file output in case
    if write:
        csv_file_name = (os.path.splitext(audio_file_name)[0]) + '.csv'
        csv_file_path = os.path.join(result_folder_path, csv_file_name)
        # since we always append to the csv file, make a reset on the file
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

    ############################################
    # Open file
    audio_file_path = os.path.join(data_folder_path, audio_file_name)
    b_format, sr = sf.read(audio_file_path)
    b_format *= np.array([1, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])  # N3D to SN3D
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

    ############################################
    # Get monophonic estimates of the event, and predict the classes
    # TODO: modify so file writting is not needed
    num_events = len(event_list)
    for event_idx in range(num_events):

        event = event_list[event_idx]
        mono_event = get_mono_audio_from_event(b_format, event, beamforming_mode, fs, frame_length)

        # Save into temp file
        fo = tempfile.NamedTemporaryFile()
        temp_file_name = fo.name+'.wav'
        sf.write(temp_file_name, mono_event, fs)

        # Predict
        class_method_string = params['class_method']
        class_method = locals()[class_method_string]
        class_method_args = params['class_method_args']

        class_idx = class_method(temp_file_name, *class_method_args)
        event.set_classID(class_idx)

        # Close (delete) file
        fo.close()

        ############################################
        # Generate metadata file from event
        if write:
            event.export_csv(csv_file_path)


    ############################################
    # Plot results
    if plot:
        plot_results(csv_file_path, params)


print('-------------- PROCESSING FINISHED --------------')
print('                                                 ')


# %% OPTIONAL EVAL

if params['mode'] == 'dev':
    print('-------------- COMPUTE DOA METRICS --------------')
    compute_metrics(gt_folder_path, result_folder_path, params)


# %%
end_time = time.time()

print('                                               ')
print('-------------- PROCESS COMPLETED --------------')
print('                                               ')
print('Elapsed time: ' + str(end_time-start_time)      )
