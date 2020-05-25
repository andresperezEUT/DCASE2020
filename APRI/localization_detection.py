'''
localization_detection.py

Estimate DOAs and onsets/offsets from the given audio files.
Return results as a list of Events.
'''




import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
plt.switch_backend('MacOSX')
import scipy.signal
import librosa
from baseline.cls_feature_class import create_folder
import os
from APRI.utils import *

# %% Parameters

params = parameter.get_params()
data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev') # path to audios
gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev') # path to annotations
fs = params['fs']

M = 4
N = 600

window_size = 2400
window_overlap = 0
nfft = window_size
D = 10 # decimate factor

plot = True
debug = True
decimate = True

diff_th = 0.2

# %% Analysis

# Iterate over all audio files
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]

# Uncomment the following lines if you want a specific file
# audio_files = ['fold6_room1_mix100_ov2.wav']
audio_files = ['fold6_room1_mix001_ov1.wav']

for audio_file_name in audio_files:

    print('------------------------')
    print(audio_file_name)

    # Compute parameters
    data, sr = sf.read(os.path.join(data_folder_path, audio_file_name))
    data *= np.array([1, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]) # N3D to SN3D
    t, f, stft = scipy.signal.stft(data.T, sr, window='boxcar', nperseg=window_size, noverlap=window_overlap, nfft=nfft)
    stft = stft[:,:-1,:-1] # round shape
    M, K, N = stft.shape

    if decimate:
        dec_stft = np.empty((M, K//D, N), dtype=complex)
        for k in range(K//D):
            dec_stft[:,k,:] = stft[:,k*D,:] # decimate
        stft = dec_stft

    DOA = doa(stft) # Direction of arrival
    diff = diffuseness(stft) # Diffuseness

    if plot:
        plot_magnitude_spectrogram(stft)
        plot_doa(DOA)
        plot_diffuseness(diff)

    diff_mask = diff <= diff_th
    plt.figure()
    plt.pcolormesh(diff_mask)

    # segment audio based on diffuseness mask
    source_activity = np.empty(N)
    for n in range(N):
        source_activity[n] = np.any(diff_mask[:,n]) # change here discriminative function
    plt.plot(source_activity*60)

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


