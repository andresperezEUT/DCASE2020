"""
generate_diffuseness_and_source_count.py

For each audio file and associated metadata file,
this script computes the DirAC diffuseness and parses the frame-wise number of active sources.

The output values are stored in a folder `num_sources` within `params['dataset_dir']`.
Inside it, a folder is created for each dataset entry, containing two files:
- diffuseness.npy, a 1200(f)x 600(t) spectrogram with values between 0 and 1.
- num_sources.npy: a 600(t) element vector with the overlapping values 0, 1 or 2.
"""


import soundfile as sf
import matplotlib.pyplot as plt
#plt.switch_backend('MacOSX')
import scipy.signal
from baseline.cls_feature_class import create_folder
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
D = 20 # decimate factor

plot = False
decimate = True

output_folder =  os.path.join(params['dataset_dir'], 'num_sources_reduced')
create_folder(output_folder)

# %% Analysis

# Iterate over all audio files
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]

# Uncomment the following lines if you want a specific file
# audio_files = ['fold6_room1_mix100_ov2.wav']
# for audio_file_name in [audio_files[0]]:

for audio_file_name in audio_files:

    print('------------------------')
    print(audio_file_name)

    # Compute diffuseness from audio file
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

    # Parse number of sources from metadata file
    gt_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
    csv = np.loadtxt(open(os.path.join(gt_folder_path, gt_file_name), "rb"), delimiter=",")

    num_sources = np.zeros(N)
    frame_indices = csv[:, 0]
    for frame_idx in frame_indices:
        num_sources[int(frame_idx)] += 1

    if plot:
        plt.figure()
        plt.plot(num_sources)

    # Save matrices
    file_name = os.path.splitext(audio_file_name)[0]
    create_folder(os.path.join(output_folder,file_name))
    np.save(os.path.join(output_folder, file_name, 'diffuseness.npy'), diff)
    np.save(os.path.join(output_folder, file_name, 'num_sources.npy'), num_sources)

