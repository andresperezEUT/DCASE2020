from APRI.localization_detection import ld_particle
from baseline import parameter
import os
from APRI.compute_metrics import compute_metrics
from APRI.utils import *
import soundfile as sf
from APRI.localization_detection import ld_particle

# %% PARAMS

plot = True


preset = 'particle'
params = parameter.get_params(preset)
data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev/') # path to audios

gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev')  # path to annotations

# audio_file_name = 'fold1_room1_mix027_ov2.wav'
audio_file_name = 'fold1_room1_mix001_ov1.wav'

metadata_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)


fs = params['fs']


# %% RUN

# Read audio
b_format, sr = sf.read(os.path.join(data_folder_path, audio_file_name))
assert sr == fs

# audio_lims = [0., 0.5] # s
# audio_lims = [34, 37] # s
# audio_lims = [39, 40] # s
audio_lims = [0, 60] # s

audio_start_samples = int(audio_lims[0] * sr)
audio_end_samples = int(audio_lims[1] * sr)
b_format = b_format[audio_start_samples:audio_end_samples]

# STFT
window = params['window']
window_size = params['window_size']
window_overlap = params['window_overlap']
nfft = params['nfft']
D = params['D']

stft = compute_spectrogram(b_format, sr, window, window_size, window_overlap, nfft, D)

# localization
diff_th = 0.05 #0.05
K_th = 5 #5
V_azi = 2 # 20  - Velocity
V_ele = 1 #10  - Velocity
in_sd = 5  #5 - standard deviation of measurement noise - [1 50] range is good
in_sdn = 50 #50 -  noise spectral density / decides how smooth the tracked signal is.
init_birth = 0.1 #0.1 - % value between [0 1] - Prior probability of birth
in_cp = 0.25 #0.25 - Noise prior - estimate of percentage of noise in the measurement data
N = 30 # 30
event_list = ld_particle(stft, diff_th, K_th, V_azi, V_ele, in_sd, in_sdn, init_birth, in_cp, N,
                         debug_plot=plot, metadata_file_path=metadata_file_path)
