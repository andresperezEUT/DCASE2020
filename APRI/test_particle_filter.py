

from baseline import parameter
import os
from APRI.compute_metrics import compute_metrics
from APRI.utils import *
import soundfile as sf


# %% PARAMS

plot = True


preset = 'mi_primerito_dia_postfilter_Q!'
params = parameter.get_params(preset)
data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev/') # path to audios

gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev')  # path to annotations

# audio_file_name = 'fold1_room1_mix027_ov2.wav'
audio_file_name = 'fold1_room1_mix001_ov1.wav'

metadata_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)


fs = params['fs']

# window = params['window']
# window_size = params['window_size']
# window_overlap = params['window_overlap']
# nfft = params['nfft']
# D = params['D'] # decimate factor
# frame_length = params['label_hop_len_s']


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
window_size = 1200*2
window_overlap = window_size//2
t, f, stft = scipy.signal.stft(b_format.T, sr, nperseg=window_size, noverlap=window_overlap)
M, K, N = stft.shape
stft = stft[:, :K//2 , :-1]
M, K, N = stft.shape

# plot_magnitude_spectrogram(stft)

DOA = doa(stft)  # Direction of arrival
diff = diffuseness(stft, dt=2)  # Diffuseness

# plot_diffuseness(diff)

# HARD THRESHOLD
diff_th = 0.05
diff_mask = diff <= diff_th
# plt.figure()
# plt.title('diff hard mask')
# plt.pcolormesh(diff_mask)


# %%
# ################### plot DOA statistics

DOA = doa(stft)
# plot_doa(doa(stft))

doa_masked = np.empty((2, K, N))
for k in range(K):
    for n in range(N):
        if diff_mask[k,n]:
            doa_masked[:, k, n] = doa(stft[:,k,n])
        else:
            doa_masked[:, k,n] = np.nan



DOA_decimated = np.empty((2, K, N//2)) # todo fix number
for n in range(N//2):
    # todo fix numbers depending on decimation factor
    DOA_decimated[:,:,n] = np.nanmean([doa_masked[:,:,n*2],doa_masked[:,:,n*2-1]], axis=0 )

    # TODO: try other methods...
    # for k in range(K):
    #     doas = np.asarray([ doa_masked[:,k,n*2], doa_masked[:,k,n*2+1] ])
    #     if not np.any(np.isnan(doas)):
    #         DOA_decimated[:, k, n] = np.mean(doas, axis=0)

    # DOA_decimated[:,:,n] = doa_masked[:,:,n*2]


# plot_doa(DOA_decimated)
doa_masked = DOA_decimated

M, K, N = doa_masked.shape
# Build broadband histogram from past and present bin

azis =  [ [] for n in range(N)]
eles =  [ [] for n in range(N)]

# filter with minimum number of estimates
K_th = 5
for n in range(N):
    a = doa_masked[0,:,n]
    e = doa_masked[1,:,n]
    azis_filtered = a[~np.isnan(a)]
    if len(azis_filtered) > K_th:
        azis[n] = azis_filtered
        eles[n] = e[~np.isnan(e)]

plt.figure()
# All estimates
for n in range(N):
    if len(azis[n]) > 0:
        a = np.mod(azis[n] * 180 / np.pi, 360)
        plt.scatter(np.ones(len(a))*n, a, marker='x', edgecolors='b')
# Circmedian
for n in range(N):
    if len(azis[n]) > 0:
        a = np.mod(azis[n] * 180 / np.pi, 360)
        plt.scatter(n, np.mod(circmedian(a,'deg'), 360), facecolors='none', edgecolors='k')

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


# for n in range(N):
#     mask = diff_mask[:,n]
#     if any(diff_mask[:,n]):
#         azis[n] = DOA[0, :, n][mask]
#         eles[n] = DOA[1, :, n][mask]




# H =  [ [] for n in range(N)]
# H_edges = [np.arange(0,2*np.pi, 2*np.pi/360.*5), np.arange(0,np.pi, np.pi/180.*5)]

# TODO: TIME AVERAGING DOAS
# TODO: PEAK PEAKING FROM 2D HIST

# Jt = 3 # neigbor frames for averaging
# side_Jt = int((Jt-1)/2)
# for n in range(side_Jt,N-side_Jt):
#
#     mask = diff_mask[:,n]
#     if any(diff_mask[:,n]):
#         azis[n] = DOA[0, :, n][mask]
#         eles[n] = DOA[1, :, n][mask]
        # H[n], _, _ = np.histogram2d(azis[n], eles[n], bins=H_edges)
        # H[n], _, _ = np.histogram2d(azis[n], eles[n])

        # todo: PEAK PEAKING, 2D HISTOGRAM, ETC
        # # compute histogram of azis
        # bin_size_azis_hist = 2*np.pi/360.*5
        # bin_size_eles_hist = np.pi/360.*5
        # azis_hist, _ = np.histogram(azis[n], bins = np.arange(-np.pi,np.pi, bin_size_azis_hist) )
        # eles_hist, _ = np.histogram(azis[n], bins = np.arange(-np.pi/2,np.pi/2, bin_size_azis_hist) )
        # # find peaks, 11
        # L_peak = 11 # toDO: TRY other lengths
        # peaks = scipy.signal.find_peaks_cwt(azis_hist, [L_peak])
        # if peaks.size == 1:
        #     azi = azis_hist[peaks] * 1.5 * bin_size_azis_hist
        #     ele = eles_hist[peaks] * 1.5 * bin_size_eles_hist # todo: indices from peaks are not same in azi and ele...
        # elif peaks.size == 2:
        #     print('2 peaks!')
        #     # TODO: do something
        #     azi = circmedian(azis_hist)
        #     ele = np.median(eles_hist)
        #     # azi = azis_hist[peaks] * 1.5 * bin_size_azis_hist
        #     # ele = eles_hist[peaks] * 1.5 * bin_size_eles_hist
        # else:
        #     warnings.warn('something happened on peak peaking!')


# %% WRITE OUTPUT FILE

import tempfile

fo = tempfile.NamedTemporaryFile()
csv_file_path = fo.name + '.csv'
output_file_path = (os.path.splitext(csv_file_path)[0]) + '.mat'

# seconds_per_frame = window_overlap / sr

with open(csv_file_path, 'a') as csvfile:
    writer = csv.writer(csvfile)
    for n in range(len(azis)):
        if len(azis[n]) > 0:  # if not empty, write
            # time = n * seconds_per_frame
            time = n * 0.1
            azi = np.mod(circmedian(azis[n]) * 180 / np.pi, 360)  # csv needs degrees, range 0..360
            ele = 90 - (np.median(eles[n]) * 180 / np.pi)  # csv needs degrees
            writer.writerow([time, azi, ele])


# %% PREPARE METADATA FOR MATLAB

csv_file_path_gt = (os.path.splitext(csv_file_path)[0]) + '_gt.csv'

# oversample_factor = int(0.1 / seconds_per_frame)
gt_csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")
with open(csv_file_path_gt, 'a') as csvfile:
    writer = csv.writer(csvfile)
    for row in gt_csv:
        # for i in range(oversample_factor):
            # time = row[0] * 0.1 + (seconds_per_frame * i)
        time = row[0] * 0.1
        azi = np.mod(row[3],360) # range 0..360
        ele = 90 - row[4] # inclination
        writer.writerow([time, azi, ele])



# %% CALL MATLAB

import matlab.engine
from scipy.io import loadmat

eng = matlab.engine.start_matlab()
this_file_path = os.path.dirname(os.path.abspath(__file__))
matlab_path = this_file_path + '/../multiple-target-tracking-master'
eng.addpath(matlab_path)

# default: [20, 10, 5, 50, 0.2, 25, 30]
V_azi = 20 # maximum velocity per frame? in degree
V_ele = V_azi//2 # maximum velocity per frame? in degree
in_sd = 5  # std_dev of measurement noise, [1, 50].
in_sdn = 50 # noise spectral density. values?
init_birth = 0.1 # probability of new birth
in_cp = 0.25 # percentage of noise in the measurement data
num_particles = 30 # number of particles: montercarlo iterations?
eng.func_tracking(csv_file_path, float(V_azi), float(V_ele), float(in_sd), float(in_sdn), init_birth, in_cp, float(num_particles), nargout=0)

# output_file = '/Users/andres.perez/source/DCASE2020/APRI/filter_output/fold1_room1_mix027_ov2.mat'
output = loadmat(output_file_path)
output_data = output['tracks'][0]
num_events = output_data.size

# each element of output_data is a different event
# order of stored data is [time][[azis][eles][std_azis][std_eles]]

# convert output data into Events
event_list = []
for n in range(num_events):
    frames = (output_data[n][0][0] / 0.1).astype(int) # frame numbers
    azis = output_data[n][1][0] * np.pi / 180.  # in rads
    eles  = (90 - output_data[n][1][1]) * np.pi/ 180. # in rads
    event_list.append(Event(-1, -1, frames, azis, eles))


#  PLOT # todo check elevation/inclination
# %%
fig = plt.figure()
plt.grid()
ax1 = fig.add_subplot(211)
ts = np.arange(0,60,0.1)
fs = np.arange(0,6000,10)
img = ax1.pcolormesh(ts, fs, doa_masked[0]*180/np.pi, cmap='rainbow', vmin=-180, vmax=180)

ax1.set_ylabel('Frequency (Hz)')

# fig.colorbar(im, orientation="horizontal", pad=-1)
# plt.show()
cax = fig.add_axes([0.125, .93, 0.775, 0.03])
fig.colorbar(img, orientation='horizontal', cax=cax)

# img.set_ylabel('Frequency')

plt.subplot(212)
# title_string = str(V_azi) +'_'+ str(V_ele) +'_'+ str(in_sd) +'_'+ str(in_sdn) +'_'+ str(init_birth) +'_'+ str(in_cp) +'_'+ str(num_particles)
# plt.title(title_string)
plt.grid()

# framewise estimates
est_csv = np.loadtxt(open(csv_file_path, "rb"), delimiter=",")
t = est_csv[:,0]
a = est_csv[:,1]

a = [az - 360 if az > 180 else az for az in a ]

plt.scatter(t, a, marker='x',  edgecolors='b')
plt.xlim(0,60)


# # groundtruth
# gt_csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")
# t = gt_csv[:,0]
# a = np.mod(gt_csv[:,3], 360)
# e = gt_csv[:,4]
# plt.scatter(t, a, facecolors='none', edgecolors='r')


# particle filter
for e_idx, e in enumerate(event_list):
    a = e.get_azis() * 180 / np.pi

    a = [az - 360 if az > 180 else az for az in a]

    plt.plot(e.get_frames()/10, a, color='chartreuse')

plt.xlabel('Time (seconds)')
plt.ylabel('Azimuth (degree)')