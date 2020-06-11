

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

audio_file_name = 'fold1_room1_mix027_ov2.wav'
# audio_file_name = 'fold1_room1_mix001_ov1.wav'

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
audio_lims = [0, 10] # s

audio_start_samples = int(audio_lims[0] * sr)
audio_end_samples = int(audio_lims[1] * sr)
b_format = b_format[audio_start_samples:audio_end_samples]

# STFT
t, f, stft = scipy.signal.stft(b_format.T, sr, nperseg=512)
M, K, N = stft.shape

K = K // 2
stft = stft[:, :K , :]
plot_magnitude_spectrogram(stft)

DOA = doa(stft)  # Direction of arrival
diff = diffuseness(stft)  # Diffuseness

plt.figure()
plot_diffuseness(diff)

# HARD THRESHOLD
diff_th = 0.3
diff_mask = diff <= diff_th
plt.title('diff hard mask')
plt.pcolormesh(diff_mask)



# %%

# # COHERENCE TEST
# J_t = 7 # frames to compute SCM (to past frames)
#
# SCM = np.empty((K, N-J_t, M, M), dtype=complex)
# eigen_stft = np.empty((K, N-J_t, M))
# for k in range(K):
#     for n in range(J_t,N):
#         scm = np.empty((J_t, M, M), dtype=complex)
#         for t in range(J_t):
#             a = stft[:, k, n-t]
#             # scm[t] = np.matmul(a, np.transpose(np.conjugate(a)))
#             scm[t] = np.outer(a, np.transpose(np.conjugate(a)))
#         mean_scm = np.mean(scm, axis=0)
#         SCM[k, n-J_t] = mean_scm
#         eigen = np.linalg.eig(SCM[k, n-J_t])[0]
#         sorted_abs_eigen = np.sort(np.real(eigen))[::-1]
#         eigen_stft[k, n-J_t] = sorted_abs_eigen
#         # eigen_stft[k, J_t] = np.sort(np.abs(np.linalg.eig(SCM[k, n-J_t])[0]))[::-1]  # sorted, reversed, abs
#
# rank = np.empty((K, N-J_t))
# for k in range(K):
#     for n in range(N-J_t):
#         rank[k, n] = np.linalg.matrix_rank(SCM[k,n])
# plt.figure()
# plt.pcolormesh(rank)
#
# eigen_diff_stft = eigen_stft[:,:,0] / eigen_stft[:,:,1]
# plt.figure()
# plt.pcolormesh(np.log10(eigen_diff_stft))
# # plt.pcolormesh(eigen_diff_stft)
# plt.colorbar()


#####
# spectrogram_smooohhing, from the paper
Jn = 2
Jk = 15
SCM_smooth = np.empty((K-Jk, N-Jn, M, M), dtype=complex)
eigen_smooth = np.empty((K-Jk, N-Jn, M))
for k in range(Jk, K):
    for n in range(Jn, N):
        stft_local = stft[:,k-Jk:k,n-Jn:n]
        out = np.empty((M, M, Jk, Jn), dtype=complex)
        for jk in range(Jk):
            for jn in range(Jn):
                a = stft_local[:,jk,jn]
                out[:,:,jk,jn] = np.outer(a, np.transpose(np.conjugate(a)))
        out_mean = np.mean(out, axis=(2,3))
        SCM_smooth[k-Jk, n-Jn] = out_mean
        eigen = np.sort(np.real(np.linalg.eig(out_mean)[0]))[::-1]
        eigen_smooth[k-Jk, n-Jn] = eigen

#####


# eigen_smooth_diff = eigen_smooth[:,:,0] / eigen_smooth[:,:,1]
# plt.figure()
# plt.pcolormesh(np.log10(eigen_smooth_diff))
# # plt.pcolormesh(eigen_diff_stft)
# plt.colorbar()
#
#
# eigen_mask = np.log10(eigen_smooth_diff) > 1
# plt.figure()
# plt.title('eigen_mask')
# plt.pcolormesh(eigen_mask)


# comedie diffuseness
comedie = np.empty((K-Jk, N-Jn))
for k in range(Jk, K):
    for n in range(Jn, N):
        eigen = eigen_smooth[k-Jk,n-Jn]
        order = 1
        g_0 = 2 * ((order + 1) ** 2 - 1)
        mean_ev = (1 / (order + 1) ** 2) * np.sum(eigen)
        g = (1 / mean_ev) * np.sum(np.abs(eigen - mean_ev))
        comedie[k-Jk,n-Jn] = 1 - g / g_0

plt.figure()
plt.pcolormesh(comedie)

comedie_th = 0.1
comedie_mask = comedie <= comedie_th
plt.figure()
plt.title('comedie_mask')
plt.pcolormesh(comedie_mask)


# %%
# ################### plot DOA statistics

DOA = doa(stft)
plot_doa(doa(stft))

doa_masked = np.empty((2, K-Jk, N-Jn))
for k in range(K-Jk):
    for n in range(N-Jn):
        if comedie_mask[k,n]:
            doa_masked[:, k, n] = doa(stft[:,k+Jk,n+Jn])
        else:
            doa_masked[:, k,n] = np.nan


plot_doa(doa_masked)

# TODO: TRY THIS WITH DEREVERBERATED SIGNAL
