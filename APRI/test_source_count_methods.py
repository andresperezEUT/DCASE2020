

from baseline import parameter
import os
from APRI.compute_metrics import compute_metrics
from APRI.utils import *
import soundfile as sf


# %% PARAMS
#
# plot = True
#
#
# preset = 'mi_primerito_dia_postfilter_Q!'
# params = parameter.get_params(preset)
# data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev/') # path to audios
#
# gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev')  # path to annotations
#
# # audio_file_name = 'fold1_room1_mix027_ov2.wav'
# audio_file_name = 'fold1_room1_mix001_ov1.wav'
#
# metadata_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
# metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)
#
#
# fs = params['fs']
#
# # window = params['window']
# # window_size = params['window_size']
# # window_overlap = params['window_overlap']
# # nfft = params['nfft']
# # D = params['D'] # decimate factor
# # frame_length = params['label_hop_len_s']
#
#
# # %% RUN
#
# # Read audio
# b_format, sr = sf.read(os.path.join(data_folder_path, audio_file_name))
# assert sr == fs
#
# # audio_lims = [0., 0.5] # s
# # audio_lims = [34, 37] # s
# # audio_lims = [39, 40] # s
# audio_lims = [0, 5] # s
#
# audio_start_samples = int(audio_lims[0] * sr)
# audio_end_samples = int(audio_lims[1] * sr)
# b_format = b_format[audio_start_samples:audio_end_samples]
#
# # STFT
# t, f, stft = scipy.signal.stft(b_format.T, sr, nperseg=512)
# M, K, N = stft.shape
#
# K = K // 2
# stft = stft[:, :K , :]
# plot_magnitude_spectrogram(stft)
#
# DOA = doa(stft)  # Direction of arrival
# diff = diffuseness(stft)  # Diffuseness
#
# plt.figure()
# plot_diffuseness(diff)
#
# # HARD THRESHOLD
# diff_th = 0.3
# diff_mask = diff <= diff_th
# plt.figure()
# plt.title('diff hard mask')
# plt.pcolormesh(diff_mask)


# %%
import soundfile as sf
import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
dimM = 4

ir_file_path0 = '/Users/andres.perez/source/dereverberation/experiment/IRs/3.wav'
# ir_file_path1 = '/Users/andres.perez/source/dereverberation/experiment/IRs/4.wav'
ir0, sr = sf.read(ir_file_path0)
# ir1, sr = sf.read(ir_file_path1)
ir1 = ir0.copy()
ir1[:,1] = ir0[:,3]
ir1[:,3] = ir0[:,1]
fs = sr

audio_file_length = 5.  # seconds
audio_file_length_samples = int(audio_file_length * fs)
audio_file_offset = 20. # seconds
audio_file_offset_samples = int(audio_file_offset * fs)

af_start = audio_file_offset_samples
af_end = audio_file_offset_samples + audio_file_length_samples

af0 = '/Volumes/Dinge/datasets/DSD100/Sources/Test/013 - Drumtracks - Ghost Bitch/drums.wav'
af1 = '/Volumes/Dinge/datasets/DSD100/Sources/Test/013 - Drumtracks - Ghost Bitch/bass.wav'

s0 = librosa.core.load(af0, sr=sr, mono=True)[0][af_start:af_end]
s1 = librosa.core.load(af1, sr=sr, mono=True)[0][af_start:af_end]

# Compute reverberant signal by FFT convolution
y0 = np.zeros((dimM, audio_file_length_samples))
y1 = np.zeros((dimM, audio_file_length_samples))
for ch in range(dimM):
    y0[ch] = scipy.signal.fftconvolve(s0, ir0[:, ch])[:audio_file_length_samples]  # keep original length
    y1[ch] = scipy.signal.fftconvolve(s1, ir1[:, ch])[:audio_file_length_samples]  # keep original length


# plt.figure()
# plt.plot(y0.T)
# plt.figure()
# plt.plot(y1.T)

# y = y0 + y1
y = y0

t, f, stft = scipy.signal.stft(y, sr, nperseg=512)
M, K, N = stft.shape


from APRI.utils import plot_magnitude_spectrogram, diffuseness, plot_diffuseness
plt.switch_backend('MacOSX')
plot_magnitude_spectrogram(stft)

diff = diffuseness(stft)  # Diffuseness
plot_diffuseness(diff)
# %%

# COHERENCE TEST
J_t = 7 # frames to compute SCM (to past frames)

SCM = np.empty((K, N-J_t, M, M), dtype=complex)
eigen_stft = np.empty((K, N-J_t, M))
for k in range(K):
    for n in range(J_t,N):
        scm = np.empty((J_t, M, M), dtype=complex)
        for t in range(J_t):
            a = stft[:, k, n-t]
            # scm[t] = np.matmul(a, np.transpose(np.conjugate(a)))
            scm[t] = np.outer(a, np.transpose(np.conjugate(a)))
        mean_scm = np.mean(scm, axis=0)
        SCM[k, n-J_t] = mean_scm
        eigen = np.linalg.eig(SCM[k, n-J_t])[0]
        sorted_abs_eigen = np.sort(np.real(eigen))[::-1]
        eigen_stft[k, n-J_t] = sorted_abs_eigen

# plt.figure()
# plt.imshow(np.abs(SCM[50,100]))

# %%
# SMOOTH
Jn = 2
J_t = Jn
Jk = 15
SCM = np.empty((K-Jk, N-Jn, M, M), dtype=complex)
eigen_stft = np.empty((K-Jk, N-Jn, M))

for k in range(Jk, K):
    for n in range(Jn, N):
        stft_local = stft[:,k-Jk:k,n-Jn:n]
        out = np.empty((M, M, Jk, Jn), dtype=complex)

        for jk in range(Jk):
            for jn in range(Jn):
                a = stft_local[:,jk,jn]
                out[:,:,jk,jn] = np.outer(a, np.transpose(np.conjugate(a)))
        out_mean = np.mean(out, axis=(2,3))
        SCM[k-Jk, n-Jn] = out_mean

        eigen = np.linalg.eig(SCM[k-Jk, n-Jn])[0]
        sorted_abs_eigen = np.sort(np.real(eigen))[::-1]
        eigen_stft[k-Jk, n-Jn] = sorted_abs_eigen


# %%
def method_et(s, t=1.2):
    """
    Chen, Weiguo, Kon Max Wong, and James P. Reilly.
    Detection of the number of signals: A predicted eigen-threshold approach.
    IEEE Transactions on Signal Processing 39.5 (1991): 1088-1098.

    t controls the noise eigenvalues floor; bigger t (around 2, 3) gives higher noise floor
    :return: estimated number of sources, estimated eigen-thresholds, eigen-values
    """

    # M = self.get_num_channels()
    # N = self.get_num_time_bins()
    # s = self.eigenvalues()

    est_s = np.zeros(M - 1)

    def l(i):
        return (1. / (M - i + 1)) * np.sum(s[i-1:M])

    for m in range(1, M):
        x = M - m

        d_u_x = ( ( (m+1) * ( ( 1 + t*np.power(N*(m+1),-0.5) )/( 1 - t*np.power(N*m,-0.5) ) ) ) - m ) * (l(x+1))
        est_s[x-1] = d_u_x

        if s[x-1] > d_u_x:
            return x, est_s, s

#
#
num_sources_et = np.empty((K-Jk, N-Jn))
for k in range(Jk, K):
    for n in range(Jn, N):
        norm_eigen = eigen_stft[k-Jk,n-Jn] / np.max(np.abs(eigen_stft[k-Jk,n-Jn]))
        num_sources_et[k-Jk,n-Jn],_,_ = method_et(norm_eigen,t=1.5)
plt.figure()
plt.pcolormesh(num_sources_et)
plt.colorbar()
plt.show()

plt.figure()
plt.pcolormesh(num_sources_et == 1)
plt.colorbar()
plt.show()



# %%
num_sources_et = np.empty(N-Jn)
mean_scm = np.mean(SCM, axis=0)

for n in range(Jn,N):

    eigen = np.linalg.eig(mean_scm[n - Jn])[0]
    sorted_abs_eigen = np.sort(np.real(eigen))[::-1]
    norm_eigen = sorted_abs_eigen / np.max(np.abs(sorted_abs_eigen))
    num_sources_et[n-Jn],_,_ = method_et(norm_eigen,t=5)

plt.figure()
plt.plot(num_sources_et)
plt.grid()
plt.show()

#
# %%
# def method_aic(s):
#     """
#     Akaike, Hirotugu.
#     A new look at the statistical model identification.
#     Selected Papers of Hirotugu Akaike.
#     Springer, New York, NY, 1974. 215-222.
#
#     Implemented from
#     Han, K., and Nehorai, A. (2013).
#     Improved source number detection and direction estimation with nested arrays and ULAs using jackknifing.
#     IEEE Transactions on Signal Processing, 61(23), 6118-6128
#     """
#     # M = self.get_num_channels()
#     # N = self.get_num_time_bins()
#     # s = self.eigenvalues()
#
#     def L(k):
#         return (N/2.) * np.log( np.power( (np.prod(np.power(s[k:],1./(M-k)))) / ( (1./(M-k))*np.sum(s[k:]) ) , M-k))
#
#     def P(k):
#         return 1 + (M * k) - (0.5 * k * (k-1))
#
#     aic = np.zeros(M)
#     for k in range(M):
#         l =  L(k)
#         p = P(k)
#         aic[k] = ( -2 * l ) + ( 2 * p )
#
#     return np.argmin(aic), aic, s
#
#
#
#
# num_sources_aic = np.empty((K, N-J_t))
# for k in range(K):
#     for n in range(J_t,N):
#         norm_eigen = eigen_stft[k,n-J_t] / np.max(np.abs(eigen_stft[k,n-J_t]))
#         num_sources_aic[k,n-J_t],_,_ = method_aic(norm_eigen)
# plt.figure()
# plt.pcolormesh(num_sources_aic)
# plt.colorbar()
# plt.show()
#
#
# # %%
# num_sources_aic = np.empty(N-J_t)
# mean_scm = np.mean(SCM, axis=0)
#
# for n in range(J_t,N):
#
#     eigen =np.linalg.eig(mean_scm[n - J_t])[0]
#     sorted_abs_eigen = np.sort(np.real(eigen))[::-1]
#     norm_eigen = sorted_abs_eigen / np.max(np.abs(sorted_abs_eigen))
#     num_sources_aic[n-J_t],_,_ = method_aic(norm_eigen)
#
# plt.figure()
# plt.plot(num_sources_aic)
# plt.grid()
# plt.show()


# %%




# %%
#
# def method_vtrs(mean_psd_matrix):
#     """
#     Jiang, Jong-Shiann, and Mary Ann Ingram.
#     Robust detection of number of sources using the transformed rotational matrix.
#     2004 IEEE Wireless Communications and Networking Conference (IEEE Cat. No. 04TH8733). Vol. 1. IEEE, 2004.
#
#     Variance of Transformed Rotational Submatrix
#     TODO: it seems it's not working properly...
#     :return: estimated number of sources, vtrs(k)
#     """
#     # 1. Compute correlation matrix
#     # M = self.get_num_channels()
#     M = 4
#     # psd_matrix = self.compute_psd_matrix()
#     # mean_psd_matrix = np.mean(psd_matrix, axis=0)
#
#     # 2. Get eigenvalues, eigenvectors
#     w, v = np.linalg.eig(mean_psd_matrix)
#
#     # Order them decreasingly
#     idx = w.argsort()[::-1]
#     w = w[idx]
#     v = v[:, idx]
#
#     # 3. Compute Phi by least squares
#     # lstsq(a,b) -> Ax = B -> Ex*Phi = Ey
#     Ex = v[:M - 1]
#     Ey = v[1:]
#     phi, residues, rank, s = scipy.linalg.lstsq(Ex, Ey)
#
#     # 4. Derive Delta
#     vtrs = np.zeros(M - 2)
#     for k in range(1, M - 1):
#         d = phi[k:M - 1, :k]
#         vtrs[k - 1] = np.power(np.linalg.norm(d, 'fro'), 2) / ((M - k - 1) * k)
#
#     return np.argmin(vtrs) + 1, vtrs
#
#
#
# num_sources_vtrs = np.empty((K, N-J_t))
# for k in range(K):
#     for n in range(J_t,N):
#         num_sources_vtrs[k,n-J_t],_ = method_vtrs(SCM[k,n-J_t])
# plt.figure()
# plt.pcolormesh(num_sources_vtrs)
# plt.colorbar()
# plt.show()
#
#
#
# # %%
# num_sources_vtrs = np.empty(N-J_t)
# mean_scm = np.mean(SCM, axis=0)
#
# for n in range(J_t,N):
#     num_sources_vtrs[n-J_t],_,= method_vtrs(mean_scm[n-J_t])
#
# plt.figure()
# plt.plot(num_sources_vtrs)
# plt.grid()
# plt.show()
