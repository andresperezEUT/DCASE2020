import csv
import os
import numpy as np
from baseline import cls_feature_class, parameter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from baseline.metrics.evaluation_metrics import cart2sph
import scipy.stats
import soundfile as sf
import warnings
import librosa.display


# %% SIGNAL

def compute_spectrogram(data, sr, window, window_size, window_overlap, nfft, D=None):

    t, f, stft = scipy.signal.stft(data.T, sr, window=window, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
    stft = stft[:,:-1,:-1] # round shape
    M, K, N = stft.shape
    # TODO: check non-integer cases
    if D is not None:
        dec_stft = np.empty((M, K//D, N), dtype=complex)
        for k in range(K//D):
            dec_stft[:,k,:] = stft[:,k*D,:] # decimate
        stft = dec_stft
    return stft


# %% STATS

def circmedian(angs, unit='rad'):
    # from https://github.com/scipy/scipy/issues/6644
    # Radians!
    pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
    if unit == 'rad':
        pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
    elif unit == 'deg':
        pdists = (pdists +180) % (360.) - 180.
    pdists = np.abs(pdists).sum(1)

    # If angs is odd, take the center value
    if len(angs) % 2 != 0:
        return angs[np.argmin(pdists)]
    # If even, take the mean between the two minimum values
    else:
        index_of_min = np.argmin(pdists)
        min1 = angs[index_of_min]
        # Remove minimum element from array and recompute
        new_pdists = np.delete(pdists, index_of_min)
        new_angs = np.delete(angs, index_of_min)
        min2 = new_angs[np.argmin(new_pdists)]
        if unit == 'rad':
            return scipy.stats.circmean([min1, min2], high=np.pi, low=-np.pi)
        elif unit == 'deg':
            return scipy.stats.circmean([min1, min2], high=180., low=-180.)



# %% PARAMETRIC SPATIAL AUDIO CODING
# Assuming ACN, SN3D data

c = 346.13  # m/s
p0 = 1.1839  # kg/m3

def intensity_vector(stft):
    P = stft[0] # sound pressure
    U = stft[1:] / (p0 * c) # particle velocity
    return np.real(U * np.conjugate(P))

def doa(stft):
    I = intensity_vector(stft)
    return np.asarray(cart2sph(I[2], I[0], I[1]))[:-1]

def energy_density(stft):
    P = stft[0]  # sound pressure
    U = stft[1:] / (p0 * c)  # particle velocity
    s1 = np.power(np.linalg.norm(U, axis=0), 2)
    s2 = np.power(abs(P), 2)
    return ((p0 / 2.) * s1) + ((1. / (2 * p0 * np.power(c, 2))) * s2)


def diffuseness(stft, dt=5):

    I = intensity_vector(stft)
    E = energy_density(stft)

    M, K, N = stft.shape
    dif = np.zeros((K, N))

    for n in range(int(dt / 2), int(N - dt / 2)):
        num = np.linalg.norm(np.mean(I[:, :, n:n + dt], axis=(2)), axis=0)
        den = c * np.mean(E[:,n:n+dt], axis=1)
        dif[:,n] = 1 - (num/den)

    # Borders: copy neighbor values
    for n in range(0, int(dt / 2)):
        dif[:, n] = dif[:, int(dt / 2)]

    for n in range(int(N - dt / 2), N):
        dif[:, n] = dif[:, int(N - (dt / 2) - 1)]

    return dif



# %% BEAMFORMING


def get_ambisonic_gains(azi, ele):
    """
    ACN, N3D
    :param azi: N vector
    :param ele: N vector
    :return:  4 x N matrix
    """
    N = azi.size
    assert N == ele.size
    return np.asarray([np.ones(N), np.sqrt(3)*np.sin(azi)*np.cos(ele), np.sqrt(3)*np.sin(ele), np.sqrt(3)*np.cos(azi)*np.cos(ele)])

def mono_extractor(b_format, azis=None, eles=None, mode='beam'):
    """
    
    :param b_format: (frames, channels)
    :param mode: 'beamforming' or 'omni'
    :return: 
    """
    frames, channels = b_format.shape

    x = np.zeros(frames)

    if mode == 'beam':
        # MaxRE decoding

        alpha = np.asarray([0.775, 0.4, 0.4, 0.4]) # MaxRE coefs
        decoding_gains = get_ambisonic_gains(azis, eles)
        w = decoding_gains*alpha[:,np.newaxis]
        x = np.sum(b_format * w.T, axis=1)

    elif mode == 'omni':
        # Just take the W channel
        x = b_format[:,0]

    return x


def get_mono_audio_from_event(b_format, event, beamforming_mode, fs, frame_length):

    frames = event.get_frames()
    w = frame_length  # frame length of the annotations
    samples_per_frame = int(w * fs)
    start_time_samples = int(frames[0] * samples_per_frame)
    end_time_samples = int((frames[-1] + 1) * samples_per_frame)  # add 1 here so we push the duration to the end
    mono_event = None

    if beamforming_mode == 'omni':
            mono_event = mono_extractor(b_format[start_time_samples:end_time_samples],
                                        mode=beamforming_mode)

    elif beamforming_mode == 'beam':
        azi_frames = event.get_azis()
        ele_frames = event.get_eles()
        # frames to samples; TODO: interpolation would be cool
        num_frames = len(frames)
        num_samples = num_frames * samples_per_frame

        assert (end_time_samples - start_time_samples == num_samples)

        azi_samples = np.zeros(num_samples)
        ele_samples = np.zeros(num_samples)
        for idx in range(num_frames):
            azi_samples[(idx * samples_per_frame):(idx + 1) * samples_per_frame] = azi_frames[idx]
            ele_samples[(idx * samples_per_frame):(idx + 1) * samples_per_frame] = ele_frames[idx]

        mono_event = mono_extractor(b_format[start_time_samples:end_time_samples],
                                    azis=azi_samples * np.pi / 180,  # deg2rad
                                    eles=ele_samples * np.pi / 180,  # deg2rad
                                    mode=beamforming_mode)

    else:
        warnings.warn('MONO METHOD NOT KNOWN"', UserWarning)

    # normalize audio to 1
    mono_event /= np.max(np.abs(mono_event))
    return mono_event


# %% DEREVERBERATION

def herm_k(X):
    return X.conj().transpose((0, 2, 1))

def transpose_k(X):
    return X.transpose((0, 2, 1))

def estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon):
    """
    GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

    :return:
    """

    dimM, dimK, dimN = y_tf.shape

    X = y_tf.transpose((1,2,0))  # [K, N, M]
    i = 0
    D = X  # [K, N, M]
    PHI = np.tile(np.identity(dimM)[np.newaxis], (dimK, 1, 1)) # [K, M, M]
    F = ita  # just for initialization
    F_k = ita  # just for initialization

    # Get recursive matrix
    Xtau = np.zeros((dimK, dimN, dimM * L), dtype=complex)  # [K, N, ML]
    for m in range(dimM):
        Xtau_m = np.zeros((dimK, dimN, L), dtype=complex) # [K, N, L]
        for l in range(L):
            for n in range(dimN):
                if n >= tau + l:  # avoid aliasing
                    Xtau_m[:, n, l] = X[:, n - tau - l, m]
        Xtau[:, :, L * m:L * (m + 1)] = Xtau_m

    # while i < i_max and F >= ita:
    while i < i_max and np.mean(F_k) >= ita:
        print('  iter',i, 'np.mean(F_k)', np.mean(F_k))

        last_D = D # [K, N, M]

        # Estimate weights
        w = np.empty((dimK, dimN), dtype='complex')  # [K, N]
        for n in range(dimN):
            d_n = last_D[:, n, :][:, :, np.newaxis]  # [K, N, 1]
            # inner = np.squeeze(np.sqrt(d_n.conj().transpose((0, 2, 1)) @ np.linalg.pinv(PHI) @ d_n)) # [K]
            inner = np.squeeze(np.sqrt(herm_k(d_n) @ np.linalg.pinv(PHI) @ d_n)) # [K]
            w[:, n] = np.power(np.power(inner, 2) + epsilon, (p / 2) - 1)

        # Estimate G
        # todo parallelize
        W = np.empty((dimK, dimN, dimN), dtype=complex) # [K, N, N]
        for k in range(dimK):
            W[k] = np.diag(w[k])

        G = np.linalg.pinv(herm_k(Xtau) @ W @ Xtau) @ (herm_k(Xtau) @ W @ X)  # [K, ML, M]

        # Estimate D
        D = X - (Xtau @ G) # [K, N, M]

        # Estimate PHI
        PHI = (1 / dimN) * (transpose_k(D) @ W @ D.conj()) # [K, M, M]

        # Estimate convergence
        # F = np.linalg.norm(D - last_D) / np.linalg.norm(D)
        # Per-band convergence
        F_k =  np.linalg.norm(D - last_D, axis=(1,2)) / np.linalg.norm(D, axis=(1,2))
        # print(F_k < ita_k)
        # print(F_k)
        # plt.figure()
        # plt.title(str(i))
        # plt.plot(F_k)
        # plt.hlines(np.mean(F_k), 0,dimK-1)
        # plt.show()
        # print(np.mean(F_k), F)

        # Update pointer
        i += 1

    return D.transpose((2, 0, 1)), G, PHI

# %% PLOT

def plot_magnitude_spectrogram(stft):
    M, K, N = stft.shape
    plt.figure()
    plt.title('STFT')
    for m in range(M):
        plt.subplot(2,2,m+1)
        plt.pcolormesh(20 * np.log10(np.abs(stft[m])))
        plt.colorbar()

def plot_doa(stft):
    plt.figure()
    # Azimuth
    plt.subplot("211")
    plt.pcolormesh(stft[0], cmap='rainbow', vmin=-np.pi, vmax=np.pi)
    plt.title('Azimuth (rad)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()

    # Elevation
    plt.subplot("212")
    plt.pcolormesh(stft[1], cmap='magma', vmin=-np.pi / 2, vmax=np.pi / 2)
    plt.title('Elevation (rad)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()

def plot_diffuseness(stft):
    plt.figure()
    plt.pcolormesh(1-stft, cmap='magma', vmin=0, vmax=1)
    plt.title('Diffuseness')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()


# %% DATA MANAGEMENT

class Event:
    def __init__(self, classID, instance, frames, azis, eles):
        self._classID = classID
        self._instance = instance
        self._frames = frames
        self._azis = azis
        self._eles = eles

    def get_classID(self):
        return self._classID

    def set_classID(self, classID):
        self._classID = classID

    def get_instance(self):
        return self._instance

    def get_frames(self):
        return self._frames

    def get_azis(self):
        return self._azis

    def get_eles(self):
        return self._eles

    def add_frame(self, frame):
        self._frames.append(frame)

    def add_azi(self, azi):
        self._azis.append(azi)

    def add_ele(self, ele):
        self._eles.append(ele)

    def print(self):
        print(self._classID)
        print(self._instance)
        print(self._frames)
        print(self._azis)
        print(self._eles)

    def export_csv(self, csv_file):
        with open(csv_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            for idx in range(len(self._frames)):
                writer.writerow([self._frames[idx],
                                 self._classID,
                                 self._instance,
                                 self._azis[idx]*180/np.pi,     # csv needs degrees
                                 self._eles[idx]*180/np.pi])    # csv needs degrees


def get_class_name_dict():
    return {
        0: 'alarm',
        1: 'crying_baby',
        2: 'crash',
        3: 'barking_dog',
        4: 'running_engine',
        5: 'female_scream',
        6: 'female_speech',
        7: 'burning_fire',
        8: 'footsteps',
        9: 'knocking_on_door',
        10:'male_scream',
        11:'male_speech',
        12:'ringing_phone',
        13:'piano'
    }



def plot_metadata(metadata_file_name):
    # Based on visualize_SELD_output.py

    def collect_classwise_data(_in_dict):
        _out_dict = {}
        for _key in _in_dict.keys():
            for _seld in _in_dict[_key]:
                if _seld[0] not in _out_dict:
                    _out_dict[_seld[0]] = []
                _out_dict[_seld[0]].append([_key, _seld[0], _seld[1], _seld[2]])
        return _out_dict

    def plot_func(plot_data, hop_len_s, ind, plot_x_ax=True, plot_y_ax=False):
        cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
        for class_ind in plot_data.keys():
            time_ax = np.array(plot_data[class_ind])[:, 0] * hop_len_s
            y_ax = np.array(plot_data[class_ind])[:, ind]
            plt.plot(time_ax, y_ax, marker='.', color=cmap[class_ind], linestyle='None', markersize=4)
        plt.grid()
        plt.xlim([0, 60])
        if not plot_x_ax:
            plt.gca().axes.set_xticklabels([])

        if not plot_y_ax:
            plt.gca().axes.set_yticklabels([])

    # --------------------------------- MAIN SCRIPT STARTS HERE -----------------------------------------
    params = parameter.get_params()


    # path of reference audio directory for visualizing the spectrogram and description directory for
    # visualizing the reference
    # Note: The code finds out the audio filename from the predicted filename automatically
    ref_dir = os.path.join(params['dataset_dir'], 'metadata_dev')

    # load the predicted output format
    feat_cls = cls_feature_class.FeatureClass(params)

    # load the reference output format
    ref_filename = os.path.basename(metadata_file_name)
    ref_dict_polar = feat_cls.load_output_format_file(os.path.join(ref_dir, ref_filename))

    ref_data = collect_classwise_data(ref_dict_polar)

    nb_classes = len(feat_cls.get_classes())

    plt.figure()
    # plt.title(ref_filename) # TODO, ADD TITLE
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0,0]), plot_func(ref_data, params['label_hop_len_s'], ind=1, plot_y_ax=True), plt.ylim(
        [-1, nb_classes + 1]), plt.title('SED reference')
    ax3 = plt.subplot(gs[1, 0]), plot_func(ref_data, params['label_hop_len_s'], ind=2, plot_y_ax=True), plt.ylim(
        [-180, 180]), plt.title('Azimuth reference')
    ax5 = plt.subplot(gs[2, 0]), plot_func(ref_data, params['label_hop_len_s'], ind=3, plot_y_ax=True), plt.ylim(
        [-180, 180]), plt.title('Elevation reference')



def plot_results(file_name, params):

    def collect_classwise_data(_in_dict):
        _out_dict = {}
        for _key in _in_dict.keys():
            for _seld in _in_dict[_key]:
                if _seld[0] not in _out_dict:
                    _out_dict[_seld[0]] = []
                _out_dict[_seld[0]].append([_key, _seld[0], _seld[1], _seld[2]])
        return _out_dict

    def plot_func(plot_data, hop_len_s, ind, plot_x_ax=False, plot_y_ax=False):
        cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
        for class_ind in plot_data.keys():
            time_ax = np.array(plot_data[class_ind])[:, 0] * hop_len_s
            y_ax = np.array(plot_data[class_ind])[:, ind]
            plt.plot(time_ax, y_ax, marker='.', color=cmap[class_ind], linestyle='None', markersize=4)
        plt.grid()
        plt.xlim([0, 60])
        if not plot_x_ax:
            plt.gca().axes.set_xticklabels([])

        if not plot_y_ax:
            plt.gca().axes.set_yticklabels([])



    # output format file to visualize
    pred = os.path.join(params['dcase_dir'], file_name)

    # path of reference audio directory for visualizing the spectrogram and description directory for
    # visualizing the reference
    # Note: The code finds out the audio filename from the predicted filename automatically
    ref_dir = os.path.join(params['dataset_dir'], 'metadata_dev')
    aud_dir = os.path.join(params['dataset_dir'], 'foa_dev')

    # load the predicted output format
    feat_cls = cls_feature_class.FeatureClass(params)
    pred_dict = feat_cls.load_output_format_file(pred)
    # pred_dict_polar = feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
    pred_dict_polar = pred_dict

    # load the reference output format
    ref_filename = os.path.basename(pred)
    ref_dict_polar = feat_cls.load_output_format_file(os.path.join(ref_dir, ref_filename))

    pred_data = collect_classwise_data(pred_dict_polar)
    ref_data = collect_classwise_data(ref_dict_polar)

    nb_classes = len(feat_cls.get_classes())

    # load the audio and extract spectrogram
    ref_filename = os.path.basename(pred).replace('.csv', '.wav')
    audio, fs = feat_cls._load_audio(os.path.join(aud_dir, ref_filename))
    stft = np.abs(np.squeeze(feat_cls._spectrogram(audio[:, :1])))
    stft = librosa.amplitude_to_db(stft, ref=np.max)

    plt.figure()
    # plt.title(ref_filename) # TODO, ADD TITLE
    gs = gridspec.GridSpec(4, 4)
    ax0 = plt.subplot(gs[0, 1:3]), librosa.display.specshow(stft.T, sr=fs, x_axis='s', y_axis='linear'), plt.xlim(
        [0, 60]), plt.xticks([]), plt.xlabel(''), plt.title('Spectrogram')
    ax1 = plt.subplot(gs[1, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=1, plot_y_ax=True), plt.ylim(
        [-1, nb_classes + 1]), plt.title('SED reference')
    ax2 = plt.subplot(gs[1, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=1), plt.ylim(
        [-1, nb_classes + 1]), plt.title('SED predicted')
    ax3 = plt.subplot(gs[2, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=2, plot_y_ax=True), plt.ylim(
        [-180, 180]), plt.title('Azimuth reference')
    ax4 = plt.subplot(gs[2, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=2), plt.ylim(
        [-180, 180]), plt.title('Azimuth predicted')
    ax5 = plt.subplot(gs[3, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=3, plot_y_ax=True), plt.ylim(
        [-180, 180]), plt.title('Elevation reference')
    ax6 = plt.subplot(gs[3, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=3), plt.ylim(
        [-180, 180]), plt.title('Elevation predicted')
    ax_lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
    # plt.savefig(os.path.join(params['dcase_dir'] , ref_filename.replace('.wav', '.jpg')), dpi=300, bbox_inches = "tight")
    plt.show()
# %%

