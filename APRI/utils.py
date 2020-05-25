import os
import numpy as np
from baseline import cls_feature_class, parameter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from baseline.metrics.evaluation_metrics import cart2sph
import scipy.stats




# %% STUFF

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
    return np.asarray(cart2sph(-I[0], -I[1], -I[2]))[:-1]

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
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0,0]), plot_func(ref_data, params['label_hop_len_s'], ind=1, plot_y_ax=True), plt.ylim(
        [-1, nb_classes + 1]), plt.title('SED reference')
    ax3 = plt.subplot(gs[1, 0]), plot_func(ref_data, params['label_hop_len_s'], ind=2, plot_y_ax=True), plt.ylim(
        [-180, 180]), plt.title('Azimuth reference')
    ax5 = plt.subplot(gs[2, 0]), plot_func(ref_data, params['label_hop_len_s'], ind=3, plot_y_ax=True), plt.ylim(
        [-90, 90]), plt.title('Elevation reference')


# %%

