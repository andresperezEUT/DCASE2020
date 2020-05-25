import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
#plt.switch_backend('MacOSX')
import scipy.signal
import librosa

# open one of the mono audio files
audio_file_path = '/home/ribanez/movidas/dcase20/dcase20_dataset/oracle_mono_signals/alarm/0.wav'
data, sr = sf.read(audio_file_path)

plt.plot(data)

# Get the spectrogram by Short-time Fourier Transform
window_size = 256
window_overlap = window_size//2
nfft = window_size
t, f, stft = scipy.signal.stft(data, sr, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

plt.figure()
plt.title('STFT')
plt.pcolormesh(20*np.log10(np.abs(stft)))
plt.colorbar()

plt.show()
# Maybe is better to use the mel-spectrogram, where the y-axis (frequency) information is more compressed and meaningful
mel = librosa.feature.melspectrogram(y=data, sr=sr)
plt.figure()
plt.title('Mel-STFT')
plt.pcolormesh(20*np.log10(np.abs(mel)))
plt.colorbar()
plt.show()