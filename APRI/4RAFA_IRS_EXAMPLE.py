import soundfile as sf
import scipy.signal
import os

# Folder with IRs, change it to yours
ir_folder_path = '/Users/andres.perez/source/DCASE2020/APRI/IR'

# e.g. let's take the first one
ir_file_name = '1.wav'
ir_file_path = os.path.join(ir_folder_path, ir_file_name)

# open it
# those irs ar
h, sr_h =  sf.read(ir_file_path)
assert sr_h == 24000 # sample rates should match

# each Impulse Response represents the acoustic response of a room
# the operation to combine a signal with an IR is the convolution.
# it "adds" the reverb to the given signal (if the signal has already reverb, the result will be the combination of all reverbs)

# open an audio signal
# for the example I take one file from the dev set, but you will use a mono_extracted audio
s, _ = sf.read('/Volumes/Dinge/datasets/DCASE2020_TASK3/foa_dev/fold1_room1_mix001_ov1.wav')
# only for the example, I take the first channel. you don't need that for the mono
s = s[:,0]

# convolution
reverberant_s = scipy.signal.fftconvolve(s, h)
# OPTIONAL: convolution adds some samples, so if you want you can manually remove them (I don't think this affects very much)
num_samples = s.size
reverberant_s = reverberant_s[:num_samples]  # optional:convolution add some
