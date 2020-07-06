'''
generate_audio_from_annotations.py

Synthesize a set of IRs for data augmentation purposes.
Adapted from masp/spherical_array_processing/test_script_mics.py.

Change the destination path to point your file system (line 75).
'''

import numpy as np
from masp import shoebox_room_sim as srs
import time
import soundfile as sf
import os

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SETUP

# Room definition
from baseline.cls_feature_class import create_folder

room = np.array([10.2, 7.1, 3.2])

# Desired RT per octave band, and time to truncate the responses
nBands = 1

# Receiver position
rec = np.array([ [4.5, 3.4, 1.5] ])
nRec = rec.shape[0]

# Mic orientations and directivities
mic_specs = np.array([ [1, 0, 0, 1] ])    # cardioid looking to the front
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RUN SIMULATOR

# Echogram
tic = time.time()

I = 10
for i in range(I):
    print(i)

    rt60 = np.array([np.random.rand() + 0.3])
    print(rt60)

    # Generate octave bands
    band_centerfreqs = np.empty(nBands)
    band_centerfreqs[0] = 1000

    # Absorption for approximately achieving the RT60 above - row per band
    abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]

    # Critical distance for the room
    _, d_critical, _ = srs.room_stats(room, abs_wall)

    # Source in random position within the room
    src = np.random.rand(1,3) * room
    print(src)
    nSrc = src.shape[0]

    maxlim = 1.  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
    limits = np.minimum(rt60, maxlim)

    # Compute echograms
    abs_echograms = srs.compute_echograms_mic(room, src, rec, abs_wall, limits, mic_specs)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # RENDERING

    # In this case all the information (receiver directivity especially) is already
    # encoded in the echograms, hence they are rendered directly to discrete RIRs
    fs = 24000
    mic_rirs = srs.render_rirs_mic(abs_echograms, band_centerfreqs, fs)

    output_path = '/Users/andres.perez/source/DCASE2020/APRI/IR'  # TODO: Change here the destination path
    ir_name = str(i)+'.wav'
    create_folder(output_path)
    sf.write(os.path.join(output_path, ir_name), mic_rirs.flatten(), fs)

toc = time.time()
print('Elapsed time is ' + str(toc-tic) + 'seconds.')
