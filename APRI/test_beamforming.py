import numpy as np
from APRI.utils import *

# %% PARAMS


preset = 'particle'
params = parameter.get_params(preset)
data_folder_path = os.path.join(params['dataset_dir'], 'foa_dev/') # path to audios

gt_folder_path = os.path.join(params['dataset_dir'], 'metadata_dev')  # path to annotations

# audio_file_name = 'fold1_room1_mix027_ov2.wav'
audio_file_name = 'fold1_room1_mix001_ov1.wav'

metadata_file_name = os.path.splitext(audio_file_name)[0] + '.csv'
metadata_file_path = os.path.join(gt_folder_path, metadata_file_name)

fs = params['fs']


audio_file_name = 'fold1_room1_mix035_ov2.wav'
# Read audio
b_format, sr = sf.read(os.path.join(data_folder_path, audio_file_name))
assert sr == fs

# limit length
audio_lims = [30, 35] # s
audio_start_samples = int(audio_lims[0] * sr)
audio_end_samples = int(audio_lims[1] * sr)
num_samples = audio_end_samples - audio_start_samples
b_format = b_format[audio_start_samples:audio_end_samples]

plt.plot(b_format)

# %% BFORMAT IS SN3D!!!

# class 8: footsteps

mono_omni_foot = mono_extractor(b_format, mode='omni')

azis = np.ones(num_samples) * (-36) * np.pi / 180.
eles = np.ones(num_samples) * (0) * np.pi / 180.
mono_foot = mono_extractor(b_format, azis=azis, eles=eles, mode='beam')
plt.figure()
plt.title('foot')

b_format_n3d = b_format * np.asarray([1, np.sqrt(3), np.sqrt(3), np.sqrt(3)])
mono_n3d_foot = mono_extractor(b_format_n3d, azis=azis, eles=eles, mode='beam')
plt.plot(mono_n3d_foot, linewidth=1, label='new')
plt.plot(mono_foot,  linewidth=1, label='old')
plt.plot(mono_omni_foot,  linewidth=0.25, linestyle='--', label='omni')
plt.legend()
sf.write('/Volumes/Dinge/foot_omni.wav', mono_omni_foot, sr)
sf.write('/Volumes/Dinge/foot_new.wav', mono_n3d_foot, sr)
sf.write('/Volumes/Dinge/foot_old.wav', mono_foot, sr)

# %% BFORMAT IS SN3D!!!

# class 13: piano

mono_omni_piano = mono_extractor(b_format, mode='omni')

azis = np.ones(num_samples) * (108) * np.pi / 180. # 149
eles = np.ones(num_samples) * (2) * np.pi / 180.   # -17
mono_piano = mono_extractor(b_format, azis=azis, eles=eles, mode='beam')
plt.figure()
plt.title('piano')
b_format_n3d = b_format * np.asarray([1, np.sqrt(3), np.sqrt(3), np.sqrt(3)])
mono_n3d_piano = mono_extractor(b_format_n3d, azis=azis, eles=eles, mode='beam')
plt.plot(mono_n3d_piano, linewidth=1, label='new')
plt.plot(mono_piano,  linewidth=1, label='old')
plt.plot(mono_omni_piano,  linewidth=0.25, linestyle='--', label='omni')
plt.legend()
sf.write('/Volumes/Dinge/piano_omni.wav', mono_omni_piano, sr)
sf.write('/Volumes/Dinge/piano_new.wav', mono_n3d_piano, sr)
sf.write('/Volumes/Dinge/piano_old.wav', mono_piano, sr)


assert np.allclose(mono_omni_piano, mono_omni_foot)