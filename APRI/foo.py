import matplotlib.pyplot as plt
import numpy as np



foa_dev = np.load('/Volumes/Dinge/datasets/DCASE2020_TASK3/feat_label/foa_dev/fold1_room1_mix001_ov1.npy')
foa_dev.shape
plt.figure()
plt.title('foa dev')
plt.pcolormesh(foa_dev.T)

foa_dev_norm = np.load('/Volumes/Dinge/datasets/DCASE2020_TASK3/feat_label/foa_dev_norm/fold1_room1_mix001_ov1.npy')
foa_dev_norm.shape
plt.figure()
plt.title('foa dev norm')
plt.pcolormesh(foa_dev_norm.T)

foa_dev_label = np.load('/Volumes/Dinge/datasets/DCASE2020_TASK3/feat_label/foa_dev_label/fold1_room1_mix001_ov1.npy')
foa_dev_label.shape
plt.figure()
plt.title('foa dev label')
plt.pcolormesh(foa_dev_label.T)