import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class_dict = ['alarm',
              'dog',
              'fire',
              'crash',
              'baby',
              'f. scream',
              'f. speech',
              'footsteps',
              'knocking',
              'm. scream',
              'm. speech',
              'piano',
              'phone',
              'engine']

# method1 is the groundtruth annotations
method1 = [769, 1306, 444, 449, 1009, 419, 345, 516, 393, 258, 315, 461, 541, 382 ]
# method2 is the estimated segmentation after particle filtering
method2 = [1048, 685, 1058, 722, 802, 454, 325, 870, 324, 347, 295, 637, 777, 826 ]

datasets = ['PAPAFIL1' for i in range(len(class_dict))] + ['PAPAFIL2' for i in range(len(class_dict))]



d = {'class': class_dict+class_dict, 'number': method1+method2, 'dataset': datasets}
df = pd.DataFrame(data=d)

sns.set(style="whitegrid")
plt.figure(figsize=(5, 3))
ax = sns.barplot(x=d['class'], y=d['number'], hue=d['dataset'])
# plt.xticks(rotation=90)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()

# tips = sns.load_dataset('tips')


############################################
############################################
# %%
import os
import soundfile as sf
import numpy as np
import pandas as pd
sr = 24000
# statistics of event durations

class_names = ['alarm',
              'barking_dog',
              'burning_fire',
              'crash',
              'crying_baby',
              'female_scream',
              'female_speech',
              'footsteps',
              'knocking_on_door',
              'male_scream',
              'male_speech',
              'piano',
              'ringing_phone',
              'running_engine']

main_path = '/Volumes/Dinge/datasets/DCASE2020_TASK3/extracted_events'

# durations = []
duration_dict = {}
for class_idx, class_name in enumerate(class_names):
    dur_class = []
    folder_path = os.path.join(main_path, class_name)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f != '.DS_Store']
    for f in files:
        data, _ = sf.read(f)
        dur_class.append(data.size/sr)
    # durations.append(np.asarray(dur_class))
    duration_dict[class_dict[class_idx]] = np.asarray(dur_class)

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in duration_dict.items() ]))

# for class_idx, class_name in enumerate(class_names):

import seaborn as sns
plt.figure(figsize=(5, 3))
ax = sns.boxplot(data=df, showfliers = False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.grid()
# ax = sns.swarmplot(data=df)
# ax = sns.violinplot(data=df[df<10], inner="quartile")