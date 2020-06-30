import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

file_path = '/Users/andres.perez/source/DCASE2020/APRI/feature_importance.pkl'
data = pd.read_pickle(file_path)

plt.figure(figsize=(5, 4))
ax = sns.barplot(y = data.index.values[::-1], x = data["relative_importance"][::-1], data = data[::-1],
                 color=sns.color_palette()[0])
ax.set_xlabel('relative importance (%)')

plt.tight_layout()




