import numpy as np



# INFO EXTRACTED FROM http://dcase.community/challenge2020/task-sound-event-localization-and-detection-results#Nguyen2020_task3_report

# er, f, le,lr

# p1_dev = [0.57,	55.6, 15.6, 66.7] #FALSE
p1_dev = [0.6, 49.8, 13.4, 54.4]
p1_eval = [0.55, 56, 12.8, 61.1]


# p2_dev = [0.44,	68.0, 13.3, 79.6] # FALSE
p2_dev = [0.57, 54.0, 13.8, 59.7]
p2_eval = [0.51, 60.1, 12.4, 65.1]


baseline_dev = [0.72, 37.4, 22.8, 60.7]
baseline_eval = [0.70, 39.5, 23.2, 62.1]

def seld_score(a):
    return np.mean([
        a[0],
        1 - a[1]/100,
        a[2] / 180,
        1 - a[3]/100]
    )


# ],],
print('==============================')
print('METHOD ER, F, LE, LR, SELD')
print('BA_DEV', baseline_dev, seld_score(baseline_dev))
print('P1_DEV', p1_dev, seld_score(p1_dev))
print('P2_DEV', p2_dev, seld_score(p2_dev))
print('-------------------------------')
print('BA_EVAL', baseline_eval, seld_score(baseline_eval))
print('P1_EVAL', p1_eval, seld_score(p1_eval))
print('P2_EVAL', p2_eval, seld_score(p2_eval))
print('==============================')



names = [ 'DU', 'NGUYEN', 'SHIMADA', 'CAO', 'PARK', 'PHAN', 'PEREZLOPEZ', 'SAMPATHKUMAR', 'PATEL', 'RONCHINI', 'NARANJO-ALCAZAR', 'SONG', 'TIAN', 'SINGLA', 'BASELINE' ]
rank = np.arange(1,16,1)
eval_scores = [
    [0.20,	84.9,	6.0,	88.5 ],
    [0.23,	82.0,	9.3,	90.0 ],
    [0.25,	83.2,	7.0,	86.2 ],
    [0.36,	71.2,	13.3,	81.1 ],
    [0.43,	65.2,	16.8,	81.9 ],
    [0.49,	61.7,	15.2,	72.4 ],
    [0.51,	60.1,	12.4,	65.1 ],
    [0.53,	56.6,	14.8,	66.5 ],
    [0.55,	55.5,	14.4,	65.5 ],
    [0.58,	50.8,	16.9,	65.5 ],
    [0.61,	49.1,	19.5,	67.1 ],
    [0.57,	50.4,	20.0,	64.3 ],
    [0.64,	47.6,	24.5,	67.5 ],
    [0.88,	18.0,	53.4,	66.2 ],
    [0.69,	41.3,	23.1,	62.4 ],
]

seld_eval_scores = [seld_score(s) for s in eval_scores]

complexity = [
    123*1e6,
    11*1e6,
    11*1e6,
    23*1e6,
    19*1e6,
    116*1e3,
    20*1e3,
    8*1e6,
    14*1e6,
    1*1e6,
    660*1e3,
    2*1e6,
    2*1e6,
    517*1e3,
    1*1e6 # TODO CHECK
]



print('==============================')
print('METHOD ER, F, LE, LR, SELD')
for i in range(15):
    print(names[i], '\t\t\t', rank[i], eval_scores[i], seld_eval_scores[i])
print('==============================')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

data = pd.DataFrame(list(zip(names, rank, seld_eval_scores, np.log10(complexity))), columns =['Name', 'Rank', 'SELD', 'log_complexity'])
print(data)

# a, b= np.polyfit(np.log(complexity), seld_eval_scores, 1)

# x = np.logspace(5, 8)
# reg_line = a*np.log(x) + b

plt.figure()

ax  = sns.regplot(x="log_complexity", y="SELD", data=data,
                  truncate=False, scatter_kws={"s": 40})
# plt.plot(x, reg_line)
# ax.set(xscale="log")


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(int(point['val'])))

label_point(data.log_complexity, data.SELD, data.Rank, plt.gca())
