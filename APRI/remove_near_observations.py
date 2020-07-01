"""
remove_near_observations.py

"""

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pandas as pd
from baseline import parameter
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats
from APRI.utils import get_class_name_dict
import numpy as np


pipeline_feature_engineering='Datasets_foa_dev_2020-06-09'
params = parameter.get_params()
df_real = pd.read_pickle(os.path.join(params['dataset_dir'], pipeline_feature_engineering, 'source_dataframes/dataframe_source_real.pkl'))

def drop_similar_observations(df_real):
    event_type = get_class_name_dict().values()
    drop_index_total=[]
    for event in event_type:
        print(event)
        df = df_real[df_real.index.str.contains(event, regex=False)]
        scl = StandardScaler()
        df_scl = scl.fit_transform(df)
        matrix = squareform(pdist(df_scl))
        drop_index = []
        i=0
        for ind in df.index:
            if i==0:
                length=np.mean(matrix[i])
            #print('######Original event: ',ind)
            aux=ind.replace(event,'')[0:22]
            j=0
            if ind not in drop_index:
                for ind2 in df.index:
                    aux2 = ind2.replace(event, '')[0:22]
                    if aux==aux2:
                        if i!=j:
                            if matrix[i][j] <length*0.8:
                                print('Similar: ',ind2)
                                drop_index.append(ind2)
                j+=1
            i+=1
        drop_index_total=drop_index_total+drop_index
        print(len(drop_index))
        print(len(drop_index_total))
    print(df_real.shape)
    print(drop_index_total[0])
    df_real=df_real.drop(drop_index_total)
    print(df_real.shape)
    print(df_real['target'].value_counts())
    return df_real

drop_similar_observations(df_real)