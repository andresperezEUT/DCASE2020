"""
This script contains methods to obtain different dataframes (pandas).

get_source_dataframes(): to build dataframes from arrays. It is used to consolidate audio feature arrays obtained in feature engineering pipeline.
It creates a different dataframe for each source: real, augmented and extra

get_dataframe_balanced_split(): to split source dataframe into train, test and validation. Using specific criteria
get_dataframe_split(): single random split into test, validation and test dataframes
get_dataframe_challenge2(): split following challenge folds

"""

from APRI.utils import get_class_name_dict
import pandas as pd
import os
import numpy as np

def get_key(val):
    classes=get_class_name_dict()
    for key, value in classes.items():
         if val == value:
              return key

def get_source_dataframes(input_path,extra=False):
    rows=[]
    if extra:
        event_type = get_class_name_dict().values()
    else:
        event_type = get_class_name_dict().values()
    for event in event_type:
        i = 0
        array_path = os.path.join(input_path, event)  # path to file
        for array in os.scandir(array_path):
            i += 1
            row = event + os.path.splitext(array.name)[0]
            rows.append(row)
            x = np.load(array)
            y = get_key(event)
            if 'data_x' in locals():
                data_x = np.vstack((data_x, x))
                data_y = np.vstack((data_y, y))
            else:
                data_x = x
                data_y = y
    data=np.hstack((data_x,data_y))
    columns = np.load(os.path.join(input_path, 'column_labels.npy')).tolist()
    columns.append('target')
    df=pd.DataFrame(data=data,index=rows,columns=columns)
    return df


def get_dataframe_balanced_split(df_real,df_aug,test_n,val_n,train_n):
    # Criteria:
    #   1. test and validation take rows from real events (test_n and val_n per class)
    #   2. train takes the rest of the real events  and augmented data until reach train_n per class
    #   3. remove augmented data obtained from those audios included in test and validation.
    
    df_test=df_real[df_real.index.str.contains("fold2", regex=False)]
    df_real=df_real.drop(df_test.index)
    df_val=df_real[df_real.index.str.contains("fold1", regex=False)]
    df_real=df_real.drop(df_val.index)
    for ind in df_test.index:
        df_aux = df_aug[df_aug.index.str.contains(str(ind)+"_", regex=False)]
        df_aug =df_aug.drop(df_aux.index)
    event_type = get_class_name_dict().values()
    for ind in df_val.index:
        df_aux = df_aug[df_aug.index.str.contains(str(ind)+"_", regex=False)]
        df_aug =df_aug.drop(df_aux.index)
    event_type = get_class_name_dict().values()
    i=0
    nmax=0
    for event in event_type:
        key=get_key(event)
        if df_real[df_real['target'] == key].shape[0]>nmax:
           nmax=df_real[df_real['target'] == key].shape[0]
    for event in event_type:
        key=get_key(event)
        if i==0:
            df_train=df_real[df_real['target']==key]
        else:
            df_aux=df_real[df_real['target']==key]
            df_train=pd.concat([df_train,df_aux])
        n=df_real[df_real['target'] == key].shape[0]
        if n<nmax:
            cont=nmax-n
            if cont>len(df_aug[df_aug['target']==key]):
                cont=len(df_aug[df_aug['target']==key])
            df_aux2=df_aug[df_aug['target']==key].sample(cont)
            df_aug = df_aug.drop(df_aux2.index)
            df_train = pd.concat([df_train, df_aux2])
        i+=1
    return df_test,df_val,df_train

def get_dataframe_split(df_real,df_aug,test_p,val_p):
    df_test = df_real.groupby('target',group_keys=False).apply(lambda x: x.sample(frac=test_p))
    df_real=df_real.drop(df_test.index)
    df_val=df_real.groupby('target',group_keys=False).apply(lambda x: x.sample(frac=val_p))
    df_train=df_real.drop(df_val.index)
    for ind in df_val.index:
        df_aux=df_aug[df_aug.index.str.contains(str(ind)+'_',regex=False)]
        df_aug = df_aug.drop(df_aux.index)
    for ind in df_test.index:
        df_aux=df_aug[df_aug.index.str.contains(str(ind)+'_',regex=False)]
        df_aug = df_aug.drop(df_aux.index)
    df_train = pd.concat([df_train, df_aug.sample(frac=1)])
    return df_test,df_val,df_train


def get_dataframe_split_challenge2(df_real,df_aug):
    df_val=df_real[df_real.index.str.contains("fold1", regex=False)]
    df_real=df_real.drop(df_val.index)
    df_test=df_real[df_real.index.str.contains("fold2", regex=False)]
    df_train=df_real
    for ind in df_val.index:
        df_aux=df_aug[df_aug.index.str.contains(str(ind)+'_',regex=False)]
        df_aug = df_aug.drop(df_aux.index)
    for ind in df_test.index:
        df_aux=df_aug[df_aug.index.str.contains(str(ind)+'_',regex=False)]
        df_aug = df_aug.drop(df_aux.index)
    df_train = pd.concat([df_train, df_aug.sample(frac=1)])
    return df_test,df_val,df_train

