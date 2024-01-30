import pathlib
import sys
sys.path.insert(0, str(pathlib.Path().resolve()))
from Sampling.src.utils import *

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import zipfile

def read_features(Path):
    Raw = np.load(Path)
    # (n, 28,28,1)
    Data = [im.flatten() for im in Raw]
    Data_df = pd.DataFrame(np.array(Data))
    
    return Data_df

dir_path = os.path.dirname(os.path.realpath(__file__))

dir_parent = os.path.dirname(dir_path)

dir_corruptions = os.path.join(dir_path,'translate')

TARGET_ZIP = 'MNIST-C2-compressed.zip'

#print(os.listdir(dir_corruptions))

archive_c = zipfile.ZipFile(os.path.join(dir_corruptions,TARGET_ZIP),'r')

print(archive_c.namelist())

with archive_c.open('test_images.npy') as out:
    features_test = read_features(out)
with archive_c.open('test_labels.npy') as out:
    labels_test = pd.DataFrame(np.load(out),columns=["label"])

with archive_c.open('train_images.npy') as out:
    features_train = read_features(out)
with archive_c.open('train_labels.npy') as out:
    labels_train =  pd.DataFrame(np.load(out),columns=["label"])

features = pd.concat([features_train,features_test])
labels = pd.concat([labels_train,labels_test])

df = pd.concat([features,labels],axis=1)

df_sorted = df.sort_values('label').reset_index(drop=True)


features = df_sorted.iloc[:,:-1]
targets = df_sorted.iloc[:,-1]


#### Load Standard MNIST Dataset Zip File

archive = zipfile.ZipFile(os.path.join(dir_parent,'mnist_compressed.zip'),'r')


with archive.open('mnist_test.csv') as out:
    df_test = pd.read_csv(out,header=None)

with archive.open('mnist_train.csv') as out:
    df_train = pd.read_csv(out,header=None)

#### Load MNIST into DataFrame

df = pd.concat([df_train,df_test], axis=0)
targets = df.iloc[:,0]
features = df.iloc[:,1:]
df_sorted = df.sort_values(0).reset_index(drop=True)
print(len(df_sorted))
features_raw = df_sorted.iloc[:,1:]
targets_raw = df_sorted.iloc[:,0]

features = pd.concat([features_train,features_test])
labels = pd.concat([labels_train,labels_test])

df = pd.concat([features,labels],axis=1)

df_sorted = df.sort_values('label').reset_index(drop=True)


features = df_sorted.iloc[:,:-1]
targets = df_sorted.iloc[:,-1]


print((targets - targets_raw).sum())

n_obs = 131
possible_lenght = np.arange(6,36,1)
classes = list(targets.unique())

N = 10
randomstates = [10,20,30,40,50,60,70,80,90,100]

datasetslist_raw = []
target_list_raw = []
annoationslist_raw = []

datasetslist_corr = []
target_list_corr = []
annoationslist_corr = []

for i in range(N):
    feat, tar = generate_samples(features,targets,n_obs,possible_lenght, randomstates[i])
    datasetslist_corr.append(feat)
    annoationslist_corr.append(find_annotations(pd.DataFrame(tar)))
    target_list_corr.append(tar)
    
    
    ### original samples
    feat, tar = generate_samples(features_raw,targets_raw,n_obs,possible_lenght, randomstates[i])
    datasetslist_raw.append(feat)
    annoationslist_raw.append(find_annotations(pd.DataFrame(tar)))
    target_list_raw.append(tar)


df_dummy_raw = [d.copy() for d in datasetslist_raw]
#print(df_dummy_raw[0])
datasetslist_merged = []
for i in range(10):
    np.random.seed(randomstates[i])
    ind = np.random.choice(df_dummy_raw[i].index,40, replace = False)
    #df_tmp = datasetslist_raw[i]
    df_tmp = df_dummy_raw[i]
    df_tmp.iloc[ind,:] = datasetslist_corr[i].iloc[ind,:]
    datasetslist_merged.append(df_tmp)

oracle_annotations = {"MNIST_C2_" + str(i): anno for i, anno in enumerate(annoationslist_corr)}


with open(os.path.join(dir_path,"annotations_MINST.json"),"w") as file:
        json.dump(oracle_annotations,file, cls=NpEncoder)


for i, file in enumerate(datasetslist_merged):

    dict_file = make_dict(file,"MNIST_"+str(i))

    with open(os.path.join(dir_path,"MNIST_C2_"+str(i)+".json"),"w") as file:
        json.dump(dict_file,file, cls=NpEncoder)


