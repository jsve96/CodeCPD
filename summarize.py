import numpy as np
import json
import pandas as pd
import os
import pathlib
from methods.python.utils import *

dir_repo = os.path.dirname(os.path.abspath(__file__))

dir_results = os.path.join(dir_repo,"results")
dir_annotations = os.path.join(dir_repo,"datasets/annotations.json")
dir_Nsample = os.path.join(dir_repo,"datasets/observations.json")

with open(dir_annotations,'r') as file:
    ANNOTATIONS = json.load(file)

with open(dir_Nsample,'r') as file:
    N_OBS = json.load(file)


#print(os.listdir(dir_results))

FLAG_ZIP = ["UCI_HAR","MNIST"]
R_Methods = ['bocpd','ecp','kcpa','WATCH']
Python_Methods = ['MMD','SWD','SWD10','SWD20']

datasets = os.listdir(dir_results)

Flagged_datasets = []

F1_RESULT = {}
Cover_RESULT = {}


def prepare_json(jsonfile):

    change_points = jsonfile['info']['cp']
    skip = False
    if not isinstance(change_points, list):

        if change_points is None:
                    skip = True
        else:
            if isinstance(change_points,int):
                if change_points == 1:
                    change_points = []
                else:
                    change_points = [change_points]

    if not change_points:
        #### Exmpty list
        next
    else: 
        change_points = [cp-1 for cp in change_points]
        change_points = [cp for cp in change_points if cp!=1]
    
    return change_points


def get_Metric_R(dir_json,annotation,nobs):
    print(dir_json)

    if not os.path.exists(dir_json):
        print('{} - does not exist -run experiments '.format(dir_json))
        print("Metric is set to 0")
        return 0,0
    
    F1 = []
    Cover = []
    for file in os.listdir(dir_json):
        with open(os.path.join(dir_json,file),"r") as file:
            result = json.load(file)
        cps = prepare_json(result)
        if cps is not None:
            F1.append(f_measure(annotation,cps))
            Cover.append(covering(annotation,cps,nobs))

    return max(F1), max(Cover)

def get_Metric_Py(dir_json,annotation,nobs):
    print(dir_json)

    if not os.path.exists(dir_json):
        print('{} - does not exist -run experiments '.format(dir_json))
        print("Metric is set to 0")
        return 0,0
    F1 = []
    Cover = []
    for file in os.listdir(dir_json):
        with open(os.path.join(dir_json,file),"r") as file:
            result = json.load(file)
        cps = result['info']['cp']
        if cps is not None:
            F1.append(f_measure(annotation,cps))
            Cover.append(covering(annotation,cps,nobs))

    return max(F1), max(Cover)



#def summarize_Metrics_Flagged(F1_RESULT,Cover_RESULT,dir_result,method):


### all datasets without subdatasets als child dir
for dataset in datasets:
    # check if datasets is flagged
    found = any(flagged in dataset for flagged in FLAG_ZIP)
    if found:
        Flagged_datasets.append(dataset)
        continue

    F1_RESULT[dataset] = []
    Cover_RESULT[dataset] = []
    annotations = ANNOTATIONS[dataset]
    n_obs = N_OBS[dataset]
    print(annotations)

    for method in R_Methods:
        dir_json = os.path.join(dir_results,dataset,'oracle_'+method)
        F1, Cov = get_Metric_R(dir_json,annotations,n_obs)
        F1_RESULT[dataset].append(np.round(F1,3))
        Cover_RESULT[dataset].append(np.round(Cov,3))
    for method in Python_Methods:
        dir_json = os.path.join(dir_results,dataset,'oracle_'+method)
        F1, Cov = get_Metric_Py(dir_json,annotations,n_obs)
        F1_RESULT[dataset].append(np.round(F1,3))
        Cover_RESULT[dataset].append(np.round(Cov,3))
    print(dataset)

#print(pd.DataFrame.from_dict(F1_RESULT,orient = 'index',columns=R_Methods+Python_Methods))

#print(pd.DataFrame.from_dict(Cover_RESULT,orient = 'index',columns=R_Methods+Python_Methods))


### all flagged datasets
for dataset in Flagged_datasets:
    # create entry in Metrics dict
    F1_RESULT[dataset] = []
    Cover_RESULT[dataset] = []
    #create tmp dict for methods
    tmp_dict_F1 = {method:[] for method in R_Methods + Python_Methods}
    tmp_dict_Cov = {method:[] for method in R_Methods + Python_Methods}
    print(tmp_dict_Cov.keys())
    #iterate over all subdatasets
    for sub_data in os.listdir(os.path.join(dir_results,dataset)):
        #if "HAR" in sub_data: 
        #     sub_data = sub_data.split("_")[0]
        #store annotations and n_obs
        if "HAR" in sub_data:
            annotations = ANNOTATIONS[sub_data.split("_")[0]]
            n_obs = N_OBS[dataset][sub_data.split("_")[0]]
        else:
            annotations = ANNOTATIONS[sub_data]
            n_obs = N_OBS[dataset][sub_data]
        for method in R_Methods:
            dir_json = os.path.join(dir_results,dataset,sub_data,'oracle_'+method)
            F1, Cov = get_Metric_R(dir_json,annotations,n_obs)
            tmp_dict_F1[method].append(np.round(F1,3))
            tmp_dict_Cov[method].append(np.round(Cov,3))
        for method in Python_Methods:
            dir_json = os.path.join(dir_results,dataset,sub_data,'oracle_'+method)
            F1, Cov = get_Metric_Py(dir_json,annotations,n_obs)
            tmp_dict_F1[method].append(np.round(F1,3))
            tmp_dict_Cov[method].append(np.round(Cov,3))
    
    #iterate over keys in tmp dicts and calculate average
    F1_AVG = []
    Cov_AVG = []
    for k in tmp_dict_F1.keys():
        F1_AVG.append(np.mean(tmp_dict_F1[k]))
        Cov_AVG.append(np.mean(tmp_dict_Cov[k]))
    
    F1_RESULT[dataset] = F1_AVG
    Cover_RESULT[dataset] = Cov_AVG




print("Best F1-scores")
print(pd.DataFrame.from_dict(F1_RESULT,orient = 'index',columns=R_Methods+Python_Methods))
print("\n")
print("Best Covering values")
print(pd.DataFrame.from_dict(Cover_RESULT,orient = 'index',columns=R_Methods+Python_Methods))
        

    
    
    