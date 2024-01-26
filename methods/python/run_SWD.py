import numpy as np
import time
import json
import os
import pathlib

from SWD.SlicedWasserstein import *
from SWD.utils import *
#from SWD.Detector import Detector
#from SWD.Detector_MMD import Detector as MMD_Detector
from SWD.Detector_TSWD import Detector as p_Detector
from itertools import product

class NpEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder,self).default(obj)


MAIN_PATH = pathlib.Path().resolve()

#PFAD noch anpassen

print(MAIN_PATH)

DIR = os.getcwd()

#print(os.listdir('/home/jsve/Github/Benchmark_SWATCH'))
with open(os.path.join(MAIN_PATH,"config.json")) as f:
    META_DATA = json.load(f)

DATASET_PATH = os.path.join(DIR,'datasets')
DATASETS = META_DATA['Datasets']
RESULTS_PATH = os.path.join(DIR,'results')
GRID = META_DATA['METHODS']['SWD']


for name in DATASETS:
    
    tmp_path = os.path.join(DATASET_PATH,name)+".json"
    
    with open(os.path.join(DATASET_PATH,'annotations')+".json","r") as f:
        annotations = json.load(f)
    if 'subject' in name:
        annotation_data = annotations[name[:-4]]
    else:
        annotation_data = annotations[name]

    if "MNIST" in name:
        data, mat = load_dataset(tmp_path,False)
    else:
        data, mat = load_dataset(tmp_path)

    print(mat.shape)
    start = time.time()
    #SWATCH = Detector(mat,K=6,eps=1.3,kappa=8,mu=15,L=100,p=2,delta=0.05)
    SWATCH = p_Detector(mat,K=9,eps=1.5,kappa=3,mu=8,L=100,p=2,delta=0.0)
   # SWATCH = MMD_Detector(mat,K=8,eps=1.0,kappa=3,mu=5,L=100,p=2)

    cps = SWATCH.run()
    end = time.time()
    print(end-start)
    print(annotation_data)
    print(cps)
    print(SWATCH.evaluate(annotation_data,cps))

    print(GRID.keys())
    start = time.time()
    results = SWATCH.grid_search(GRID,annotation_data)
    end = time.time()
    print("time per iteration {}".format((end-start)/420))
    print("### Values corresponding to max F1 ###")
    max_ind = results['F1'].index(max(results['F1']))
    print("F1 score:",results['F1'][max_ind])
    for k,p in zip(GRID.keys(),results['parameter'][max_ind]):
        print("{} : {:.2f}".format(k,p))
    print("CP ids:",results['cp_id'][max_ind])
    print("covering:",results['covering'][max_ind])

    print('### Values correpsonding to max Covering ###')
    max_ind_cov = results['covering'].index(max(results['covering']))
    print("F1 score:",results['F1'][max_ind_cov])
    for k,p in zip(GRID.keys(),results['parameter'][max_ind_cov]):
        print("{} : {:.2f}".format(k,p))
    print("CP ids:",results['cp_id'][max_ind_cov])
    print("covering:",results['covering'][max_ind_cov])

    #print(len(list(product(*GRID.values()))))



    



       



