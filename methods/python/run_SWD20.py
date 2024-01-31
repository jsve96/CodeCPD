import numpy as np
import time
import json
import os
import pathlib
import zipfile
import shutil

from SWD.SlicedWasserstein import *
from SWD.utils import *
#from SWD.Detector_MMD import Detector as MMD_Detector
from SWD.Detector_TSWD import Detector as p_Detector
from itertools import product

def isZIP(path):
    return zipfile.is_zipfile(os.path.join(path+'.zip'))

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

with open(os.path.join(MAIN_PATH,"config.json")) as f:
    META_DATA = json.load(f)

DATASET_PATH = os.path.join(DIR,'datasets')
DATASETS = META_DATA['Datasets']
RESULTS_PATH = os.path.join(DIR,'results')
GRID = META_DATA['METHODS']['SWD']


for name in DATASETS:
    
    tmp_path = os.path.join(DATASET_PATH,name)#+".json"

    with open(os.path.join(DATASET_PATH,'annotations')+".json","r") as f:
        annotations = json.load(f)

    #check if ZIP File
    print(name)
    if isZIP(tmp_path):
        with zipfile.ZipFile(os.path.join(tmp_path+'.zip'),"r") as archive:
            FILES = archive.namelist()
            if not os.path.exists(os.path.join(MAIN_PATH,'tmp')):
                os.mkdir(os.path.join(MAIN_PATH,'tmp'))
            if not os.path.exists(os.path.join(RESULTS_PATH,name)):
                os.mkdir(os.path.join(RESULTS_PATH,name))
            for file in FILES[4:5]:

                if file.endswith('.json'):
                    archive.extract(file, path=os.path.join(MAIN_PATH,'tmp'))   
                    #mat, dat = load_dataset(os.path.join(MAIN_PATH,'tmp',file))
                    fname = os.path.splitext(file)[0]

                    if fname.endswith('HAR'):
                        ground_truth = annotations[fname[:-4]]
                        dat, mat = load_dataset(os.path.join(MAIN_PATH,'tmp',file))
                    else: 
                        ground_truth = annotations[fname]
                        if 'MNIST' in fname:
                            data, mat = load_dataset(os.path.join(MAIN_PATH,'tmp',file),normalize=False)
                        else:
                            data, mat = load_dataset(os.path.join(MAIN_PATH,'tmp',file))
                print(mat.shape)
                print(ground_truth)

                SWATCH = p_Detector(mat,K=4,eps=1.1,kappa=5,mu=8,L=100,p=2,delta=0.2)
                start = time.time()
                cps = SWATCH.run()
                end = time.time()
                print(end-start)

                print(GRID.keys())
                print(fname)
                print(ground_truth)

                start = time.time()
                results = SWATCH.grid_search(GRID,ground_truth)
                end = time.time()
                #print(end-start)

                #results = grid_search(GRID,ground_truth,mat)
                #results = SWATCH.solo_grid(GRID,ground_truth)
                #print(results['F1'])
                print(results)

                for ind in range(len(results['F1'])):
                    out_dict = {'Setting':{'name':fname}, 'info':{'Method':'SWD20','params':results['parameter'][ind],'cp':results['cp_id'][ind],'F1':results['F1'][ind],'Covering':results['covering'][ind],'runtime':results['runtime'][ind]}}
                    print(out_dict)
                    cur_out_dir = os.path.join(RESULTS_PATH,name,fname)
                    if not os.path.exists(cur_out_dir):
                        os.mkdir(os.path.join(cur_out_dir))

                    SWD_dir = os.path.join(cur_out_dir,"oracle_SWD20")
                    if not os.path.exists(SWD_dir):
                        os.mkdir(os.path.join(SWD_dir))


                    with open(os.path.join(SWD_dir,'SWD'+str(ind)+'.json'), "w") as f:
                        json.dump(out_dict,f, indent=4)

            #remove tmp directory with unziped files
            shutil.rmtree(os.path.join(MAIN_PATH,'tmp'))
    else:
        tmp_path = os.path.join(DATASET_PATH,name)+".json"

        ground_truth = annotations[name]
        dat, mat = load_dataset(tmp_path)
        SWATCH = p_Detector(mat,K=9,eps=1.5,kappa=3,mu=8,L=100,p=2,delta=0.2)
        start = time.time()
        results = SWATCH.grid_search(GRID,ground_truth)
        #results = SWATCH.solo_grid(GRID,ground_truth)
        end = time.time()
        print(end-start)

        for ind in range(len(results['F1'])):
            out_dict = {'Setting':{'name':name}, 'info':{'Method':'SWD20','params':results['parameter'][ind],'cp':results['cp_id'][ind],'F1':results['F1'][ind],'Covering':results['covering'][ind],'runtime':results['runtime'][ind]}}
            cur_out_dir = os.path.join(RESULTS_PATH,name)
            if not os.path.exists(cur_out_dir):
                os.mkdir(os.path.join(cur_out_dir))

            SWD_dir = os.path.join(cur_out_dir,"oracle_SWD20")
            if not os.path.exists(SWD_dir):
                os.mkdir(os.path.join(SWD_dir))


            with open(os.path.join(SWD_dir,'SWD'+str(ind)+'.json'), "w") as f:
                json.dump(out_dict,f, indent=4)


    

    



       



