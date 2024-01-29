import numpy as np
import time
import json
import os
import pathlib
import zipfile
import shutil

from SWD.SlicedWasserstein import *
#from SWD.utils import *
#from SWD.Detector import Detector
#from SWD.Detector_MMD import Detector as MMD_Detector
from SWD.Detector_TSWD import Detector as p_Detector
from itertools import product
from utils import *

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


# if zipfile.is_zipfile(os.path.join(DATASET_PATH,'UCI_HAR'+'.zip')):
#     with zipfile.ZipFile(os.path.join(DATASET_PATH,'UCI_HAR'+'.zip'),"r") as archive:
#         archive.printdir()



GRID =  {
            "K": [
                5,
                6
            ],
            "eps": [
                1.0,
                1.1
            ],
            "mu": [
                5
            ],
            "kappa": [
                2
            ]
        }

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
            for file in FILES[:2]:

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

                SWATCH = p_Detector(mat,K=9,eps=1.5,kappa=3,mu=8,L=100,p=2,delta=0.0)
                print(GRID.keys())
                start = time.time()
                results = SWATCH.grid_search(GRID,ground_truth)
                end = time.time()

                for ind in range(len(results['F1'])):
                    out_dict = {'Setting':{'name':fname}, 'info':{'Method':'SWD','params':results['parameter'][ind],'cp':results['cp_id'][ind],'F1':results['F1'][ind],'Covering':results['covering'][ind],'runtime':results['runtime']}}
                    print(out_dict)
                    cur_out_dir = os.path.join(RESULTS_PATH,name,fname)
                    if not os.path.exists(cur_out_dir):
                        os.mkdir(os.path.join(cur_out_dir))

                    SWD_dir = os.path.join(cur_out_dir,"oracle_SWD")
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
        SWATCH = p_Detector(mat,K=9,eps=1.5,kappa=3,mu=8,L=100,p=2,delta=0.0)
        start = time.time()
        results = SWATCH.grid_search(GRID,ground_truth)
        end = time.time()

        for ind in range(len(results['F1'])):
            out_dict = {'Setting':{'name':fname}, 'info':{'Method':'SWD','params':results['parameter'][ind],'cp':results['cp_id'][ind],'F1':results['F1'][ind],'Covering':results['covering'][ind],'runtime':results['runtime']}}
            print(out_dict)
            cur_out_dir = os.path.join(RESULTS_PATH,name)
            if not os.path.exists(cur_out_dir):
                os.mkdir(os.path.join(cur_out_dir))

            SWD_dir = os.path.join(cur_out_dir,"oracle_SWD")
            if not os.path.exists(SWD_dir):
                os.mkdir(os.path.join(SWD_dir))


            with open(os.path.join(SWD_dir,'SWD'+str(ind)+'.json'), "w") as f:
                json.dump(out_dict,f, indent=4)


    
    # with open(os.path.join(DATASET_PATH,'annotations')+".json","r") as f:
    #     annotations = json.load(f)
    # if 'subject' in name:
    #     annotation_data = annotations[name[:-4]]
    # else:
    #     annotation_data = annotations[name]

    # if "MNIST" in name:
    #     data, mat = load_dataset(tmp_path,False)
    # else:
    #     data, mat = load_dataset(tmp_path)

#     print(mat.shape)
#     start = time.time()
#     #SWATCH = Detector(mat,K=6,eps=1.3,kappa=8,mu=15,L=100,p=2,delta=0.05)
#     SWATCH = p_Detector(mat,K=9,eps=1.5,kappa=3,mu=8,L=100,p=2,delta=0.0)
#    # SWATCH = MMD_Detector(mat,K=8,eps=1.0,kappa=3,mu=5,L=100,p=2)

#     cps = SWATCH.run()
#     end = time.time()
#     print(end-start)
#     print(annotation_data)
#     print(cps)
#     print(SWATCH.evaluate(annotation_data,cps))

#     print(GRID.keys())
#     start = time.time()
#     results = SWATCH.grid_search(GRID,annotation_data)
#     end = time.time()
#    




    



       



