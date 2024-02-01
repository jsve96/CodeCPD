import numpy as np
import time
import json
import os
from bocpdms import CpModel, BVARNIG, Detector
from itertools import product
import concurrent.futures
from SWD.utils import load_dataset, f_measure,covering
import zipfile
import shutil


import pathlib

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


print(MAIN_PATH)


DIR = os.getcwd()

with open(os.path.join(MAIN_PATH,"config.json")) as f:
    META_DATA = json.load(f)

DATASET_PATH = os.path.join(DIR,'datasets')
DATASETS = META_DATA['Datasets']
RESULTS_PATH = os.path.join(DIR,'results')
GRID = META_DATA['METHODS']['BOCPDMS']





print(GRID)

#parameter grid
param_comb = product(*GRID.values())

# default_params = {'lambda': 100, "prior_a": 1, 'prior_b': 1}



class BOCPDMS_Detector:

    def __init__(self, data, params,hyperparameter,dir_out,name):
        self.params = params
        self.mat = data
        self.dir_out = dir_out
        self.hyperparameter = hyperparameter
        self.data_name = name
    
    def run(self):
        AR_models = []
        for lag in range(self.params["lower_AR"], self.params["upper_AR"] + 1):
            AR_models.append(
                BVARNIG(
                    prior_a=self.hyperparameter["prior_a"],
                    prior_b=self.hyperparameter["prior_b"],
                    S1=self.params["S1"],
                    S2=self.params["S2"],
                    prior_mean_scale=self.params["prior_mean_scale"],
                    prior_var_scale=self.params["prior_var_scale"],
                    intercept_grouping=self.params["intercept_grouping"],
                    nbh_sequence=[0] * lag,
                    restriction_sequence=[0] * lag,
                    hyperparameter_optimization="online",
                ))
        cp_model = CpModel(self.hyperparameter["lambda"])

        model_universe = np.array(AR_models)
        model_prior = np.array([1 / len(AR_models) for m in AR_models])

        detector = Detector(
        data=self.mat,
        model_universe=model_universe,
        model_prior=model_prior,
        cp_model=cp_model,
        S1=self.params["S1"],
        S2=self.params["S2"],
        T=self.mat.shape[0],
        store_rl=True,
        store_mrl=True,
        trim_type="keep_K",
        threshold=self.params["threshold"],
        save_performance_indicators=True,
        generalized_bayes_rld="kullback_leibler",
        # alpha_param_learning="individual",  # not sure if used
        # alpha_param=0.01,  # not sure if used
        # alpha_param_opt_t=30,  # not sure if used
        # alpha_rld_learning=True,  # not sure if used
        loss_der_rld_learning="squared_loss",
        loss_param_learning="squared_loss",
        )   
        start = time.time()
        detector.run()
        end = time.time()
        runtime = end - start

        locations = [x[0] for x in detector.CPs[-2]]
        locations = [loc - 1 for loc in locations]

        # convert to Python ints
        locations = [int(loc) for loc in locations]

        return locations, runtime


    def grid_search_single(self,params,annotations):

        param_combination = dict(zip(self.hyperparameter.keys(), params))
        #print(params)
        self.hyperparameter = param_combination
        # defaults_params["prior_a"] = param_comb['prior_a'],
        # defaults_params["prior_b"] = param_comb['prior_b'],
        # defaults_params["intensity"] = param_comb['intensity'],
        try:
            locations, run_time = self.run()

            #locations = [x[0] for x in detector.CPs[-2]]
            #locations = [loc - 1 for loc in locations]


            #locations = [int(loc) for loc in locations]
            f1 = f_measure(annotations, locations)
            cov = covering(annotations,locations,mat.shape[0])

        except Exception as err:
            locations = None
            f1 = 0
            cov = 0
            print(err)
            run_time = 0

        out_dict =  {'Name':self.data_name, 'Method':"BOCPDMS", 'params': self.hyperparameter, 'cp':locations, 'F1':f1, 'covering':cov,"runtime": run_time}
        
        #name = str(param_combination['prior_a']) +"_"+ str(param_combination['prior_b']) + "_" + str(param_combination['lambda']) +".json"
        # with open(os.path.join(self.dir_out,name), "w") as f:
        #     json.dump(out_dict,f, indent=4)

        return out_dict

    def grid_search(self, param_grid, annotations):
        
        self.param_grid = param_grid
        n_repeats = len(list(product(*param_grid.values())))
        print(n_repeats)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            out_dicts = list(executor.map(self.grid_search_single, product(*param_grid.values()), [annotations]*n_repeats))

        return out_dicts




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
                defaults = {
                                "S1": mat.shape[1],
                                "S2": 1,
                                "intercept_grouping": None,
                                "prior_mean_scale": 0,  # data is standardized except MNIST
                                "prior_var_scale": 1,
                                "threshold":0  # data is standardized except MNIST
                            }
                T = mat.shape[0]
                Lmin = 1
                Lmax = int(pow(T / np.log(T), 0.25) + 1)
                defaults["lower_AR"] = Lmin
                defaults["upper_AR"] = Lmax

                ind = 0
                for parameter_sample in product(*GRID.values()):
                    ind +=1 
                    sample_dict = {'prior_a':parameter_sample[1],'prior_b':parameter_sample[2],'lambda':parameter_sample[0]}
                    BOCPDMS_detector = BOCPDMS_Detector(mat, defaults, sample_dict, None,name)
                    locations, run_time = BOCPDMS_detector.run()
                    out_dict =  {'Setting':{'name':fname, 'Method':"BOCPDMS", 'params': sample_dict}, 'info':{'cp':locations, 'F1':f_measure(ground_truth, locations), 'covering':covering(ground_truth,locations,mat.shape[0]),"runtime": run_time}}

                    cur_out_dir = os.path.join(RESULTS_PATH,name,fname)
                    if not os.path.exists(cur_out_dir):
                        os.mkdir(os.path.join(cur_out_dir))

                    SWD_dir = os.path.join(cur_out_dir,"oracle_bocpdms")
                    if not os.path.exists(SWD_dir):
                        os.mkdir(os.path.join(SWD_dir))


                    with open(os.path.join(SWD_dir,'bocpdms'+str(ind)+'.json'), "w") as f:
                        json.dump(out_dict,f, indent=4)

                # print(GRID.keys())
                # print(fname)
                # print(ground_truth)

                
                # print(results)

                # for ind in range(len(results['F1'])):
                #     out_dict = {'Setting':{'name':fname}, 'info':{'Method':'SWD','params':results['parameter'][ind],'cp':results['cp_id'][ind],'F1':results['F1'][ind],'Covering':results['covering'][ind],'runtime':results['runtime'][ind]}}
                #     print(out_dict)
                #     cur_out_dir = os.path.join(RESULTS_PATH,name,fname)
                #     if not os.path.exists(cur_out_dir):
                #         os.mkdir(os.path.join(cur_out_dir))

                #     SWD_dir = os.path.join(cur_out_dir,"oracle_SWD")
                #     if not os.path.exists(SWD_dir):
                #         os.mkdir(os.path.join(SWD_dir))


                #     with open(os.path.join(SWD_dir,'SWD'+str(ind)+'.json'), "w") as f:
                #         json.dump(out_dict,f, indent=4)

            #remove tmp directory with unziped files
            shutil.rmtree(os.path.join(MAIN_PATH,'tmp'))
    else:
        tmp_path = os.path.join(DATASET_PATH,name)+".json"

        ground_truth = annotations[name]
        dat, mat = load_dataset(tmp_path)
        defaults = {
        "S1": mat.shape[1],
        "S2": 1,
        "intercept_grouping": None,
        "prior_mean_scale": 0,  # data is standardized except MNIST
        "prior_var_scale": 1,
        "threshold":0  # data is standardized except MNIST
    }
        T = mat.shape[0]
        Lmin = 1
        Lmax = int(pow(T / np.log(T), 0.25) + 1)
        defaults["lower_AR"] = Lmin
        defaults["upper_AR"] = Lmax
        ind = 0
        for parameter_sample in product(*GRID.values()):
            ind +=1 
            sample_dict = {'prior_a':parameter_sample[1],'prior_b':parameter_sample[2],'lambda':parameter_sample[0]}
            BOCPDMS_detector = BOCPDMS_Detector(mat, defaults, sample_dict, None,name)
            locations, run_time = BOCPDMS_detector.run()
            out_dict =  {'Setting':{'name':name, 'Method':"BOCPDMS", 'params': sample_dict}, 'info':{'cp':locations, 'F1':f_measure(ground_truth, locations), 'covering':covering(ground_truth,locations,mat.shape[0]),"runtime": run_time}}
        # for ind in range(len(results['F1'])):
        #     out_dict = {'Setting':{'name':name}, 'info':{'Method':'SWD','params':results['parameter'][ind],'cp':results['cp_id'][ind],'F1':results['F1'][ind],'Covering':results['covering'][ind],'runtime':results['runtime'][ind]}}
            cur_out_dir = os.path.join(RESULTS_PATH,name)
            if not os.path.exists(cur_out_dir):
                os.mkdir(os.path.join(cur_out_dir))

            SWD_dir = os.path.join(cur_out_dir,"oracle_bocpdms")
            if not os.path.exists(SWD_dir):
                os.mkdir(os.path.join(SWD_dir))


            with open(os.path.join(SWD_dir,'bocpdms'+str(ind)+'.json'), "w") as f:
                json.dump(out_dict,f, indent=4)



# ##### main loop
# DATASETS = ['apple']
# for name in DATASETS:

#     tmp_path = os.path.join(DATASET_PATH,name)+".json"
    
#     with open(os.path.join(DATASET_PATH,'annotations')+".json","r") as f:
#         annotations = json.load(f)
#     if 'subject' in name:
#         annotation_data = annotations[name[:-4]]
#     else:
#         annotation_data = annotations[name]

#     if "MNIST" in name:
#         data, mat = load_dataset(tmp_path,False)
#     else:
#         data, mat = load_dataset(tmp_path)

#     defaults = {
#         "S1": mat.shape[1],
#         "S2": 1,
#         "intercept_grouping": None,
#         "prior_mean_scale": 0,  # data is standardized except MNIST
#         "prior_var_scale": 1,
#         # "prior_a":1,
#         # "prior_b":1,
#         # "intensity": 100,
#         "threshold":0  # data is standardized except MNIST
#     }

#     #default hyperparameter
#     hyperparameter ={
#         "prior_a":1,
#         "prior_b":1,
#         "lambda": 100
#     }


#     T = mat.shape[0]
#     Lmin = 1
#     Lmax = int(pow(T / np.log(T), 0.25) + 1)
#     defaults["lower_AR"] = Lmin
#     defaults["upper_AR"] = Lmax


#     outputdir = os.path.join(MAIN_PATH,'results',name,'default')
#     print(outputdir)
#     BOCPDMS_detector = BOCPDMS_Detector(mat, defaults, hyperparameter, outputdir,name)

#     locations, run_time = BOCPDMS_detector.run()
#     print(locations, run_time)
#     print(f_measure(annotation_data,locations))

#     out_dict =  {'Name':name, 'Method':"BOCPDMS", 'params': defaults, 'cp':locations, 'F1':f_measure(annotation_data, locations), 'covering':covering(annotation_data,locations,mat.shape[0]),"time": run_time}

#     # # outputdir = os.path.join(MAIN_PATH,'results',name,'default')

#     with open(os.path.join(outputdir,'default_BOCPDMS.json'), "w") as f:
#         json.dump(out_dict,f, indent=4)

#     outputdir = os.path.join(MAIN_PATH,'results',name,'oracle_BOCPDMS')

#     # BOCPDMS_detector.dir_out = outputdir
#     print(GRID)

#     ###Grid Search 
#     i=1
#     for parameter_sample in product(*GRID.values()):
#         print(i)
#         sample_dict = {'prior_a':parameter_sample[1],'prior_b':parameter_sample[2],'lambda':parameter_sample[0]}
#         print(sample_dict)
#         BOCPDMS_detector = BOCPDMS_Detector(mat, defaults, sample_dict, outputdir,name)
#         locations, run_time = BOCPDMS_detector.run()
#         out_dict =  {'setting':{'Name':name, 'Method':"BOCPDMS", 'params': sample_dict}, 'info':{'cp':locations, 'F1':f_measure(annotation_data, locations), 'covering':covering(annotation_data,locations,mat.shape[0]),"runtime": run_time}}
#         print(run_time)
#         if not os.path.exists(outputdir):
#             os.mkdir(outputdir)

#             with open(os.path.join(outputdir,'oracle_BOCPDMS'+str(i)+'.json'), "w") as f:
#                 json.dump(out_dict,f, indent=4)
#         else:
#             with open(os.path.join(outputdir,'oracle_BOCPDMS'+str(i)+'.json'), "w") as f:
#                 json.dump(out_dict,f, indent=4)
#         i+=1
        

        #results = BOCPDMS_detector.grid_search(GRID,annotation_data)

    #save experiment in results folder

    # if not os.path.exists(outputdir):
    #     os.mkdir(outputdir)
    #     with open(os.path.join(outputdir,'oracle_BOCPDMS.json'), "w") as f:
    #         json.dump(results,f, indent=4)
    # else:
    #     with open(os.path.join(outputdir,'oracle_BOCPDMS.json'), "w") as f:
    #         json.dump(results,f, indent=4)




