import numpy as np
import time
import json
import os
from rbocpdms import CpModel, Detector, BVARNIGDPD
from itertools import product
import concurrent.futures
from utils import load_dataset, f_measure,covering

import pathlib

# with open('/home/jsve/Github/Benchmark_SWATCH/config.json') as f:
#     META_DATA = json.load(f)

# DATASET_PATH = os.path.join(DIR,'datasets')
# DATASETS = META_DATA['Datasets']
# RESULTS_PATH = os.path.join(DIR,'results')
# GRID = META_DATA['METHODS']['RBOCPDMS']
MAIN_PATH = pathlib.Path().resolve()

DIR = os.getcwd()

with open(os.path.join(MAIN_PATH,"config.json")) as f:
    META_DATA = json.load(f)

DATASET_PATH = os.path.join(DIR,'datasets')
DATASETS = META_DATA['Datasets']
RESULTS_PATH = os.path.join(DIR,'results')
GRID = META_DATA['METHODS']['BOCPMS']





print(GRID)

#parameter grid
# param_comb = product(*GRID.values())

# default_params = {'lambda': 100, "prior_a": 1, 'prior_b': 1}




# class RBOCPDMS_Detector:

#     def __init__(self, data, params,hyperparameter,dir_out,name):
#         self.params = params
#         self.mat = data
#         self.dir_out = dir_out
#         self.hyperparameter = hyperparameter
#         self.data_name = name
    
#     def run(self):
#         S1 = self.params['S1']
#         S2 = self.params['S2']

#         model_universe = [
#             BVARNIGDPD(
#             prior_a=self.hyperparameter["prior_a"],
#             prior_b=self.hyperparameter["prior_b"],
#             S1=S1,
#             S2=S2,
#             alpha_param=self.params["alpha_param"],
#             prior_mean_beta=self.params["prior_mean_beta"],
#             prior_var_beta=self.params["prior_var_beta"],
#             prior_mean_scale=self.params["prior_mean_scale"],
#             prior_var_scale=self.params["prior_var_scale"],
#             general_nbh_sequence=[[[]]] * S1 * S2,
#             general_nbh_restriction_sequence=[[0]],
#             general_nbh_coupling="weak coupling",
#             hyperparameter_optimization="online",
#             VB_window_size=self.params["VB_window_size"],
#             full_opt_thinning=self.params["full_opt_thinning"],
#             SGD_batch_size=self.params["SGD_batch_size"],
#             anchor_batch_size_SCSG=self.params["anchor_batch_size_SCSG"],
#             anchor_batch_size_SVRG=None,
#             first_full_opt=self.params["first_full_opt"],
#         )
#         ]
#         model_universe = np.array(model_universe)
#         model_prior = np.array([1 / len(model_universe)] * len(model_universe))
#         cp_model = CpModel(self.hyperparameter["lambda"])

#         detector = Detector(
#         data=self.mat,
#         model_universe=model_universe,
#         model_prior=model_prior,
#         cp_model=cp_model,
#         S1=self.params["S1"],
#         S2=self.params["S2"],
#         T=self.mat.shape[0],
#         store_rl=True,
#         store_mrl=True,
#         trim_type="keep_K",
#         threshold=self.params["threshold"],
#         save_performance_indicators=True,
#         generalized_bayes_rld=self.params["rld_DPD"],
#         alpha_param_learning="individual",
#         alpha_param=self.params["alpha_param"],
#         alpha_param_opt_t=100,
#         alpha_rld=self.params["alpha_rld"],
#         alpha_rld_learning=True,
#         loss_der_rld_learning=self.params["loss_der_rld_learning"],
#     )

#         start = time.time()
#         detector.run()
#         end = time.time()
#         runtime = end - start

#         locations = [x[0] for x in detector.CPs[-2]]
#         locations = [loc - 1 for loc in locations]

#         # convert to Python ints
#         locations = [int(loc) for loc in locations]

#         return locations, runtime


#     def grid_search_single(self,params,annotations):

#         param_combination = dict(zip(self.hyperparameter.keys(), params))
#         #print(params)
#         self.hyperparameter = param_combination
#         # defaults_params["prior_a"] = param_comb['prior_a'],
#         # defaults_params["prior_b"] = param_comb['prior_b'],
#         # defaults_params["intensity"] = param_comb['intensity'],
#         try:
#             locations, run_time = self.run()

#             #locations = [x[0] for x in detector.CPs[-2]]
#             #locations = [loc - 1 for loc in locations]


#             #locations = [int(loc) for loc in locations]
#             f1 = f_measure(annotations, locations)
#             cov = covering(annotations,locations,mat.shape[0])

#         except Exception as err:
#             locations = None
#             f1 = 0
#             cov = 0
#             print(err)
#             run_time = 0

#         out_dict =  {'Name':self.data_name, 'Method':"RBOCPDMS", 'params': self.hyperparameter, 'cp':locations, 'F1':f1, 'covering':cov,"time": run_time}
        
#         #name = str(param_combination['prior_a']) +"_"+ str(param_combination['prior_b']) + "_" + str(param_combination['lambda']) +".json"
#         # with open(os.path.join(self.dir_out,name), "w") as f:
#         #     json.dump(out_dict,f, indent=4)

#         return out_dict

#     def grid_search(self, param_grid, annotations):
        
#         self.param_grid = param_grid
#         n_repeats = len(list(product(*param_grid.values())))
#         print(n_repeats)

#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             out_dicts = list(executor.map(self.grid_search_single, product(*param_grid.values()), [annotations]*n_repeats))

#         return out_dicts
    





# ##### main loop
# DATASETS = ['subject1_HAR']#, 'subject30_HAR']
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
#             "S1": mat.shape[1],
#             "S2": 1,
#             "SGD_batch_size": 10,
#             "VB_window_size": 360,
#             "anchor_batch_size_SCSG": 25,
#             "first_full_opt": 10,
#             "full_opt_thinning": 20,
#             "intercept_grouping": None,
#             "loss_der_rld_learning": "absolute_loss",
#             "prior_mean_beta": None,
#             "prior_mean_scale": 0,  # data has been standardized
#             "prior_var_beta": None,
#             "prior_var_scale": 1.0,  # data has been standardized
#             "rld_DPD": "power_divergence",  # this ensures doubly robust
#             "alpha_param": 0.5,
#             "alpha_rld": 0.5,
#             "threshold": 100
#         }

#     hyperparameter ={
#         "prior_a":1,
#         "prior_b":1,
#         "lambda": 100
#     }


#     outputdir = os.path.join(MAIN_PATH,'results',name,'default')
#     print(outputdir)
#     RBOCPDMS_detector = RBOCPDMS_Detector(mat, defaults, hyperparameter, outputdir,name)

#     locations, run_time = RBOCPDMS_detector.run()
#     print(locations, run_time)
#     print(f_measure(annotation_data,locations))

#     out_dict =  {'Name':name, 'Method':"RBOCPDMS", 'params': defaults, 'cp':locations, 'F1':f_measure(annotation_data, locations), 'covering':covering(annotation_data,locations,mat.shape[0]),"time": run_time}

#     #outputdir = os.path.join(MAIN_PATH,'results',name,'default')

#     with open(os.path.join(outputdir,'default_RBOCPDMS.json'), "w") as f:
#         json.dump(out_dict,f, indent=4)

#     outputdir = os.path.join(MAIN_PATH,'results',name,'oracle_RBOCPDMS')
