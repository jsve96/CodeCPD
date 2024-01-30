import numpy as np
import json
import os
import shutil

#DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

print(DIR_PATH)

RESULTS_PATH = os.path.join(DIR_PATH,'results')

Dataset_DIR = os.path.join(DIR_PATH,'datasets')

R_UTILS_DIR = os.path.join(DIR_PATH,'methods/utils.R')

print(RESULTS_PATH)
#delete old results
# if os.listdir(RESULTS_PATH):
#     for dir in os.listdir(RESULTS_PATH):
#         shutil.rmtree(os.path.join(RESULTS_PATH,dir))
#         print('delete: {} from {}'.format(dir,RESULTS_PATH))
# else:
#     print("deleting skipped")
# # delete old config files

config_path = os.path.join(DIR_PATH,"config.json")
#os.remove(config_path)

DATASETS = ['MNIST']#['MNIST_C3_'+str(i) for i in range(10)]#['subject'+str(i)+"_HAR" for i in range(1,31)]#['Libras']#['MNIST_1','MNIST_2','MNIST_3','MNIST_4','MNIST_5','MNIST_6','MNIST_7','MNIST_8','MNIST_9']#['batch6']#,'batch6']#['bee_waggle_6', 'occupancy','run_log']#['MNIST_0','MNIST_1','MNIST_2','MNIST_3','MNIST_4','MNIST_5','MNIST_6','MNIST_7','MNIST_8','MNIST_9']
#['apple','bee_waggle_6', 'occupancy','run_log', 'subject1_HAR','subject2_HAR', 'subject3_HAR', 'subject4_HAR','subject5_HAR', 'subject6_HAR','subject7_HAR','subject8_HAR',
            #'subject9_HAR', 'subject10_HAR','subject11_HAR', 'subject12_HAR', 'subject13_HAR','subject14_HAR','subject15_HAR','subject16_HAR','subject17_HAR','subject18_HAR','subject19_HAR','subject20_HAR', 'subject21_HAR','subject22_HAR','subject23_HAR','subject24_HAR','subject25_HAR','subject26_HAR','subject27_HAR','subject28_HAR','subject29_HAR','subject30_HAR']#['batch3','batch4','batch5','batch6']
#  ['apple', 'bee_waggle_6', 'occupancy','run_log', 'subject1_HAR','subject2_HAR', 'subject3_HAR', 'subject4_HAR','subject5_HAR', 'subject6_HAR','subject7_HAR','subject8_HAR',
#     'subject9_HAR', 'subject10_HAR', 'subject11_HAR', 'subject12_HAR', 'subject13_HAR',
# 'subject14_HAR','subject15_HAR','subject16_HAR','subject17_HAR','subject18_HAR','subject19_HAR','subject20_HAR',
#'subject21_HAR','subject22_HAR','subject23_HAR','subject24_HAR','subject25_HAR','subject26_HAR','subject27_HAR','subject28_HAR']
#['apple', 'bee_waggle_6']# 'occupancy','run_log', 'subject1_HAR','subject29_HAR', 'subject30_HAR']
METHODS = ['ecp','kcpa','BOCPD','WATCH','BOCPDMS',"SWD"]#['BOCPD']#["WATCH"]#["ecp","kcpa"]


bocpd_intensities = [10, 50, 100, 200]
bocpd_prior_a = [0.01, 0.1, 1.0, 10, 100]
bocpd_prior_b = [0.01, 0.1, 1.0, 10, 100]
bocpd_prior_k = [0.01, 0.1, 1.0, 10, 100]

cpt_manual_penalty = list(np.logspace(-3, 3, 101))


Params = {"METHODS":{
     "BOCPD": 
        {
            "lambda": bocpd_intensities ,
            "prior_a": bocpd_prior_a,
            "prior_b": bocpd_prior_b,
            "prior_k": bocpd_prior_k,
        }
    ,
    "BOCPDMS":{
        "lambda": bocpd_intensities,
        'prior_a': bocpd_prior_a,
        'prior_b':bocpd_prior_b
    },

    "RBOCPDMS":{
        "lambda": bocpd_intensities,
        'prior_a': bocpd_prior_a,
        'prior_b':bocpd_prior_b
    },
    "ECP":
        {"algorithm": ["e.agglo", "e.divisive"],
         "siglvl": [0.01, 0.05],
         "minsize": [2, 30],
         "alpha": [0.5, 1.0, 1.5]}
    ,

    "KCPA":
        {"maxcp": ['max', 'default'],
         "cost":  cpt_manual_penalty},

    "WATCH":
        { "K": [3,4,5,6,7,8,9],
          "eps": [1.0,1.1,1.2,1.3,1.4,1.5],
          "mu": [3,4,5,6,7,8],
          "kappa": [2,3,4]
        },
    "SWD":{
        "K": [3,4,5,6,7,8,9],
        "eps": [1.0,1.1,1.2,1.3,1.4,1.5],
        "mu": [3,4,5,6,7,8],
        "kappa": [2,3,4]
    }
}
    ,
    "Dataset_DIR": Dataset_DIR,
    "Results_DIR":RESULTS_PATH,
    'Datasets' : DATASETS,
    'UTILS': R_UTILS_DIR
}


#create config.json
# print(DIR_PATH)
with open(os.path.join(DIR_PATH,'config.json'), 'w') as fp:
     json.dump(Params, fp, indent=4)
