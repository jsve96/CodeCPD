import sys
import numpy as np
from SWD.utils import *
from SWD.SlicedWasserstein import *
import os
from itertools import product
import timeit
import concurrent.futures
import time

class Detector:
    
    """
    Description
    
    Attributes:
        data: numpy array; 
            a NxD array of N instances with D features
        K : int;
           Length of batches
        L : int;
           Number of Monte Carlo Samples for Sliced Wasserstein Distance
        eps: float > 0;
           Threshold for Sliced Wasserstein Distance between Batches
        kappa: int;
           Minmimum number of mini-batches in distribution buffer
        mu: int;
           Maximum number of mini-batches in distribution buffer
        p: int;
           Order of Sliced Wasserstein Distance
    """
    def __init__(self, data,K,eps,kappa,mu,L,p,delta=0):

        """
        Initializes the Detector object.

        Parameters:
        - data (numpy array): A NxD array of N instances with D features.
        - K (int): Length of batches.
        - L (int): Number of Monte Carlo Samples for Sliced Wasserstein Distance.
        - eps (float): Threshold for Sliced Wasserstein Distance between batches.
        - kappa (int): Minimum number of mini-batches in the distribution buffer.
        - mu (int): Maximum number of mini-batches in the distribution buffer.
        - p (int): Order of Sliced Wasserstein Distance.
        """

        self.data = data
        self.K =  K
        self.eps = eps
        self.kappa = kappa
        self.mu = mu
        self.L = L
        self.p = p
        self.fimp_data = None
        self.cp_list = None
        self.delta = delta


    def run(self):

        """
        Executes the change point detection algorithm using the provided data and parameters.
        """

        data_split = sequence_data(self.data, self.K)

        T = int(len(data_split))

        current_distribution = []
        #self.fimp_list = []
        fimp_id =[]
        cps = []

        for i in range(T):
            if len(current_distribution)< self.kappa:
                current_distribution.append(data_split[i])

                if len(current_distribution) >= self.kappa:
                        threshold = get_threshold(current_distribution,self.eps,projections = self.L,delta=self.delta)
            else:
                p = p_test_SWD(data_split[i], current_distribution, T = threshold,  feature_imp = True, projections = self.L, delta=self.delta)
                if p > 0.95:
                    cp = i*self.K
                    cps.append(cp)
                    current_distribution =[data_split[i]]
                else:
                    if len(current_distribution) < self.mu:
                        current_distribution.append(data_split[i])
                        threshold = get_threshold(current_distribution,self.eps, projections = self.L,delta = self.delta)
                    else:
                        current_distribution.pop(0)
                        current_distribution.append(data_split[i])
                        threshold = get_threshold(current_distribution, self.eps, projections=self.L, delta = self.delta)

        #self.fimp_data = {'values': self.fimp_list, 'id':fimp_id}

        return cps
    
    def evaluate(self, annotations, cps):

        """
        Evaluates the performance of the change point detection using annotations (ground truth labels).

        Parameters:
        - annotations (numpy.ndarray): Annotations or ground truth labels.

        Returns:
        - tuple: F1 score and covering value.
        """

        F1 = f_measure(annotations,cps)
        cover = covering(annotations, cps,self.data.shape[0])

        return F1, cover
    
    
    def grid_search_single(self, params, annotations):
        """
        Perform grid search for a single parameter combination.

        Parameters:
        - params (tuple): A tuple of parameter values.
        - annotations (numpy.ndarray): Annotations or ground truth labels.

        Returns:
        - tuple: A tuple containing parameter combination, change point IDs, F1 score, and covering value.
        """
        param_combination = dict(zip(self.param_grid.keys(), params))
        if param_combination['kappa'] > param_combination['mu']:
            return None
        else:
            self.K = int(param_combination['K'])
            self.eps = param_combination['eps']
            self.kappa = param_combination['kappa']
            self.mu = param_combination['mu']
            start = time.time()
            cps = self.run()
            end = time.time()
            run_time = end-start
            return (
                tuple(param_combination.values()),
                cps,
                f_measure(annotations, cps),
                covering(annotations, cps, self.data.shape[0]),
                run_time

            )

    def grid_search(self, param_grid, annotations):
        """
        Perform a grid search over the specified parameter grid in parallel.

        Parameters:
        - param_grid (dict): A dictionary where keys are parameter names and values are lists of possible values for each parameter.
        - annotations (numpy.ndarray): Annotations or ground truth labels.

        Returns:
        - dict: A dictionary containing the results of the grid search.
        - 'F1' (list): List of F1 scores for each parameter combination.
        - 'parameter' (list): List of parameter combinations.
        - 'cp_id' (list): List of change point IDs for each parameter combination.
        - 'covering' (list): List of covering values for each parameter combination.
        """
        results = {'F1': [], "parameter": [], "cp_id": [], "covering": [],"runtime":[]}

        self.param_grid = param_grid  # Store the param_grid as an attribute
        n_repeats = len(list(product(*param_grid.values())))
        # Use ProcessPoolExecutor to parallelize the grid search
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Map the grid_search_single function to the parameter combinations
            results_list = list(executor.map(self.grid_search_single, product(*param_grid.values()), [annotations]*n_repeats))

        # Filter out None results (when kappa > mu)
        results_list = [result for result in results_list if result is not None]

        # Unpack the results into separate lists
        results['parameter'], results['cp_id'], results['F1'], results['covering'],results['runtime'] = zip(*results_list)

        return results
    
    def solo_grid(self, param_grid,annotations):
        results = {'F1': [], "parameter": [], "cp_id": [], "covering": [],"runtime":[]}
        param_comb ={'kappa':0,'eps':0,'mu':0,'K':0}
        for param in list(product(*param_grid.values())):
            param_comb['kappa'] = param[0]
            param_comb['eps'] = param[1]
            param_comb['mu'] = param[2]
            param_comb['K'] = param[3]

            if param_comb['kappa'] > param_comb['mu']:
                return None
            else:
                self.K = int(param_comb['K'])
                self.eps = param_comb['eps']
                self.kappa = param_comb['kappa']
                self.mu = param_comb['mu']
                start = time.time()
                cps = self.run()
                end = time.time()
                run_time = end-start

                results['F1'].append(f_measure(annotations, cps))
                results['covering'].append(covering(annotations, cps, self.data.shape[0]))
                results['cp_id'].append(cps)
                results['parameter'].append(param_comb)
                results['runtime'].append(run_time)
                print(run_time)


            
        return results

    



   

# import sys
# import numpy as np
# from SWD.utils import *
# from SWD.SlicedWasserstein import *
# import os
# from itertools import product
# import timeit
# import concurrent.futures
# import time
# import multiprocessing

# class Detector:
    
#     """
#     Description
    
#     Attributes:
#         data: numpy array; 
#             a NxD array of N instances with D features
#         K : int;
#            Length of batches
#         L : int;
#            Number of Monte Carlo Samples for Sliced Wasserstein Distance
#         eps: float > 0;
#            Threshold for Sliced Wasserstein Distance between Batches
#         kappa: int;
#            Minmimum number of mini-batches in distribution buffer
#         mu: int;
#            Maximum number of mini-batches in distribution buffer
#         p: int;
#            Order of Sliced Wasserstein Distance
#     """
#     def __init__(self, data,K,eps,kappa,mu,L,p,delta=0):

#         """
#         Initializes the Detector object.

#         Parameters:
#         - data (numpy array): A NxD array of N instances with D features.
#         - K (int): Length of batches.
#         - L (int): Number of Monte Carlo Samples for Sliced Wasserstein Distance.
#         - eps (float): Threshold for Sliced Wasserstein Distance between batches.
#         - kappa (int): Minimum number of mini-batches in the distribution buffer.
#         - mu (int): Maximum number of mini-batches in the distribution buffer.
#         - p (int): Order of Sliced Wasserstein Distance.
#         """

#         self.data = data
#         self.K =  K
#         self.eps = eps
#         self.kappa = kappa
#         self.mu = mu
#         self.L = L
#         self.p = p
#         self.fimp_data = None
#         self.cp_list = None
#         self.delta = delta


#     def run(self):

#         """
#         Executes the change point detection algorithm using the provided data and parameters.
#         """

#         data_split = sequence_data(self.data, self.K)

#         T = int(len(data_split))

#         current_distribution = []
#         #self.fimp_list = []
#         fimp_id =[]
#         cps = []

#         for i in range(T):
#             if len(current_distribution)< self.kappa:
#                 current_distribution.append(data_split[i])

#                 if len(current_distribution) >= self.kappa:
#                         threshold = get_threshold(current_distribution,self.eps,projections = self.L,delta=self.delta)
#             else:
#                 p = p_test_SWD(data_split[i], current_distribution, T = threshold,  feature_imp = True, projections = self.L, delta=self.delta)
#                 if p > 0.95:
#                     cp = i*self.K
#                     cps.append(cp)
#                     current_distribution =[data_split[i]]
#                 else:
#                     if len(current_distribution) < self.mu:
#                         current_distribution.append(data_split[i])
#                         threshold = get_threshold(current_distribution,self.eps, projections = self.L,delta = self.delta)
#                     else:
#                         current_distribution.pop(0)
#                         current_distribution.append(data_split[i])
#                         threshold = get_threshold(current_distribution, self.eps, projections=self.L, delta = self.delta)

#         #self.fimp_data = {'values': self.fimp_list, 'id':fimp_id}

#         return cps
    
#     def evaluate(self, annotations, cps):

#         """
#         Evaluates the performance of the change point detection using annotations (ground truth labels).

#         Parameters:
#         - annotations (numpy.ndarray): Annotations or ground truth labels.

#         Returns:
#         - tuple: F1 score and covering value.
#         """

#         F1 = f_measure(annotations,cps)
#         cover = covering(annotations, cps,self.data.shape[0])

#         return F1, cover




# def build_run(params,keys,annotation,data,delta=0):

#     parameter = dict(zip(keys, params))
#     if parameter['kappa'] > parameter['mu']:
#             return None
#     else:
#         SWD = Detector(data,K=parameter['K'],eps=parameter['eps'],kappa=parameter['kappa'],mu=parameter['kappa'],L=100,p=2,delta=delta)
#         start = time.time()
#         cps = SWD.run()
#         end = time.time()
#         f1,cov = SWD.evaluate(annotation,cps)
#         run_time = end-start
#         print(run_time)
#         return (
#                 tuple(parameter),
#                 cps,
#                 f1,
#                 cov,
#                 run_time
#             )

# def grid_search(grid,annotations,data):
#     results = {'F1': [], "parameter": [], "cp_id": [], "covering": [],"runtime":[]}

#     n_repeats = len(list(product(*grid.values())))
#     keys = ['K','eps','mu','kappa']
#     print(n_repeats)

#     # Use multiprocessing.Pool to parallelize the grid search
#     with multiprocessing.Pool() as pool:
#         # Map the grid_search_single function to the parameter combinations
#         results_list = pool.starmap(build_run, [(params, keys,annotations,data) for params in product(*grid.values())])

#         # Filter out None results (when kappa > mu)
#     results_list = [result for result in results_list if result is not None]

#     # Unpack the results into separate lists
#     results['parameter'], results['cp_id'], results['F1'], results['covering'],results['runtime'] = zip(*results_list)

#     return results