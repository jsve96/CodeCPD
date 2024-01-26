import sys
import numpy as np
from src.utils import *
from src.SlicedWasserstein import *
import os
from itertools import product
import timeit
import concurrent.futures
import time
from src.Distances import *

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
    def __init__(self, data,K,eps,kappa,mu,L,p):

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

    def run(self):

        """
        Executes the change point detection algorithm using the provided data and parameters.
        """

        data_split = sequence_data(self.data, self.K)

        T = int(len(data_split))

        current_distribution = []
        self.fimp_list = []
        fimp_id =[]
        cps = []

        for i in range(T):
            if len(current_distribution)< self.kappa:
                current_distribution.append(data_split[i])

                if len(current_distribution) >= self.kappa:
                        threshold = MMD_Threshold(current_distribution,eps=self.eps)
            else:
                M_stat = US_MMD(np.vstack(current_distribution), np.array(data_split[i]))
               #swd, fimp = get_swd(data_split[i], current_distribution, feature_imp = True, projections = self.L)
                #swd= get_swd(data_split[i], current_distribution, feature_imp = False, projections = self.L)
                #self.fimp_list.append(fimp)
                #fimp_id.append(i)
                if M_stat > threshold:
                    cp = i*self.K
                    cps.append(cp)
                    current_distribution =[data_split[i]]
                else:
                    if len(current_distribution) < self.mu:
                        current_distribution.append(data_split[i])
                        threshold = MMD_Threshold(current_distribution,eps=self.eps)
                    else:
                        current_distribution.pop(0)
                        current_distribution.append(data_split[i])
                        threshold = MMD_Threshold(current_distribution,eps=self.eps)

       # self.fimp_data = {'values': self.fimp_list, 'id':fimp_id}

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
    
    # def grid_search(self,param_grid,annotations):
    #     """
    #     Perform a grid search over the specified parameter grid.

    #     Parameters:
    #     - param_grid (dict): A dictionary where keys are parameter names and values are lists of possible values for each parameter.
    #     - annotations (numpy.ndarray): Annotations or ground truth labels.

    #     Returns:
    #     - dict: A dictionary containing the results of the grid search.
    #     - 'F1' (list): List of F1 scores for each parameter combination.
    #     - 'parameter' (list): List of parameter combinations.
    #     - 'cp_id' (list): List of change point IDs for each parameter combination.
    #     - 'covering' (list): List of covering values for each parameter combination.

    #     Examples:
    #     ```python
    #     # Example usage:
    #     >>> param_grid = {"kappa": [0.1, 0.2], "eps": [0.01, 0.02], "mu": [0.05, 0.1], "K": [5, 10]}
    #     >>> annotations = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1])
        
    #     >>> your_instance = YourClass(...)
    #     >>> results = your_instance.grid_search(param_grid, annotations)
        
    #     >>> print("Grid Search Results:")
    #     >>> print("F1 Scores:", results['F1'])
    #     >>> print("Parameter Combinations:", results['parameter'])
    #     >>> print("Change Point IDs:", results['cp_id'])
    #     >>> print("Covering Values:", results['covering'])
    #     ```
    #     """
    #     results ={'F1':[],"parameter":[],"cp_id":[],"covering":[]}


    #     for params in product(*param_grid.values()):
    #         param_combination = dict(zip(param_grid.keys(),params))
    #         if param_combination['kappa']> param_combination['mu']:
    #             next
    #         else:
    #             self.K = int(param_combination['K'])
    #             self.eps=param_combination['eps']
    #             self.kappa=param_combination['kappa']
    #             self.mu=param_combination['mu']
    #             self.run()
    #             results['cp_id'].append(self.cp_list)
    #             results['parameter'].append(tuple(param_combination.values()))
    #             results['F1'].append(f_measure(annotations,self.cp_list))
    #             results['covering'].append(covering(annotations,self.cp_list,self.data.shape[0]))
    #     return results
    
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
            cps = self.run()
            return (
                tuple(param_combination.values()),
                cps,
                f_measure(annotations, cps),
                covering(annotations, cps, self.data.shape[0])
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
        results = {'F1': [], "parameter": [], "cp_id": [], "covering": []}

        self.param_grid = param_grid  # Store the param_grid as an attribute
        n_repeats = len(list(product(*param_grid.values())))
        # Use ProcessPoolExecutor to parallelize the grid search
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Map the grid_search_single function to the parameter combinations
            results_list = list(executor.map(self.grid_search_single, product(*param_grid.values()), [annotations]*n_repeats))

        # Filter out None results (when kappa > mu)
        results_list = [result for result in results_list if result is not None]

        # Unpack the results into separate lists
        results['parameter'], results['cp_id'], results['F1'], results['covering'] = zip(*results_list)

        return results
        


    


if __name__ == '__main__':

    print(0)