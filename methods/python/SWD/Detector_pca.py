import numpy as np
from utils import *
from SlicedWasserstein import *
import os
from itertools import product
from sklearn.decomposition import PCA
from collections import Counter
import math
import sys


def get_important_feature(X,n_pc,pca_obj=None):

    """
    Identify the most important features from a PCA transformation.

    Parameters:
    - X : numpy array;
          Input data.
    - n_pc: int;
          Number of principal components to consider.
    - pca_obj: pca object (sklearn);
          Fitted PCA transformer object from scikit-learn, or None to fit a new one.

    Returns:
    - numpy.ndarray: Indices of the most important features.

    This function maps the most important features for the first n_pc principal components back to their original indices in X.
    """
    if pca_obj is None:
        pca_transformer = PCA(n_components=n_pc, svd_solver="randomized")
        pca_transformer.fit(X)
        most_imp = [np.abs(pca_transformer.components_[i]).argmax() for i in range(n_pc)]
    else:
        most_imp = [np.abs(pca_obj.components_[i]).argmax() for i in range(n_pc)]
    init_features = np.arange(X.shape[1])
    most_imp_names = [init_features[most_imp[i]] for i in range(n_pc)]

    overview_pca = [most_imp_names[i] for i in range(n_pc)]

    res, ind = np.unique(overview_pca,return_index=True)

    overview_pca = res[np.argsort(ind)]

    return overview_pca


def cosine_similarity(a,b):

    """
    Calculate the cosine similarity between two vectors represented as dictionaries.

    Parameters:
    - a (dict): A dictionary representing the first vector.
    - b (dict): A dictionary representing the second vector.

    Returns:
    - float: Cosine similarity between the two vectors.
    """

    terms = set(a).union(b)
    dotprod = sum(a.get(k,0) * b.get(k,0) for k in terms)
    magA = math.sqrt(sum(a.get(k,0)**2 for k in terms))
    magB = math.sqrt(sum(b.get(k,0)**2 for k in terms))

    return dotprod/(magA*magB)


class Detector_pca:
    
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
    def __init__(self, data,K,eps,kappa,mu,L,p,pca_args,sim_threshold):
        self.data = data
        self.K =  K
        self.eps = eps
        self.kappa = kappa
        self.mu = mu
        self.L = L
        self.p = p
        self.fimp_data = None
        self.cp_list = None
        self.pca_transformer = PCA(**pca_args)
        self.sim_threshold = sim_threshold



    def run(self):

        data_split = sequence_data(self.data, self.K)

        T = int(len(data_split))

        current_distribution = []
        initial_distribution = []
        self.fimp_list = []
        fimp_id =[]
        self.cp_list = []

        for i in range(T):
            if len(initial_distribution)< self.kappa:
                initial_distribution.append(data_split[i])

                if len(initial_distribution) >= self.kappa:
                        self.pca_transformer.fit(np.vstack(tuple(initial_distribution)))
                        current_distribution = [self.pca_transformer.transform(data) for data in initial_distribution]
                        threshold = get_threshold(current_distribution,self.eps, projections = self.L)
                        
            else:
                data_split_transformed = self.pca_transformer.transform(data_split[i])
                swd, fimp = get_swd(data_split_transformed, current_distribution, feature_imp = True, projections = self.L)
                self.fimp_list.append(fimp)
                fimp_id.append(i)
                if swd > threshold:
                    current_pcas = get_important_feature(np.vstack(tuple(initial_distribution)),n_pc=self.pca_transformer.n_components,pca_obj=self.pca_transformer)
                    new_pcas = get_important_feature(np.vstack((np.vstack(tuple(initial_distribution[1:])),data_split[i])),n_pc=self.pca_transformer.n_components)
                    if cosine_similarity(Counter(current_pcas), Counter(new_pcas)) < self.sim_threshold:
                        cp = i*self.K
                        self.cp_list.append(cp)
                        initial_distribution = [data_split[i]]
                    else:
                       # initial_distribution.pop(0)
                        initial_distribution.append(data_split[i])
                        #current_distribution.pop(0)
                        self.pca_transformer.fit(np.vstack(tuple(initial_distribution)))
                        current_distribution.append(self.pca_transformer.transform(data_split[i]))
                        current_distribution = [self.pca_transformer.transform(data) for data in initial_distribution]
                        threshold = get_threshold(current_distribution,self.eps)
                else:
                    if len(current_distribution) < self.mu:
                        initial_distribution.append(data_split[i])
                        current_distribution.append(self.pca_transformer.transform(data_split[i]))
                        threshold = get_threshold(current_distribution,self.eps, projections = self.L)
                    else:
                        initial_distribution.pop(0)
                        initial_distribution.append(data_split[i])
                        current_distribution.pop(0)
                        self.pca_transformer.fit(np.vstack(tuple(initial_distribution)))
                        current_distribution = [self.pca_transformer.transform(data) for data in initial_distribution]
                        threshold = get_threshold(current_distribution, self.eps, projections=self.L)

        self.fimp_data = {'values': self.fimp_list, 'id':fimp_id}

        return self.cp_list
    
    def evaluate(self, cps, annotations):
        F1 = f_measure(annotations,cps)
        cover = covering(annotations, cps,self.data.shape[0])
        return F1, cover
    
    def grid_search(self,param_grid,annotations):
        """
        Perform a grid search over the specified parameter grid.

        Parameters:
        - param_grid (dict): A dictionary where keys are parameter names and values are lists of possible values for each parameter.
        - annotations (numpy.ndarray): Annotations or ground truth labels.

        Returns:
        - dict: A dictionary containing the results of the grid search.
        - 'F1' (list): List of F1 scores for each parameter combination.
        - 'parameter' (list): List of parameter combinations.
        - 'cp_id' (list): List of change point IDs for each parameter combination.
        - 'covering' (list): List of covering values for each parameter combination.

        Examples:
        ```python
        # Example usage:
        >>> param_grid = {"kappa": [0.1, 0.2], "eps": [0.01, 0.02], "mu": [0.05, 0.1], "K": [5, 10]}
        >>> annotations = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1])
        
        >>> your_instance = YourClass(...)
        >>> results = your_instance.grid_search(param_grid, annotations)
        
        >>> print("Grid Search Results:")
        >>> print("F1 Scores:", results['F1'])
        >>> print("Parameter Combinations:", results['parameter'])
        >>> print("Change Point IDs:", results['cp_id'])
        >>> print("Covering Values:", results['covering'])
        ```
        """
        results ={'F1':[],"parameter":[],"cp_id":[],"covering":[]}


        for params in product(*param_grid.values()):
            param_combination = dict(zip(param_grid.keys(),params))
            if param_combination['kappa']> param_combination['mu']:
                next
            else:
                self.K = int(param_combination['K'])
                self.eps = param_combination['eps']
                self.kappa = param_combination['kappa']
                self.mu = param_combination['mu']
                self.pca_transformer = PCA(**param_combination['pca_args'])
                self.run()
                results['cp_id'].append(self.cp_list)
                results['parameter'].append(tuple(param_combination.values()))
                results['F1'].append(f_measure(annotations,self.cp_list))
                results['covering'].append(covering(annotations,self.cp_list,self.data.shape[0]))
        return results


if __name__ == '__main__':
    pca_args =  {'n_components': 4, 'svd_solver': "randomized"}

    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    print(script_directory)
    print(os.listdir(os.path.join(script_directory,"Datasets","Gas_Sensor")))
    filename = "gas_sensor_sample_6_1.json"
    file_path = os.path.join(script_directory,"Datasets","Gas_Sensor",filename)
    annotations_path = os.path.join(script_directory,"Datasets","Gas_Sensor","annotations_gas.json")
    data, mat = load_dataset(file_path)
    #mat = np.load("Libras_features.npy")

    with open(annotations_path, "r") as fp:
       gas_annotations = json.load(fp)
    annotations = gas_annotations[filename[:-5]]

    #annotations = {"1": np.load("libra_cps.npy")}
    print(annotations)

    SwATCH_pca  = Detector_pca(mat,5,1.45,4,8,100,2,pca_args,0.4)
    print(SwATCH_pca.run())
    print(SwATCH_pca.evaluate(annotations))

    List_K = np.arange(3,10,1,dtype=int)
    List_eps = np.arange(1.0,1.4,0.1)
    List_Kappa = np.arange(4,7,1)
    List_MU = np.arange(4,8,1)
    List_pca_args = [{'n_components': 3, 'svd_solver': "randomized"},{'n_components': 4, 'svd_solver': "randomized"},{'n_components': 5, 'svd_solver': "randomized"}]
    param_grid ={"kappa":List_Kappa, "eps":List_eps, "mu":List_MU,"K":List_K, "pca_args": List_pca_args}

    print("Simulation for : {}".format(filename))

    results = SwATCH_pca.grid_search(param_grid,annotations)
    
    print("### Values corresponding to max F1 ###")
    max_ind = results['F1'].index(max(results['F1']))
    print("F1 score:",results['F1'][max_ind])
    print("Parameter:",results['parameter'][max_ind])
    print("CP ids:",results['cp_id'][max_ind])
    print("covering:",results['covering'][max_ind])

    print('### Values correpsonding to max Covering ###')
    max_ind_cov = results['covering'].index(max(results['covering']))
    print("F1 score:",results['F1'][max_ind_cov])
    print("Parameter:",results['parameter'][max_ind_cov])
    print("CP ids:",results['cp_id'][max_ind_cov])
    print("covering:",results['covering'][max_ind_cov])



