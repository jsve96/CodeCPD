import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def generate_samples(features,targets,n_obs,possible_lenght,seed_in):
    np.random.seed(seed_in)
    classes = list(targets.unique())
    possible_classes = classes
    c = np.random.choice(possible_classes,)
    
    N = np.random.choice(possible_lenght)
   
    sampled_targets = np.repeat(c,N)
    tmp_targets = targets[targets == c].sample(N,random_state=seed_in)
    sampled_features = features.iloc[list(tmp_targets.index)]

    #update targets (remove sampled indicies from original data)
    targets = targets.drop(tmp_targets.index)
    #features = features.drop(tmp_targets.index).reset_index(drop=True)
    prev_c = c
    possible_classes.remove(c)
    while sampled_targets.shape[0] < n_obs:
        c = np.random.choice(possible_classes)
        N = np.random.choice(possible_lenght)
        tmp_targets = targets[targets == c].sample(N,random_state=seed_in)
        possible_classes.remove(c)
        possible_classes.append(prev_c)
        prev_c = c
        sampled_features = pd.concat([sampled_features,features.iloc[tmp_targets.index]],axis=0)
        sampled_targets = np.hstack((sampled_targets,np.repeat(c,N)))
        targets = targets.drop(tmp_targets.index)
        #features = features.drop(tmp_targets.index).reset_index(drop=True)

    return sampled_features.reset_index(drop=True), sampled_targets


def find_annotations(targets):
    annoations = {"1": list(targets[targets[0].diff()!=0].index[1:])}
    return annoations


def plot_samples(df):
    fig = plt.figure(figsize=(18,24))
    plt.subplots_adjust(hspace=0.5)
    N_rows_plot = df.shape[0]//10+1
    for row in df.T:
        ax = plt.subplot(N_rows_plot, 10, row + 1)
        plt.imshow(np.array(df.iloc[row]).reshape(28,28), cmap=plt.get_cmap('gray'),axes=ax)
    plt.tight_layout()
    return fig


class NpEncoder(json.JSONEncoder):
    """
    A custom JSON encoder class that handles NumPy data types when converting to JSON format.
    """

    def default(self,obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder,self).default(obj)
    
def make_dict(df,name):
    series_list = []
    for col in range(df.shape[1]):
        series_list.append({"label":col, "type": "int", "raw": list(df.iloc[:,col])})


    json_out = {"name":"MINST", "longname":name, "n_obs": int(df.shape[0]), 
            "n_dim": 784,
            "time": { "index": np.arange(int(df.shape[0]))},
            "series": series_list}

    return json_out


#oracle_annotations = {"MNIST_IS_" + str(i): anno for i, anno in enumerate(annoationslist_corr)}

#oracle_annotations