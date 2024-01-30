import numpy as np
from itertools import combinations

def sw_vectorized(X, Y, L=100, p=2, feature_importance = False, delta = 0.0):

    """
    Computes the Sliced Wasserstein Distance between two samples sets.

    Parameters:
    - X (numpy.ndarray): A 2D array (matrix) representing the first dataset with shape (N, d).
    - Y (numpy.ndarray): A 2D array (matrix) representing the second dataset with shape (N, d).
    - L (int): The number of random projections. Default is 100.
    - p (float): The power of the Wasserstein distance. Default is 2.

    Returns:
    - float: The Vectorized Sliced Wasserstein Distance between the datasets.

    Examples:
    ```python
    # Example usage:
    import numpy as np
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 10)
    result = sw_vectorized(x, y, 1000, 2.0)
    print("Result:", result)
    ```
    """

    N, d = X.shape
    theta = np.random.randn(L,d)
    theta_norm = np.linalg.norm(theta, axis=1)
    theta_normed = theta / theta_norm[:, np.newaxis]
    
    x_proj = np.dot(X, theta_normed.T)
    y_proj = np.dot(Y, theta_normed.T)

    qs = np.linspace(0+delta,1-delta,50)
    xp_quantiles = np.quantile(x_proj, qs, axis=0, method="inverted_cdf")
    yp_quantiles = np.quantile(y_proj, qs, axis=0, method="inverted_cdf")

    dist_p = np.abs(xp_quantiles - yp_quantiles)**p
    wd = dist_p.mean(axis=0)

    if feature_importance:
        #max_wd_index = list(wd).index(max(wd))
        ind = np.argpartition(wd,-25)[-25:]
        return (np.mean(wd))**(1/p), np.mean(np.abs(theta_normed[ind]),axis=0)
        

    return (np.mean(wd))**(1/p)


def wasserstein_1d(X,Y,p=2):
    x_sorted = np.sort(X)
    y_sorted = np.sort(Y)

    return np.mean(np.abs(x_sorted-y_sorted)**p)**(1/p)



def get_threshold(list_of_data,eps,projections=100,p=2,delta=0.0):

    """
    Computes the threshold for detecting changes in sequential data (mini-batches).

    Parameters:
    - list_of_data (list of numpy.ndarray): A list of time series data, where each element is a 2D array (matrix) with shape (N, d).
    - eps (float): A user-defined constant.
    - projections (int): The number of random projections. Default is 100.
    - p (float): The power parameter for the Wasserstein distance. Default is 2.

    Returns:
    - float: threshold.

    Examples:
    ```python
    # Example usage:
    import numpy as np
    data_list = [np.random.randn(100, 10) for _ in range(5)]
    threshold = get_threshold(data_list, 0.1, 100, 2.0)
    print("Threshold:", threshold)
    ```
    """
     
    maximum = 0
    nc = np.array(list_of_data[0]).shape[1]

    # for pair in combinations(list_of_data,2):
    #     if nc > 1:
    #         swd = eps*sw_vectorized(pair[0],pair[1],L=projections)
    #     else: 
    #         swd = eps*wasserstein_1d(pair[0],pair[1],p=p)
    for i, ref_batch in enumerate(list_of_data):

        new_data = np.vstack([v for j, v in enumerate(list_of_data) if j != i])

        if nc >1:
            swd = eps*sw_vectorized(ref_batch, new_data, L=projections,delta=delta)
        else: 
            qs = np.linspace(0+delta,1-delta,100)
            x_qs = np.quantile(ref_batch, qs, axis=0, method="inverted_cdf")
            bootstrap_swd = []
            for k in range(25):
                subset = np.random.choice(new_data.flatten(),max(ref_batch.shape[0],int(new_data.shape[0]*0.3)))
                y_qs = np.quantile(subset, qs, axis=0, method="inverted_cdf")
                bootstrap_swd.append((np.mean(np.abs(x_qs - y_qs)**p))**(1/p))
            #y_qs = np.quantile(new_data, qs, axis=0, method="inverted_cdf")
            #swd = eps*(np.mean(np.abs(x_qs - y_qs)**p))**(1/p)
            swd = eps*np.mean(bootstrap_swd)
            #print(swd)
        if swd >= maximum:
            maximum = swd
    return maximum


def get_swd(batch, list_of_data,projections=100,p=2, feature_imp=False,delta=0.0):

    """
    Computes the Sliced Wasserstein Distance (SWD) between a batch of time series data and a list of data (batches).

    Parameters:
    - batch (numpy.ndarray): The batch of time series data with shape (N, d).
    - list_of_data (list of numpy.ndarray): A list of time series data, where each element is a 2D array (matrix) with shape (N, d).
    - projections (int): The number of random projections. Default is 100.
    - p (float): The power parameter for the Wasserstein distance. Default is 2.
    - feature_imp (bool): If True, returns feature importance. Default is False.

    Returns:
    - float or numpy.ndarray: The computed SWD or feature importance.

    Examples:
    ```python
    # Example usage:
    import numpy as np
    batch_data = np.random.randn(100, 10)
    data_list = [np.random.randn(100, 10) for _ in range(5)]
    
    swd_result = get_swd(batch_data, data_list, 100, 2.0)
    print("SWD Result:", swd_result)
    
    feature_imp_result = get_swd(batch_data, data_list, 100, 2.0, feature_imp=True)
    print("Feature Importance Result:", feature_imp_result)
    ```
    """

    maximum = 0
    nc = np.array(batch).shape[1]
    # for data in list_of_data:
    #     if nc >1:
    #         swd = sw_vectorized(batch, data, L=projections)
    #     else:
    #         swd = wasserstein_1d(batch,data,p=p)
    #     if swd >= maximum:
    #         maximum = swd
    #         max_data = data
    if nc > 1:
        swd = sw_vectorized(batch, np.vstack(list_of_data), L=projections)
    else: 
        qs = np.linspace(0+delta,1-delta,100)
        x_qs = np.quantile(batch, qs, axis=0, method="inverted_cdf")
        y_s = np.vstack(list_of_data)
        bootstrap_swd = []
        for k in range(25):
            subset = np.random.choice(y_s.flatten(),max(batch.shape[0],int(y_s.shape[0]*0.3)))
            y_qs = np.quantile(subset, qs, axis=0, method="inverted_cdf")
            bootstrap_swd.append((np.mean(np.abs(x_qs - y_qs)**p))**(1/p))
        swd = np.mean(bootstrap_swd)
        #print(swd)
        #swd = (np.mean(np.abs(x_qs - y_qs)**p))**(1/p)

    if feature_imp:
        return sw_vectorized(batch, np.vstack(list_of_data), L=projections, feature_importance=feature_imp,delta=delta)
    
    return swd


def p_test_SWD(batch, list_of_data,T,N = 50,projections = 50, feature_imp = False, delta = 0):
    y_s = np.vstack(list_of_data)
    bootstrap = []
    features_list = []
    if feature_imp:
        for trial in range(N):
            subset_ind = np.random.choice(range(y_s.shape[0]),max(batch.shape[0],int(y_s.shape[0]*0.3)))
            subset = y_s[subset_ind,:]
            swd, features = sw_vectorized(batch,subset,L=projections,feature_importance=True,delta=delta)
            bootstrap.append(swd)
            features_list.append(features)
        return np.mean((np.array(bootstrap) > T)*1), np.mean(np.vstack(features_list),axis=0)
        
    else:
        for trial in range(N):
            subset_ind = np.random.choice(range(y_s.shape[0]), max(batch.shape[0],int(y_s.shape[0]*0.3)))
            subset = y_s[subset_ind]
            swd = sw_vectorized(batch, subset, L = projections, feature_importance=False, delta=delta)
            bootstrap.append(swd)
        return np.mean((np.array(bootstrap)> T)*1)