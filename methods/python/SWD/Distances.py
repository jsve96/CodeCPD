import numpy as np
from sklearn import metrics

def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()



def US_MMD(X_ref,Y,nb_b=10, mmd_function=mmd_rbf):
    mmd = 0
    for b in range(nb_b):
        ind = np.arange(0,X_ref.shape[0],1)
        samples = X_ref[np.random.choice(ind,Y.shape[0],replace=bool),:]
        mmd+=mmd_function(samples,Y)

    return mmd/nb_b

def MMD_Threshold(X_ref,nb_b=10,mmd_function=mmd_rbf, eps=1.0):
    threshold = 0 
    #X_ref list of lists
    dump = X_ref
    for i,x in enumerate(X_ref):
        #print(x)
        dump = X_ref[:i] + X_ref[i + 1:]
        threhsold_candidate = eps*US_MMD(np.vstack((dump)),np.array(x),mmd_function=mmd_function)
        if threhsold_candidate > threshold:
            threshold = threhsold_candidate
        
    return threshold


if __name__=='__main__':
    np.random.seed(0)
    X1 = np.random.multivariate_normal([0,0], np.array([[1,0],[0,1]]),size=15)
    X2 = np.random.multivariate_normal([1,3], np.array([[1,0],[0,1]]),size=15)
    X3 = np.random.multivariate_normal([0,0], np.array([[1,0],[0,1]]), size=15)

    

    X_ref = [X1,X2,X3]
    print(MMD_Threshold(X_ref))
    #Y = np.random.normal(1,0.5,5).reshape(-1,1)

    