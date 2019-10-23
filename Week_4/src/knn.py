import numpy as np
import pandas as pd
from collections import Counter


def euclidean_distance(a, b):
    """Compute the euclidean distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    distance: float
    """

    nrow = b.shape[0]
    dist = np.empty([nrow,1])
    
    for i in range(nrow):

        dist[i] = np.linalg.norm(a-b[i])

    return dist

def cosine_distance(a, b):
    """Compute the cosine dissimilarity between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    distance: float
    """

    nrow = b.shape[0]
    dist = np.empty([nrow,1])
    
    for i in range(nrow):

        dist[i] = 1 - a.dot(b[i])/(np.linalg.norm(a)*np.linalg.norm(b[i]))

    return dist
    


def manhattan_distance(a, b):
    """Compute the manhattan (L1) distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    distance: float
    """

    nrow = b.shape[0]
    dist = np.empty([nrow,1])
    
    for i in range(nrow):

    dist[i] = np.sum(np.absolute((a - b[i])))

    return dist

class KNNRegressor:
    """Regressor implementing the k-nearest neighbors algorithm.

    Parameters
    ----------
    k: int, optional (default = 5)
        Number of neighbors that are included in the prediction.
    distance: function, optional (default = euclidean)
        The distance function to use when computing distances.
    """

    def __init__(self, k=5, distance=euclidean_distance):
        """Initialize a KNNRegressor object."""
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        According to kNN algorithm, the training data is simply stored.

        Parameters
        ----------
        X: numpy array, shape = (n_observations, n_features)
            Training data.
        y: numpy array, shape = (n_observations,)
            Target values.

        Returns
        -------
        self
        """

        self.X = X
        self.y = y

        return self

    def predict(self, X):
        """Return the predicted values for the input X test data.

        Assumes shape of X is [n_test_observations, n_features] where
        n_features is the same as the n_features for the input training
        data.

        Parameters
        ----------
        X: numpy array, shape = (n_observations, n_features)
            Test data.

        Returns
        -------
        result: numpy array, shape = (n_observations,)
            Predicted values for each test data sample.

        """

        nrow = X.shape[0]
        result = np.empty([nrow,1])
        res = np.empty([nrow,1])

        for i in range(nrow):

            res = self.distance(X[i], self.X)

            KNN = np.argsort(res, axis = 0)[0:self.k,:]

            result[i] = np.mean(self.y[KNN])
            


        return result
            
                             
if __name__ == '__main__':
    pass
