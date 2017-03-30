import numpy as np
import pandas as pd 

class KNNLearner(object):
    """
    The class builds a KNN model object and provides a method
        to make prediction on new input points.

    """
    
    def __init__(self, k=3, verbose=False):
        #initialize the k in KNN model.
        self.k = k
        
    def fit(self, dataX, dataY):
        """
        This method stores the training X and Y respectively for the 
            use of the future prediction.

        @param dataX: 2-d array
        @param dataY: Series
        """
        self.X = dataX
        self.Y = dataY
        
        
    def predict(self, points):
        """
        The method takes a 2-d array as input and return a 1-d array
            as the prediction based on the KNN algorithm.
        
        @param points: must be 2d -array, if not reshape it
        @return: 1d-array, predicted values
        """

        #reshape points to 2-d array if points is 1-d array
        if len(points.shape) == 1:
            points = points.reshape(1, -1)

        ans = np.array([])
        for i in range(points.shape[0]):
            xi = points[i]
            dists = np.sqrt(((self.X - xi) ** 2).sum(axis=1))
            idxes = dists.argsort()[:self.k]
            pred_y = self.Y[idxes].mean()
            ans = np.append(ans, [pred_y])
        return ans