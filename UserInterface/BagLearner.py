from KNN import *
import numpy as np

class BagLearner(object):
    """
    The class builds a bag model that contains many base regression models.
    The base regression models in the bag is KNN model.
    Each model is built based on the data set bootstraped from the training data.
    The default number of KNN models in the bag is 20, and it can be changed 
        during the initialization.
    The class also provides a method to make predictions on new inputs.
    """
    
    def __init__(self, learner=KNNLearner, kwargs={"k":3}, n_learners=20, random_state=817):
    	"""
    	@param leaner: base leaner class, defaulst is the KNN model 
    	@param n_leaners: int, number of base leaners in the bag, default is 20

    	self.leaners: list, store leaners
    	"""
        self.random_state = random_state
        self.learners = []
        self.n_learners = n_learners
        for i in range(self.n_learners):
            self.learners.append(learner(**kwargs))
        
    
    def fit(self, dataX, dataY):
    	"""
    	For each leaner in the bag, the function bootstrap
    	   the training data and fit the model.

    	@param dataX: 2d-array
    	@param dataY: Series
    	"""
        n = dataX.shape[0]
        # n is the number of training instances

        #set seed for reproduce result
        np.random.seed(self.random_state)

        for i in range(self.n_learners):
        	#bootstrap n indices
            idxes = np.random.choice(n, n)
        
            X = dataX[idxes] #bootstrap training X
            Y = dataY[idxes] #bootstrap training Y
            self.learners[i].fit(X, Y)
    
    def predict(self, points):
    	"""
    	@param points: ndarray
    	@return: 1-d array
    	"""
        ans = self.learners[0].predict(points)
        # ans is a 1-d array

        for i in range(1, self.n_learners):
            ans += self.learners[i].predict(points)
        
        #average the predictions of learners in the bag    
        ans = ans / self.n_learners
        return ans
