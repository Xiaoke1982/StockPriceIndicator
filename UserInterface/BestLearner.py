from BagLearner import *
from KNN import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


class BestLearner(object):
	"""
	The class object is initialized by inputing a training dataframe.
	The input dataframe contains stock indicators and rate of change as target variable 
		and stock prices and future actual prices of a specific symbol.
	The training set is divided into a smaller training set and a validation set.
	Several models are built based on the smaller training set.
	The models are KNN, BagLearner, Regression Tree and Random Forest.
	These models are built using different tunning parameters.
	These models are evaluated based on the validation set and the optimal model with 
		the optimal tunning parameters is obtained.
	The model evaluation metrics used is the mean absolute percentage error (mape) on 
		the validation set.
	Using the optimal tunning parameters, the optimal model is built again based on the 
		whole training set (combine small training set and validation set). 
	This final optimal model is used for the future prediction.
	"""

	def __init__(self, df_train):
		"""
		@param df_train: processed dataframe of a specific symbol
						It contains stock indicators and 
						rate of change as target variable and
						stock prices and future actual prices.
		@initialization:
		self.df_train: processed dataframe, df_train, including xs and y and prices 

		self.mean:          columns means, for normalization
		self.std:           columns std, for normalization
		self.trainX_normed: 2-d array, normalized training x
		self.trainY:        Series, training y 

		self.small_trainX_normed: 2-d array, normalized small training x
		self.small_trainY:        Series, small training y

		self.validationX_normed: 	 2-d array, normalized validation x
		self.validationPrices:   	 prices of the symbol in validation set
		self.validationFuturePrices: future actual prices of the symbol in the validation set 
		"""
		self.df_train = df_train
		trainX = self.df_train[self.df_train.columns[:-3]].values
		self.mean = trainX.mean(axis = 0)
		self.std = trainX.std(axis = 0)
		self.trainX_normed = (trainX - self.mean) / self.std
		self.trainY = self.df_train["Y"]

		n = trainX.shape[0]
		small_train_size = int(n * 0.75)

		small_trainX = trainX[:small_train_size]
		small_mean = small_trainX.mean(axis = 0)
		small_std = small_trainX.std(axis = 0)
		self.small_trainX_normed = (small_trainX - small_mean) / small_std
		#self.small_trainX is a ndarray

		self.small_trainY = self.trainY[:small_train_size]
		#self.small_trainY is a Series

		validationX = trainX[small_train_size:]
		self.validationX_normed = (validationX - small_mean) / small_std
		#self.validationX is a ndarray

		self.validationPrices = self.df_train["price"][small_train_size:]
		#self.validationPrices is a Series

		self.validationFuturePrices = self.df_train["future_price"][small_train_size:]
		#self.validationFuturePrices is a Series	

	def bestKNN(self):
		"""
		The method builds several KNN learners with different k based on small training set.
		Then these learners are evaluated based on the validation set.
		The optimal learner with optimal k then is selected. 

		@return: optimal int k, which denotes the optimal k of the optimal KNN learner 
				 that give the best performance on validation set, 
				 and the corresponding mean absolute percentage error (mape)
				
		The performance metrics is the mean absolute percentage error (mape)
		between predicted future prices and actual future prices of validation set.

		The choices of k are integers ranging from 3 to 20.
		"""
		optimal_mape = 1
		optimal_k = 3

		for k in range(3, 20):
			knn = KNNLearner(k = k)
			knn.fit(self.small_trainX_normed, self.small_trainY)
			predicted_roc = knn.predict(self.validationX_normed)
			predicted_prices = self.validationPrices * (1 + predicted_roc)
			mape = np.mean(np.abs(predicted_prices - self.validationFuturePrices) / self.validationFuturePrices)
			if mape < optimal_mape:
				optimal_mape = mape
				optimal_k = k

		return optimal_k, optimal_mape 

	def bestBagLearner(self):
		"""
		The method builds several BagLearner models based on small training set. 
		The base model in the BagLearner is the KNN learner.
		The k in KNN learner is obtained by the method self.bestKNN()
		These BagLearner models have different n_learners parameter, which denotes number
		of base learners in the bag. 
		The optimal model with the optimal n_learners will be obtained based on validation set.

		@return: optimal integer n_learners, which denotes the number of learners in the bag

		The choices of n_learners are 20, 30, 40, ..., 100
		"""
		k = self.bestKNN()[0]
		optimal_n = 20
		optimal_mape = 1
		for n_learners  in range(20, 101, 10):
			bag = BagLearner(learner=KNNLearner, kwargs={"k":k}, n_learners=n_learners)
			bag.fit(self.small_trainX_normed, self.small_trainY)
			predicted_roc = bag.predict(self.validationX_normed)
			predicted_prices = self.validationPrices * (1 + predicted_roc)
			mape = np.mean(np.abs(predicted_prices - self.validationFuturePrices) / self.validationFuturePrices)
			if mape < optimal_mape:
				optimal_mape = mape
				optimal_n = n_learners

		return optimal_n, optimal_mape

	def bestTree(self):
		"""
		The method builds a set of regression tree models with different tunning paramters 
			based on the small training set.
		The tunning parameter used is the min_samples_split, which is a parameter for controling 
			the complexity of the tree.
		The values of min_samples_split are: 5, 10, 15, 20, 25, 30.
		The optimal tree with the optimal min_samples_split is obtained based on validation set.
		
		@return: optimal min_samples_split, the mape on validation set
		"""
		optimal_min_samples_split = 5
		optimal_mape = 1

		for min_samples_split in range(5, 31, 5):
			tree = DecisionTreeRegressor(min_samples_split=min_samples_split, random_state=817)
			tree.fit(self.small_trainX_normed, self.small_trainY)
			predicted_roc = tree.predict(self.validationX_normed)
			predicted_prices = self.validationPrices * (1 + predicted_roc)
			mape = np.mean(np.abs(predicted_prices - self.validationFuturePrices) / self.validationFuturePrices)
			if mape < optimal_mape:
				optimal_mape = mape
				optimal_min_samples_split = min_samples_split

		return optimal_min_samples_split, optimal_mape

	def bestForest(self):
		"""
		The method builds several random forest regression models with different n_estimators values 
			based on small training set.
		The tunning parameter n_estimators denotes the number of trees in the forest.
		These forest models then are evaluated based on the validation set.
		The optimal n_estimators is obtained.

		@return: optimal n_estimators, corresponding mape on validation set
		"""
		optimal_mape = 1
		optimal_n_trees = 20

		for n_trees in range(20, 101, 10):
			forest = RandomForestRegressor(n_estimators=n_trees, random_state=817)
			forest.fit(self.small_trainX_normed, self.small_trainY)
			predicted_roc = forest.predict(self.validationX_normed)
			predicted_prices = self.validationPrices * (1 + predicted_roc)
			mape = np.mean(np.abs(predicted_prices - self.validationFuturePrices) / self.validationFuturePrices)
			if mape < optimal_mape:
				optimal_mape = mape
				optimal_n_trees = n_trees

		return optimal_n_trees, optimal_mape


	def get_best_learner(self):
		"""
		The method compares optimal KNN, BagLearner, tree and random forest based on their mapes
			and the best model among these 4 models is selected as the final best model.
		The final best model is fit again using the whole training set, not just the small training set.
		The best model is stored in self.best_model. 
		Also, the values of the tunning parameters in the best model are stored in self.best
		"""
		k, mape_knn = self.bestKNN()

		n, mape_bag = self.bestBagLearner()

		mss, mape_tree = self.bestTree()

		n_trees, mape_forest = self.bestForest()

		best = "knn"
		best_mape = mape_knn

		if mape_bag < best_mape:
			best = "bag"
			best_mape = mape_bag

		if mape_tree < best_mape:
			best = "tree"
			best_mape = mape_tree

		if mape_forest < best_mape:
			best = "forest"
			best_mape = mape_tree

		if best == "knn":
			self.best_model = KNNLearner(k = k)
			self.best = {"model":best, "k":k, "mape on validation": best_mape}

		elif best == "bag":
			self.best_model = BagLearner(learner=KNNLearner, kwargs={"k":k})
			self.best = {"model":best, "n_learners":n, "k":k, "mape on validation": best_mape}

		elif best == "tree":
			self.best_model = DecisionTreeRegressor(min_samples_split=mss, random_state=817)
			self.best = {"model":best, "min_samples_split":mss, "mape on validation": best_mape}

		else:
			self.best_model = RandomForestRegressor(n_estimators=n_trees, random_state=817)
			self.best = {"model":best, "n_trees":n_trees, "min_samples_split":mss, "mape on validation": best_mape}

		self.best_model.fit(self.trainX_normed, self.trainY)


	def predict(self, points):
		"""
		@param points: must be 2-d array, if 1-d array, then reshape it to 2-d array
		@return: 1-d array

		The method takes a 2-d array of technical indicator values as input.
		The self.best_model is used to make the prediction of the future price based on the input.
		"""
		
		# points needs to be normalized first before making the prediction
		points_normed = (points - self.mean) / self.std
		
		return self.best_model.predict(points_normed)













