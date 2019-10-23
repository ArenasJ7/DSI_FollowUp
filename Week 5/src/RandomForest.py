from DecisionTree import DecisionTree
from TreeNode import TreeNode
import numpy as np
from scipy import stats


class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees,
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_features):
        '''
        Return a list of num_trees DecisionTrees built using bootstrap samples
        and only considering num_features features at each branch.
        '''
        r = []
        
        for _ in range(num_trees):
            dt = DecisionTree(num_features)
            bootstrap_sample = np.random.choice(range(X.shape[0]), len(X), True)
            dt.fit(X[bootstrap_sample,:], y[bootstrap_sample])
            r.append(dt)
            
        return r

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''
        res = np.empty([X.shape[0],1])
        
            
        for m in self.forest:
                
            r = np.array([m.predict(X)])
            
            res = stats.mode(r)[0][0]
                    
        return res                    
                

    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        
        res = self.predict(X)
        
        for i in range(len(y)):
            
            if y[i] == res[i]:
                
                s += 1
             
        r = s / len(y)    
        
        return r

