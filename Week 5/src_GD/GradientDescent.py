import numpy as np
import logistic_regression_functions as f

class GradientDescent(object):
    """Perform the gradient descent optimization algorithm for an arbitrary
    cost function.
    """

    def __init__(self, cost, gradient, predict_func, 
                 alpha=0.01,
                 num_iterations=10000, fit_intercept = True, standardize = True):
        """Initialize the instance attributes of a GradientDescent object.

        Parameters
        ----------
        cost: The cost function to be minimized.
        gradient: The gradient of the cost function.
        predict_func: A function to make predictions after the optimization has
            converged.
        alpha: The learning rate.
        num_iterations: Number of iterations to use in the descent.

        Returns
        -------
        self: The initialized GradientDescent object.
        """
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.standardize = standardize

    def fit(self, X, y, step_size = None):
        """Run the gradient descent algorithm for num_iterations repetitions.

        Parameters
        ----------
        X: A two dimensional numpy array.  The training data for the
            optimization.
        y: A one dimensional numpy array.  The training response for the
            optimization.

        Returns
        -------
        self:  The fit GradientDescent object.
        """

        cost_step = []
        
        if self.standardize:
            
            X = self.scale_X(X)
        
        if self.fit_intercept:
            
            X = f.add_intercept(X)
        
        self.coeffs = np.zeros(X.shape[1])
        
        if step_size == None:
        
            for _ in range(self.num_iterations):
                
                grad = self.gradient(X, y, self.coeffs)
                self.coeffs = self.coeffs - self.alpha * grad
            
            
        else:
            while True:
            
                self.coeffs = self.coeffs - self.alpha * self.gradient(X, y, self.coeffs)
                cost_step.append(self.cost(X,y,self.coeffs)) 
            
                if ((cost_step[-2] - cost_step[-1]) <= step_size):
                    print(cost_step[-1], cost_step[-2], count)
                    break
            
            
        

    def predict(self, X):
        """Call self.predict_func to return predictions.

        Parameters
        ----------
        X: Data to make predictions on.

        Returns
        -------
        preds: A one dimensional numpy array of predictions.
        """
        
        if self.standardize:
            
             X = self.scale_X(X)
        
        if self.fit_intercept:
            
            X = f.add_intercept(X)
        
        preds = self.predict_func(X, self.coeffs)
        
        return preds
    
    def scale_y(self, y):
        
        y = (y - np.mean(y)) / np.std(y)
        
        return y
        
        
    def scale_X(self, X):
        
        X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
        
        return X
