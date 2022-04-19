import numpy as np
import pandas as pd

class gradient_des:
    """Using gradient descent to solve linear regression problems."""
    
    def __init__(self, features, target):
        """
        features: numpy.ndarray = contains data with m-rows with n-features
        target: numpy.ndarray = contains m-values that we want to be
        predicted"""
        
        self.features = features
        self.target = target
        
    def check_data(self):
        """add feature 0 to the current features, if there's none in the features.
        Also convert features and target into a numpy.matrix"""
        try:
            #extract first column
            first_column = self.features[:,0]
            check_one = [value==1 for value in first_column]
            if (sum(check_one)/first_column.size)==1:
                pass
            else:
                new_features = []
                for each in self.features:
                    one = [1]
                    one.extend(each)
                    new_features.append(one)
                
                setattr(self, "features", np.matrix(new_features))
        except IndexError:
            new_features = []
            for each in self.features:
                new_features.append([1, each])
        
            setattr(self, "features", np.matrix(new_features))
            
        try:
            if self.target.shape[0]>1 & self.target.shape[1]==1:
                pass
            
            setattr(self, "target", np.matrix(self.target))
        except IndexError:
            new_target = []
            for each in self.target:
                new_target.append([each])
        
            setattr(self, "target", np.matrix(new_target))
            
        
    def set_params(self, learn_rate, thetas):
        """
        Set parameters as an initial value to run gradient descent.
        learn_rate: float = initial guess of learning rate to use in gradient
        descent model
        thetas: list = contains initial guesses of all theta as a starting 
        point for gradient descent model to be converge"""
        
        temp = []
        for value in thetas:
            temp.append([value])
        temp_m = np.matrix(temp)
        
        setattr(self, "thetas", temp_m)
        setattr(self, "learning_rate", learn_rate)
        
    def calc_cost(self):
        """Calculate cost function for the given data: self.features and self.target"""
        cost_temp = (self.features*self.thetas - self.target)
        for idx,value in enumerate(cost_temp):
            cost_temp[idx] = value**2
        cost = sum(cost_temp)/(2*self.target.size)
        return cost.item(0)
    
    def fit_model(self, iteration):
        """fit gradient descent to our data using initial parameters"""
        self.check_data()
        initial_cost = self.calc_cost()
        setattr(self, "initial_cost", initial_cost)
        
        num_iter = np.array([i+1 for i in range(iteration)])
        y = self.target
        X = self.features
        J_iter = []
        for i in range(iteration):
            temp_diff = (self.learning_rate/y.size) * (X.transpose()*((X*self.thetas)-y))
            self.thetas = self.thetas - temp_diff
            J_iter.append(self.calc_cost())
        
        J_iter = np.array(J_iter)
        return (num_iter, J_iter)

    def get_params(self, num: int=2):
        """show resulting parameters after fitting gradient descent for about
        m-iteration
        num: int = number of parameters want to show
        """
        result_thetas = np.asarray(self.thetas).flatten()
        for idx,each in enumerate(result_thetas):
            print("THETA {}: {}".format(idx, each))
        return result_thetas #if you want the result to be an array
    
    def pred_cost(self):
        """Make prediction of target from features and thetas obtained from fitting
        gradient descent"""
        final_cost = self.calc_cost()
        y_pred = self.features * self.thetas
        print("Initial cost function: {}".format(self.initial_cost))
        print("Final cost function: {}".format(final_cost))
        return (final_cost, y_pred)