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
        
    def calc_diff(self, index):
        """take derivate of each theta and calculate the difference of h(thetas)
        compared to our data"""
        calc_h = self.features * self.thetas
        diff = calc_h - self.target
        result_diff = diff.transpose()*self.features[:,index]
        
        return result_diff
    
    def fit_model(self, iteration):
        """fit gradient descent to our data using initial parameters"""
        self.check_data()
        num = 0
        while True:
            if num == iteration:
                break
                
            for idx,value in enumerate(np.asarray(self.thetas).flatten()):
                diff = self.learning_rate*(self.target.size)*self.calc_diff(idx)
                self.thetas[idx] = self.thetas[idx] - diff
            num += 1

    def show_params(self, num: int=2):
        """show resulting parameters after fitting gradient descent for about
        m-iteration
        num: int = number of parameters want to show
        """
        result_thetas = np.asarray(self.thetas).flatten()
        for idx,each in enumerate(result_thetas):
            print("THETA: {}: {}".format(idx, each))
        return result_thetas