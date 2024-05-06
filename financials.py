import pandas as pd
import numpy as np

class financials:

    def residual_returns(time_slice):
        """
        Takes in a k x i matrix of asset prices
        where k := number of assets and
        n := number of days and return
        residual returns for days 1 to i
        as pd df
        """
        if time_slice.shape[1] <= 1:
            return pd.DataFrame()
        if time_slice.shape[0] <= 0:
            return pd.DataFrame()

        residuals = time_slice.pct_change(axis = 1) # calculate percent change
        residuals = residuals.drop(residuals.columns[0], axis = 1) # drop NaN column
        return residuals
    
    def simple_moving_average(time_slice_vector):
        """
        Takes a 1 x i time slice vector of residuals
        where i := the number of days and calculates
        a simple moving average
        """
        return time_slice_vector.mean()
    
    def expo_moving_average(time_slice_vector):
        """
        Takes a 1 x i time time slice vector of residuals
        where i :=  number of days and calculates
        a base 2 expodential moving average
        """
        i = len(time_slice_vector) # get number of days
        w = (pd.Series(np.cumprod([1/2] * i))[::-1]).transpose() # calculates base 2 expodential weights
        return (w * time_slice_vector).sum()


    def upper_boiler_band(time_slice_vector):
        """
        Takes a 1 x i time slice vector of residuals
        where i := number of days and confidence
        coeffecient gamma and calculates the upper 
        boiler band of an expodential moving average
        """
        sd = time_slice_vector.std()
        gamma = 1
        return financials.expo_moving_average(time_slice_vector) + gamma * sd

    def lower_boiler_band(time_slice_vector):
        """
        Takes a 1 x i time slice vector of residuals
        where i := number of days and confidence
        coeffecient gamma and calculates the lower 
        boiler band of an expodential moving average
        """
        sd = time_slice_vector.std()
        gamma = 1
        return financials.expo_moving_average(time_slice_vector) - gamma * sd

    def signal_to_noise(time_slice_vector):
        """
        Takes a 1 x i time slice vector of residuals
        where i := number of days and calculates
        the signal to noise ratio (m / v) with
        m := expo moving average and v :=  variance 
        (W.R.T. time_slice)
        """
        v = time_slice_vector.var()
        return financials.expo_moving_average(time_slice_vector) / v

class fin_handlers:

    def create_features(data_matrix, fin_functions, labels = None):
        """
        Takes an n x i data matrix of raw prices where
        i := time slice and n := number of observations
        (observations may be of the same asset),
        list financial functions of size f, and
        optional labels. Returns an n x f feature matrix.
        """
        if data_matrix.shape[0] <= 0:
            return pd.DataFrame()
        if data_matrix.shape[1] <= 1:
            return pd.DataFrame()
        f = len(fin_functions)
        if f <= 0:
            return pd.DataFrame()
        if labels == None:
            labels = []
            for i in range(0, f):
                labels.append("f_" + str(i))
        
        def row_wise_apply(row):

             return pd.Series([func(row) for func in fin_functions])

        data_matrix = financials.residual_returns(data_matrix)
        feature_matrix = data_matrix.apply(row_wise_apply, axis=1)
        feature_matrix.columns = labels
        return feature_matrix


