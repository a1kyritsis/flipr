import pandas as pd
import numpy as np

class financials:

    def residual_returns(time_slice):
        """
        Takes in a k x i matrix of asset prices
        where k := number of assets and
        i := number of days and return
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
        time_slice_vector = time_slice_vector.reset_index(drop = True)
        i = time_slice_vector.size # get number of days
        w = np.exp(np.arange(i))
        w = pd.Series(w / np.sum(w))
        return (w * time_slice_vector).sum()


    def upper_boiler_band(time_slice_vector):
        """
        Takes a 1 x i time slice vector of residuals
        where i := number of days and confidence
        coeffecient gamma and calculates the upper 
        boiler band of an expodential moving average
        """
        m = financials.expo_moving_average(time_slice_vector)
        time_slice_vector = time_slice_vector.reset_index(drop = True)
        sd = time_slice_vector.std()
        gamma = 1
        return m + gamma * sd

    def lower_boiler_band(time_slice_vector):
        """
        Takes a 1 x i time slice vector of residuals
        where i := number of days and confidence
        coeffecient gamma and calculates the lower 
        boiler band of an expodential moving average
        """
        m = financials.expo_moving_average(time_slice_vector)
        time_slice_vector = time_slice_vector.reset_index(drop = True)
        sd = time_slice_vector.std()
        gamma = 1
        return m - gamma * sd

    def signal_to_noise(time_slice_vector):
        """
        Takes a 1 x i time slice vector of residuals
        where i := number of days and calculates
        the signal to noise ratio (m / v) with
        m := expo moving average and v :=  variance 
        (W.R.T. time_slice)
        """
        m = financials.expo_moving_average(time_slice_vector)
        time_slice_vector = time_slice_vector.reset_index(drop = True)
        v = time_slice_vector.var()
        return  m / v

class fin_handlers:

    def create_features(data_matrix, fin_functions, labels = None):
        """
        Takes an n x i data matrix of residuals where
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

        indices = np.arange(data_matrix.shape[1])

        def row_wise_apply(row):

            return pd.Series([func(row) for func in fin_functions])
        
        feature_matrix = data_matrix.apply(row_wise_apply, axis=1)
        feature_matrix.columns = labels
        return feature_matrix


