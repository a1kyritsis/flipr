import random
import numpy as np
import pandas as pd
from financials import financials
from financials import fin_handlers
# !!! include potential models in imports !!!
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier

SHORT = 0
LONG = 1
CLEAR = 2
DATE_BUFFER = 2


class Fin_model:

    def __init__(self, model, fin_functions, labels = None):
        """
        fin_functions := a list of metrics we will train on
        labels := name of financial metrics
        """
        self.model = model()
        self.fin_functions = fin_functions
        self.labels = labels

    def label(self, target_day, epsilon = None):
        """
        Takes 1 x n vector of target day residuals (r)
        and labels SHORT if r_i < -epsilon and
        LONG if r_i > epsilon and CLEAR otherwise
        where epsilon := movement tolerance.
        """
        if epsilon == None:
            epsilon = .1

        labels = np.where(target_day < -epsilon, SHORT, np.where(target_day > epsilon, LONG, CLEAR))
        return pd.Series(labels)

    def get_sequential_features(self, asset_indices, Delta, training_data):
        """
        Takes the indices of assets to train of length n, a collection window
        Delta, and training data. Assume there are d days in the training_data
        Returns a (d x n) x p matrix of training data.
        """
        string_days = pd.Series(training_data.columns.tolist())
        max_days = len(string_days)

        if Delta + DATE_BUFFER > max_days:
            return pd.DataFrame(), pd.Series()
        
        fixed_indices = -1 * np.arange(Delta + 2) # fixed_indices[0] := prediction day, fixed_indices[Delta + 2] := buffer day to calculate residual 
        training_features = pd.DataFrame()
        training_labels = pd.Series()

        for day in range(Delta + DATE_BUFFER, max_days):

            date_indices = (fixed_indices + day)[::-1] # gets the indices in "chronological" order
            dates = string_days.loc[date_indices] # grab the actual date strings
            time_slice = training_data.loc[asset_indices, dates] # get the raw prices
            residuals_i = financials.residual_returns(time_slice) # calculate the residuals using financials
            target_day = residuals_i.iloc[:, Delta] # get the day we are predicting on
            residuals_i.drop(residuals_i.columns[Delta], axis = 1) # remove it, since we don't want to train on it
            labels_i = self.label(target_day, epsilon = .01) # label the target day
            features_i = fin_handlers.create_features(residuals_i, self.fin_functions, self.labels) # get the features
            training_labels = pd.concat([training_labels, labels_i])
            training_features = pd.concat([training_features, features_i], axis = 0) # append to training set

        return training_features.reset_index(drop = True), training_labels.reset_index(drop = True)

    def get_sample_features(self, asset_indices, Delta, k_samples, training_data):
        """
        Takes the indices of assets to train of length n, a collection window
        Delta, the number of samples to conduct k := k_samples, along 
        with the training data. Returns a (k x n) x p matrix of training
        data.
        """
        string_days = pd.Series(training_data.columns.tolist())
        max_days = len(string_days)

        if Delta + DATE_BUFFER > max_days:
            return pd.DataFrame(), pd.Series()
        
        fixed_indices = -1 * np.arange(Delta + 2) # fixed_indices[0] := prediction day, fixed_indices[Delta + 2] := buffer day to calculate residual 
        training_features = pd.DataFrame()
        training_labels = pd.Series()

        for _ in range(0, k_samples):

            random_day = random.randint(Delta + DATE_BUFFER, max_days) # select a random day
            date_indices = (fixed_indices + random_day)[::-1] # gets the indices in "chronological" order
            dates = string_days.loc[date_indices] # grab the actual date strings
            time_slice = training_data.loc[asset_indices, dates] # get the raw prices
            residuals_i = financials.residual_returns(time_slice) # calculate the residuals using financials
            target_day = residuals_i.iloc[:, Delta] # get the day we are predicting on
            residuals_i.drop(residuals_i.columns[Delta], axis = 1) # remove it, since we don't want to train on it
            labels_i = self.label(target_day, epsilon = .01) # label the target day
            features_i = fin_handlers.create_features(residuals_i, self.fin_functions, self.labels) # get the features
            training_labels = pd.concat([training_labels, labels_i])
            training_features = pd.concat([training_features, features_i], axis = 0) # append to training set
            
        
        return training_features.reset_index(drop = True), training_labels.reset_index(drop = True)
    
    def train(self, asset_indices, Delta, training_data, k_samples = None):
        """
        Trains the model on the selected data with selected asset_indices,
        collection window Delta, and trianing data. If k_samples is None,
        defaults to sequential training the data. O.W. , random sampling
        is deployed for number of samples specefied.
        """
        if k_samples == None:
            training_features, labels = self.get_sequential_features(asset_indices, Delta, training_data)
        else:
            training_features, labels = self.get_sample_features(asset_indices, Delta, k_samples, training_data)
        
        if training_features.empty:
            print("Training failed.")
            return
        
        self.model.fit(training_features, labels)

        train_accuracy = self.model.score(training_features, labels)

        print("Accuracy on training data:", train_accuracy)

        return

    def train_with_feature_map(self, asset_indices, Delta, training_data, k_samples = None):

        pass

    def get_prediction_features(self, day, Delta, asset_indices, data_matrix):

        string_days = pd.Series(data_matrix.columns.tolist())
        fixed_indices = -1 * np.arange(Delta + 1) # fixed_indices[0] := prediction day, fixed_indices[Delta + 2] := buffer day to calculate residual 
        date_indices = (fixed_indices + day)[::-1] # gets the indices in "chronological" order
        dates = string_days.loc[date_indices] # grab the actual date strings
        time_slice = data_matrix.loc[asset_indices, dates] # get the raw prices
        residuals_i = financials.residual_returns(time_slice) # calculate the residuals using financials
        return fin_handlers.create_features(residuals_i, self.fin_functions, self.labels) # get the features

    def predict(self, day, Delta, asset_indices, data_matrix):
        """"
        Takes a data_matrix of raw price data
        at time step i and returns predictions.
        """
        if data_matrix.shape[0] <= 0:
            return CLEAR * np.ones(data_matrix.shape[1])
        
        feature_matrx = self.get_prediction_features(day, Delta, asset_indices, data_matrix)
        preds = self.model.predict(feature_matrx)
        return preds

