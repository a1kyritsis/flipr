import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from functions import loaddata, MACD, RSI
from portfolio import Portfolio


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return 0
    else:
        return 2
    
def my_log(x):
    if x != 0 and np.isnan(x) == False:
        return np.log(x)
    else:
        return x
    
def stock_prediction_model(
        training_period_start = 20180101, 
        training_period_end = 20190101,
        testing_period_start = 20190101,
        testing_period_end = 20200101,
        total_market = True,
        individual = False,
        polynomial_features = 1,
        print_results = True
    ):
    
    poly = PolynomialFeatures(polynomial_features)
    total = []
    total1 = []
    chk = []
    test = []
    pls = []
    indices = []

    rtxm_ti2 = loaddata("rtxm_ti2", training_period_start, training_period_end)
    rtxm_ti1 = loaddata("rtxm_ti1", training_period_start, training_period_end)
    r_ti2 = loaddata("r_ti2", training_period_start, training_period_end)
    volume = loaddata("volall_day", training_period_start, training_period_end)
    mid_close = loaddata("mid_close", training_period_start, training_period_end)
    risk = loaddata("bfast_totrisk", training_period_start, training_period_end)

    rtxm_ti2_test = loaddata("rtxm_ti2", testing_period_start, testing_period_end)
    rtxm_ti1_test = loaddata("rtxm_ti1", testing_period_start, testing_period_end)
    r_ti2_test = loaddata("r_ti2", testing_period_start, testing_period_end)
    volume_test = loaddata("volall_day", testing_period_start, testing_period_end)
    mid_close_test = loaddata("mid_close", testing_period_start, testing_period_end)
    mid_open_test = loaddata("mid_open", testing_period_start, testing_period_end)

    
    if total_market == True:
        indices = range(mid_close.shape[0])
    elif total_market == False:
        average_risk = risk.mean(axis=1, skipna=True).dropna()
        bottom_third = average_risk.quantile(.3333)
        indices = average_risk[average_risk < bottom_third].index

    #Check to see if we are trying to train a model for individual companies or the entire market
    #Entire Market Model
    if individual == False:
        """ TRAINING DATA AND MODEL """
        #Iterates though every company in the designated market
        #4752 companies if total_market = True, around 1200-1300 companies if total_market = False
        #Total Market vs. Low Volatility 
        for i in indices:
            #Gets the daily closing price
            price_i = mid_close.iloc[i]
            #Gets the overnight residual returns minus market influence
            overnight_i = rtxm_ti1.iloc[i, 1:]
            #Gets yesterday's intraday residual returns minus market influence
            intraday_i = rtxm_ti2.iloc[i,:-1]
            #Gets yesterday's volume and maps it to log space
            volume_i = volume.iloc[i, :-1].map(lambda x: my_log(x), na_action="ignore")
            #Calculates the RSI indicator
            rsi_12 = RSI(rtxm_ti2.iloc[i], 12)[:-1]
            #Calculates the MACD indicator
            macd_26_12_9 = MACD(price_i, 26, 12, 9)[:-1]
            #Creates our labels for whether the raw residual returns will go down, up, or stay the same
            y = r_ti2.iloc[i, 1:].map(lambda x: sign(x), na_action="ignore")
            #Creates a dataframe for the data and drops all days where we do not have sufficient data
            data = {'X1': np.array(overnight_i), 'X2': np.array(intraday_i), "X3": np.array(volume_i), 
                    "X4": np.array(rsi_12), "X5": np.array(macd_26_12_9),
                    "Y": np.array(y)}
            data_train = pd.DataFrame(data)
            new_train = data_train.dropna()
            #Threshold of 100 trading days in a year for the company to be trained on.
            if new_train.shape[0] > 100:
                total1.append(new_train)
        #Concatenates all the companies data together into a dataframe for easy shuffling and training
        final_df = pd.concat(total1)

        #Initialize our model with 1000 epochs for training
        LR = LogisticRegression(max_iter=1000)
        #Randomize our data (keeps x and y together but shuffles rows)
        temp = final_df.sample(frac=1).reset_index(drop=True)
        #Set aside our predictor variable and labels
        final_x_training = temp[["X1","X2","X3","X4","X5"]]
        final_y_training = temp[["Y"]]
        #Creates a polynomial transformation of our feature matrix
        attempt = poly.fit_transform(final_x_training)
        #Fit our model to the data
        LR.fit(attempt, np.ravel(final_y_training))

        """ TESTING DATA AND MODEL """
        #Check shape of data and makes sure there is no cheating
        if total_market == True:
            indices = range(mid_close_test.shape[0])
        elif total_market == False:
            average_risk = risk.mean(axis=1, skipna=True).dropna()
            bottom_third = average_risk.quantile(.3333)
            indices = average_risk[average_risk < bottom_third].index
        
        #Iterates though every company in the designated market
        for i in indices:
            #Initialize amount made per company
            total_made = 0
            #Initializes out test data
            price_i_test = mid_close_test.iloc[i]
            open_price_i_test = mid_open_test.iloc[i]
            overnight_i_test = rtxm_ti1_test.iloc[i, 1:]
            intraday_i_test = rtxm_ti2_test.iloc[i,:-1]
            volume_i_test = volume_test.iloc[i, :-1].map(lambda x: my_log(x), na_action="ignore")
            rsi_12_test = RSI(rtxm_ti2_test.iloc[i], 12)[:-1]
            macd_26_12_9_test = MACD(price_i_test, 26, 12, 9)[:-1]
            y_test = r_ti2_test.iloc[i, 1:].map(lambda x: sign(x), na_action="ignore")
            data_test = {'X1': np.array(overnight_i_test), 'X2': np.array(intraday_i_test), "X3": np.array(volume_i_test), 
                "X4": np.array(rsi_12_test), "X5": np.array(macd_26_12_9_test),
                "Y": np.array(y_test)}
            data_test = pd.DataFrame(data_test, index=y_test.index)
            new_test = data_test.dropna()
            X_test = new_test[["X1","X2","X3","X4","X5"]]  
            y_test = new_test[["Y"]]
            #Check the size of the company
            if X_test.shape[0] > 100:
                #Creates a polynomial transformation of our feature matrix
                x_test = poly.fit_transform(X_test)
                #Records how accurate our model is
                chk.append(LR.score(x_test, y_test))
                #Predicts labels for time frame of one company
                signals = LR.predict(x_test)
                count = 0
                #Take the dates for each label
                signal_dates = new_test.index
                #Iterates through the timeframe
                for j in range(len(signals)):
                    #Collects the open and close price for a given day
                    price1 = open_price_i_test.loc[signal_dates[j]]
                    price2 = price_i_test.loc[signal_dates[j]]
                    #Records how often our y-data was correctly recorded
                    pls.append(int(price1 - price2 < 0) == np.array(y_test)[j][0])

                    #Short
                    if signals[j] == 0:
                        gain = price1 - price2
                        test.append(gain > 0)
                        total_made += gain
                    #Long
                    elif signals[j] == 1:
                        gain = price2 - price1
                        test.append(gain > 0)
                        total_made += gain
                #Record the total made/lost for each company
                total.append(total_made)

    #Individual Models for each company
    elif individual == True:
        #Iterates though every company in the designated market
        for i in indices:
            total_made = 0
            #TRAINING DATA
            price_i = mid_close.iloc[i]
            overnight_i = rtxm_ti1.iloc[i, 1:]
            intraday_i = rtxm_ti2.iloc[i,:-1]
            volume_i = volume.iloc[i, :-1].map(lambda x: my_log(x), na_action="ignore")
            rsi_12 = RSI(rtxm_ti2.iloc[i], 12)[:-1]
            macd_26_12_9 = MACD(price_i, 26, 12, 9)[:-1]
            y = r_ti2.iloc[i, 1:].map(lambda x: sign(x), na_action="ignore")
            data = {'X1': np.array(overnight_i), 'X2': np.array(intraday_i), "X3": np.array(volume_i), 
                    "X4": np.array(rsi_12), "X5": np.array(macd_26_12_9),
                    "Y": np.array(y)}
            data_train = pd.DataFrame(data)
            new_train = data_train.dropna()
            temp = new_train.sample(frac=1).reset_index(drop=True)
            X_train = temp[["X1","X2","X3","X4","X5"]]
            y_train = temp[["Y"]]

            #TESTING DATA
            price_i_test = mid_close_test.iloc[i]
            open_price_i_test = mid_open_test.iloc[i]
            overnight_i_test = rtxm_ti1_test.iloc[i, 1:]
            intraday_i_test = rtxm_ti2_test.iloc[i,:-1]
            volume_i_test = volume_test.iloc[i, :-1].map(lambda x: my_log(x), na_action="ignore")
            rsi_12_test = RSI(rtxm_ti2_test.iloc[i], 12)[:-1]
            macd_26_12_9_test = MACD(price_i_test, 26, 12, 9)[:-1]
            y_test = r_ti2_test.iloc[i, 1:].map(lambda x: sign(x), na_action="ignore")
            data_test = {'X1': np.array(overnight_i_test), 'X2': np.array(intraday_i_test), "X3": np.array(volume_i_test), 
                "X4": np.array(rsi_12_test), "X5": np.array(macd_26_12_9_test),
                "Y": np.array(y_test)}
            data_test = pd.DataFrame(data_test, index=y_test.index)
            new_test = data_test.dropna()
            X_test = new_test[["X1","X2","X3","X4","X5"]]  
            y_test = new_test[["Y"]]

            #Train and Test individual models
            if new_train.shape[0] > 100 and new_test.shape[0] > 100:
                LR = LogisticRegression(max_iter=1000)
                x_train = poly.fit_transform(X_train)
                x_test = poly.fit_transform(X_test)
                LR.fit(x_train, np.ravel(y_train))
                chk.append(LR.score(x_test, y_test))

                signals = LR.predict(x_test)
                count = 0

                signal_dates = new_test.index

                for j in range(len(signals)):
                    price1 = open_price_i_test.loc[signal_dates[j]]
                    price2 = price_i_test.loc[signal_dates[j]]
                    pls.append(int(price1 - price2 < 0) == np.array(y_test)[j][0])
                    #Short
                    if signals[j] == 0:
                        gain = price1 - price2
                        test.append(gain > 0)
                        total_made += gain
                    #Long
                    elif signals[j] == 1:
                        gain = price2 - price1
                        test.append(gain > 0)
                        total_made += gain
                total.append(total_made)

    if print_results == True:
        print("Mean Score: ", sum(chk)/len(chk))
        print("Max Score: ", max(chk))
        print("Min Score: ", min(chk))
        print("Proportion of correct labels: ", sum(pls)/len(pls))
        print("Proportion of correct guesses: ", sum(test)/len(test))
        print("Total Money Made: ", sum(total))
        #The following 2 metrics should be taken with a grain of salt because the initial funds are an extremely crude estimation
        print("Initial Funds: ", mid_open_test.iloc[:,0].dropna().sum())
        print("YoY: ", sum(total) / mid_open_test.iloc[:,0].dropna().sum())
    else:
        return sum(total), sum(chk)/len(chk), max(chk), min(chk)