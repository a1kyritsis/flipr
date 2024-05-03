import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loaddata(name: str, start_date = 20100104, end_date = 20211231):
    dates = pd.read_csv("Data_Files/dates.csv", header=None)
    data = pd.read_csv("Data_Files/"+  name + ".csv", header=None, names=dates.T.values.tolist()[0])

    return data.loc[:, start_date:end_date]

def MACD(data: pd.DataFrame, ema_long: int, ema_short: int, signal_length: int):

    long = data.dropna().ewm(span=ema_long, ignore_na=True).mean()
    short = data.dropna().ewm(span=ema_short, ignore_na=True).mean()
    macd = short - long
    signal = macd.ewm(span=signal_length, ignore_na=True).mean()

    return macd - signal

def WWMA(data: pd.DataFrame, ma_period: int):
    data = data.dropna()
    new = [0 for i in range(len(data))]
    for i,x in enumerate(data.to_numpy()):
        if i != 0:
            new[i] = new[i-1] + ((1/ma_period) * (x - new[i-1]))
        else:
            new[i] = x
    print(new)
    return pd.DataFrame(new, index=data.index)

def RSI(data: pd.DataFrame, period: int):
    rsi_values = []
    cleaned = data.dropna()
    av_gain_array = []
    av_loss_array = []
    for i in range(len(cleaned)):
        if i + 1 < period:
            batch = cleaned.iloc[:i+1]
            positives = batch[batch >= 0]
            negatives = batch[batch < 0]
            av_gain_array.append((positives.sum()+.001) / (i+1))        
            av_loss_array.append(((-1*negatives.sum())+.001) / (i+1))
        else:
            current_move = cleaned.iloc[i]
            pos = current_move if current_move >= 0 else 0
            neg = -1*(current_move) if current_move < 0 else 0
            av_gain_array.append(((av_gain_array[i-1] * (period-1)) + pos)/period)
            av_loss_array.append(((av_loss_array[i-1] * (period-1)) + neg)/period)
    #print(np.array(av_gain_array), np.array(av_loss_array) )
    RS = np.array(av_gain_array) / np.array(av_loss_array)
    return 100 - (100/(1+RS))
