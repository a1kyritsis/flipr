import pandas as pd
import numpy as np

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

