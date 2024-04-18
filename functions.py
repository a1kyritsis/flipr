import pandas as pd

def loaddata(name: str, start_date = 20100104, end_date = 20211231):
    dates = pd.read_csv("dates.csv", header=None)
    data = pd.read_csv(name + ".csv", header=None, names=dates.T.values.tolist()[0])

    return data.loc[:, start_date:end_date]

