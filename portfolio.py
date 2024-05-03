import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Portfolio:
    """
    Tracks model network and asset holdings.
    """

    def __init__(self, asset_indices, starting_capital):
        """
        Initialize with starting capital, asset group (fixed over simulation duration)
        self.holdings is defined as follows:
        asset_index <- index of asset in our portfolio
        holdings <- number of asset we are holding
        price <- price at which call was made
        position <- type of hold "LONG" "SHORT" "CLEAR"
        """
        self.value = starting_capital
        self.cash = starting_capital
        self.position_size = starting_capital
        self.holdings = pd.DataFrame()
        self.holdings["asset_index"] = asset_indices
        self.holdings["holdings"] = 0
        self.holdings["position"] = ""

    def update(self, decision_data, target_returns = None):   
        """
        Updates the portfolio and each time step s_i.
        Decision data is a panadas df where
            pred := k x 1 column of model predictions in the form of 1 = BUY & 0 = SELL
            price := k x 1 column of prices at s_i
            market_cap := 1 x k column of market capitalizatioin at s_i
        (optional) target_returns := target returns of portfolio. if blank, allocation is based on market capitalization weights 
        """
        short_group = decision_data[decision_data["pred"] == 0]
        long_group =  decision_data[decision_data["pred"] == 1]
        self.clear_position(decision_data)
        self.position_size = self.cash
        self.value = self.cash
        if ((short_group.shape[0] + long_group.shape[0]) == 0):
            percent_alloc = 0
        else:
            percent_alloc = 1 / (short_group.shape[0] + long_group.shape[0])
        self.cash += self.short(short_group, percent_alloc)
        self.cash -= self.long(long_group, percent_alloc)

    def long(self, buy_group, percent_alloc):
        """
        Takes a pandas dataframe of stocks to be bought
        along with current prices. Updates fact in portfolio holdings
        """
        if (buy_group.shape[0]) == 0: # nothing to be done
            return 0
        if (self.position_size) <= 0: # return if we have no money to make allocations
            return 0
        indices = buy_group["asset_index"] # get the indices of stocks to be bought
        w = percent_alloc * np.ones(buy_group.shape[0]) # get the portfolio weights
        target_allocations = w * self.position_size # calculate target allocations as weight vector * cash
        buy_quantity = np.floor(target_allocations / buy_group["price"]) # get the quantity of the stocks
        self.holdings.loc[indices, "holdings"] = buy_quantity # update the current holdings
        self.holdings.loc[indices, "position"] = "LONG"
        return (buy_quantity * buy_group["price"]).sum()
    
    def short(self, short_group, percent_alloc):
        """
        Takes a pandas dataframe of stocks to be shorted
        along with prices at current prices. Updates fact in portfolio holdings
        """
        if (short_group.shape[0]) <= 0: # nothing to be done
            return 0
        if (self.position_size) <= 0: # no money to make calls
            return 0
        indices = short_group["asset_index"] # grab the indices
        w = percent_alloc * np.ones(short_group.shape[0]) # define portfolio weights
        target_allocations = w * self.position_size # find magnitude of allocations
        short_quantity = np.floor(target_allocations / short_group["price"]) # find the quantity
        self.holdings.loc[indices, "holdings"] = short_quantity # update the holdings
        self.holdings.loc[indices, "short_price"] = short_group["price"] # note the short price
        self.holdings.loc[indices, "position"] = "SHORT" # update portfolio position
        return (short_quantity * short_group["price"]).sum() # return magnitude of position

    def sell_long(self, long_sell_group):
        """
        liquidates long assets.
        Updates fact in portfolio holdings
        """
        if (long_sell_group.shape[0] == 0): #nothing to do
            return
        indices = long_sell_group["asset_index"] # get the indices of the stocks to be sold
        self.cash += (self.holdings.loc[indices, "holdings"]  * long_sell_group["price"]).sum() # add the profit/loss to the cash
        return
    
    def cover_short(self, short_cover_group):
        """
        Covers short assets.
        Updates fact in portfolio holdings
        """
        if short_cover_group.shape[0] <= 0: #nothing to do
            return
        indices = short_cover_group["asset_index"]
        self.cash -= (self.holdings.loc[indices, "holdings"] * short_cover_group["price"]).sum()
        return

    def clear_position(self, clear_group):
        """
        Handler for cover_short and sell_long.
        Updates position in holdings.
        """
        indices = clear_group["asset_index"] # grab the indices of the group
        clear_group["position"] = self.holdings.loc[indices, "position"]
        self.sell_long(clear_group[clear_group["position"] == "LONG"])
        self.cover_short(clear_group[clear_group["position"] == "SHORT"])
        self.holdings.loc[indices, "holdings"] = 0 # set the holdings to 0 since these assets have been sold
        self.holdings.loc[indices, "position"] = "CLEAR" # update the holdings to reflect the new position
        return


    def portfolio_weights(self, d, target_return):
        """
        Returns 1 x k vector of portfolio weights
        d := number of assets bought at s_i
        """

        return (1 / d) * np.ones(d)
