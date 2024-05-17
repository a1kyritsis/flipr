import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHORT = 0
LONG = 1
CLEAR = 2

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
        self.value = starting_capital # the current value is 
        self.cash = starting_capital # upon itiialization the portfolio is completely liquid
        self.position_size = starting_capital
        self.holdings = pd.DataFrame()
        self.holdings["asset_index"] = asset_indices # asset the index corresponding to the assets in our portfolio
        self.holdings["holdings"] = 0 # since we are holding nothing upon initialization
        self.holdings["position"] = CLEAR # 0 = short, 1 = long, 2 nothing
        self.holdings["short_price"] = 0

    def update(self, decision_data, target_returns = None):   
        """
        Updates the portfolio and each time step s_i.
        Decision data is a panadas df where
            pred := k x 1 column of model predictions in the form of 1 = BUY & 0 = SELL
            price := k x 1 column of prices at s_i
            market_cap := 1 x k column of market capitalizatioin at s_i
        (optional) target_returns := target returns of portfolio. if blank, allocation is based on market capitalization weights 
        """
        self.update_value(decision_data)
        stock_updates = decision_data[decision_data["position"] != self.holdings["position"]] # grab all the stocks that need to have positions changed
        clear_group = stock_updates[stock_updates["position"] == CLEAR] # get the group for which positions we are clearing
        short_group = stock_updates[stock_updates["position"] == SHORT] # group for which positions are being shorted
        long_group = stock_updates[stock_updates["position"] == LONG] # group for which positions are long
        clear_short_group = short_group[self.holdings.loc[short_group["asset_index"], "position"] == LONG] # for those assets that we are shorting but are currently long, we need to clear the position
        clear_long_group = long_group[self.holdings.loc[long_group["asset_index"], "position"] == SHORT] # for those assets that we are buying but are currently short, we need to clear the position
        self.clear_position(clear_group) # clear the position of the clear group
        self.clear_position(clear_short_group) # clear the position of the short gorup
        self.clear_position(clear_long_group) # clear the position of the long group
        self.position_size = self.cash # the position size will be the remaining liquidity in the portfolio

        if ((short_group.shape[0] + long_group.shape[0]) == 0): # we are not making any updates 
            percent_alloc = 0 # the allocation size is 0
        else:
            percent_alloc = 1 / (short_group.shape[0] + long_group.shape[0]) # else, devide equally among the positions we are taking #self.holdings.shape[0]

        self.cash -= self.short(short_group, percent_alloc) # if we are shorting, we see the stock at its curret price
        self.cash -= self.long(long_group, percent_alloc) # if we are going long, we buy the stocks

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
        self.holdings.loc[indices, "position"] = LONG
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
        self.holdings.loc[indices, "holdings"] = short_quantity # update the holdings                       #self.holdings.loc[indices, "holdings"]
        self.holdings.loc[indices, "short_price"] = short_group["price"] # note the short price             # dfmi.loc[:, ('one', 'second')]
        self.holdings.loc[indices, "position"] = SHORT # update portfolio position
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
        self.cash += (self.holdings.loc[indices, "holdings"] * (2 * self.holdings.loc[indices, "short_price"] - short_cover_group.loc[indices, "price"])).sum()
        return

    def clear_position(self, clear_group):
        """
        Handler for cover_short and sell_long.
        Updates position in holdings.
        """
        if clear_group.shape[0] <= 0:
            return

        indices = clear_group["asset_index"] # grab the indices of the group
        clear_group["position"] = self.holdings.loc[indices, "position"]
        self.sell_long(clear_group[clear_group["position"] == LONG])
        self.cover_short(clear_group[clear_group["position"] == SHORT])
        self.holdings.loc[indices, "holdings"] = 0 # set the holdings to 0 since these assets have been sold
        self.holdings.loc[indices, "position"] = CLEAR # update the holdings to reflect the new position
        return

    def update_value(self, decision_data):

        self.value = 0
        long_group = self.holdings[self.holdings["position"] == LONG]
        long_indices = long_group["asset_index"]
        short_group = self.holdings[self.holdings["position"] == SHORT]
        short_indices = short_group["asset_index"]
        self.value += (short_group["holdings"] * short_group["short_price"] - decision_data.loc[short_indices, "price"]).sum()
        self.value += (long_group["holdings"] * decision_data.loc[long_indices, "price"]).sum()
        self.value += self.cash

    def reset(self, starting_capital):
        """
        Resets portfolio instance
        to factory settings.
        """
        self.value = starting_capital # the current value is 
        self.cash = starting_capital # upon itiialization the portfolio is completely liquid
        self.position_size = starting_capital
        self.holdings["holdings"] = 0 # since we are holding nothing upon initialization
        self.holdings["position"] = CLEAR
        self.holdings["short_price"] = 0



    def portfolio_weights(self, d, target_return):
        """
        Returns 1 x k vector of portfolio weights
        d := number of assets bought at s_i
        """

        pass
