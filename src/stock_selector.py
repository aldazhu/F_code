
from abc import ABC, abstractmethod
# stock selector abstract class
class StockSelector:
    def __init__(self):
        pass
    
    @abstractmethod
    def select(self, stock_list):
        raise NotImplementedError("StockSelector::select() is not implemented")
    
class RSIStockSelector(StockSelector):
    def __init__(self, days, high_thresh, low_thresh):
        self.days = days
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

    def select(self, stock_list):
        # stock_list is a list of stock data
        # stock data is a dictionary with the following keys
        # open, high, low, close, volume, adj_close
        # return a list of stock symbols
        pass

class PETTMStockSelector(StockSelector):
    def __init__(self):
        pass
        
    def select(self, stock_list):
        # stock_list is a list of stock data
        # stock data is a dictionary with the following keys
        # open, high, low, close, volume, adj_close
        # return a list of stock symbols
        pass