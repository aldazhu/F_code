import strategy as sg
import dataPro as dp

from abc import ABC, abstractmethod

class StrategyStatus:
    def __init__(self, strategy_name, status):
        self.strategy_name = strategy_name
        self.status = status

# stock selector abstract class
class StockSelector:
    def __init__(self):
        pass
        self.selected_stocks_ = {} # key: stock code, value: string of strategy name and buy/sell,date_index的flag
        self.is_flag_calculated_ = False
        self.stock_flags_ = {}
        
    @abstractmethod
    def select(self, date_index):
        raise NotImplementedError("StockSelector::select() is not implemented")
    
    @abstractmethod
    def calculate_flag(self, stock_list):
        raise NotImplementedError("StockSelector::calculate_flag() is not implemented")
    
class RSIStockSelector(StockSelector):
    def __init__(self,  days, high_thresh, low_thresh):
        super().__init__()
        self.days = days
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        
    def calculate_flag(self, stock_list):
        for stock in stock_list:
            data = dp.readData(stock)
            data = dp.getNormalData(data)
            flagList = sg.RSIStrategy(data, self.days, self.high_thresh, self.low_thresh)
            self.stock_flags_[stock] = flagList
        self.is_flag_calculated_ = True
        

    def select(self, date_index):
        self.selected_stocks_ = {}
        if not self.is_flag_calculated_:
            raise Exception("StockSelector::calculate_flag() is not called before select()")
        for stock, flagList in self.stock_flags_.items():
            if date_index < len(flagList):
                if flagList[date_index] != 0:
                    self.selected_stocks_[stock] = StrategyStatus("RSI", flagList[date_index])
        return self.selected_stocks_
    
class CCIThreshStockSelector(StockSelector):
    def __init__(self, days, CCI_thresh, after_Ndays):
        super().__init__()
        self.days = days
        self.CCI_thresh = CCI_thresh
        self.after_Ndays = after_Ndays
        
    def calculate_flag(self, stock_list):
        for stock in stock_list:
            data = dp.readData(stock)
            data = dp.getNormalData(data)
            flagList = sg.CCIThresh(data, self.days, self.CCI_thresh, self.after_Ndays)
            self.stock_flags_[stock] = flagList
        self.is_flag_calculated_ = True
        
    def select(self, date_index):
        self.selected_stocks_ = {}
        if not self.is_flag_calculated_:
            raise Exception("StockSelector::calculate_flag() is not called before select()")
        for stock, flagList in self.stock_flags_.items():
            if date_index < len(flagList):
                if flagList[date_index] != 0:
                    self.selected_stocks_[stock] = StrategyStatus("CCI", flagList[date_index])
        return self.selected_stocks_
    
class OSCStockSelector(StockSelector):
    def __init__(self, short, long):
        super().__init__()
        self.short = short
        self.long = long
        
    def calculate_flag(self, stock_list):
        for stock in stock_list:
            data = dp.readData(stock)
            data = dp.getNormalData(data)
            flagList = sg.OSCStrategy(data, self.short, self.long)
            self.stock_flags_[stock] = flagList
        self.is_flag_calculated_ = True
        
    def select(self, date_index):
        self.selected_stocks_ = {}
        if not self.is_flag_calculated_:
            raise Exception("StockSelector::calculate_flag() is not called before select()")
        for stock, flagList in self.stock_flags_.items():
            if date_index < len(flagList):
                if flagList[date_index] != 0:
                    self.selected_stocks_[stock] = StrategyStatus("OSC", flagList[date_index])
        return self.selected_stocks_
    

