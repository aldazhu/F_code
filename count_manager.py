import pandas as pd
import os
from my_logger import logger
import dataPro as dp


class StockStock:
    def __init__(self, stock_code,buy_price) -> None:
        self.stock_code = stock_code
        self.buy_price = buy_price
        self.buy_date = ""
        self.sell_date = ""
        self.sell_price = 0
        self.hold_days = 0
        self.current_price = 0

    def update_current_price(self, price):
        self.current_price = price
    
    def get_earning_ratio(self):
        if self.buy_price == 0:
            return 0
        
        if self.sell_price == 0:
            return (self.current_price - self.buy_price) / self.buy_price

        return (self.sell_price - self.buy_price) / self.buy_price
        


class MyCount:
    def __init__(self) -> None:
        self.hold_stocks = {}
        self.total_earning = 0
        self.total_earning_ratio = 0
        self.total_cost_ratio = 0
        self.fllow_stop_loss_ratio = -0.08
        self.data_pool = {}
        self.buy_fee_ratio = 0.0003
        self.sell_fee_ratio = 0.0003

    def status_update(self, stock_code, date_index, action, price):
        if action == "buy":
            if stock_code not in self.hold_stocks:
                self.hold_stocks[stock_code] = StockStock(stock_code,price)
                self.hold_stocks[stock_code].buy_date = date_index
                self.hold_stocks[stock_code].buy_price = price
                logger.info("date: %s, stock: %s, buyPrice: %f", date_index, stock_code, price)
        elif action == "sell":
            if stock_code in self.hold_stocks:
                self.hold_stocks[stock_code].sell_date = date_index
                self.hold_stocks[stock_code].sell_price = price
                self.hold_stocks[stock_code].hold_days = date_index - self.hold_stocks[stock_code].buy_date
                self.total_earning += (self.hold_stocks[stock_code].sell_price - self.hold_stocks[stock_code].buy_price)
                earning_ratio = (self.hold_stocks[stock_code].sell_price - self.hold_stocks[stock_code].buy_price) / self.hold_stocks[stock_code].buy_price
                self.total_earning_ratio += earning_ratio
                logger.info("date: %s, stock: %s, sellPrice: %f, earning: %f \%", date_index, stock_code, price, earning_ratio*100)
                del self.hold_stocks[stock_code]
        else:
            pass
    
    def everyday_update(self, date_index):
        for stock_code in self.hold_stocks:
            if stock_code not in self.data_pool:
                data = dp.readData(stock_code)
                data = dp.getNormalData(data)
                self.data_pool[stock_code] = data
            
            data = self.data_pool[stock_code]
            price = data['close'][date_index]
            self.hold_stocks[stock_code].update_current_price(price)
            earning_ratio = self.hold_stocks[stock_code].get_earning_ratio()
            if earning_ratio < self.fllow_stop_loss_ratio:
                self.status_update(stock_code, date_index, "sell", price)
                logger.info("date: %s, stock: %s, stop loss", date_index, stock_code)
            else:
                logger.info("date: %s, stock: %s, earning_ratio: %f", date_index, stock_code, earning_ratio)
                pass
            

