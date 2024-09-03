import backtrader as bt
import backtrader.feeds as btfeeds
import datetime
import numpy as np
import matplotlib.pyplot as plt

import backtrader.talib as ta
from backtrader_indicator import RSRS, RSRS_Norm, Diff
from my_logger import logger


from xgboost import XGBClassifier, XGBRegressor

class MyData(btfeeds.GenericCSVData):
    params = (
        ('fromdate', datetime.datetime(2022, 11, 1)),
        ('todate', datetime.datetime(2023, 12, 31)),
        ('dtformat', ('%Y-%m-%d')),
        # ('tmformat', ('%H.%M.%S')),
        ('datetime', 1),
        # ('time', -1),
        ('open', 3),
        ('high', 4),
        ('low', 5),
        ('close', 6),
        ('volume', 8),
        # ('openinterest', -1)
    )

class MyMinutelyData(btfeeds.GenericCSVData):
    params = (
        ('fromdate', datetime.datetime(2022, 11, 1)),
        ('todate', datetime.datetime(2023, 12, 31)),
        ('dtformat', ('%Y%m%d%H%M%S%f')),
        # ('tmformat', ('%Y%m%d%H%M%S%f')),
        ('datetime', 2), # 20220104103000000
        # ('time', 2),
        ('open', 4),
        ('high', 5),
        ('low', 6),
        ('close', 7),
        ('volume', 8),
        ('openinterest', -1)
    )

class MySeaData(btfeeds.GenericCSVData):
    params = (
        ('fromdate', datetime.datetime(2022, 11, 1)),
        ('todate', datetime.datetime(2023, 12, 31)),
        ('dtformat', ('%Y-%m-%d')),
        # ('tmformat', ('%H.%M.%S')),
        ('datetime', 0),
        # ('time', -1),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 6),
        # ('openinterest', -1)
    )




# https://www.backtrader.com/docu/quickstart/quickstart/#customizing-the-strategy-parameters
class QuickGuideStrategy(bt.Strategy):
    params = (
        ('exitbars', 5),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.dataclose[-1]:
                    # current close less than previous close

                    if self.dataclose[-1] < self.dataclose[-2]:
                        # previous close less than the previous close

                        # BUY, BUY, BUY!!! (with default parameters)
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])

                        # Keep track of the created order to avoid a 2nd order
                        self.order = self.buy()

        else:

            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + self.params.exitbars):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

# a class of stock data , buy, sell, hold days, earning ratio etc
class StockStatus:
    def __init__(self, stock_code, strategy_name):
        self.stock_code = stock_code
        self.strategy_name = strategy_name
        self.status = 0 # 0 means nothing, 1 means long the stock, -1 means short the stock
        self.buy_date = ""
        self.sell_date = ""
        self.buy_price = 0
        self.sell_price = 0
        self.hold_days = 0
        self.earning_ratio = 0

    def calculate_earning_ratio(self):
        if self.status == 1:
            self.earning_ratio = (self.sell_price - self.buy_price) / self.buy_price
        elif self.status == -1:
            self.earning_ratio = (self.sell_price - self.buy_price) / self.sell_price

        return self.earning_ratio
    
    def calculate_hold_days(self):
        if self.buy_date == "" or self.sell_date == "":
            return -999

        if self.status == 1:
            self.hold_days = (self.sell_date - self.buy_date).days
        elif self.status == -1:
            self.hold_days = (self.buy_date - self.sell_date).days

        return self.hold_days

    def get_status_in_string(self):
        self.calculate_earning_ratio()
        self.calculate_hold_days()
        if self.status == 1:
            status_str = (
                f"Stock code: {self.stock_code}, Strategy: {self.strategy_name}, Status: {self.status},"
                f"Buy date: {self.buy_date}, Buy price: {self.buy_price}, "
                f"Sell date: {self.sell_date},Sell price: {self.sell_price}, "
                f"Hold days: {self.hold_days}, Earning ratio: {self.earning_ratio}"
            )
        elif self.status == -1:
            status_str = (
                f"Stock code: {self.stock_code}, Strategy: {self.strategy_name}, Status: {self.status},"
                f"Sell date: {self.sell_date},Sell price: {self.sell_price}, "
                f"Buy date: {self.buy_date}, Buy price: {self.buy_price}, "
                f"Hold days: {self.hold_days}, Earning ratio: {self.earning_ratio}"
            )

        return status_str


    def print(self):
                
        print(self.get_status_in_string())
        logger.info(self.get_status_in_string())
        
       

class HoldPool:
    def __init__(self):
        self.pool = {}

    def add_record(self, record):
        """
        args:
            record: StockStatus
        """
        self.pool[record.stock_code] = record

    def remove_record(self, stock_code):
        self.pool.pop(stock_code)

    def get_record(self, stock_code):
        return self.pool.get(stock_code)

    def print(self):
        for stock_code, record in self.pool.items():
            record.print()

class StragegyTemplate(bt.Strategy):
    params = (('stop_loss', 0.305), ('stop_earning', 0.35))

    history_records = []

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.logger = logger

        self.max_price_from_buy = 0

        self.change_percent = 0
        self.change_percent_final = 0

        self.hold_pool = HoldPool()

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        time = self.datas[0].datetime.time(0)
        # print('%s T %s, %s' % (dt.isoformat(),time, txt))
        self.logger.info('%s T %s, %s' % (dt.isoformat(),time, txt))

    def query_holding_number(self):
        holding_number = len(self.hold_pool.pool)
        self.logger.info("Holding number: %s" % holding_number)

    def stop_loss_watch_dog(self):
        for i, data in enumerate(self.datas):
            if data._name in self.hold_pool.pool:
                record = self.hold_pool.get_record(data._name)
                if record:
                    if record.buy_price * (1 - self.params.stop_loss) > data.close[0]:
                        self.order = self.sell(data)
                        self.log(f"Stop loss triggered, sell the {data._name} ")
                        
    def stop_eaning_watch_dog(self):
        for i, data in enumerate(self.datas):
            if data._name in self.hold_pool.pool:
                record = self.hold_pool.get_record(data._name)
                if record:
                    if record.buy_price * (1 + self.params.stop_earning) < data.close[0]:
                        self.order = self.sell(data)
                        self.log(f"Stop earning triggered, sell the {data._name}, today's price: {data.close[0]}")
   
    def analyze_the_history(self):
        success_count = 0
        total_count = len(self.history_records)
        buy_price_sum = 0
        sell_price_sum = 0
        earning_ratio = []
        stock_trade_dict = {} # stock_code , record_list
        for record in self.history_records:
            record.print()
            if record.earning_ratio > 0:
                success_count += 1
            buy_price_sum += record.buy_price
            sell_price_sum += record.sell_price
            earning_ratio.append(record.earning_ratio)
            if record.stock_code in stock_trade_dict.keys():
                stock_trade_dict[record.stock_code].append(record)
            else:
                stock_trade_dict[record.stock_code] = [record]

        for stock_code, record_list in stock_trade_dict.items():
            self.logger.info("==============================\n stock code : %s", stock_code)
            long_earning_ratio = 0
            short_earning_ratio = 0
            long_count = 0
            short_count = 0
            for record in record_list:
                
                if record.status == 1:
                    long_count += 1
                    long_earning_ratio += record.earning_ratio
                elif record.status == -1:
                    short_count += 1
                    short_earning_ratio += record.earning_ratio
                self.logger.info("record, %s",record.get_status_in_string())
                # print(record)
            self.logger.info("Long count: %s, Short count: %s", long_count, short_count)
            self.logger.info("Long earning ratio: %s, Short earning ratio: %s", long_earning_ratio , short_earning_ratio )
            self.logger.info("final earning ratio: %s", long_earning_ratio + short_earning_ratio)

        if buy_price_sum == 0:
            self.logger.info("No record in the history")
            return

        if total_count == 0:
            print("No record to analyze")
            return
        final_earning_ratio = (sell_price_sum - buy_price_sum) / buy_price_sum
        self.logger.info("buy price sum: %s, sell price sum: %s, earning ratio: %s", buy_price_sum, sell_price_sum, final_earning_ratio)
        # print(f"buy price sum: {buy_price_sum}, sell price sum: {sell_price_sum}, earning ratio: {final_earning_ratio}")
        self.logger.info("Total count: %s, success count: %s, success rate: %s", total_count, success_count, success_count / total_count)
        # print(f"Total count: {total_count}, success count: {success_count}, success rate: {success_count / total_count}")

        sharp_ratio = np.mean(earning_ratio) / np.std(earning_ratio)
        # print(f"Sharp ratio: {sharp_ratio}")
        
        self.logger.info("mean: %s, std: %s", np.mean(earning_ratio), np.std(earning_ratio))
        self.logger.info("Sharp ratio: %s", sharp_ratio)

        # plot holding days and earning ratio
        holding_days = [record.hold_days for record in self.history_records]
        earning_ratio = [record.earning_ratio for record in self.history_records]
        
        days = np.array(holding_days)
        ratio = np.array(earning_ratio)
        plt.scatter(days, ratio)
        plt.xlabel('Holding days')
        plt.ylabel('Earning ratio')

        # plt histgram
        plt.figure()
        plt.hist(ratio, bins=50)
        plt.xlabel('Earning ratio')
        plt.ylabel('Frequency')

        plt.show()


    def stop(self):
        self.analyze_the_history()

    def get_final_change_percent(self):
        return self.change_percent_final
    
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                if order.data._name not in self.hold_pool.pool:
                    # stock not in hold pool, buy it
                    record = StockStatus(order.data._name, self.__class__.__name__)
                    record.status = 1 # 1 means bought
                    record.buy_date = self.datas[0].datetime.date(0)
                    record.buy_price = order.executed.price
                    self.hold_pool.add_record(record)
                else:
                    # stock already in hold pool, it has been sold before, buy it back
                    record = self.hold_pool.get_record(order.data._name)
                    record.buy_date = self.datas[0].datetime.date(0)
                    record.buy_price = order.executed.price
                    self.hold_pool.remove_record(order.data._name)
                    self.history_records.append(record)

                self.log(
                    'name : %s , BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.data._name,
                     order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                # self.buyprice = order.executed.price
                # self.buycomm = order.executed.comm
                # self.max_price_from_buy = order.executed.price
            else:  # Sell

                if order.data._name in self.hold_pool.pool:
                    # stock in hold pool, sell it
                    record = self.hold_pool.get_record(order.data._name)
                    if record is not None and record.status == 1:
                        record.sell_date = self.datas[0].datetime.date(0)
                        record.sell_price = order.executed.price
                        # record.hold_days = len(self)
                        # record.earning_ratio = (record.sell_price - record.buy_price) / record.buy_price
                        record.print()
                        # erase the record from hold pool
                        self.hold_pool.remove_record(order.data._name)
                        self.history_records.append(record)
                else:
                    # stock not in hold pool, sell it
                    record = StockStatus(order.data._name, self.__class__.__name__)
                    record.status = -1 # -1 means sold
                    record.sell_date = self.datas[0].datetime.date(0)
                    record.sell_price = order.executed.price
                    self.hold_pool.add_record(record)
                    
                self.log('name : %s , SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.data._name,
                          order.executed.price,
                          order.executed.value,
                          order.executed.comm))
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
    def next(self):
        print("This is a template strategy, please implement your own strategy.")



class MovingAverageStrategy(StragegyTemplate):
    params = (('ma_period', 15), ) # class variable , can be accessed by self.params.ma_period
    def __init__(self):
        super().__init__()
        self.ma = []
        for i, data in enumerate(self.datas):
            self.ma.append(bt.indicators.SimpleMovingAverage(data.close, period=self.params.ma_period))
        
    def next(self):
        # check if there is an unfinished order
        if self.order:
            return

        # check if in the market
        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                # if the closing price is above the moving average, buy
                if data.close[0] > self.ma[i][0]:
                    self.order = self.buy(data)
            else:
                # if the closing price is below the moving average, sell
                if data.close[0] < self.ma[i][0]:
                    self.order = self.sell(data)
        

class RSIStrategy(StragegyTemplate):
    params = (('rsi_period', 15), ('rsi_upper', 70), ('rsi_lower', 30),('high_period', 20),('stop_loss', 0.2))
    def __init__(self):
        super().__init__()
        self.min_price = []
        self.rsi = []
        self.highest = []
        for i, data in enumerate(self.datas):
            self.min_price.append(bt.indicators.Lowest(data.low, period=self.params.high_period))
            self.rsi.append(bt.indicators.RSI_Safe(data.close, period=self.params.rsi_period))
            self.highest.append(bt.indicators.Highest(data.high, period=self.params.high_period))

    def next(self):
        
        # self.log("Close: %.2f, RSI: %.2f, Change percent: %.2f" % (self.dataclose[0], self.rsi[0], change_percent) )
        # check if there is an unfinished order
        if self.order:
            return

        for i, data in enumerate(self.datas):
            if self.getposition(data).size > 0:
                if self.rsi[i][0] < self.params.rsi_upper and self.rsi[i][-1] >= self.params.rsi_upper:
                    self.order = self.sell(data)
            else:
                if self.rsi[i][0] > self.params.rsi_lower and self.rsi[i][-1] <= self.params.rsi_lower:
                    self.order = self.buy(data)
        self.query_holding_number()
        
        

class CCIStrategy(StragegyTemplate):
    params = (('cci_period', 15), ('cci_upper', 150), ('cci_lower', -150), ('high_period', 20),)
    def __init__(self):
        super().__init__()
        self.cci = []
        self.highest = []
        for i, data in enumerate(self.datas):
            print(f"data name: {data._name}, num: {i}")
            self.cci.append(bt.indicators.CCI(data, period=self.params.cci_period))
            self.highest.append(bt.indicators.Highest(data.high, period=self.params.high_period))
            print(f"done with {data._name}")

        
        
    def next(self):
        
        # check if there is an unfinished order
        if self.order:
            return

        for i, data in enumerate(self.datas):
            print(f"data name: {data._name}, num: {i}")
            if self.getposition(data).size > 0:
                if self.cci[i][0] < self.params.cci_upper and self.cci[i][-1] >= self.params.cci_upper:
                    self.order = self.sell(data)
            else:
                if self.cci[i][0] > self.params.cci_lower and self.cci[i][-1] <= self.params.cci_lower:
                    self.order = self.buy(data)
            print(f"done with {data._name}")
        
                

class OSCStrategy(StragegyTemplate):
    params = (('short', 10), ('long', 25))
    def __init__(self):
        super().__init__()
        self.short_ma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.params.short)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.params.long)
        self.osc = self.short_ma - self.long_ma

    def next(self):
        change = self.dataclose[0] - self.dataclose[-1]
        change_percent = change / self.dataclose[-1] * 100
        self.log("Close: %.2f, OSC: %.2f, Change percent: %.2f" % (self.dataclose[0], self.osc[0], change_percent) )
        # check if there is an unfinished order
        if self.order:
            return

        # check if in the market
        if self.position:
            # if the OSC is above 0, sell
            if self.osc[0] > 0:
                self.order = self.sell()
        else:
            # if the OSC is below 0, buy
            if self.osc[0] < 0:
                self.order = self.buy()

        if self.position:
            self.stop_loss_watch_dog(self.data_close[0])

class DoubleMAStrategy(StragegyTemplate):
    params = (('short', 10), ('long', 25))
    def __init__(self):
        super().__init__()
        self.short_ma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.params.short)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.params.long)

    def next(self):
        change = self.dataclose[0] - self.dataclose[-1]
        change_percent = change / self.dataclose[-1] * 100
        self.log("Close: %.2f, Short MA: %.2f, Long MA: %.2f, Change percent: %.2f" % (self.dataclose[0], self.short_ma[0], self.long_ma[0], change_percent) )
        # check if there is an unfinished order
        if self.order:
            return

        # check if in the market
        if not self.position:
            # if the short MA crosses the long MA from below, buy
            if self.short_ma[0] > self.long_ma[0] and self.short_ma[-1] < self.long_ma[-1]:
                self.order = self.buy()
        else:
            # if the short MA crosses the long MA from above, sell
            if self.short_ma[0] < self.long_ma[0] and self.short_ma[-1] > self.long_ma[-1]:
                self.order = self.sell()

        if self.position:
            self.stop_loss_watch_dog(self.data_close[0])



class GroupStrategy(StragegyTemplate):
    params = (
        ('ema_short_period', 10),
        ('ema_long_period', 30),
        ('rsi_period',14),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
        ('cci_period', 14),
        ('cci_upper', 150),
        ('cci_lower', -150),
    )
    def __init__(self):
        super().__init__()
        
        # Add indicators
        self.ema_short = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=self.params.ema_short_period)
        self.ema_long = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=self.params.ema_long_period)

        self.rsi = bt.indicators.RSI(self.datas[0].close, period=self.params.rsi_period)
        self.cci = bt.indicators.CCI(self.datas[0], period=self.params.cci_period)

    def next(self):
        change = self.dataclose[0] - self.dataclose[-1]
        change_percent = change / self.dataclose[-1] * 100
        # self.log("Close: %.2f, EMA Short: %.2f, EMA Long: %.2f, RSI: %.2f, CCI: %.2f, Change percent: %.2f" % (self.dataclose[0], self.ema_short[0], self.ema_long[0], self.rsi[0], self.cci[0], change_percent) )
        # check if there is an unfinished order
        if self.order:
            return

        rsi_signal = 0
        cci_signal = 0
        double_emas_signal = 0
        if self.rsi[0] < self.params.rsi_upper and self.rsi[-1] >= self.params.rsi_upper:
            rsi_signal = -1
            self.log('rsi %.2f , rsi sell signal')
        elif self.rsi[0] > self.params.rsi_lower and self.rsi[-1] <= self.params.rsi_lower:
            rsi_signal = 1
            self.log('rsi %.2f , rsi buy signal')

        if self.cci[0] < self.params.cci_upper and self.cci[-1] >= self.params.cci_upper:
            cci_signal = -1
            self.log('cci %.2f , cci sell signal')
        elif self.cci[0] > self.params.cci_lower and self.cci[-1] <= self.params.cci_lower:
            cci_signal = 1
            self.log('cci %.2f , cci buy signal')

        if self.ema_short[0] > self.ema_long[0] and self.ema_short[-1] < self.ema_long[-1]:
            double_emas_signal = 1
            self.log('double ema buy signal')
        elif self.ema_short[0] < self.ema_long[0] and self.ema_short[-1] > self.ema_long[-1]:
            double_emas_signal = -1
            self.log('double ema sell signal')

        sell_flag = False
        rsi_signal = 0
        cci_signal = 0
        if self.position.size > 0: # self.position.size表示当前持仓，小于0表示做空，
            if rsi_signal == -1 or cci_signal == -1 :
                pass
                self.order = self.sell()
                # sell_flag = True
        elif self.position.size <= 0:
            if rsi_signal == 1 or cci_signal == 1 or double_emas_signal == 1:
                self.order = self.buy()


        # if self.position.size > 0 and not sell_flag: # 当天没有卖出，否则可能导致指标卖出一次，stoploss卖出一次
        #     self.stop_loss_watch_dog(self.data_close[0])

    

class CombinedIndicatorStrategy(StragegyTemplate):
    params = (
        ('ma_period', 50),
        ('rsi_period', 14),
        ('cci_period', 20),
        ('cci_upper', 150),
        ('cci_lower', -150),
        ('rsi_upper', 70),
        ('rsi_lower', 30)
    )

    def __init__(self):
        super().__init__()
        self.ma = []
        self.rsi = []
        self.cci = []
        for i, data in enumerate(self.datas):
            self.ma.append(bt.indicators.SimpleMovingAverage(data.close, period=self.params.ma_period))
            self.rsi.append(bt.indicators.RSI(data.close, period=self.params.rsi_period))
            self.cci.append(bt.indicators.CCI(data, period=self.params.cci_period))

    def next(self):
        if self.order:
            return

        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                if self.rsi[i][0] < self.params.rsi_lower or self.cci[i][0] < self.params.cci_lower:
                    self.order = self.buy(data)
            else:
                if self.rsi[i][0] > self.params.rsi_upper or self.cci[i][0] > self.params.cci_upper:
                    self.order = self.sell(data)
        
class DoubleEmaStrategy(StragegyTemplate):
    params = (
        ('fast_ema_period', 7),
        ('slow_ema_period', 15),
    )

    def __init__(self):
        super().__init__()

        self.fast_ema = []
        self.slow_ema = []
        for i, data in enumerate(self.datas):
            self.fast_ema.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.fast_ema_period))
            self.slow_ema.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.slow_ema_period))


    def next(self):
        if self.order:
            return

        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                if self.fast_ema[i][0] > self.slow_ema[i][0] and self.fast_ema[i][-1] < self.slow_ema[i][-1]:
                    self.order = self.buy(data)
            else:
                if self.fast_ema[i][0] < self.slow_ema[i][0] and self.fast_ema[i][-1] > self.slow_ema[i][-1]:
                    self.order = self.sell(data)

class NewHighStrategy(StragegyTemplate):
    params = (
        ('highest_window', 30),
        ('lowest_window', 15),
        ('ema_period', 10),
        ('ema_sell_period', 10)
    )

    def __init__(self):
        super().__init__()
        self.high = []
        self.low = []
        self.ema = []
        self.ema_sell = []
        self.diff = [] # high - low
        self.rsi = []
        for i, data in enumerate(self.datas):
            self.high.append(bt.indicators.Highest(data.high, period=self.params.highest_window))
            self.low.append(bt.indicators.Lowest(data.low, period=self.params.lowest_window))
            self.ema.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_period))
            self.ema_sell.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_sell_period))
            self.diff.append(Diff(data,ema_period=self.params.ema_period))
            self.rsi.append(bt.indicators.RSI_Safe(data.close, period=14))

    def next(self):
        if self.order:
            return

        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0 :
                if data.close[0] > self.high[i][-1] and self.diff[i][0] > 0 and data.close[0] > self.ema[i][0]:
                    # print(f"{data.datetime.date(0)}: name : {data._name} buy , today coloe at {data.close[0]}")
                    self.order = self.buy(data)
            else:
                if data.close[0] < self.ema[i][0]  or (self.rsi[i][-1] > 70 and self.rsi[i][0] < 70):
                    # print(f"{data.datetime.date(0)}: name : {data._name} sell , today close at {data.close[0]}")
                    self.order = self.sell(data)
        # stop loss
        # self.stop_loss_watch_dog()
        # self.stop_eaning_watch_dog()
        self.query_holding_number()

class NewLowStrategy(StragegyTemplate):
    params = (
        ('highest_window', 15),
        ('lowest_window', 50),
        ('ema_period', 5),
        ('ema_sell_period', 10)
    )

    def __init__(self):
        super().__init__()
        self.high = []
        self.low = []
        self.ema = []
        self.ema_sell = []
        for i, data in enumerate(self.datas):
            self.high.append(bt.indicators.Highest(data.high, period=self.params.highest_window))
            self.low.append(bt.indicators.Lowest(data.low, period=self.params.lowest_window))
            self.ema.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_period))
            self.ema_sell.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_sell_period))

    def next(self):
        if self.order:
            return

        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0 :
                if self.low[i][0] < self.low[i][-1] and data.close[0] > data.open[0] and data.close[0] > self.ema[i][0] :
                    print(f"{data.datetime.date(0)}: name : {data._name} buy , today coloe at {data.close[0]}")
                    self.order = self.buy(data)
            else:
                # hold_days = (data.datetime.date(0) - self.hold_pool.get_record(data._name).buy_date).days
                if data.close[0] > self.high[i][-1] :
                    print(f"{data.datetime.date(0)}: name : {data._name} sell , today close at {data.close[0]}")
                    self.order = self.sell(data)
        self.query_holding_number()
        # stop loss
        # self.stop_loss_watch_dog()
        # self.stop_eaning_watch_dog()
        
class MACDTrendFollowingStrategy(StragegyTemplate):
    params = (('macd1', 12), ('macd2', 26), ('macdsig', 14), ('highest_window', 20),)

    def __init__(self):
        super().__init__()
        self.macd = []
        self.crossover = []
        self.highest = []
        for i, data in enumerate(self.datas):
            self.macd.append(bt.indicators.MACD(data.close, period_me1=self.params.macd1, period_me2=self.params.macd2, period_signal=self.params.macdsig))
            self.crossover.append(bt.indicators.CrossOver(self.macd[i].macd, self.macd[i].signal))
            self.highest.append(bt.indicators.Highest(data.high, period=self.params.highest_window))


    def next(self):
        if self.order:
            return
        
        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                if self.crossover[i] > 0 and data.close[0] > self.highest[-1]:
                    self.order = self.buy(data)
            else:
                if self.crossover[i] < 0 or data.close[0] < self.highest[-1] * 0.80:
                    self.order = self.sell(data)


class BollingerBandsStrategy(StragegyTemplate):
    params = (('period', 20), ('devfactor', 2.0),)

    def __init__(self):
        super().__init__()
        self.boll = []
        for i, data in enumerate(self.datas):
            self.boll.append(bt.indicators.BollingerBands(data, period=self.params.period, devfactor=self.params.devfactor))
        
    def next(self):
        if self.order:
            return

        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                if data.close[-1] <= self.boll[i].lines.bot and data.close[0] > self.boll[i].lines.bot:
                    self.order = self.buy(data)
            else:
                if data.close[-1] >= self.boll[i].lines.top and data.close[0] < self.boll[i].lines.top:
                    self.order = self.sell(data)

        

class RSRSStrategy(StragegyTemplate):
    params = (('N', 18), ('value', 5),('rsrs_norm_thresh',0.8))

    def __init__(self):
        super().__init__()

        self.rsrs = []
        self.rsrs_norm = []
        self.rsrs_r2 = []
        self.beta_right = []
        for i, data in enumerate(self.datas):
            self.rsrs.append(RSRS(data))
            self.rsrs_norm.append(RSRS_Norm(data))
            self.rsrs_r2.append(self.rsrs_norm[i] * self.rsrs[i].R2)
            self.beta_right.append(self.rsrs[i] * self.rsrs_r2[i])

    def next(self):
        if self.order:
            return

        # print(f"{self.datas[0].datetime.date(0)}, beta_right: {self.beta_right[0]}, rsrs_norm: {self.rsrs_norm[0]}, rsrs_r2: {self.rsrs_r2[0]}")
        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                if self.rsrs_norm[i][0] > self.params.rsrs_norm_thresh:
                    self.order = self.buy(data)
            else:
                if self.rsrs_norm[i][0] < - self.params.rsrs_norm_thresh:
                    self.order = self.sell(data)

# this strategy is based on the following code: 
# https://github.com/paperswithbacktest/awesome-systematic-trading/blob/main/static/strategies/short-term-reversal-in-stocks.py
# This strategy is based on the short-term reversal effect in stocks. The main idea is that stocks that
# have performed poorly in the past may perform well in the future, and stocks that have performed well 
# in the past may perform poorly in the future.

# The specific implementation steps are as follows:

# First, the strategy selects the 500 stocks with the best liquidity as the initial portfolio.

# Then, from these 500 stocks, the 100 stocks with the largest market value are selected.

# For these 100 stocks, the strategy calculates their returns over the past month and sorts them from high to low based on the returns.

# At the same time, the strategy also calculates the returns of these 100 stocks over the past week and sorts them from low to high based on the returns.

# The strategy will buy the 10 worst-performing stocks over the past week (expecting them to rebound in the future) and short the 10 best-performing stocks over the past month (expecting them to fall in the future).

# This portfolio is rebalanced every week, that is, stocks are reselected every week.
class ShortTermReversalEffectinStocks(StragegyTemplate):
    params = (
        ('liquidity_threshold', 100),
        ('market_value_threshold', 50),
        ('long_short_threshold', 10),
        ('rebalance_period', 5),
    )

    def __init__(self):
        self.liquidity = []
        self.month_return = []
        self.week_return = []
        self.market_value = []
        # self.month
        # self.week
        self.rebalance_period = self.params.rebalance_period
        self.rebalance_counter = 0

        for i, data in enumerate(self.datas):
            self.liquidity.append(data.liquidity)
            self.market_value.append(data.market_value)
            self.month_return.append(data.month_return)
            self.week_return.append(data.week_return)

    def next(self):
        self.rebalance_counter += 1
        if self.rebalance_counter % self.rebalance_period == 0:
            self.rebalance_counter = 0
            self.rebalance()

    def rebalance(self):
        # select the 500 stocks with the best liquidity as the initial portfolio
        self.liquidity = sorted(self.datas, key=lambda x: x.liquidity, reverse=True)[:self.params.liquidity_threshold]
        # select the 100 stocks with the largest market value
        self.market_value = sorted(self.liquidity, key=lambda x: x.market_value, reverse=True)[:self.params.market_value_threshold]
        # calculate the returns of these 100 stocks over the past month and sort them from high to low
        self.month_return = sorted(self.market_value, key=lambda x: x.month_return, reverse=True)
        # calculate the returns of these 100 stocks over the past week and sort them from low to high
        self.week_return = sorted(self.market_value, key=lambda x: x.week_return, reverse=False)
        # buy the 10 worst-performing stocks over the past week and short the 10 best-performing stocks over the past month
        long_stocks = self.week_return[:self.params.long_short_threshold]
        short_stocks = self.month_return[:self.params.long_short_threshold]
        for stock in long_stocks:
            self.buy(stock)
        for stock in short_stocks:
            self.sell(stock)


# The idea behind this strategy is that once a trend is established, it is more likely to continue 
# in that direction than to move against the trend. This is based on the assumption that assets that 
# have performed well in the past will continue to perform well in the future, and vice versa.
class PriceMomumentStrategy(StragegyTemplate):
    params = (
        ('period', 30),
        ('ema_period', 30),
        ('top_k', 10),
    )
    def __init__(self):
        super().__init__()
        self.momentum = [] # momentum in percentage
        self.ema = []
        # self.hold_pool = HoldPool()
        for i, data in enumerate(self.datas):
            print(f"{i}  {data._name}")
            self.momentum.append(bt.indicators.Momentum(data.close, period=self.params.period) / data.close[-self.params.period] * 100)
            self.ema.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_period))
            print(f"{data._name} done")

    def next(self):
        pass
        
        #  calculate the momentum of each stock
        moment_list = [] # item is (i, momentum)
        for i, data in enumerate(self.datas):
            moment_list.append((i, self.momentum[i][0]))
        
        # sort the momentum list by momentum
        moment_list = sorted(moment_list, key=lambda x: x[1], reverse=True)
        # select the top k stocks with the highest momentum
        top_k_stocks = moment_list[:self.params.top_k]
        # buy the top k stocks
        for stock in top_k_stocks:
            print(f"top {self.params.top_k} {self.datas[stock[0]]._name}, momentum: {stock[1]}")
            now_price = self.datas[stock[0]].close[0]
            last_price = self.datas[stock[0]].close[-self.params.period]
            cal_momentum = (now_price - last_price) / last_price * 100
            print(f"now price: {now_price}, last price: {last_price}, calculated momentum: {cal_momentum}")
            # print(f"next close price: {self.datas[stock[0]].close[1]}, next open price: {self.datas[stock[0]].open[1]}")
            # print(f"yesterday price: {self.datas[stock[0]].close[-1]}")
            stock_code = self.datas[stock[0]]._name
            if self.hold_pool.get_record(stock_code) is None :
                self.buy(self.datas[stock[0]])
        # sell the stocks that are not in the top k
        for i, data in enumerate(self.datas):
            if (i, self.momentum[i][0]) not in top_k_stocks:
                stock_code = self.datas[i]._name
                if self.hold_pool.get_record(stock_code) is not None:
                    
                    self.sell(data)
        
        # input("Press Enter to continue...")
        self.query_holding_number()


        
class InvertPriceMomumentStrategy(StragegyTemplate):
    params = (
        ('period', 30),
        ('top_k', 10),
    )
    def __init__(self):
        super().__init__()
        self.momentum = [] # momentum in percentage
        # self.hold_pool = HoldPool()
        for i, data in enumerate(self.datas):
            # print(f"{data._name}")
            self.momentum.append(bt.indicators.Momentum(data.close, period=self.params.period) / data.close[-self.params.period] * 100)
            # print(f"{data._name} done")

    def next(self):
        pass
        
        #  calculate the momentum of each stock
        moment_list = [] # item is (i, momentum)
        for i, data in enumerate(self.datas):
            moment_list.append((i, self.momentum[i][0]))
        
        # sort the momentum list by momentum
        moment_list = sorted(moment_list, key=lambda x: x[1], reverse=False)
        # select the top k stocks with the highest momentum
        top_k_stocks = moment_list[:self.params.top_k]
        # buy the top k stocks
        for stock in top_k_stocks:
            print(f"top k {self.datas[stock[0]]._name}, momentum: {stock[1]}")
            now_price = self.datas[stock[0]].close[0]
            last_price = self.datas[stock[0]].close[-self.params.period]
            cal_momentum = (now_price - last_price) / last_price * 100
            print(f"now price: {now_price}, last price: {last_price}, calculated momentum: {cal_momentum}")
            stock_code = self.datas[stock[0]]._name
            if self.hold_pool.get_record(stock_code) is None:
                self.buy(self.datas[stock[0]])
        # sell the stocks that are not in the top k
        for i, data in enumerate(self.datas):
            if (i, self.momentum[i][0]) not in top_k_stocks:
                stock_code = self.datas[i]._name
                if self.hold_pool.get_record(stock_code) is not None:
                    
                    self.sell(data)
        
        self.query_holding_number()


# short the topk stocks with the highest return in the past month
# long the topk stocks with the lowest return in the past week
class PriceMomumentStrategyForUS(StragegyTemplate):
    params = (
        ('long_period', 20),
        ('short_period', 15),
        ('top_k', 10),
        ('volume_period', 20),
        ('volume_topk', 100), # the top k stocks with the highest volume
    )
    def __init__(self):
        super().__init__()
        self.long_momentum = [] # momentum in percentage
        self.short_momentum = []
        self.valume = []
        self.ema = []
        # self.hold_pool = HoldPool()
        for i, data in enumerate(self.datas):
            print(f"{i}  {data._name}")
            self.long_momentum.append(bt.indicators.Momentum(data.close, period=self.params.long_period) / data.close[-self.params.long_period] * 100)
            self.short_momentum.append(bt.indicators.Momentum(data.close, period=self.params.short_period) / data.close[-self.params.short_period] * 100)
            print(f"{data._name} done")

    def next(self):
        pass

        sum_volume = [] # item is (i, volume)
        for i, data in enumerate(self.datas):
            mean_volume = sum([data.volume[j]*data.close[j] for j in range(-self.params.volume_period, 0)]) / self.params.volume_period
            sum_volume.append((i, mean_volume))


        # sort volume list by volume
        sum_volume = sorted(sum_volume, key=lambda x: x[1], reverse=True) # highest first

        sum_volume = sum_volume[:self.params.volume_topk]

        #  calculate the momentum of each stock
        short_moment_list = [] # item is (i, momentum)
        long_moment_list = []
        for i, data in sum_volume:
            short_moment_list.append((i, self.short_momentum[i][0]))
            long_moment_list.append((i, self.long_momentum[i][0]))

            
        # sort the momentum list by momentum
        short_moment_list = sorted(short_moment_list, key=lambda x: x[1], reverse=False) # lowest first
        long_moment_list = sorted(long_moment_list, key=lambda x: x[1], reverse=True) # highest first
        
        buy_top_k_stocks = short_moment_list[:self.params.top_k]
        sell_top_k_stocks = long_moment_list[:self.params.top_k]

        # buy 
        for stock in buy_top_k_stocks:
            if self.hold_pool.get_record(self.datas[stock[0]]._name) is None:
                print(f"buy {self.datas[stock[0]]._name}, momentum: {stock[1]}")
                self.buy(self.datas[stock[0]])
            else:
                if self.hold_pool.get_record(self.datas[stock[0]]._name).status == -1:
                    print(f"buy {self.datas[stock[0]]._name}, momentum: {stock[1]}")
                    self.buy(self.datas[stock[0]])

        # sell 
        for stock in sell_top_k_stocks:
            if self.hold_pool.get_record(self.datas[stock[0]]._name) is None:
                print(f"sell {self.datas[stock[0]]._name}, momentum: {stock[1]}")
                self.sell(self.datas[stock[0]])
            else:
                if self.hold_pool.get_record(self.datas[stock[0]]._name).status == 1:
                    print(f"sell {self.datas[stock[0]]._name}, momentum: {stock[1]}")
                    self.sell(self.datas[stock[0]])

        # process the hold stocks
        for i, data in enumerate(self.datas):
            if (i, self.long_momentum[i][0]) not in sell_top_k_stocks and (i, self.short_momentum[i][0]) not in buy_top_k_stocks:
                if self.hold_pool.get_record(data._name) is not None:
                    if self.hold_pool.get_record(data._name).status == 1:
                        self.sell(data)
                    else:
                        self.buy(data)
        
        self.query_holding_number()

        # input("Press Enter to continue...")



class EMATrendStrategy(StragegyTemplate):
    params = (
        ('ema_period', 20),
        ('ema_period2', 10),
    )

    def __init__(self):
        super().__init__()
        self.ema = []
        self.ema2 = []
        for i, data in enumerate(self.datas):
            self.ema.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_period))
            self.ema2.append(bt.indicators.ExponentialMovingAverage(data.volume, period=self.params.ema_period2))


    def next(self):
        if self.order:
            return

        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                if self.ema[i][0] >= self.ema[i][-1] and self.ema[i][-1] >= self.ema[i][-2] and self.ema2[i][0] > self.ema2[i][-1]:
                    self.order = self.buy(data)
            else:
                if self.ema[i][0] < self.ema[i][-1] and self.ema[i][-1] < self.ema[i][-2]:
                    self.order = self.sell(data)
        self.query_holding_number()
        self.stop_loss_watch_dog()



#Long Lower Shadow Candlestick
class LongLowerShadowCandlestickStrategy(StragegyTemplate):
    params = (
        ('low_shadow_length', 3),
        ('min_bar_length', 0.001),
        ('ema_period', 5),
        ('hold_days', 3),
    )

    def __init__(self):
        super().__init__()
        self.ema = []
        for i, data in enumerate(self.datas):
            self.ema.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_period))
        
    def is_long_lower_shadow(self, data_index):
        if self.ema[data_index][-1] < self.ema[data_index][-2] :
            if self.datas[data_index].close[0] > self.datas[data_index].open[0]:
                shadow_length = min(self.datas[data_index].open[0], self.datas[data_index].close[0]) - self.datas[data_index].low[0]
                bar_length = abs(self.datas[data_index].close[0] - self.datas[data_index].open[0])
                bar_length_ratio = bar_length / self.datas[data_index].close[0]
                if shadow_length > bar_length * self.params.low_shadow_length  and bar_length_ratio > self.params.min_bar_length:
                    return True
        return False
        

    def next(self):
        if self.order:
            return
        
        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                if self.is_long_lower_shadow(i):
                    self.order = self.buy(data)
            else:
                hold_days = (data.datetime.date(0) - self.hold_pool.get_record(data._name).buy_date).days
                if hold_days >= self.params.hold_days:
                    self.order = self.sell(data)
        
        self.query_holding_number()

class DiffStrategy(StragegyTemplate):
    params = (
        ('ema_period', 15),
        ('diff_period', 20),
        ('diff_threshold', 0.1),
        ('hold_days', 30),
    )

    def __init__(self):
        super().__init__()
        self.ema = []
        self.diff = []
        for i, data in enumerate(self.datas):
            self.ema.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_period))
            self.diff.append(Diff(data, ema_period=self.params.diff_period))

    def next(self):
        if self.order:
            return
        
        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                # print(f"self.diff[i][0]: {self.diff[i][0]}, self.diff[i][-1]: {self.diff[i][-1]}")
                if self.diff[i][-1] < 0.0 - self.params.diff_threshold and self.diff[i][0] > 0.0 - self.params.diff_threshold:
                    self.order = self.buy(data)
            else:
                hold_days = (data.datetime.date(0) - self.hold_pool.get_record(data._name).buy_date).days
                if self.data.close[0] > self.ema[i][0]  or hold_days >= self.params.hold_days:
                    self.order = self.sell(data)
        
        self.query_holding_number()

class XGBoostStrategy(StragegyTemplate):
    params = (
        ('rsi_period',14),
        ('cci_period',14),
        ('adx_period',14),
        ('mom_period',10),
        ('atr_period',14),
        ('roc_period',10),
        ('min_start_index', 60),
    )

    def __init__(self):
        super().__init__()
        self.rsi = []
        self.cci = []
        # self.fastk = []
        self.slowk = []
        self.slowd = []
        self.adx = []
        self.mom = []
        self.macd = []
        self.atr = []
        self.roc = []
        self.indictors_list = []
        self.model_path = "model/xgboost/xgboost_model_regressor.json"
        self.bst = XGBRegressor()
        self.bst.load_model(self.model_path)
        self.pre_days = 9
        self.first_obv = []
        self.emas = []
        for i, data in enumerate(self.datas):
            # RSI
            self.emas.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.min_start_index))
            indictors = {}
            indictors["RSI"] = ta.RSI(data.close, timeperiod=self.params.rsi_period)

            # CCI
            indictors["CCI"] = ta.CCI(data.high, data.low, data.close, timeperiod=self.params.cci_period)

            # Stochastic Oscillator
            # indictors["slowk"], indictors["slowd"] = ta.STOCH(data.high, data.low, data.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

            stoch = ta.STOCH(data.high, data.low, data.close,
                  fastk_period=5, slowk_period=3, slowk_matype=0,
                  slowd_period=3, slowd_matype=0)
            # print(stoch)
            # print attibute
            
            # 获取 %K 和 %D 线
            indictors["slowk"] = stoch.slowk
            indictors["slowd"] = stoch.slowd

            # ADX
            indictors["ADX"] = ta.ADX(data.high, data.low, data.close, timeperiod=self.params.adx_period)

            # Momentum
            indictors["MOM"] = ta.MOM(data.close, timeperiod=self.params.mom_period)

            indictors["OBV"] = ta.OBV(data.close, data.volume)

            # MACD
            macd = ta.MACD(data.close, fastperiod=12, slowperiod=26, signalperiod=9)
            indictors["macd"], indictors["signal"], indictors["hist"] = macd.macd, macd.macdsignal, macd.macdhist
            # ATR
            indictors["ATR"] = ta.ATR(data.high, data.low, data.close, timeperiod=self.params.atr_period)

            # ROC
            indictors["ROC"] = ta.ROC(data.close, timeperiod=self.params.roc_period)
            # start_index = 0
            # length = len(indictors["RSI"])
            # for index in range(0, length):
            #     for key in indictors:
            #         if np.isnan(indictors[key][index]) :
            #             start_index += 1
            #             break
            # print(f"start_index: {start_index}")
            # indictors["OBV"] = (indictors["OBV"] - indictors["OBV"][start_index]) / indictors["OBV"][start_index]
            self.indictors_list.append(indictors)


    def get_predays_indictor(self, data_index, time_index):
        x_i = []
        
        for key in self.indictors_list[0]:
            # if key == "OBV":
            #     x = []
            #     for i in range(self.pre_days):
            #         normal_obv = (self.indictors_list[data_index][key][time_index-self.pre_days+i] - self.first_obv[data_index]) / self.first_obv[data_index]
            #         x.append(normal_obv)
                
            #     x_i.extend(x)
            # else:
            #     # it does not support the clice operation
            x = []
            for i in range(self.pre_days):
                x.append(self.indictors_list[data_index][key][time_index-self.pre_days+i])
            x_i.extend(x)

        if np.isnan(x_i).any() or np.isinf(x_i).any():
            return None
        
        return x_i


    def next(self):
        if self.order:
            return
        
        is_first = True
        if is_first:
            for i, data in enumerate(self.datas):
                self.first_obv.append(self.indictors_list[i]["OBV"][0])
            is_first = False

        
        for i, data in enumerate(self.datas):
            x_i = self.get_predays_indictor(i, 0)
            if x_i is None:
                continue
            x_i = np.array([x_i])
            # print(f"x_i shae: {x_i.shape}")
            y_i = self.bst.predict(x_i)

            # print(f"x_i: \n {x_i}")
            # print(f"y_i_pred: \n {y_i}")
            if self.getposition(data).size <= 0:
                if y_i[0] > 0.3:
                    self.order = self.buy(data)
            else:
                hold_days = (data.datetime.date(0) - self.hold_pool.get_record(data._name).buy_date).days
                if hold_days >= 30 :
                    self.order = self.sell(data)
        
        self.query_holding_number()

        