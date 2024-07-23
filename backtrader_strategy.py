import backtrader as bt
import backtrader.feeds as btfeeds
import datetime
import numpy as np

from backtrader_indicator import RSRS, RSRS_Norm, Diff

class MyData(btfeeds.GenericCSVData):
    params = (
        ('fromdate', datetime.datetime(2022, 11, 1)),
        ('todate', datetime.datetime(2023, 12, 31)),
        ('dtformat', ('%Y-%m-%d')),
        ('tmformat', ('%H.%M.%S')),
        ('datetime', 1),
        ('time', -1),
        ('open', 3),
        ('high', 4),
        ('low', 5),
        ('close', 6),
        ('volume', 8),
        ('openinterest', -1)
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
        self.status = 0
        self.buy_date = ""
        self.sell_date = ""
        self.buy_price = 0
        self.sell_price = 0
        self.hold_days = 0
        self.earning_ratio = 0

    def print(self):
        self.hold_days = (self.sell_date - self.buy_date).days
        print(
            f"Stock code: {self.stock_code}, Strategy: {self.strategy_name}, Status: {self.status}", 
            f"Buy date: {self.buy_date}, Buy price: {self.buy_price} ", 
            f"Sell date: {self.sell_date},Sell price: {self.sell_price}",
            f"Hold days: {self.hold_days}, Earning ratio: {self.earning_ratio}"
            )

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
    params = (('stop_loss', 0.08),)

    history_records = []

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        time = self.datas[0].datetime.time(0)
        print('%s T %s, %s' % (dt.isoformat(),time, txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.max_price_from_buy = 0

        self.change_percent = 0
        self.change_percent_final = 0

        self.hold_pool = HoldPool()

    def analyze_the_history(self):
        success_count = 0
        total_count = len(self.history_records)
        buy_price_sum = 0
        sell_price_sum = 0
        earning_ratio = []
        for record in self.history_records:
            record.print()
            if record.earning_ratio > 0:
                success_count += 1
            buy_price_sum += record.buy_price
            sell_price_sum += record.sell_price
            earning_ratio.append(record.earning_ratio)

        if total_count == 0:
            print("No record to analyze")
            return
        final_earning_ratio = (sell_price_sum - buy_price_sum) / buy_price_sum
        print(f"buy price sum: {buy_price_sum}, sell price sum: {sell_price_sum}, earning ratio: {final_earning_ratio}")
        
        print(f"Total count: {total_count}, success count: {success_count}, success rate: {success_count / total_count}")

        sharp_ratio = np.mean(earning_ratio) / np.std(earning_ratio)
        print(f"Sharp ratio: {sharp_ratio}")

    def stop(self):
        self.analyze_the_history()

    def get_final_change_percent(self):
        return self.change_percent_final

    def stop_loss_watch_dog(self, price):
        if self.position.size > 0:
            if self.data_high[0] > self.max_price_from_buy:
                self.max_price_from_buy = self.data_high[0]

        else:
            self.max_price_from_buy = 0 

        if self.max_price_from_buy:
            if price < self.max_price_from_buy * (1 - self.params.stop_loss):
                self.log("Stop loss triggered, sell at price: %.2f" % price)
                if self.position.size > 0:
                    self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                record = StockStatus(order.data._name, self.__class__.__name__)
                record.status = 1
                record.buy_date = self.datas[0].datetime.date(0)
                record.buy_price = order.executed.price
                self.hold_pool.add_record(record)

                self.log(
                    'name : %s , BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.data._name,
                     order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.max_price_from_buy = order.executed.price
            else:  # Sell
                record = self.hold_pool.get_record(order.data._name)
                if record:
                    record.status = -1
                    record.sell_date = self.datas[0].datetime.date(0)
                    record.sell_price = order.executed.price
                    record.hold_days = len(self)
                    record.earning_ratio = (record.sell_price - record.buy_price) / record.buy_price
                    record.print()
                    # self.hold_pool[order.data._name] = record
                    # erase the record from hold pool
                    self.hold_pool.remove_record(order.data._name)
                    self.history_records.append(record)
                    
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
            self.rsi.append(bt.indicators.RSI(data.close, period=self.params.rsi_period))
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
        

class CCIStrategy(StragegyTemplate):
    params = (('cci_period', 15), ('cci_upper', 150), ('cci_lower', -150), ('high_period', 20),)
    def __init__(self):
        super().__init__()
        self.cci = []
        self.highest = []
        for i, data in enumerate(self.datas):
            self.cci.append(bt.indicators.CCI(data, period=self.params.cci_period))
            self.highest.append(bt.indicators.Highest(data.high, period=self.params.high_period))

    def next(self):
        
        # check if there is an unfinished order
        if self.order:
            return

        for i, data in enumerate(self.datas):
            if self.getposition(data).size > 0:
                if self.cci[i][0] < self.params.cci_upper and self.cci[i][-1] >= self.params.cci_upper:
                    self.order = self.sell(data)
            else:
                if self.cci[i][0] > self.params.cci_lower and self.cci[i][-1] <= self.params.cci_lower:
                    self.order = self.buy(data)
        
                

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
        ('lowest_window', 10),
        ('ema_period', 120),
        ('ema_sell_period', 10)
    )

    def __init__(self):
        super().__init__()
        self.high = []
        self.low = []
        self.ema = []
        self.ema_sell = []
        self.diff = [] # high - low
        for i, data in enumerate(self.datas):
            self.high.append(bt.indicators.Highest(data.high, period=self.params.highest_window))
            self.low.append(bt.indicators.Lowest(data.low, period=self.params.lowest_window))
            self.ema.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_period))
            self.ema_sell.append(bt.indicators.ExponentialMovingAverage(data.close, period=self.params.ema_sell_period))
            self.diff.append(Diff(data,ema_period=self.params.ema_period))

    def next(self):
        if self.order:
            return

        for i, data in enumerate(self.datas):
            if self.getposition(data).size <= 0:
                if data.close[0] > self.high[i][-1] and self.diff[i][0] > 0:
                    print(f"{data.datetime.date(0)}: name : {data._name} buy , today coloe at {data.close[0]}")
                    self.order = self.buy(data)
            else:
                if data.close[0] < self.low[i][-1] or data.close[0] < self.ema_sell[i][0]:
                    print(f"{data.datetime.date(0)}: name : {data._name} sell , today close at {data.close[0]}")
                    self.order = self.sell(data)

        
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

