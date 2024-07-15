import backtrader as bt
import backtrader.feeds as btfeeds
import datetime

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

class StragegyTemplate(bt.Strategy):
    params = (('stop_loss', 0.08),)
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

        self.max_price_from_buy = 0

        self.change_percent = 0
        self.change_percent_final = 0

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
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.max_price_from_buy = order.executed.price
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                if self.buyprice:
                    self.change_percent = 100 * (order.executed.price - self.buyprice) / self.buyprice
                    self.change_percent_final += self.change_percent
                    self.max_price_from_buy = 0

            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f , change_percent: %.2f, change_percent_final: %.2f' %
                 (trade.pnl, trade.pnlcomm, self.change_percent, self.change_percent_final))

    def next(self):
        print("This is a template strategy, please implement your own strategy.")



class MovingAverageStrategy(StragegyTemplate):
    params = (('ma_period', 15), ) # class variable , can be accessed by self.params.ma_period
    def __init__(self):
        super().__init__()
        self.ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_period)
        
    def next(self):
        # check if there is an unfinished order
        if self.order:
            return

        # check if in the market
        if not self.position:
            # if the closing price is above the moving average, buy
            if self.data_close[0] > self.ma[0]:
                self.order = self.buy()
        else:
            # if the closing price is below the moving average, sell
            if self.data_close[0] < self.ma[0]:
                self.order = self.sell()
        
        if self.position:
            self.stop_loss_watch_dog(self.data_close[0])

class RSIStrategy(StragegyTemplate):
    params = (('rsi_period', 14), ('rsi_upper', 70), ('rsi_lower', 30))
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.datas[0].close, period=self.params.rsi_period)
        
    def next(self):
        change = self.dataclose[0] - self.dataclose[-1]
        change_percent = change / self.dataclose[-1] * 100
        self.log("Close: %.2f, RSI: %.2f, Change percent: %.2f" % (self.dataclose[0], self.rsi[0], change_percent) )
        # check if there is an unfinished order
        if self.order:
            return

        # check if in the market
        if self.position:
            # if the RSI is above the upper bound, sell
            if self.rsi[0] > self.params.rsi_upper:
                self.order = self.sell()
        else:
            # if the RSI is below the lower bound, buy
            if self.rsi[0] < self.params.rsi_lower:
                self.order = self.buy()
        
        if self.position:
            self.stop_loss_watch_dog(self.data_close[0])

class CCIStrategy(StragegyTemplate):
    params = (('cci_period', 14), ('cci_upper', 150), ('cci_lower', -150))
    def __init__(self):
        super().__init__()
        self.cci = bt.indicators.CCI(self.datas[0], period=self.params.cci_period)
        
    def next(self):
        change = self.dataclose[0] - self.dataclose[-1]
        change_percent = change / self.dataclose[-1] * 100
        # self.log("Close: %.2f, CCI: %.2f, Change percent: %.2f" % (self.dataclose[0], self.cci[0], change_percent) )
        # check if there is an unfinished order
        if self.order:
            return

        # check if in the market
        if self.position:
            # if the CCI is above the upper bound, sell
            if self.cci[0] > self.params.cci_upper :
                self.order = self.sell()
                
        else:
            # if the CCI is below the lower bound, buy
            if self.cci[0] < self.params.cci_lower:
                self.order = self.buy()
        
        
                

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
            if rsi_signal == -1 or cci_signal == -1 or double_emas_signal == -1:
                pass
                # self.order = self.sell()
                # sell_flag = True
        elif self.position.size <= 0:
            if rsi_signal == 1 or cci_signal == 1 or double_emas_signal == 1:
                self.order = self.buy()


        if self.position.size > 0 and not sell_flag: # 当天没有卖出，否则可能导致指标卖出一次，stoploss卖出一次
            self.stop_loss_watch_dog(self.data_close[0])

    