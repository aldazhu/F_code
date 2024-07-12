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
        if not self.position:
            # if the RSI is above the upper bound, sell
            if self.rsi[0] > self.params.rsi_upper:
                self.order = self.sell()
        else:
            # if the RSI is below the lower bound, buy
            if self.rsi[0] < self.params.rsi_lower:
                self.order = self.buy()

class CCIStrategy(StragegyTemplate):
    params = (('cci_period', 14), ('cci_upper', 100), ('cci_lower', -100))
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
        if not self.position:
            # if the CCI is above the upper bound, sell
            if self.cci[0] > self.params.cci_upper:
                self.order = self.sell()
        else:
            # if the CCI is below the lower bound, buy
            if self.cci[0] < self.params.cci_lower:
                self.order = self.buy()
