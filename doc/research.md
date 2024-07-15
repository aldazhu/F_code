
hs300
startDate = "2020-01-01"
endDate = "2024-07-15"
## 1. cci
```python
class CCIStrategy(StragegyTemplate):
    params = (('cci_period', 15), ('cci_upper', 150), ('cci_lower', -150), ('high_period', 20),)
    def __init__(self):
        super().__init__()
        self.cci = bt.indicators.CCI(self.datas[0], period=self.params.cci_period)
        self.high = bt.indicators.Highest(self.datas[0].high, period=self.params.high_period)
        
    def next(self):
        change = self.dataclose[0] - self.dataclose[-1]
        change_percent = change / self.dataclose[-1] * 100
        # self.log("Close: %.2f, CCI: %.2f, Change percent: %.2f" % (self.dataclose[0], self.cci[0], change_percent) )
        # check if there is an unfinished order
        if self.order:
            return

        # check if in the market
        if self.position.size > 0:
            # if the CCI is above the upper bound, sell
            if self.cci[0] < self.params.cci_upper and self.cci[-1] >= self.params.cci_upper:
                self.order = self.sell()
            # elif self.dataclose[0] < self.high[-1] * 0.80:
            #     self.order = self.sell()
                
        else:
            # if the CCI is below the lower bound, buy
            if self.cci[0] > self.params.cci_lower and self.cci[-1] <= self.params.cci_lower:
                self.order = self.buy()
```
result：
```
Total gain: 171035.88855302427
217 / 299 acc: 0.725752508361204
```

## 2. new high
```python
class NewHighStrategy(StragegyTemplate):
    params = (
        ('window', 20),
        ('ema_period', 50),
        ('ema_sell_period', 50)
    )

    def __init__(self):
        super().__init__()
        self.high = bt.indicators.Highest(self.datas[0].high, period=self.params.window)
        self.low = bt.indicators.Lowest(self.datas[0].low, period=self.params.window)
        self.ema = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.ema_period)
        self.sell_ema = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.ema_sell_period)

    def next(self):
        if self.order:
            return

        if not self.position:  # not in the market
            if self.dataclose[0] > self.high[-1] and self.dataclose[0] > self.ema[0]:
                self.order = self.buy()
        else:  # in the market
            if self.dataclose[0] < self.low[-1] or self.dataclose[0] < self.high[-1] * (1 - 0.20) or self.dataclose[0] < self.sell_ema[0]:
                self.order = self.sell()

```

```txt
Total gain: 163876.14867420745
163 / 300 acc: 0.5433333333333333
```

compare the two strategies, the cci strategy has a higher accuracy and total gain. but the new high strategy has a lower accuracy and get higher average profit per trade, and new high strategy has a shorter holding period.

## 3. rsi

```python

class RSIStrategy(StragegyTemplate):
    params = (('rsi_period', 15), ('rsi_upper', 70), ('rsi_lower', 30),('high_period', 20),('stop_loss', 0.2))
    def __init__(self):
        super().__init__()
        self.min_price = 0.0
        self.rsi = bt.indicators.RSI(self.datas[0].close, period=self.params.rsi_period)
        self.highest = bt.indicators.Highest(self.datas[0].high, period=self.params.high_period)
        
    def next(self):
        change = self.dataclose[0] - self.dataclose[-1]
        change_percent = change / self.dataclose[-1] * 100
        # self.log("Close: %.2f, RSI: %.2f, Change percent: %.2f" % (self.dataclose[0], self.rsi[0], change_percent) )
        # check if there is an unfinished order
        if self.order:
            return

        # check if in the market
        if self.position.size > 0:
            # if the RSI is above the upper bound, sell
            if self.rsi[0] <= self.params.rsi_upper and self.rsi[-1] > self.params.rsi_upper:
                self.order = self.sell()
            # elif self.dataclose[0] < self.highest[-1] * (1 - self.params.stop_loss):
            #     self.order = self.sell()
            
        else:
            # if the RSI is below the lower bound, buy
            if self.rsi[0] >= self.params.rsi_lower and self.rsi[-1] < self.params.rsi_lower:
                self.order = self.buy()
```

```txt
Total gain: 63572.84902108788
169 / 298 acc: 0.5671140939597316
```