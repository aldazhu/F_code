import backtrader as bt
import datetime

import os

from backtrader_strategy import *

def test_backtrader(data, strategy, cash=100000.0, commission=0.001,stake=100):
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(strategy)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(cash)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)

    # Set the commission
    cerebro.broker.setcommission(commission=commission)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())


def get_data(data_name, from_date, to_date):
    if not os.path.exists(data_name):
        print(f"Data file {data_name} does not exist.")
        raise "Data file does not exist."

    data = MyData(dataname=data_name,
                fromdate=from_date,
                todate=to_date)
    return data

def demo_of_simple_strategy():
    # Create a Data Feed
    data_name = 'data/sz.300628.csv'
    from_date = datetime.datetime(2022, 1, 1)
    to_date = datetime.datetime(2024, 12, 31)
    data = get_data(data_name, from_date, to_date)

    # test_backtrader(data, strategy=MovingAverageStrategy, cash=100000.0, commission=0.001, stake=100)

    # test_backtrader(data, strategy=QuickGuideStrategy, cash=100000.0, commission=0.001, stake=100)

    # test_backtrader(data, strategy=RSIStrategy, cash=100000.0, commission=0.001, stake=100)

    test_backtrader(data, strategy=CCIStrategy, cash=100000.0, commission=0.001, stake=100)


if __name__ == '__main__':
    demo_of_simple_strategy()