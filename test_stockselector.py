import stock_selector as ss
import dataPro as dp
import os
import numpy as np
import pandas as pd

from my_logger import logger

from count_manager import MyCount


def demo_of_stock_selector():
    # 1. get stock list
    folder = "./data"
    start_date_index = 50
    file_list = os.listdir(folder)
    file_list = [os.path.join(folder, file) for file in file_list]

    # 2. define the stock selector

    RSI_selector = ss.RSIStockSelector(days=14, high_thresh=70, low_thresh=30)
    CCI_selector = ss.CCIThreshStockSelector(days=5, CCI_thresh=100, after_Ndays=10)
    # OSC_selector = ss.OSCStockSelector(short=12, long=26)
    selector_list = [RSI_selector, CCI_selector]

    # 3. calculate the flag
    for selector in selector_list:
        selector.calculate_flag(file_list)
    
    # 4. select the stock
    data = dp.readData(file_list[0])
    date_list = data.index.tolist()
    my_count = MyCount()
    for date_index in range(start_date_index, 100):
        date = date_list[date_index]
        logger.info("Date: %s", date)
        for selector in selector_list:
            selector.select(date_index)
            selected_stocks = selector.selected_stocks_
            for stock, status in selected_stocks.items():
                logger.info("Stock: %s, Strategy: %s, status: %s", stock, status.strategy_name, status.status)
                action = "buy" if status.status == 1 else "sell"
                action_price = data['open'][date_index]
                my_count.status_update(stock, date_index, action, action_price)
        my_count.everyday_update(date_index)

if __name__ == "__main__":
    demo_of_stock_selector()