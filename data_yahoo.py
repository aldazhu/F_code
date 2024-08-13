import backtrader as bt
import pandas as pd
import datetime

import yfinance as yf

def demo_of_download_data():
    stock = 'AAPL'
    data = yf.download(stock, start="2020-01-01", end="2021-01-01")
    print(type(data))
    print(data.head())
    print(data.tail())

    data.to_csv(f"{stock}.csv")

def demo_of_download_sp500():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    start_date = "2020-01-01"
    end_date = "2024-07-01"
    df = table[0]
    print(df.head())
    print(df.tail())

    # save df to txt
    df.to_csv("doc/sp500_info.txt", index=False )

    tickers = df['Symbol'].tolist()
    total_tickers = len(tickers)
    for i, ticker in enumerate(tickers):
        try:
            print(f"Downloading {ticker}, {i}/{total_tickers}")
            data = yf.download(ticker, start=start_date, end=end_date)
            data.to_csv(f"data_us/{ticker}.csv")
        except Exception as e:
            print(f"Error in downloading {ticker}, {e}")
        

if __name__ == "__main__":
    # demo_of_download_data()
    demo_of_download_sp500()