import math
import struct
import pandas as pd
import os

# 获取日期字符串的函数
def get_date_str(h1, h2):
    year = math.floor(h1 / 2048) + 2004
    month = math.floor(h1 % 2048 / 100)
    day = h1 % 2048 % 100
    hour = math.floor(h2 / 60)
    minute = h2 % 60
    return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"

# 获取数据的函数
def get_min_data(file_path):
    with open(file_path, 'rb') as ofile:
        buf = ofile.read()
    num = len(buf)
    no = num // 32
    b = 0
    e = 32
    dl = []
    for i in range(no):
        a = struct.unpack('HHffffllf', buf[b:e])
        date_str = get_date_str(a[0], a[1])
        dl.append([date_str, a[2], a[3], a[4], a[5], a[6], a[7]])
        b += 32
        e += 32
    df = pd.DataFrame(dl, columns=['date', 'open', 'high', 'low', 'close', 'amount', 'volume'])
    return df

def get_daily_data(file_path):
    with open(file_path, 'rb') as ofile:
        buf = ofile.read()
    num = len(buf)
    no = num // 32
    b = 0
    e = 32
    data_list = []
    for i in range(int(no)):
        a = struct.unpack('IIIIIfII', buf[b:e])
        year = int(a[0] / 10000)
        month = int((a[0] % 10000) / 100)
        day = int((a[0] % 10000) % 100)
        date_str = f"{year}-{month:02d}-{day:02d}"
        open_price = a[1] / 100.0
        high = a[2] / 100.0
        low = a[3] / 100.0
        close = a[4] / 100.0
        amount = a[5]
        volume = a[6]
        print(a)
        data_list.append([date_str, open_price, high, low, close, amount, volume])
        b += 32
        e += 32
    df = pd.DataFrame(data_list, columns=['date', 'open', 'high', 'low', 'close', 'amount', 'volume'])
    return df

def temp_fun(file_path):
    with open(file_path, 'rb') as ofile:
        buf = ofile.read()
    num = len(buf)
    for i in range(0, num, 8):
        for j in range(8):
            print(buf[i+j], end=' ')

if __name__ == '__main__':
    data_root = r'D:\myapp\中信证券\vipdoc\sh\lday'
    file_paths = [os.path.join(data_root, f) for f in os.listdir(data_root)]
    print(f"file number : {len(file_paths)}")
    
    # 1分钟数据
    # df = get_min_data(r'C:/new_tdx/vipdoc/sh/lday/sh600000.day')
    # print(df)

    # 日线数据
    print(file_paths[2])
    df = get_daily_data(file_paths[2])
    print(df)

    temp_fun(file_paths[2])