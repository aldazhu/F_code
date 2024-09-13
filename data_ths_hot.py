

#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Date: 2023/1/11 16:11
Desc: 问财-热门股票排名
https://www.iwencai.com/unifiedwap/home/index
"""
import pandas as pd
import requests
from py_mini_racer import py_mini_racer
from tqdm import tqdm

from akshare.datasets import get_ths_js


def get_stock_concept_rot_rank():
    url='https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/plate?'
    params={
        'type': 'concept'
    }
    headers={
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
        }
    res=requests.get(url=url,params=params,headers=headers)
    text=res.json()
    status_code=text['status_code']
    if int(status_code)==0:
        df=pd.DataFrame(text['data']['plate_list'])
        #columns=['热度','概念代码','涨跌幅','上榜统计','热度变化',
                    #'市场id','概念名称','概念统计','排序','etf_rise_and_fall',
                    #'etf_product_id','etf_name','etf_market_id']
        df=df.rename(columns={"code":"概念代码","name":"概念名称"})
        #df.columns=columns
        return df
    else:
        print('失败')
        return False

def _get_file_content_ths(file: str = "ths.js") -> str:
    """
    获取 JS 文件的内容
    :param file:  JS 文件名
    :type file: str
    :return: 文件内容
    :rtype: str
    """
    setting_file_path = get_ths_js(file)
    with open(setting_file_path) as f:
        file_data = f.read()
    return file_data


def stock_hot_rank_wc(date: str = "20210430") -> pd.DataFrame:
    """
    问财-热门股票排名
    https://www.iwencai.com/unifiedwap/result?w=%E7%83%AD%E9%97%A85000%E8%82%A1%E7%A5%A8&querytype=stock&issugs&sign=1620126514335
    :param date: 查询日期
    :type date: str
    :return: 热门股票排名
    :rtype: pandas.DataFrame
    """
    url = "http://www.iwencai.com/unifiedwap/unified-wap/v2/result/get-robot-data"
    js_code = py_mini_racer.MiniRacer()
    js_content = _get_file_content_ths("ths.js")
    js_code.eval(js_content)
    v_code = js_code.call("v")
    headers = {
        "hexin-v": v_code,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
    }
    params = {
        "question": f"{date}热门5000股票",
        "perpage": "5000",
        "page": "1",
        "secondary_intent": "",
        "log_info": '{"input_type":"click"}',
        "source": "Ths_iwencai_Xuangu",
        "version": "2.0",
        "query_area": "",
        "block_list": "",
        "add_info": '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json"}',
    }
    big_df = pd.DataFrame()
    for page in tqdm(range(1, 11), leave=False):
        params.update({
            "page": page,
        })
        r = requests.post(url, data=params, headers=headers)
        # r = requests.get(url, params=params, headers=headers)
        print(f"r.status_code: {r.status_code}")
        print(f"r.text: {r.text}")
        data_json = r.json()
        temp_df = pd.DataFrame(
            data_json["data"]["answer"][0]["txt"][0]["content"]["components"][0][
                "data"
            ]["datas"]
        )
        big_df = pd.concat([big_df, temp_df], ignore_index=True)

    big_df.reset_index(inplace=True)
    big_df["index"] = range(1, len(big_df) + 1)
    try:
        rank_date_str = big_df.columns[1].split("[")[1].strip("]")
    except:
        try:
            rank_date_str = big_df.columns[2].split("[")[1].strip("]")
        except:
            rank_date_str = date
    big_df.rename(
        columns={
            "index": "序号",
            f"个股热度排名[{rank_date_str}]": "个股热度排名",
            f"个股热度[{rank_date_str}]": "个股热度",
            "code": "股票代码",
            "market_code": "_",
            "最新涨跌幅": "涨跌幅",
            "最新价": "现价",
            "股票代码": "_",
        },
        inplace=True,
    )
    big_df = big_df[
        [
            "序号",
            "股票代码",
            "股票简称",
            "现价",
            "涨跌幅",
            "个股热度",
            "个股热度排名",
        ]
    ]
    big_df["涨跌幅"] = big_df["涨跌幅"].astype(float).round(2)
    big_df["排名日期"] = rank_date_str
    big_df["现价"] = pd.to_numeric(big_df["现价"], errors="coerce")
    return big_df

def demo_of_stock_concept_rot_rank():
    df=get_stock_concept_rot_rank()
    print(df)
    df.to_csv('stock_concept_rot_rank.csv',index=False)


def demo_of_stock_hot_rank_wc():
    stock_hot_rank_wc_df = stock_hot_rank_wc(date="20240909")
    print(stock_hot_rank_wc_df)

if __name__ == "__main__":
    demo_of_stock_concept_rot_rank()
    # demo_of_stock_hot_rank_wc()