import akshare as ak
import os

list_etf = ["512690"]

index_funds = [
    {"index_name": "上证综指", "index_code": "000001.SH", "fund_code": "510050.SH"},
    {"index_name": "深证成指", "index_code": "399001.SZ", "fund_code": "159901.SZ"},
    {"index_name": "沪深300指数", "index_code": "000300.SH", "fund_codes": ["510300.SH", "159919.SZ"]},
    {"index_name": "中证500指数", "index_code": "000905.SH", "fund_codes": ["510500.SH", "510510.SH"]},
    {"index_name": "创业板指数", "index_code": "399006.SZ", "fund_code": "159915.SZ"},
    {"index_name": "上证50指数", "index_code": "000016.SH", "fund_code": "510050.SH"},
    {"index_name": "科创50指数", "index_code": "000688.SH", "fund_code": "588000.SH"},
    {"index_name": "中证1000指数", "index_code": "000852.SH", "fund_code": "512100.SH"},
    # {"index_name": "恒生指数", "index_code": "HSI.HI", "fund_code": "2800.HK"},
    {"index_name": "恒生中国企业指数", "index_code": "HSCEI.HI", "fund_code": "510900.SH"},
    {"index_name": "x", "index_code": "x.HI", "fund_code": "512690.SH"},
]

def get_large_market_cap_etfs(fund_code):
    os.makedirs("data_etf", exist_ok=True)
    fund_code = fund_code.split(".")[0]
    etf_info_df = ak.fund_etf_hist_em(fund_code, adjust="qfq")
    etf_info_df.to_csv(f"data_etf/{fund_code}.csv")
    print(etf_info_df)


def demo_of_download_etf():
    for index_fund in index_funds:
        if "fund_code" in index_fund:
            get_large_market_cap_etfs(index_fund["fund_code"])
        elif "fund_codes" in index_fund:
            for fund_code in index_fund["fund_codes"]:
                get_large_market_cap_etfs(fund_code)

if __name__ == "__main__":
    demo_of_download_etf()
