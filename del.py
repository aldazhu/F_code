import requests
 
url = 'http://27.push2.eastmoney.com/api/qt/clist/get'
for i in range(1, 10):
    data = {
        'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152',
        'pz': 1000,         # 每页条数
        'pn': i,            # 页码
        'fs': 'm:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048'
    }
    response = requests.get(url, data)
    response_json = response.json()
    print(i, response_json)
    # 返回数据为空时停止循环
    if response_json['data'] is None:
        break
    input('press ENTER to continue...')
    for j, k in response_json['data']['diff'].items():
        code = k['f12']         # 代码
        name = k['f14']         # 名称
        price = k['f2']         # 股价
        pe = k['f9']            # 动态市盈率
        pb = k['f23']           # 市净率
        total_value = k['f20']          # 总市值
        currency_value = k['f21']       # 流通市值
        price = round(price/100, 2)     # 价格转换为正确值（保留2位小数）
        pe = round(pe/100, 2)           # 市盈率转换为正确值（保留2位小数）
        pb = round(pb/100, 2)           # 市净率转换为正确值（保留2位小数）
        total_value = round(total_value / 100000000, 2)         # 总市值转换为亿元（保留2位小数）
        currency_value = round(currency_value / 100000000, 2)   # 流通市值转换为亿元（保留2位小数）
        print('代码: %s, 名称: %s, 现价: %s, 动态市盈率: %s, 市净率: %s, 总市值: %s亿, 流通市值: %s' % (code, name, price, pe, pb, total_value, currency_value))