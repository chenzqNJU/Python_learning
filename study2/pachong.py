
#导入需要使用到的模块
import urllib
import re
import pandas as pd
# import pymysql
import os
import numpy as np
import datetime
import tushare as ts

import tushare as ts
df = ts.get_realtime_quotes(['000581','000005'])

#爬虫抓取网页函数
def getHtml(url):
    html = urllib.request.urlopen(url).read()
    html = html.decode('gbk')
    return html

#抓取网页股票代码函数
def getStackCode(html):
    s = r'<li><a target="_blank" href="http://quote.eastmoney.com/\S\S(.*?).html">'
    pat = re.compile(s)
    code = pat.findall(html)
    return code

Url = 'http://quote.eastmoney.com/stocklist.html'#东方财富网股票数据连接地址
url=Url

#实施抓取
code = getStackCode(getHtml(Url))

len(code)
#获取所有股票代码（以6开头的，应该是沪市数据）集合
CodeList = []


for item in code:
    if item[0]=='6':
        CodeList.append('sh'+item)
    if item[0] == '0' or item[0] == '3':
        CodeList.append('sz' + item)

len(CodeList)

stock = CodeList[40:70]

# 从 新浪财经 爬取数据：价格、日期、时间
url = 'http://hq.sinajs.cn/list='
price=[];time=[];date=[]

starttime = datetime.datetime.now()
for stockID in stock:
    request = urllib.request.urlopen(url + str(stockID)).read()
    request = request.decode('gbk')
    request = request.split(',')
    price = price+ [request[3]]
    time = time + [request[31]]
    date = date + [request[30]]
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)

# 运用 tushare 500个也能很快的读取实时数据
stock = [x[2:] for x in CodeList[:500]]
df = ts.get_realtime_quotes(stock)
df = df[['code','price','date','time']]
df.price = df.price.astype(float)
df = df[df.price>0]