from queue import Queue
import threading
import os
import datetime
import pandas as pd
import tushare as ts
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import types
import myfunc
from imp import reload

reload(myfunc)

myfunc.search('strp')

# # 获取所有股票数据，利用股票代码获取复权数据
# stock_basics = ts.get_stock_basics()
# cal_dates = ts.trade_cal()  # 返回交易所日历，类型为DataFrame, calendarDate  isOpen


# stock_basics.to_csv('e:\\data\\stock_basics.csv',encoding='GBK')
# cal_dates.to_csv('e:\\data\\cal_dates.csv',encoding='GBK')
stock_basics  = pd.read_csv('e:\\data\\stock_basics.csv',encoding='GBK')
stock_basics.index = stock_basics.code.astype('str').str.zfill(6)
cal_dates  = pd.read_csv('e:\\data\\cal_dates.csv',encoding='GBK').iloc[:,-2:]

# 本地实现判断市场开市函数
# date: str类型日期eg.'2017-11-23'
def get_date_list(begin_date, end_date):
    date_list = []
    while begin_date <= end_date:
        # date_str = str(begin_date)
        date_list.append(begin_date)
        begin_date += datetime.timedelta(days=1)
    return date_list


def is_open_day(date):
    if date in cal_dates['calendarDate'].values:
        return cal_dates[cal_dates['calendarDate'] == date].iat[0, 1] == 1
    return False

dates = get_date_list(datetime.date(2019, 2, 25), datetime.date.today())
dates = [str(x) for x in dates if is_open_day(str(x))]

date_index = -4
date = dates[date_index]
# 获取复权数据
def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result


test = 'e:\\data\\stk\\'+ date
result = all_path(test)

df1 = pd.DataFrame()
for a in result[100:300]:
    try:
        df = pd.read_csv(a)
    except:
        df = pd.read_csv(a, encoding='gbk')
    df['date'] = a.split('\\')[3]
    df1 = df1.append(df)

df1 = df1.iloc[:,1:]
df1.code = df1.code.astype('str').str.zfill(6)

# df2 = df1[df1.code==600094]

codeList = list(set(df1.code.values))
codeList.sort()

#导入需要使用到的模块
import urllib
import re
import pandas as pd
import pymysql
import os

#爬虫抓取网页函数
def getHtml(url):
    try:
        html = urllib.request.urlopen(url,timeout=5).read()
        html = html.decode('gbk')
        return html
    except Exception as e:  # 抛出超时异常
        print('a', str(e))
    return None


#实施抓取

date = '20190301'
code = '600096'

def get_attr(date,attr):
    date=date.replace('-','')
    col_info =['交易日期', '指数代码', '指数名称', '收盘价', '最高价', '最低价', '开盘价', '前收盘', '涨跌额', '涨跌幅', '换手率', '成交量', '成交额', '总市值', '流通市值']
    pos = col_info.index(attr)
    ret = []
    for i,code in enumerate(codeList):
        if i%10==0:print(i)
        url = 'http://quotes.money.163.com/service/chddata.html?code=' + str(1-int(code[0]=='6'))+code + '&start=' + date + '&end='+date+'&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
        data = getHtml(url)
        if data==None:rate=''
        else:         rate = data.split('\r\n')[1].split(',')[pos]
        ret  += [rate]
    tmp = pd.DataFrame()
    tmp['code'] = codeList
    tmp[attr] = ret
    return tmp

post_ret = get_attr(dates[date_index+1],'涨跌幅').set_index('code')

pre_close = get_attr(dates[date_index-1],'收盘价').set_index('code')
today_close = get_attr(dates[date_index],'收盘价').set_index('code')
post_open = get_attr(dates[date_index+1],'开盘价').set_index('code')

len(post_ret.dropna())
len(post_ret)

post_ret=post_ret[post_ret!=['None']]
post_ret.iloc[:,0]=post_ret.iloc[:,0].astype(float)
post_ret=post_ret.dropna()
f3=post_ret.iloc[:,0].quantile(0.3)
f7=post_ret.iloc[:,0].quantile(0.7)

post_ret['flag']=0
post_ret.iloc[np.where(post_ret.iloc[:,0]<f3)[0],1]=-1
post_ret.iloc[np.where(post_ret.iloc[:,0]>f7)[0],1]=1
post_ret=post_ret.sort_values('涨跌幅')
post_ret.iloc[:,0].mean()





pre_close = pre_close.set_index('code')
pre_close = pre_close.set_index('code')
pre_close = pre_close.set_index('code')
pre_close = pre_close.set_index('code')

a=pre_close.join(today_close.rename(columns={'ret':'ret'}, inplace = True))

df2 = pd.merge(df1, tmp, on='code', how='left')

myfunc.search('rename')

import sys
sys.exit()
sys.exit(0)
sys.exit(1)

a = pd.DataFrame([['a', 1], ['b', 2]], columns=['A', 0])
b = pd.DataFrame([['a', 1], ['b', 2]], columns=['B', 0])
c = pd.concat([pre_close, today_close,post_open,post_ret], axis=1, join_axes=[pre_close.index])
print(df3)

############################################# 统计
i=0.99       # 取前 十分之一
t=df1.groupby('code')['amount'].quantile(i).to_frame()
df2 = pd.merge(df1, t.rename(columns={'amount':'a'}), on='code', how='left')
df2 = df2[df2.amount>df2.a]

df2=df1.sort_values(['code','amount'])
df2=df2.groupby('code').tail(20)

df=df2
z=df.groupby(['code','type'])['amount'].sum()
z=z.reset_index()
z=z[z.type!='中性盘']
z1=z.pivot(index='code',columns='type',values='amount')
z1['rate1']=np.round((z1.iloc[:,0]-z1.iloc[:,1])/z1.iloc[:,1]*100,2)
z1=z1.join(post_ret)
z1=z1.sort_values('涨跌幅',ascending=True)

z1.groupby('flag')['rate1'].mean()
z1.rate1.mean()

z1=z1.sort_values('涨跌幅',ascending=False)
z1=z1.sort_values('rate1',ascending=False)


import matplotlib.pyplot as plt

code = z1.index[4]
t=df1[df1.code==code]

t.price.plot(figsize=(10,5))
plt.show()

myfunc.search('matplotlib')
# page=data.split('\r\n')
# col_info=page[0].split(',')   #各列的含义
# index_data=page[1:]     #真正的数据

# #为了与现有的数据库对应，这里我还修改了列名，大家不改也没关系
# col_info[col_info.index('日期')]='交易日期'   #该段更改列名称
# col_info[col_info.index('股票代码')]='指数代码'
# col_info[col_info.index('名称')]='指数名称'
# col_info[col_info.index('成交金额')]='成交额'
#
# index_data=[x.replace("'",'') for x in index_data]  #去掉指数编号前的“'”
# index_data=[x.split(',') for x in index_data]
#
# index_data=index_data[0:index_data.__len__()-1]   #最后一行为空，需要去掉
# pos1=col_info.index('涨跌幅')
# pos2=col_info.index('涨跌额')
# posclose=col_info.index('收盘价')
# index_data[index_data.__len__()-1][pos1]=0      #最下面行涨跌额和涨跌幅为None改为0
# index_data[index_data.__len__()-1][pos2]=0
# for i in range(0,index_data.__len__()-1):       #这两列中有些值莫名其妙为None 现在补全
#     if index_data[i][pos2]=='None':
#         index_data[i][pos2]=float(index_data[i][posclose])-float(index_data[i+1][posclose])
#     if index_data[i][pos1]=='None':
#         index_data[i][pos1]=(float(index_data[i][posclose])-float(index_data[i+1][posclose]))/float(index_data[i+1][posclose])

