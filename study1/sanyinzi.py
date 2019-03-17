import numpy as np
import pandas as pd
import scipy
import math
import statsmodels

import WindPy
import datetime



WindPy.w.start()

#每月月初第一个交易日调仓
initial=1000000

#获取交易日
date=WindPy.w.tdays("2015-06-30","2016-11-25").Data[0]

#建立持仓变量
stock_position=pd.DataFrame(index=date[1:],columns=["1","2","3","4","5","6","7","8","9","10"])

#建立每日资产价值变量
Portfolio_Value=pd.DataFrame(index=date[1:],columns=["value"])

#确定每日持仓
i=4
for i in range(1,10):#range(1,len(date)):

    #判断是否为月初第一个交易日，不是的话，持仓等于前一天持仓并跳出循环
    '''if date[i].month==date[i-1].month :
        stock_position.loc[date[i]] = stock_position.loc[date[i-1]]
        continue'''
    last_trading_day=date[i-1]

    if date[i].month == 1 or date[i].month == 2 or date[i].month == 3:
        season_end = datetime(date[i].year - 1, 12, 31)
    elif date[i].month == 4 or date[i].month == 5 or date[i].month == 6:
        season_end = datetime(date[i].year, 3, 31)
    elif date[i].month == 7 or date[i].month == 8 or date[i].month == 9:
        season_end = datetime.datetime(date[i].year, 6, 30)
    else:
        season_end = datetime(date[i].year, 9, 30)

    season_end = season_end.strftime('%Y%m%d')
    last_trading_day=last_trading_day.strftime('%Y%m%d')

#获取全部A股代码
    a = WindPy.w.wset("SectorConstituent", u"date=" + last_trading_day + ";sector=全部A股")
    code=a.Data[1]

#获取EPS因子
    c=WindPy.w.wss(code, "eps_basic","rptDate="+season_end+";currencyType=")
    a = pd.DataFrame(data=c.Data, columns=c.Codes, index=c.Fields).T

#获取反转因子
    b1=WindPy.w.wss(code, "MA", "tradeDate="+last_trading_day+";MA_N=5;priceAdj=U;cycle=D").Data[0]
    b2=WindPy.w.wss(code, "MA", "tradeDate="+last_trading_day+";MA_N=10;priceAdj=U;cycle=D").Data[0]
    for j in range(0,len(code)):
        b2[j]=(b2[j]-b1[j])/b1[j]
    a['MA']=b2

#获取缩量量因子
    c1=WindPy.w.wss(code, "vol_nd", "days=-30;tradeDate="+last_trading_day).Data[0]
    c2=WindPy.w.wss(code, "vol_nd", "days=-1;tradeDate="+last_trading_day).Data[0]
    for k in range(0,len(code)):
        c2[k]=math.log(c1[k]/(30*c2[k]))
    a['VOLUME']=c2

#标准去极值化
    a['EPS_BASIC']=(a['EPS_BASIC']-a['EPS_BASIC'].mean())/a['EPS_BASIC'].std()
    a['MA'] = (a['MA'] - a['MA'].mean()) / a['MA'].std()
    a['VOLUME'] = (a['VOLUME'] - a['VOLUME'].mean()) / a['VOLUME'].std()
    a=a[a['VOLUME']<=3][a['MA']<=3][a['EPS_BASIC']<=3]

#因子加权并排序
    weight=[0.3,0.4,0.3]
    a['score']=0.3*a['EPS_BASIC']+0.4*a['MA']+0.3*a['VOLUME']
  #  a=a.sort(columns='score',ascending=False)


    a=a.sort_values(by=['score'], ascending=False)
    #取总得分前十只股票，确定调仓日持仓
    stock_position.loc[date[i]]=a.index[0:10]

#针对每日持仓求得累积收益率
i=1
current = initial
for i in range(1,10):#range(1, len(date)):
    now_position = stock_position.loc[date[i]].tolist()
    today = date[i].strftime('%Y%m%d')
    Return = WindPy.w.wss(now_position, "pct_chg", "tradeDate=" + today + ";cycle=D").Data[0]
    Portfolio_Return = sum(Return) / (100*len(Return))
    current = current * (1 + Portfolio_Return)
    Portfolio_Value.loc[date[i]]=current
    process_bar.show_process()
print(Portfolio_Value.head(9))

del stock_position

import jindu
import time
max_steps = 100

process_bar = ShowProcess(max_steps)

for i in range(max_steps + 1):
    process_bar.show_process()
    time.sleep(0.05)
process_bar.close()