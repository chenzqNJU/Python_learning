# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 09:53:42 2017


"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import common
import datetime
from datetime import datetime
from math import log
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import constr_factors_df
from sklearn import linear_model

benchmark_code = '000300.SH'
transfer_interval = 20
# %%

'''
获取昨日日期，以备指数成份股的提取
'''

def get_yesterday():
    tran_date = datetime.datetime.now()
    new_date = datetime.date(tran_date.year, tran_date.month, tran_date.day-1)
    return new_date
type(tran_date)
datetime.date(1,1)
# %%
'''
每次调仓期初获取行业标签代码备用
数据库表待更新每天的行业标签数据
'''


def get_sector_id():
    sector_id_list = pd.read_sql('SELECT sec_id FROM "WIND"."TABLE_COMM_1000" ',
con=common.DB_SQL_CONN_PYODBC)
    sector_id_list = list(sector_id_list['sec_id'])
    return sector_id_list

def calc_beta(start_date,transfer_interval,end_date):
    trade_date = pd.read_sql('''SELECT date FROM "WIND"."COMMON_CN_3010" ORDER BY date''',
con=common.DB_SQL_CONN_PYODBC)
    start_date = datetime.date(datetime.strptime(start_date,'%Y-%m-%d'))
#    end_date = datetime.date(datetime.strptime('2016-04-18','%Y-%m-%d'))
#    time_slice = trade_date['date'][start_date:end_date]
    start_loca = trade_date.loc[trade_date.date==start_date]
    loca_index = start_loca.index[0]
    beta_start_date = trade_date.loc[loca_index-transfer_interval,'date']
    beta_end_date = trade_date.loc[loca_index-1,'date']

    trade_date.index = trade_date['date']
    sz_rewards = trade_date[beta_start_date:beta_end_date]

    sz_close = pd.read_sql('''SELECT close FROM "WIND"."STOCK_INDEX_CN_1002" WHERE date between '{0}'
and '{1}' and windcode = '{2}' '''.format(beta_start_date,beta_end_date,'000001.SH'),
con=common.DB_SQL_CONN_PYODBC)
    sz_open = pd.read_sql('''SELECT open FROM "WIND"."STOCK_INDEX_CN_1002" WHERE date = '{0}' and
windcode = '{1}' '''.format(beta_start_date,'000001.SH'), con=common.DB_SQL_CONN_PYODBC).iloc[0,0]

    sz_rewards['sz_reward'] = 1.0

    for i in range(len(sz_close)):
        sz_rewards['sz_reward'][i] += log(sz_close.loc[i,'close'])-log(sz_open)

    set_date = end_date + ' 00:00:00.005'
#    set_date = '2017-03-20 00:00:00.005'
    sql_price ='''
               SELECT "WIND"."STOCK_CN_1002".windcode, "WIND"."STOCK_CN_1002".date,
"WIND"."STOCK_CN_1002".open, "WIND"."STOCK_CN_1002".close
               FROM "WIND"."STOCK_CN_1002"
               INNER JOIN "WIND"."INDEX_CN_6010"
               ON "WIND"."STOCK_CN_1002".windcode = "WIND"."INDEX_CN_6010".windcode
               WHERE "WIND"."INDEX_CN_6010".date = '{0}'
               AND "STOCK_CN_1002".date between '{1}' and '{2}'
               and "WIND"."INDEX_CN_6010".index_windcode = '{3}'
               '''
    price_df = pd.read_sql(sql_price.format(set_date,beta_start_date,beta_end_date,benchmark_code),
con=common.DB_SQL_CONN_PYODBC)
    price_df.index = price_df['date']

#    for j in range(len(price_df)):
#        price_df['close'][j] = log(price_df['close'][j])

    price = price_df['close'].tolist()
    price = [log(m) for m in price]
    price_df['close'] = price


    stock_beta = DataFrame(price_df['windcode'].drop_duplicates())
    stock_beta =  stock_beta.reset_index(drop = True)
    stock_beta['beta'] = 0.0
    stock_list = price_df['windcode'].drop_duplicates().tolist()

    for stock in stock_list:
        stock_price = price_df[price_df.windcode == stock]
        stock_price['reward'] = 1.0
        stock_price['reward'] += stock_price['close'] - log(stock_price['open'][0])
        beta_df = pd.merge(stock_price,sz_rewards, on = ['date'])
        stock_beta['beta'][stock_beta.windcode == stock] = scs.linregress(beta_df['reward'].tolist
(),beta_df['sz_reward'].tolist())[0]
#np.linalg.lstsq(x,y)[0]
    stock_beta = stock_beta.dropna()
    return stock_beta


'''
生成包含总市值因子和28个行业示性函数值的大表，以待回归
'''

def creat_sector_table(start_date,end_date):
    set_date = end_date + ' 00:00:00.005'
#    set_date = '2017-03-20 00:00:00.005'
    sql_ev = '''
             SELECT "WIND"."STOCK_CN_1002".windcode, "WIND"."STOCK_CN_1002".ev
             FROM "WIND"."STOCK_CN_1002"
             INNER JOIN "WIND"."INDEX_CN_6010"
             ON "WIND"."STOCK_CN_1002".windcode = "WIND"."INDEX_CN_6010".windcode
             WHERE "WIND"."INDEX_CN_6010".date = '{0}'
             AND "STOCK_CN_1002".date between '{1}' and '{2}'
             AND "WIND"."INDEX_CN_6010".index_windcode = '{3}'
             ORDER BY windcode
             '''
    ev = pd.read_sql(sql_ev.format(set_date,start_date,end_date, benchmark_code),
con=common.DB_SQL_CONN_PYODBC)
    ev = ev.dropna()
    ev_mean = DataFrame(ev['windcode'].drop_duplicates())
    ev_mean = ev_mean.reset_index(drop = True)
    code_list = list(ev_mean['windcode'])
#    ev_mean.index = ev_mean['windcode']
    ev_mean['ev'] = 0.0
    mean_value = []
    for i in range(len(code_list)):
        mean_value.append(log(ev['ev'][ev.windcode == code_list[i]].mean()))

    for i in range(len(mean_value)):
#        if mean_value[i] > np.mean(mean_value) + 3*np.std(mean_value):      # 去极值
#            mean_value[i] = np.mean(mean_value) + 3*np.std(mean_value)
#        elif mean_value[i] < np.mean(mean_value) + 3*np.std(mean_value):
#            mean_value[i] = np.mean(mean_value) - 3*np.std(mean_value)

        mean_value[i] = (mean_value[i]-np.mean(mean_value))/np.std(mean_value) # 标准化 计算z-score
    ev_mean['ev'] = mean_value

    sector_id_list = get_sector_id()
    indus_indic_fun_value = DataFrame(ev_mean['windcode'])
    for i in sector_id_list:
        indus_indic_fun_value[i] = 0
        codes_in_indus = pd.read_sql('''SELECT windcode FROM "WIND"."SECTOR_CN_6001" WHERE sector_id =
'{0}' and date = '{1}' '''.format(i,start_date), con=common.DB_SQL_CONN_PYODBC)
        codes_in_indus = list(codes_in_indus['windcode'])
        indus_indic_fun_value[i][indus_indic_fun_value['windcode'].isin(codes_in_indus)] = 1

    df_result = pd.merge(indus_indic_fun_value,ev_mean, on = ['windcode'])
    return df_result

# %%

'''
导入因子总表数据，并进行回归，得到行业和市值中性化后的因子值
'''

def neu_regression(start_date,end_date,factors):
    df_factors = constr_factors_df.constr_factors_df(start_date,end_date,factors)
    df_result = creat_sector_table(start_date,end_date)
    beta = calc_beta(start_date,transfer_interval,end_date)
    df_sector = pd.merge(df_result,beta,on = ['windcode'])
    df = pd.merge(df_factors,df_sector,on=['windcode'])
    df_neu_factors = DataFrame(df['windcode'])
    sector_id_list = get_sector_id()

    x = np.array(df.loc[:,['ev','beta'] + sector_id_list])

#    factors =
['mkt_cap_float','pb','pe','roa2','roe','asset_on_liab','eps','net_profit_rate','yoyocf','yoyeps_basic
','yoyocfps','quick','current','bps']
    for i in factors:
        df_neu_factors[i] = 0.0
        y = list(df[i])
        slope = np.linalg.lstsq(x,y)[0]
        for j in range(len(y)):
            y[j] = y[j] - np.dot(slope,x[j].T)
        df_neu_factors[i] = y
    return df_neu_factors

