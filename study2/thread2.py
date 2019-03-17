from queue import Queue
import threading
import os
import datetime
import pandas as pd
import tushare as ts
from sqlalchemy import create_engine
from sqlalchemy import types


# 获取所有股票数据，利用股票代码获取复权数据
stock_basics = ts.get_stock_basics()
# stock_basics = ts.get_hs300s() #获取上证50成份股  返回值为DataFrame：code股票代码 name股票名称

cal_dates = ts.trade_cal()  # 返回交易所日历，类型为DataFrame, calendarDate  isOpen

stock_basics  = pd.read_csv('e:\\data\\stock_basics.csv',encoding='GBK')
stock_basics.index = stock_basics.code.astype('str').str.zfill(6)
cal_dates  = pd.read_csv('e:\\data\\cal_dates.csv',encoding='GBK').iloc[:,-2:]


# 本地实现判断市场开市函数
#date: str类型日期eg.'2017-11-23'
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
dates = get_date_list(datetime.date(2018, 6, 4), datetime.date(2019, 3, 4))
dates = [x  for x in dates if is_open_day(str(x))]

dates=dates[-3:-1]

stock_code_queue = Queue()
# for code in stock_basics.index[100:400]:
#     stock_code_queue.put(code)
for date in dates:
    for code in stock_basics.index:
        stock_code_queue.put((code,date))
# for date in dates[2:]:
#     for code in stock_basics.code:
#         stock_code_queue.put((code,date))

task_qeue=stock_code_queue

dir = 'e:\\data\\stk\\'
# 获取复权数据
def process_data(task_qeue):
    #queueLock.acquire()
    while not task_qeue.empty():
        t = task_qeue.get()
        data = t[0];date=t[1]
        print("正在获取%s;数据还有%s条:" %(data,task_qeue.qsize()))
        date = str(date)
        if not os.path.exists(dir + date):
            os.makedirs(dir + date)
        try:
            # qfq_day = ts.get_h_data(data,start = str(date_begin),end=str(date_end),autype='qfq',drop_factor=False)
            qfq_day = ts.get_tick_data(data, date, pause=0.1, src='tt')
            # ts.get_h_data('002337', start='2015-01-01', end='2015-03-16')

            qfq_day['code'] = data
            # qfq_day.to_sql('fq_day',engine,if_exists='append',dtype=type_fq_day)
            qfq_day.to_csv(dir + date + '\\' +data+'.csv',encoding='gbk')
            # qfq_day.to_csv('e:\\'+data+'.csv',encoding='gbk',index=None)

        except:
            # task_qeue.put(data)  # 如果数据获取失败，将该数据重新存入到队列，便于后期继续执行
            print(data + '  '+ date +  '  None')


# import pandas as pd
# file = r'e:\a.h5'
#
# hdf5_file = pd.HDFStore(file, 'w', complevel=4, complib='blosc')
# hdf5_file['data'] = qfq_day
# hdf5_file.close()
# pd.read_hdf(file)

# 重写线程类，用户获取数据
class get_qfq(threading.Thread):

    def __init__(self, name, queue):
        threading.Thread.__init__(self)
        self.name = name
        self.queue = queue
    def run(self):
        process_data(self.queue)
        print("Exiting " + self.name)

# 声明线程锁
#queueLock = threading.Lock()
# old_date = get_old_date()
# date = datetime.date(2018,6,4)

# 生成10个线程
starttime = datetime.datetime.now()
threads = []
for i in range(20):
    thread = get_qfq('thread'+ str(i), stock_code_queue)
    thread.start()
    threads.append(thread)
for thread in threads:
    thread.join()
endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)
print((endtime - starttime))
os.system('shutdown -s -f -t 10')
#
#
# import time
# time.sleep(1)
# help(threading.Thread.start)
#
#
# thread._Thread__stop()
#
# import signal
# import os
# import threading
# from time import sleep
# def f(a,b):
#     # print 'kill me'
#     os.kill(os.getpid(),signal.SIGKILL)
#
# def tf():
#     sleep(20)
#
# signal.signal(signal.SIGINT,f)
# p=threading.Thread(target=tf)
# p.start()
# p.join()
#
# threading.enumerate()
# threading.current_thread()
#
#
# import time
# def worker():
#     print("worker")
#     time.sleep(10)
#     print("worker1")
#     return
#
# starttime = datetime.datetime.now()
# for i in range(5):
#     t = threading.Thread(target=worker)
#     t.start()
# endtime = datetime.datetime.now()
# # print((endtime - starttime).seconds)
# print((endtime - starttime))
#
# "current has %d threads" % (threading.activeCount() - 1)


# def worker():
#     time.sleep(3)
#     print("worker")
#
#
# t = threading.Thread(target=worker)
# t.setDaemon(True)
# t.start()


# ###################################################
# import numpy as np
# import pandas as pd
# import tushare as ts
# import datetime
# import time
# import tushare as ts
# import os
# import tables
#
# data_dir = 'E:\\stk\\'  # 下载数据的存放路径
#
# # ts.get_sz50s() #获取上证50成份股  返回值为DataFrame：code股票代码 name股票名称
#
# cal_dates = ts.trade_cal()  # 返回交易所日历，类型为DataFrame, calendarDate  isOpen
#
#
# # 本地实现判断市场开市函数
# #date: str类型日期eg.'2017-11-23'
#
# def is_open_day(date):
#     if date in cal_dates['calendarDate'].values:
#         return cal_dates[cal_dates['calendarDate'] == date].iat[0, 1] == 1
#     return False
#
#
# # 从TuShare获取tick data数据并保存到本地
# # @symbol: str类型股票代码 eg.600030
# # @date: date类型日期
# # get_save_tick_data(stock, date)
#
# # stock='600848'
# # str_date='2018-05-30'
# # date = datetime.datetime.strptime(str_date,'%Y-%m-%d').date()
# # a=datetime.date(2012,11,19)
# # b=datetime.datetime(2012,11,19)
# # date=datetime.datetime.strptime(str_date,'%Y-%m-%d')
#
# #
# # is_open_day(str_date)
# # d = ts.get_tick_data(symbol, str_date, src='tt')
# # sleep_time = 10
#
#     # 获取从起始日期到截止日期中间的的所有日期，前后都是封闭区间
# def get_date_list(begin_date, end_date):
#     date_list = []
#     while begin_date <= end_date:
#         # date_str = str(begin_date)
#         date_list.append(begin_date)
#         begin_date += datetime.timedelta(days=1)
#     return date_list
#
#
#     # 获取感兴趣的所有股票信息，这里只获取沪深300股票
# def get_all_stock_id():
#     stock_info = ts.get_hs300s()
#     return stock_info['code'].values
#
#
#     # 从TuShare下载感兴趣的所有股票的历史成交数据，并保存到本地HDF5压缩文件
#     # dates=get_date_list(datetime.date(2017,11,6), datetime.date(2017,11,12))
# # queue=stock_code_queue
# stock='600036'
# def get_save_tick_data(queue, date):
#     while not queue.empty():
#         stock = queue.get()
#         print("正在获取%s;数据还有%s条:" % (stock, queue.qsize()))
#
#
#         str_date = str(date)
#         dir = data_dir + stock
#         file = dir + '\\' + stock + '_' + str_date + '_tick_data.h5'
#         if is_open_day(str_date):
#             if not os.path.exists(dir):
#                 os.makedirs(dir)
#             if not os.path.exists(file):
#                 try:
#                     d = ts.get_tick_data(stock, str_date,pause=0.1, src='tt')
#                     hdf5_file = pd.HDFStore(file, 'w', complevel=4, complib='blosc')
#                     hdf5_file['data'] = d
#                     hdf5_file.close()
#                     # sleep_time = max(sleep_time / 2, 2)  # 每次成功下载后sleep_time变为一半，但是至少2s
#                     print("Successfully download and save file: " + file)
#                 except:
#                     print('none')
#                     # sleep_time = min(sleep_time * 2, 128)  # 每次下载失败后sleep_time翻倍，但是最大128s
#                     # print('Get tick data error: symbol: ' + stock + ', date: ' + str_date)
#             else:
#                 print("Data already downloaded before, skip " + file)
#
#
#
#
# # str_date='2018-06-04'
#
# # d = ts.get_tick_data(stock, str_date, src='tt')
#
# sleep_time = 2
# dates = get_date_list(datetime.date(2018, 6, 4), datetime.date(2018, 12, 30))
# stocks = get_all_stock_id()
# stock = stocks[1]
# date = dates[2]
#
# import threading
# from queue import Queue
# stock_code_queue = Queue()
# for code in stocks:
#     stock_code_queue.put(code)
# task_qeue=stock_code_queue
#
#
#
#
# class get_qfq(threading.Thread):
#     def __init__(self, name, queue, date):
#         threading.Thread.__init__(self)
#         self.name = name
#         self.queue = queue
#         self.date = date
#     def run(self):
#         get_save_tick_data(self.queue, self.date)
#         print("Exiting " + self.name)
#
#
# starttime = datetime.datetime.now()
# threads = []
# for i in range(5):
#     thread = get_qfq('thread'+ str(i), stock_code_queue,date)
#     thread.start()
# #     threads.append(thread)
# # for thread in threads:
# #     thread.join()
# endtime = datetime.datetime.now()
# # print((endtime - starttime).seconds)
