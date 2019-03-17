import tushare as ts

del ts
# 历史分笔
a = ts.get_tick_data('000005', date='2018-12-24')

help(ts)
# 当日历史分
b = ts.get_today_ticks('601333', date='2017-01-09')

# 实时分笔
ts.get_realtime_quotes('000581')

# 历史分笔 和 当日历史分 返回结果
time：时间
price：成交价格
pchange：涨跌幅
change：价格变动
volume：成交手
amount：成交金额(元)
type：买卖类型【买盘、卖盘、中性盘】


import easyquotation

import tf

from urllib.request import urlretrieve

f = open('SHA.csv', 'r')
for line in f:
    data = line.split(',')
    stock_no = '0' + data[0].strip()
    start_date = data[1].strip().replace('-', '')
    url = 'http://quotes.money.163.com/service/chddata.html?code=' + stock_no + '&start=' + start_date + '&end=20181103&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    filename = stock_no + '.csv'
    print(url)
    urlretrieve(url, filename)

http: // quotes.money
.163.com / service / chddata.html?code = 1000005 & start = 20180903 & end = 20181103 & fields = TCLOSE;
HIGH;
LOW;
TOPEN;
LCLOSE;
CHG;
PCHG;
TURNOVER;
VOTURNOVER;
VATURNOVER;
TCAP;
MCAP
'

#############################################################
#############################################################
import urllib.request
import re

##def downback(a,b,c):
##    ''''
##    a:已经下载的数据块
##    b:数据块的大小
##    c:远程文件的大小
##   '''
##    per = 100.0 * a * b / c
##    if per > 100 :
##        per = 100
##    print('%.2f%%' % per)
stock_CodeUrl = 'http://quote.eastmoney.com/stocklist.html'

url = stock_CodeUrl


# 获取股票代码列表
def urlTolist(url):
    allCodeList = []
    html = urllib.request.urlopen(url).read()
    html = html.decode('gbk')
    s = r'<li><a target="_blank" href="http://quote.eastmoney.com/\S\S(.*?).html">'
    pat = re.compile(s)
    code = pat.findall(html)
    for item in code:
        if item[0] == '6' or item[0] == '3' or item[0] == '0':
            allCodeList.append(item)
    return allCodeList


allCodelist = urlTolist(stock_CodeUrl)
start = '20181201'
end = '20181230'
code = '000005'

for code in allCodelist:
    print('正在获取%s股票数据...' % code)
    if code[0] == '6':
        url = 'http://quotes.money.163.com/service/chddata.html?code=0' + code + '&start=' + start + \
              '&end=' + end + '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    else:
        url = 'http://quotes.money.163.com/service/chddata.html?code=1' + code + '&start=' + start + \
              '&end=' + end + '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    urllib.request.urlretrieve(url, 'E:\\stock\\' + code + '.csv')  # 可以加一个参数dowmback显示下载进度

##########################################################
import tushare as ts
cal_dates = ts.trade_cal()

df = ts.get_tick_data('000756', '2014-12-27', src='tt')

df = ts.get_tick_data('600848', date='2014-12-12', src='tt')

print(ts.__version__)

###################################################
import numpy as np
import pandas as pd
import tushare as ts
import datetime
import time
import tushare as ts
import os
import tables

data_dir = 'E:\\stock\\'  # 下载数据的存放路径

# ts.get_sz50s() #获取上证50成份股  返回值为DataFrame：code股票代码 name股票名称

cal_dates = ts.trade_cal()  # 返回交易所日历，类型为DataFrame, calendarDate  isOpen


# 本地实现判断市场开市函数
#date: str类型日期eg.'2017-11-23'

def is_open_day(date):
    if date in cal_dates['calendarDate'].values:
        return cal_dates[cal_dates['calendarDate'] == date].iat[0, 1] == 1
    return False


# 从TuShare获取tick data数据并保存到本地
# @symbol: str类型股票代码 eg.600030
# @date: date类型日期
# get_save_tick_data(stock, date)
stock='600848'
str_date='2018-05-30'
date = datetime.datetime.strptime(str_date,'%Y-%m-%d').date()
# a=datetime.date(2012,11,19)
# b=datetime.datetime(2012,11,19)
# date=datetime.datetime.strptime(str_date,'%Y-%m-%d')

#
# is_open_day(str_date)
# d = ts.get_tick_data(symbol, str_date, src='tt')
# sleep_time = 10

    # 获取从起始日期到截止日期中间的的所有日期，前后都是封闭区间
def get_date_list(begin_date, end_date):
    date_list = []
    while begin_date <= end_date:
        # date_str = str(begin_date)
        date_list.append(begin_date)
        begin_date += datetime.timedelta(days=1)
    return date_list


    # 获取感兴趣的所有股票信息，这里只获取沪深300股票
def get_all_stock_id():
    stock_info = ts.get_hs300s()
    return stock_info['code'].values


    # 从TuShare下载感兴趣的所有股票的历史成交数据，并保存到本地HDF5压缩文件
    # dates=get_date_list(datetime.date(2017,11,6), datetime.date(2017,11,12))

def get_save_tick_data(stock, date):
    global sleep_time, data_dir
    res = True
    str_date = str(date)
    dir = data_dir + stock + '\\' + str(date.year) + '\\' + str(date.month)
    file = dir + '\\' + stock + '_' + str_date + '_tick_data.h5'
    if is_open_day(str_date):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(file):
            try:
                d = ts.get_tick_data(stock, str_date,pause=0.1, src='tt')
                if d is None:
                    return 'None'
                # print(d)
                # df = ts.get_tick_data('600848', date='2014-12-12', src='tt')

            except IOError as msg:
                print(str(msg))
                sleep_time = min(sleep_time * 2, 128)  # 每次下载失败后sleep_time翻倍，但是最大128s
                print('Get tick data error: symbol: ' + stock + ', date: ' + str_date + ', sleep time is: ' + str(sleep_time))
                return res
            else:
                hdf5_file = pd.HDFStore(file, 'w', complevel=4, complib='blosc')
                hdf5_file['data'] = d
                hdf5_file.close()
                sleep_time = max(sleep_time / 2, 2)  # 每次成功下载后sleep_time变为一半，但是至少2s
                print("Successfully download and save file: " + file + ', sleep time is: ' + str(sleep_time))
                return res
        else:
            print("Data already downloaded before, skip " + file)
            res = False
            return res



dates = get_date_list(datetime.date(2017, 10, 30), datetime.date(2017, 11, 4))
stocks = get_all_stock_id()
stock = stocks[1]
date = dates[1]

for stock in stocks:
    for date in dates:
        if get_save_tick_data(stock, date):
            time.sleep(sleep_time)

sleep_time=10
date = datetime.date(2018,6,4)
for stock in stocks:
    get_save_tick_data(stock, date)


sleep_time=10
stock = stocks[1]

date = datetime.date(2018,6,1)
str_date='2018-06-04'
d = ts.get_tick_data(stock, str_date, src='tt')

d==None


##########################################
# 在python3下测试

import sys
import requests
import threading
import datetime

# 传入的命令行参数，要下载文件的url
url = sys.argv[1]

url = 'http://quotes.money.163.com/service/chddata.html?code=0' + '000001' + '&start=' + '20180101' + \
              '&end=' + '20181230' + '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'

def Handler(start, end, url, filename):
    headers = {'Range': 'bytes=%d-%d' % (start, end)}
    r = requests.get(url, headers=headers, stream=True)

    # 写入文件对应位置
    with open(filename, "r+b") as fp:
        fp.seek(start)
        var = fp.tell()
        fp.write(r.content)


def download_file(url, num_thread=5):
    r = requests.head(url)
    try:
        file_name = url.split('/')[-1]
        file_size = int(
            r.headers['content-length'])  # Content-Length获得文件主体的大小，当http服务器使用Connection:keep-alive时，不支持Content-Length
    except:
        print("检查URL，或不支持对线程下载")
        return

    #  创建一个和要下载文件一样大小的文件
    fp = open(file_name, "wb")
    fp.truncate(file_size)
    fp.close()

    # 启动多线程写文件
    part = file_size // num_thread  # 如果不能整除，最后一块应该多几个字节
    for i in range(num_thread):
        start = part * i
        if i == num_thread - 1:  # 最后一块
            end = file_size
        else:
            end = start + part

        t = threading.Thread(target=Handler, kwargs={'start': start, 'end': end, 'url': url, 'filename': file_name})
        t.setDaemon(True)
        t.start()

    # 等待所有线程下载完成
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
        t.join()
    print('%s 下载完成' % file_name)


if __name__ == '__main__':
    start = datetime.datetime.now().replace(microsecond=0)
    download_file(url)
    end = datetime.datetime.now().replace(microsecond=0)
    print("用时: ", end='')
    print(end - start)

###################################################
import numpy as np
import pandas as pd
import tushare as ts
import datetime
import time
import tushare as ts
import os
import tables

data_dir = 'E:\\stk\\'  # 下载数据的存放路径

# ts.get_sz50s() #获取上证50成份股  返回值为DataFrame：code股票代码 name股票名称

cal_dates = ts.trade_cal()  # 返回交易所日历，类型为DataFrame, calendarDate  isOpen


# 本地实现判断市场开市函数
#date: str类型日期eg.'2017-11-23'

def is_open_day(date):
    if date in cal_dates['calendarDate'].values:
        return cal_dates[cal_dates['calendarDate'] == date].iat[0, 1] == 1
    return False


# 从TuShare获取tick data数据并保存到本地
# @symbol: str类型股票代码 eg.600030
# @date: date类型日期
# get_save_tick_data(stock, date)

# stock='600848'
# str_date='2018-05-30'
# date = datetime.datetime.strptime(str_date,'%Y-%m-%d').date()
# a=datetime.date(2012,11,19)
# b=datetime.datetime(2012,11,19)
# date=datetime.datetime.strptime(str_date,'%Y-%m-%d')

#
# is_open_day(str_date)
# d = ts.get_tick_data(symbol, str_date, src='tt')
# sleep_time = 10

    # 获取从起始日期到截止日期中间的的所有日期，前后都是封闭区间
def get_date_list(begin_date, end_date):
    date_list = []
    while begin_date <= end_date:
        # date_str = str(begin_date)
        date_list.append(begin_date)
        begin_date += datetime.timedelta(days=1)
    return date_list


    # 获取感兴趣的所有股票信息，这里只获取沪深300股票
def get_all_stock_id():
    stock_info = ts.get_hs300s()
    return stock_info['code'].values


    # 从TuShare下载感兴趣的所有股票的历史成交数据，并保存到本地HDF5压缩文件
    # dates=get_date_list(datetime.date(2017,11,6), datetime.date(2017,11,12))
# queue=stock_code_queue
stock='600036'
def get_save_tick_data(queue, date):
    while not queue.empty():
        stock = queue.get()
        print("正在获取%s;数据还有%s条:" % (stock, queue.qsize()))


        str_date = str(date)
        dir = data_dir + stock
        file = dir + '\\' + stock + '_' + str_date + '_tick_data.h5'
        if is_open_day(str_date):
            if not os.path.exists(dir):
                os.makedirs(dir)
            if not os.path.exists(file):
                try:
                    d = ts.get_tick_data(stock, str_date,pause=0.1, src='tt')
                    hdf5_file = pd.HDFStore(file, 'w', complevel=4, complib='blosc')
                    hdf5_file['data'] = d
                    hdf5_file.close()
                    # sleep_time = max(sleep_time / 2, 2)  # 每次成功下载后sleep_time变为一半，但是至少2s
                    print("Successfully download and save file: " + file)
                except:
                    print('none')
                    # sleep_time = min(sleep_time * 2, 128)  # 每次下载失败后sleep_time翻倍，但是最大128s
                    # print('Get tick data error: symbol: ' + stock + ', date: ' + str_date)
            else:
                print("Data already downloaded before, skip " + file)




# str_date='2018-06-04'

# d = ts.get_tick_data(stock, str_date, src='tt')

sleep_time = 2
dates = get_date_list(datetime.date(2018, 6, 4), datetime.date(2018, 12, 30))
stocks = get_all_stock_id()
stock = stocks[1]
date = dates[2]

import threading
from queue import Queue
stock_code_queue = Queue()
for code in stocks:
    stock_code_queue.put(code)
task_qeue=stock_code_queue




class get_qfq(threading.Thread):
    def __init__(self, name, queue, date):
        threading.Thread.__init__(self)
        self.name = name
        self.queue = queue
        self.date = date
    def run(self):
        get_save_tick_data(self.queue, self.date)
        print("Exiting " + self.name)


starttime = datetime.datetime.now()
threads = []
for i in range(4):
    thread = get_qfq('thread'+ str(i), stock_code_queue,date)
    thread.start()
    threads.append(thread)
for thread in threads:
    thread.join()
endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)
print((endtime - starttime))