# from queue import Queue
# import threading
# import os
# import datetime
#
# import tushare as ts
# from sqlalchemy import create_engine
# from sqlalchemy import types
#
#
# # 创建myql数据库引擎，便于后期链接数据库
# mysql_info = {'host':'localhost','port':3306,'user':'******','passwd':'******','db':'stock','charset':'utf8'}
# engine = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=%s' %(mysql_info['user'],mysql_info['passwd'],
#                         mysql_info['host'],mysql_info['port'],mysql_info['db'],mysql_info['charset']),echo=False)
#
# # 获取所有股票数据，利用股票代码获取复权数据
# stock_basics = ts.get_stock_basics()
# stock_basics.columns
#
# # 获取数据库现有数据的时间日期
# # def get_old_date():
# #     con = engine.connect()
# #     sql1 = 'show tables;'
# #     tables = con.execute(sql1)
# #     if ('fq_day',) not in tables:
# #         date_old = datetime.date(2001,1,1)
# #         return date_old
# #     sql2 = 'select max(date) from fq_day;'
# #     date_old = con.execute(sql2).fetchall()[0][0].date()
# #     if date_old < datetime.date.today() - datetime.timedelta(1):
# #         return date_old
# #     else:
# #         con.close()
# #         print('今天已经获取过数据，不需重新获取')
# #         os._exit(1)
#
# # 声明队列，用于存取股票以代码数据，以便获取复权明细
#
# stock_code_queue = Queue()
# for code in stock_basics.index[100:400]:
#     stock_code_queue.put(code)
#
# # type_fq_day = {'code':types.CHAR(6),'open':types.FLOAT,'hige':types.FLOAT,'close':types.FLOAT,'low':types.FLOAT,
# #               'amount':types.FLOAT,'factor':types.FLOAT}
#
# task_qeue=stock_code_queue
#
# date_begin=datetime.date(2018,12,1)
# old_date = datetime.date(2018,12,4)
# date_end = datetime.date(2018,12,20)
# # 获取复权数据
# def process_data(old_date,task_qeue):
#     #queueLock.acquire()
#     while not task_qeue.empty():
#         data = task_qeue.get()
#         print("正在获取%s;数据还有%s条:" %(data,task_qeue.qsize()))
#         #queueLock.release()
#         date_begin = old_date + datetime.timedelta(1)
#         # date_end = datetime.date.today()
#         try:
#             qfq_day = ts.get_h_data(data,start = str(date_begin),end=str(date_end),autype='qfq',drop_factor=False)
#             # ts.get_h_data('002337', start='2015-01-01', end='2015-03-16')
#
#             qfq_day['code'] = data
#             # qfq_day.to_sql('fq_day',engine,if_exists='append',dtype=type_fq_day)
#             qfq_day.to_csv('e:\\stk\\'+data+'.csv')
#         except:
#             # task_qeue.put(data)  # 如果数据获取失败，将该数据重新存入到队列，便于后期继续执行
#             print('None')
#     #else:
#         #queueLock.release()
#
# # starttime = datetime.datetime.now()
# # while not task_qeue.empty():
# #     data = task_qeue.get()
# #     print("正在获取%s;数据还有%s条:" % (data, task_qeue.qsize()))
# #     # queueLock.release()
# #     date_begin = old_date + datetime.timedelta(1)
# #     # date_end = datetime.date.today()
# #     try:
# #         qfq_day = ts.get_h_data(data, start=str(date_begin), end=str(date_end), autype='qfq', drop_factor=False)
# #         qfq_day['code'] = data
# #         # qfq_day.to_sql('fq_day',engine,if_exists='append',dtype=type_fq_day)
# #         qfq_day.to_csv('e:\\stk\\' + data + '.csv')
# #     except:
# #         # task_qeue.put(data)  # 如果数据获取失败，将该数据重新存入到队列，便于后期继续执行
# #         print('None')
# # endtime = datetime.datetime.now()
# # # print((endtime - starttime).seconds)
# # print((endtime - starttime))
#
# # 重写线程类，用户获取数据
# class get_qfq(threading.Thread):
#
#     def __init__(self, name, queue, date_begin):
#         threading.Thread.__init__(self)
#         self.name = name
#         self.queue = queue
#         self.begin = date_begin
#
#     def run(self):
#         process_data(self.begin, self.queue)
#         print("Exiting " + self.name)
#
# # 声明线程锁
# #queueLock = threading.Lock()
# # old_date = get_old_date()
# # old_date = datetime.date(2018,12,1)
#
# # 生成10个线程
# starttime = datetime.datetime.now()
# threads = []
# for i in range(20):
#     thread = get_qfq('thread'+ str(i), stock_code_queue,old_date)
#     thread.start()
#     threads.append(thread)
# for thread in threads:
#     thread.join()
# endtime = datetime.datetime.now()
# # print((endtime - starttime).seconds)
# print((endtime - starttime))
#
# #
# #
# # import time
# # time.sleep(1)
# # help(threading.Thread.start)
# #
# #
# # thread._Thread__stop()
# #
# # import signal
# # import os
# # import threading
# # from time import sleep
# # def f(a,b):
# #     # print 'kill me'
# #     os.kill(os.getpid(),signal.SIGKILL)
# #
# # def tf():
# #     sleep(20)
# #
# # signal.signal(signal.SIGINT,f)
# # p=threading.Thread(target=tf)
# # p.start()
# # p.join()
# #
# # threading.enumerate()
# # threading.current_thread()
# #
# #
# # import time
# # def worker():
# #     print("worker")
# #     time.sleep(10)
# #     print("worker1")
# #     return
# #
# # starttime = datetime.datetime.now()
# # for i in range(5):
# #     t = threading.Thread(target=worker)
# #     t.start()
# # endtime = datetime.datetime.now()
# # # print((endtime - starttime).seconds)
# # print((endtime - starttime))
# #
# # "current has %d threads" % (threading.activeCount() - 1)
#
#
# # def worker():
# #     time.sleep(3)
# #     print("worker")
# #
# #
# # t = threading.Thread(target=worker)
# # t.setDaemon(True)
# # t.start()
