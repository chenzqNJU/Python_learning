def search(key,dirFlag=0,file_dir=r'C:\Users\Administrator\Documents\PycharmProjects\code'):
    import os
    temp = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in ['.py','.txt']:
                temp.append(os.path.join(root, file))
    file_dir = r'C:\Users\Administrator\Documents\PycharmProjects'
    files = os.listdir(file_dir)
    for fi in files:
        fi_d = os.path.join(file_dir, fi)
        if os.path.splitext(fi_d)[1] in ['.py','.txt']:
            temp.append(fi_d)
    for dir in temp:
        s = 0
        #dir=file[8]
        try:
            with open(dir,encoding='utf-8') as f:
                for i in f:
                    if key in i:
                        print(i),
                        s+=1
                        if dirFlag==1:
                            print(dir)
                #if s == 0:
                    #print("don't match it！")
        except:
            try:
                with open(dir, encoding='gbk') as f:
                    for i in f:
                        if key in i:
                            print(i),
                            s += 1
                            if dirFlag == 1:
                                print(dir)
                    # if s == 0:
                    # print("don't match it！")
            except:
                print(dir)

    # import os
    # def gci(filepath):
    #     filepath=file_dir
    #     # 遍历filepath下所有文件，包括子目录
    #     files = os.listdir(filepath)
    #     fi= files[2]
    #     for fi in files:
    #         fi_d = os.path.join(filepath, fi)
    #         if os.path.isdir(fi_d):
    #             gci(fi_d)
    #         else:
    #             print(os.path.join(filepath, fi_d))

def delete():
    z = list(globals().keys())
    print(z)
    for key in z:
        if not key.startswith("__"):
            print(key)
            globals().pop(key)
def __Delete(t):
    z = list(globals().keys())
    print(z)
    z1=len(t)
    for key in z:
        if key.startswith(t) and ((len(key)==z1) or not key[z1].isalpha()):
            print(key)
            globals().pop(key)

def add(x, y):
    z=x*y
    return z

import sys, time

class ShowProcess():
    import sys, time

    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 1 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 1

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        import numpy as np
        if i is not None:
            self.i = i
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        if np.floor(percent)!=np.floor(percent-100.0 / self.max_steps):
            sys.stdout.write(process_bar) #这两句打印字符到终端
            sys.stdout.flush()
        self.i += 1

    def close(self, words='done'):
        print ('')
        print (words)
        self.i = 1
    '''
    if __name__=='__main__':
        max_steps = 100
    
        process_bar = ShowProcess(max_steps)
    
        for i in range(max_steps + 1):
            process_bar.show_process()
           # time.sleep(0.05)
        process_bar.close()   '''


def flag_jishu(df,key='date'):
    import numpy as np
    t=df[key].values
    t1=t[1:]>t[:-1]
    t1=np.concatenate(([True],t1))
    t2=np.cumsum(t1)
    df['flag']=t2

def zhangdie(Price):
    import numpy as np
    temp = Price[:1].copy();
    temp[:] = np.nan
    temp1 = round((Price + 0.000001) * 1.1, 2)
    temp1 = temp.append(temp1[:-1])
    temp1.index = Price.index
    zhangting = temp1 == Price
    temp2 = round((Price + 0.000001) * 0.9, 2)
    temp2 = temp.append(temp2[:-1])
    temp2.index = Price.index
    dieting = (temp2 == Price)
    return zhangting.astype(int)-dieting.astype(int)


def kxiantu(df,code=''):
    from matplotlib import dates as mdates
    import datetime as dt
    from matplotlib.finance import candlestick_ohlc
    from matplotlib import ticker as mticker
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.close('all')
    t=df.columns
    date = t[t.str.contains('ate')][0]
    open =t[t.str.contains('pen')][0]
    high =t[t.str.contains('igh')][0]
    low  =t[t.str.contains('ow')][0]
    close =t[t.str.contains('lose')][0]
    t=df[date].reset_index(drop=True)  #标签
    df[date] = pd.to_datetime(df[date])
    df.loc[:,'DateTime'] = mdates.date2num(df.loc[:,date].astype(dt.date))
    df.drop(date, axis=1, inplace=True)
    # 调整顺序
    df = df.reindex(columns=['DateTime', open, high, low, close])
    SP = len(df)
    fig = plt.figure(facecolor='#07000d', figsize=(15, 10))
    ax1 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4, axisbg='#07000d')
    # 蜡烛图
    t1=df.iloc[0,0]
    df.DateTime=range(len(df))+t1
    candlestick_ohlc(ax1, df.values[-SP:], width=0.8, colorup='#ff1717', colordown='#53c156')

    ax1.grid(True, color='blue')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.label.set_color("blue")

    #######################避免k线出现断裂（周末）采用label标签的方法
    label=[]
    for x in ax1.get_xticks():
        if x >= t1 and x <= t1+len(df)-1:label+=[t[x-t1]]
        elif x <= t1+len(df)-1:label+=[mdates.num2date(x).strftime("%Y-%m-%d")]
        else:label+=['']
    ax1.set_xticklabels(label)
    # 上下左右的边框线为蓝色
    ax1.spines['bottom'].set_color("#5998ff")
    ax1.spines['top'].set_color("#5998ff")
    ax1.spines['left'].set_color("#5998ff")
    ax1.spines['right'].set_color("#5998ff")
    ax1.tick_params(axis='y', colors='blue')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax1.tick_params(axis='x', colors='blue')
    plt.ylabel('Stock price and Volume')
    plt.title(code,fontsize='large', color='blue')
    return fig
    plt.show()

def kdj(df):
    #df=data
    import pandas as pd
    t=df.columns
    date = t[t.str.contains('ate')][0]
    open =t[t.str.contains('pen')][0]
    high =t[t.str.contains('igh')][0]
    low  =t[t.str.contains('ow')][0]
    close =t[t.str.contains('lose')][0]
    ticker=['code','ticker'][t.str.contains('ticker').any()]
    t=df[date].reset_index(drop=True)
    df[date] = pd.to_datetime(df[date])

    Price = pd.pivot_table(df, index=date, columns=ticker, values=close)

    for i in ['openPrice', 'lowestPrice', 'highestPrice']:
        locals()[i] = pd.read_csv(i + '.csv').iloc[-100:, :10]
        locals()[i].tradeDate = pd.to_datetime(locals()[i].tradeDate, format='%Y-%m-%d')
        locals()[i].set_index('tradeDate', inplace=True)
        locals()[i] = locals()[i].loc[Price.index]

    RSV = (Price - lowestPrice) / (highestPrice - lowestPrice) * 100
    RSV.replace(np.inf, np.nan, inplace=True)
    # 涨跌停
    temp = Price[:1].copy();
    temp[:] = np.nan
    temp1 = round((Price + 0.000001) * 1.1, 2)
    temp1 = temp.append(temp1[:-1])
    temp1.index = Price.index
    zhangting = temp1 == Price
    temp2 = round((Price + 0.000001) * 0.9, 2)
    temp2 = temp.append(temp2[:-1])
    temp2.index = Price.index
    dieting = (temp2 == Price)

    RSV[dieting] = 0
    RSV[zhangting] = 100
    ####计算k值
    for n, i in enumerate(RSV.index):
        # n=1;i=Price.index[n]
        if n == 0:
            K = np.ones((1, RSV.shape[1])) * 50;
            tempK = K[0].copy()
            D = K.copy()
            tempD = tempK.copy()
            RSV_ = RSV.values
        else:
            x = np.where(isOpen.loc[i] == 1)
            tempK[x] = RSV_[n, x] * 1 / 3 + tempK[x] * 2 / 3
            # temp[np.isnan(temp)^np.isnan(Price_[n,])]=50
            K = np.row_stack((K, tempK))
            tempD[x] = tempK[x] * 1 / 3 + tempD[x] * 2 / 3
            D = np.row_stack((D, tempD))

    K = pd.DataFrame(K, index=Price.index, columns=Price.columns).round(2)
    D = pd.DataFrame(D, index=Price.index, columns=Price.columns).round(2)
    K_ = pd.DataFrame(K.unstack(), columns=['K'])
    D_ = pd.DataFrame(D.unstack(), columns=['D'])
    kdj = pd.concat([K_, D_], axis=1)
    kdj['J'] = kdj.K * 2 - kdj.D

# date: str类型日期eg.'2017-11-23'
def get_date_list(begin_date=None, end_date=None):
    '''
    dates = get_date_list(datetime.date(2019, 2, 18), datetime.date.today())
    get_date_list()
    :param begin_date:
    :param end_date:
    :return:
    '''
    from pandas import read_csv
    import datetime
    date_list=[];dates=[]
    if end_date==None:
        end_date = datetime.date.today()
    if begin_date==None:
        begin_date = end_date - datetime.timedelta(days=365)
    while begin_date <= end_date:
        # date_str = str(begin_date)
        date_list.append(begin_date)
        begin_date += datetime.timedelta(days=1)
    # cal_dates = ts.trade_cal()
    cal_dates = read_csv('e:\\data\\cal_dates.csv', encoding='GBK').iloc[:, -2:]
    def is_open_day(date):
        if date in cal_dates['calendarDate'].values:
            return cal_dates[cal_dates['calendarDate'] == date].iat[0, 1] == 1
        return False
    dates = [str(x) for x in date_list if is_open_day(str(x))]
    return dates





def formula(t):
    import sympy
    if isinstance(t,str):
        for i in range(10,1,-1):
            try:
                t=eval(t)
                break
            except Exception as e:
                a = str(e).split("'")[1]
                exec(a + '=sympy.Symbol(a)')

    t = sympy.latex(t)
    t = t.center(len(t) + 2, '$')
    # list_i = list(t)    # str -> list
    # list_i.insert(0, '$')   # 注意不用重新赋值
    # list_i.insert(len(t)+1, '$')
    # str_i = ''.join(list_i)    # list -> str
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim([1,10])
    ax.set_ylim([1,5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.text(3,3,t,fontsize=28)
    plt.show()

    #t='$\int_a^b f(x)\mathrm{d}x$'

