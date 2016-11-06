# coding: utf-8
import os
import sys
import numpy as np
import pandas as pd
from itertools import izip

from statsmodels.tsa.arima_model import ARMA, _arma_predict_out_of_sample
from statsmodels.tsa.arima_model import ARIMA

basedir = os.getcwd()
filename = os.path.join(basedir, 'stock_fanny.xlsx')
xls = pd.ExcelFile(filename)

df_train = xls.parse('Sheet4', index_col='Date') # train
df_train.index = pd.to_datetime(df_train.index)
print "length df_train = ",len(df_train)

class BuildModel:
    def __init__(self,
        p=2,                 # Auto Regression (AR), lag p
        d=1,                 # Differencing NaN shifting for ARIMA, lag d
        q=0,                 # Moving Average (MA), lag q
        n_days=1,            # How many days a head to be predicted
        fc=1,                # iteration for n days forecast
        kdj_short_period=3,  # to calculate D (Divergen) K
        kdj_long_period=5):  # to calculate K

        # ARMA/ ARIMA
        self.p = p
        self.d = d
        self.q = q
        
        # how many days a head to be forecasted
        self.n_days = n_days

        # KDJ
        self.kdj_short_period = kdj_short_period
        self.kdj_long_period  = kdj_long_period

        # fc
        self.fc = fc

    # -------------------------------------------------------------
    # 1st Section
    # ARIMA/ ARMA
    
    def predict_arima_next_days(self, item):
        ts = df_train[item]
        ts = ts.sort_index() # sorting index Date
        ts_last_day = ts[self.fc] # real last data
        ts = ts[0:self.fc] # index 0 until last data - 1

        model = ARIMA(ts, order=(self.p, self.d, self.q)) # build a model
        fitting = model.fit(disp=False)

        # n_days forecasting
        forecast, fcasterr, conf_int = fitting.forecast(steps=self.n_days, alpha=.05)
        # ts:          history until 1 day before self.fc
        # ts[self.fc]: last day
        # forecast:    1 day forecast (time equalto ts[self.fc])
        return ts, ts_last_day, forecast
    
    def predict_arma_next_days(self, item):
        ts = df_train[item]
        ts = ts.sort_index() # sorting index Date
        ts_last_day = ts[self.fc] # real last data
        ts = ts[0:self.fc] # index 0 until last data - 1

        model = ARMA(ts, order=(self.p, self.q), freq='D') # build a model
        fitting = model.fit(disp=False)
        params = fitting.params
        residuals = fitting.resid
        p = fitting.k_ar
        q = fitting.k_ma
        k_exog = fitting.k_exog
        k_trend = fitting.k_trend

        # n_days forecasting
        forecast = _arma_predict_out_of_sample(params, self.n_days, residuals, p, q, k_trend, k_exog, endog=ts, exog=None, start=len(ts))
        # ts:          history until 1 day before self.fc
        # ts[self.fc]: last day
        # forecast:    1 day forecast (time equalto ts[self.fc])
        return ts, ts_last_day, forecast

    # -------------------------------------------------------------
    # 2nd Section
    # KDJ Indicator
    
    def rsv_indicator(self, cl, lo_prev, hi_prev):
        return 100 * (cl-np.min(lo_prev))/(np.max(hi_prev)-np.min(lo_prev))
    
    def kdj_indicator(self, rsv):
        rsv = pd.DataFrame(rsv)
        # get K by EMA
        K = rsv.ewm(ignore_na=False, span=self.kdj_long_period, min_periods=0, adjust=True).mean()
        # get D (Divergen of K) by MA
        K.dropna(inplace=True)
        D = K.rolling(window=self.kdj_short_period, center=False).mean()
        # get J
        J = 3*K - 2*D
        # print ( 'Calculate KDJ Indicator.' )
        return K, D, J
    
    def kdj_calc(self, open, high, low, close, volume):
        # print ( 'Length data before KDJ in: {}'.format(len(open)) )
        # X is total input list
        # y is output list
        rsv_list, X, y = [], [], []
        for i, op in enumerate(open):
            hi, lo, cl, vo = high[i], low[i], close[i], volume[i]
            rsv = 0.0
            # lo_prev is a list for previous low price
            # hi_prev is a list for previous high price
            lo_prev, hi_prev = [], []
            if i > kdj_long_period-1:
                for k in range(1, kdj_long_period+1):
                    _lo, _hi = low[i-k], high[i-k]
                    lo_prev.append(_lo)
                    hi_prev.append(_hi)
                rsv = self.rsv_indicator(cl, lo_prev, hi_prev)
            else: pass

            rsv_list.append(rsv)
            X.append([ op, hi, lo, cl, vo ])
            
            y.append([ cl ])


        K, D, J = self.kdj_indicator(rsv_list)
        # Convert from pandas DataFrame to a list
        K = map( float, K[0].tolist() )
        D = map( float, D[0].tolist() )
        J = map( float, J[0].tolist() )
                
        # Remove Index value in X and y before index self.kdj_long_period to get rid of NaN
        # X_new and Y_new for list input and output new X and y with index started from 'self.kdj_long_period'
        X_new, y_new = [], []
        for i, row in enumerate(X[self.kdj_long_period:], start=self.kdj_long_period):
            op, hi, lo, cl, vo = row
            _K, _D, _J = K[i], D[i], J[i]
                
            # X new for input list
            X_new.append([ op, hi, lo, cl, vo, _K, _D, _J ])
                
            # y new for output list
            y_new.append([ y[i][0] ])
            
        X, y = X_new, y_new
        # print ( 'Length data after KDJ in: {}'.format(len(X)) )
        return X, y




# -----------------------------------------------------------
# Get the forecast each of items/price & volume
# instance: is instance of class BuildModel
# item: is stock prices and volume:
#       e.g: 'Open', 'High', 'Low', 'Close', 'Volume'
# d: is from p, d, q:
#       d=0 for ARMA, and d=1 for ARIMA
def item_prediction(instance, d, item):
    ts, n_days_real, n_days_forecast = None, None, None
    if d == 0:
        ts, n_days_real, n_days_forecast = instance.predict_arma_next_days(item)
    else:
        ts, n_days_real, n_days_forecast = instance.predict_arima_next_days(item)
    
    ts_real = ts
    ts_forecast = ts
  
    # n_days_real : output format is float
    # n_days_forecast: output format is a list
    ts_real = ts_real.append(pd.DataFrame([n_days_real])) # ts + n days real
    ts_forecast = ts_forecast.append(pd.DataFrame(n_days_forecast)) #ts + n days fc
    
    ts_real = map( float, ts_real[0].tolist() ) # n days real
    ts_forecast = map( float, ts_forecast[0].tolist() ) # n days fc 

    return ts_real, ts_forecast, n_days_real, n_days_forecast[0]


# -----------------------------------------------------------
print 'Started...'
# Configuration ARMA/ARIMA, KDJ and Output target price
# NOTICE:
#    d=1 , automatically use ARIMA
#    d=0 , automatically use ARMA
p, d, q = 2, 1, 0
n_days = 1 # How many 1 days prediction/ DONOT CHANGE
kdj_short_period = 3
kdj_long_period = 5

# Start Produce Weight for GA
print 'Produce trX and trY for GA.'

trX, trY = [], []
start = 27
stop = 1258
fc = 0
an = BuildModel(p, d, q, n_days, fc, kdj_short_period, kdj_long_period)
for fc in range(start, stop): # fc = forecast n days
    an.fc = fc
    ts_real_op, ts_fc_op, n_days_real_op, n_days_fc_op = item_prediction(an, d, 'Open')
    ts_real_hi, ts_fc_hi, n_days_real_hi, n_days_fc_hi = item_prediction(an, d, 'High')
    ts_real_lo, ts_fc_lo, n_days_real_lo, n_days_fc_lo = item_prediction(an, d, 'Low')
    ts_real_cl, ts_fc_cl, n_days_real_cl, n_days_fc_cl = item_prediction(an, d, 'Close')
    ts_real_vo, ts_fc_vo, n_days_real_vo, n_days_fc_vo = item_prediction(an, d, 'Volume')
    

    X, y = an.kdj_calc(ts_real_op, ts_real_hi, ts_real_lo, ts_real_cl, ts_real_vo)
    # get the last row of X and then extract the all items.
    # print "X.shape = ",len(X)
    op, hi, lo, cl, vo, K, D, J = X[-1]

    # get the 2nd last row of X then extract all the items.
    op2, hi2, lo2, cl2, vo2, K2, D2, J2 = X[-2]
    
    # delta close
    d_close = cl-cl2
    
    # delta close power
    d2_close = np.power(d_close, 2)
    
    trX.append([ op, hi, lo, cl, vo, K, D, J, d_close, d2_close ])

    
    
print "trX = ",trX[-1]

print "trX befor changing = ",(len(trX),len(trX[0]))
#the ninth number d2_close in trX is wrong,correct it
length = len(trX)

for i in xrange(length):
    if i == 0:
        trX[i][9] = trX[1][8] - trX[0][8]
    else:
        trX[i][9] = trX[i][8] - trX[i - 1][8]
print "trX after changing = ",(len(trX),len(trX[0]))

# min_close,max_close = min(trX[:][8]),max(trX[:][8])
# min_d1,max_d1 = min(trX[:][8])
# trY length    is    trX length-1
# The last row does not have value i+1
for i in range(0, len(trX)-1):
    op, hi, lo, cl, vo, K, D, J, d_close, d2_close = trX[i]
    op2, hi2, lo2, cl2, vo2, K2, D2, J2, d_close2, d2_close2 = trX[i+1]
    trY.append([ cl2 ])

print 'Length trX: %s' % len(trX)
print 'Length trY: %s' % len(trY)
# Delete trX last index
del trX[-1]

import os
outdir = './finalfile/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

x_train_real = np.asarray(trX)
y_train_real = np.asarray(trY)

np.save(outdir + 'x_train_real',np.asarray(trX))
np.save(outdir + 'y_train_real',np.asarray(trY))

X_train = np.asarray(trX)
Xmax = np.amax(X_train, axis=0)
Xmin = np.amin(X_train, axis=0)
X_train = (X_train - Xmin) / (Xmax - Xmin)

y_train = np.asarray(trY)
ymax = np.amax(y_train, axis=0)
ymin = np.amin(y_train, axis=0)
y_train = (y_train - ymin) / (ymax - ymin)

print "X_train.shape = ",X_train.shape
print "y_train.shape = ",y_train.shape

x_train_normalized = X_train
y_train_normalized = y_train

np.save(outdir + 'x_train_normalized',X_train)
np.save(outdir + 'y_train_normalized',y_train)

import pickle

f = open(outdir + 'ymax.txt','wb')
parameters = {'ymax':ymax,"ymin":ymin}
pickle.dump(parameters,f)


# ===================================================================
# =====================================================================

import numpy as np
import tensorflow as tf
from itertools import izip

outdir = './finalfile/'
x_sample = np.float32(np.load(outdir + 'x_train_normalized.npy'))
y_sample = np.float32(np.load(outdir + 'y_train_normalized.npy'))
print "y_sample.shape = ",y_sample.shape
print "x_sample.shape = ",x_sample.shape
# print x_sample[0:10,:]
# print "--------------------------------------"
# print y_sample[0:10,]

train_start = 600
train_end = 1150
y_train = y_sample[train_start:train_end,:]
x_train = x_sample[train_start:train_end,:]
sample_number = x_train.shape[0]

test_start = 1000
test_end = 1200
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]


nn_input = 10
layer_one = 100
layer_two = 50
nn_output = 1
learn_rate = 0.001

erro_rate = 0.0128
# 初始化权重
def init_weight(shape,name = None):
    return tf.Variable(tf.random_normal(shape, stddev=learn_rate),name = name)

def init_bias(shape,name = None):
    init = tf.zeros(shape)
    return tf.Variable(init, name=name)


def model(X, W, B):
    m = tf.matmul(X, W) + B
    # RELU for instead sigmoid, Sigmoid only for Final
    L = tf.nn.tanh(m)
    return L

# 定义占位符+ 初始化变量
X = tf.placeholder("float", [None, nn_input])
Y = tf.placeholder("float", [None, nn_output])


W1 = init_weight([nn_input, layer_one], 'W1')
B1 = init_bias([layer_one], 'B1')

W2 = init_weight([layer_one, layer_two], 'W2')
B2 = init_bias([layer_two], 'B2')


W3 = init_weight([layer_two, nn_output], 'W3')
B3 = init_bias([nn_output], 'B3')
# -------------------------------------
L2 = model(X,  W1, B1)
L3 = model(L2, W2, B2)

y_out = tf.nn.relu(tf.matmul(L3, W3) + B3)


cost = tf.reduce_mean(tf.square((Y - y_out)))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out, Y)) 
# train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
pridict_op = tf.nn.relu(tf.matmul(L3, W3) + B3)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print "running ..."
sess.run(train_op, feed_dict={X: x_train, Y: y_train})
y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
erro_pridict = y_train - y_pridict
old_erro = np.abs(erro_pridict).mean()


run_times = 0
while(old_erro > erro_rate):
    try:
        sess.run(train_op, feed_dict={X:x_train, Y:y_train})
    except Exception,ex:
        print "[WARMING]exception happens when run train model"
        continue
    y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
    erro_pridict = y_train - y_pridict
    old_erro = np.abs(erro_pridict).mean()

    if run_times % 300 == 0:
        print "old_erro = ",old_erro
    run_times = run_times + 1
    print "run_times = %d ,old_erro = %f" % (run_times,old_erro)

print "I have trianed %d times !!!!" % (run_times)

# save train result
import os 
outdir = './finalfile/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
np.save(outdir + 'x_train',x_train)
np.save(outdir + 'y_train',y_train)
np.save(outdir + 'y_train_predict',y_pridict)

# save model
save_path = outdir + '/Arimann4Model.ckpt'
saver = tf.train.Saver()
saver.save(sess,save_path)


# save test result
y_test_pridict = sess.run(pridict_op,feed_dict = {X:x_test})
np.save(outdir + 'x_test',x_test)
np.save(outdir + 'y_test',y_test)
np.save(outdir + 'y_test_pridict',y_test_pridict)

# restore nn model
outdir = './finalfile/'
save_path = outdir + '/Arimann4Model.ckpt'
saver = tf.train.Saver()
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
saver.restore(sess,save_path)
y_test_pridict = sess.run(pridict_op,feed_dict = {X:x_test})

print "session restore ok"



# show result
# ============================================================================
# ===============================================================================
import os 
outdir = './finalfile/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

x_real = np.load(outdir + 'x_train_real.npy')
y_real = np.load(outdir + 'y_train_real.npy')

ymax = np.amax(y_real, axis=0)
ymin = np.amin(y_real, axis=0)
print "ymax = %f,ymin = %f " % (ymax,ymin)

x_normalized = np.load(outdir+ 'x_train_normalized.npy')
y_normalized = np.load(outdir + 'y_train_normalized.npy')

train_start = 600
train_end = 1150

test_start = 1000
test_end = 1200

y_train_real = y_real[train_start:train_end,:]
y_test_real = y_real[test_start:test_end,:]


# y_train = (y_train - ymin) / (ymax - ymin)
y_train_predict = np.load(outdir + 'y_train_predict.npy')
y_train_denormalized = y_train_predict*(ymax - ymin) + ymin

y_test_predict = np.load(outdir + 'y_test_predict.npy')
y_test_denormalized = y_test_predict*(ymax - ymin) + ymin

from matplotlib import pyplot as plt
outdir = './image/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

# ---------------------------plot train dataset result---------------------------
# plt.plot(y_train_real,'ro')
# plt.plot(y_train_predict,'bo')
plt.plot(y_train_real,'r-')
plt.plot(y_train_denormalized,'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('arima-nn denormalized training predict')
plt.legend(['y_train_real','y_train_denormalized'])
plt.grid(True)
plt.savefig(outdir + 'y_train_denormalized.jpg')
plt.show()


# ---------------------------plot test dataset result---------------------------
plt.plot(y_test_real,'ro')
plt.plot(y_test_denormalized,'bo')
plt.plot(y_test_real,'r-')
plt.plot(y_test_denormalized,'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('arima-nn denormalized test predict')
plt.legend(['y_test_real','y_test_denormalized'])
plt.grid(True)
plt.savefig(outdir + 'y_test_denormalized.jpg')
plt.show()

# ---------------------------plot train dataset acurracy---------------------------
train_acurracy = y_train_denormalized - y_train_real
shape = train_acurracy.shape
for i in xrange(shape[0]):
    train_acurracy[i,0] = (y_train_denormalized[i,0] - y_train_real[i,0])/y_train_real[i,0]
plt.plot(train_acurracy,'b-')
plt.xlabel('index')
plt.ylabel('train_acurracy')
plt.title('arima-nn train predict acurracy ')
plt.legend(['train_acurracy',])
plt.grid(True)
plt.savefig(outdir + 'train_acurracy.jpg')
plt.show()
print "mean train acurracy = ",np.mean(train_acurracy)


# ---------------------------plot train dataset acurracy---------------------------
test_acrracy = y_test_denormalized - y_test_real
shape = test_acrracy.shape
for i in xrange(shape[0]):
    test_acrracy[i,0] = (y_test_denormalized[i,0] - y_test_real[i,0])/y_test_real[i,0]

plt.plot(test_acrracy,'b-')
plt.xlabel('index')
plt.ylabel('test_acrracy')
plt.title('arima-nn test predict acurracy ')
plt.legend(['test_acrracy',])
plt.grid(True)
plt.savefig(outdir + 'test_acurracy.jpg')
plt.show()

print "mean train acurracy = ",np.mean(test_acrracy)