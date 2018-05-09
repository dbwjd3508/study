
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import plotly.offline as py
import plotly.graph_objs as go
from matplotlib.pylab import rcParams
import time as tm
#py.init_notebook_mode(connected=True)
#get_ipython().magic(u'matplotlib inline')
rcParams['figure.figsize'] = (15, 6)
rcParams['axes.grid']=True

data = pd.read_csv(filepath_or_buffer="../DL/PM10_LSTM.csv", parse_dates=["date"], index_col="date")

start_time = tm.time()


from pandas.plotting import lag_plot
lag_plot(data)


def test_stationary(timeseries):

#rolmean = pd.rolling_mean(timeseries, window=24)
#rolstd = pd.rolling_std(timeseries, window=24)

#orig = plt.plot(timeseries, color='blue', label='Original')
#mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#std = plt.plot(rolstd, color='black', label='Rolling Std')
 #plt.legend(loc='best')
 #plt.title('Rolling Mean & Standard Deviation')
 #plt.show(block=False)

 print 'Results of Dickey-Fuller Test: '
 timeseries = timeseries.iloc[:,0].values
 dftest = adfuller(timeseries, autolag='AIC')
 dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lag Used', 'Number of Observations Used'])
 for key,value in dftest[4].items():
  dfoutput['Critical Value (%s)'%key] = value
 print dfoutput

test_stationary(data)


# In[5]:


from statsmodels.tsa.stattools import acf, pacf

ts_log = np.log(data)
moving_avg = pd.rolling_mean(ts_log,24)
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
expwighted_avg = pd.ewma(ts_log, halflife=24)
ts_log_ewma_diff = ts_log - expwighted_avg
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)

"""
lag_acf = acf(ts_log_diff, nlags=24)
lag_pacf = pacf(ts_log_diff, nlags=24, method='ols')

# Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()
#data2 = data.astype('float64')
#print data2.info()
"""

# In[13]:


from statsmodels.tsa.arima_model import ARIMA

#AR Model
model = ARIMA(ts_log, order=(1, 1, 1))
results_AR = model.fit(disp=-1)
#plt.plot(ts_log_diff)
#plt.plot(results_AR.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
#plt.show()


# In[14]:


predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
#print predictions_ARIMA_diff.head()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#print predictions_ARIMA_diff_cumsum.head()


# In[15]:


predictions_ARIMA_log = pd.Series(data['PM10'], index=data.index)
#print predictions_ARIMA_log.head()
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#print predictions_ARIMA_log.head()


# In[16]:


predictDates = data.tail(len(data)).index
actual_chart = go.Scatter(x=predictDates, y=data['PM10'], name= 'Actual PM10')
predict_chart = go.Scatter(x=predictDates, y=predictions_ARIMA_log, name= 'Predict PM10')
#py.iplot([actual_chart, predict_chart])


# In[17]:


error = sqrt(mean_squared_error(data['PM10'], predictions_ARIMA_log))
print('Test ARIMA RMSE: %.3f' % error)


# In[18]:


def determine_level(pred, actual):
    true_val = 0
    false_val = 0
    
    for pm_pred, pm_act in zip(pred, actual):
        level = 0
        levelA = 0
        if pm_pred >= 16.0 and pm_pred <=35.0:
            level = 1
        elif pm_pred >= 36.0 and pm_pred <=75.0:
            level = 2
        else:
            level = 3
        
        if pm_act >= 16.0 and pm_act <=35.0:
            levelA = 1
        elif pm_act >= 36.0 and pm_act <=75.0:
            levelA = 2
        else:
            levelA = 3
        if level == levelA:
            true_val+=1
        else:
            false_val+=1
    #print ("%d" % (true_val+false_val))
    acc = float(true_val)/float(true_val+false_val)
    return acc

print ("ARIMA accuracy: %.3f" % determine_level(predictions_ARIMA_log, data['PM10']))

print("--- %s seconds ---" % (tm.time() - start_time))

