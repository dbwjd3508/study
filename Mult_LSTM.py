
# coding: utf-8

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
#import seaborn as sns
#py.init_notebook_mode(connected=True)
#get_ipython().magic(u'matplotlib inline')


# In[3]:


data = pd.read_csv(filepath_or_buffer="../DL/pm10_col.csv", index_col="date")
data.info()


# In[4]:


data.head()


# In[6]:


#btc_trace = go.Scatter(x=data.index, y=data['PM10'], name= 'PM10')
#py.iplot([btc_trace])


# In[7]:


data['PM10'].replace(0, np.nan, inplace=True)
data['PM10'].fillna(method='ffill', inplace=True)


# In[8]:


from sklearn.preprocessing import MinMaxScaler
values = data['PM10'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# In[9]:


train_size = int(len(scaled) * 0.7)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
#print(len(train), len(test))


# In[10]:


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)


# In[11]:


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[12]:


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[15]:


model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=10, batch_size=120, validation_data=(testX, testY), verbose=0, shuffle=False)


# In[16]:


#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()


# In[17]:


yhat = model.predict(testX)
#pyplot.plot(yhat, label='predict')
#pyplot.plot(testY, label='true')
#pyplot.legend()
#pyplot.show()


# In[18]:


yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))


# In[19]:


rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
print('Test RMSE: %.3f' % rmse)


# In[20]:


#pyplot.plot(yhat_inverse, label='predict')
#pyplot.plot(testY_inverse, label='actual', alpha=0.5)
#pyplot.legend()
#pyplot.show()


# In[21]:


predictDates = data.tail(len(testX)).index


# In[22]:


testY_reshape = testY_inverse.reshape(len(testY_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))


# In[23]:


#actual_chart = go.Scatter(x=predictDates, y=testY_reshape, name= 'Actual Price')
#predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name= 'Predict Price')
#py.iplot([predict_chart, actual_chart])


# In[24]:


#sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)


# In[25]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[28]:


values = data[['PM10'] + ['pm10(1hour)'] + ['pm10(2hours)']].values
values = values.astype('float32')


# In[29]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# In[30]:


reframed = series_to_supervised(scaled, 1, 1)
#reframed.head()


# In[31]:


reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
#print(reframed.head())


# In[32]:


values = reframed.values
n_train_hours = int(len(values) * 0.7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[34]:


multi_model = Sequential()
multi_model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_model.add(Dense(1))
multi_model.compile(loss='mae', optimizer='adam')
multi_history = multi_model.fit(train_X, train_y, epochs=10, batch_size=120, validation_data=(test_X, test_y), verbose=0, shuffle=False)


# In[35]:


#pyplot.plot(multi_history.history['loss'], label='multi_train')
#pyplot.plot(multi_history.history['val_loss'], label='multi_test')
#pyplot.legend()
#pyplot.show()


# In[36]:


yhat = multi_model.predict(test_X)
#pyplot.plot(yhat, label='predict')
#pyplot.plot(test_y, label='true')
#pyplot.legend()
#pyplot.show()


# In[37]:


test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


# In[38]:


rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[39]:


#actual_chart = go.Scatter(x=predictDates, y=inv_y, name= 'Actual Price')
#multi_predict_chart = go.Scatter(x=predictDates, y=inv_yhat, name= 'Multi Predict Price')
#predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name= 'Predict Price')
#py.iplot([predict_chart, multi_predict_chart, actual_chart])

