
# coding: utf-8

# In[4]:


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
import seaborn as sns
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

data = pd.read_csv(filepath_or_buffer="../DL/PM10_LSTM.csv", index_col="date")

start_time = time.time()


data['PM10'].replace(0, np.nan, inplace=True)
data['PM10'].fillna(method='ffill', inplace=True)


from sklearn.preprocessing import MinMaxScaler
values = data['PM10'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
train_size = int(len(scaled) * 0.983)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    #print(len(dataY))
    return np.array(dataX), np.array(dataY)
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1, input_dim=4, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=300, batch_size=24, validation_data=(testX, testY), verbose=0, shuffle=False)

yhat = model.predict(testX)

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
print('Test Predict RMSE: %.3f' % rmse)

print("--- %s seconds ---" % (time.time() - start_time))


predictDates = data.tail(len(testX)).index
testY_reshape = testY_inverse.reshape(len(testY_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))
actual_chart = go.Scatter(x=predictDates, y=testY_reshape, name= 'Actual PM10')
predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name= 'Predict PM10')

start_time = time.time()


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


# In[20]:


values = data[['PM10']].values
values = values.astype('float32')


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


reframed = series_to_supervised(scaled, 1, 1)


values = reframed.values
n_train_hours = int(len(values) * 0.983)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


multi_model = Sequential()
multi_model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_model.add(Dense(1))
multi_model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
multi_history = multi_model.fit(train_X, train_y, epochs=300, batch_size=24, validation_data=(test_X, test_y), verbose=0, shuffle=False)


# In[28]:


yhat = multi_model.predict(test_X)
scores = multi_model.evaluate(test_X, test_y)


test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test Multi Predict RMSE: %.3f' % rmse)

print("--- %s seconds ---" % (time.time() - start_time))

actual_chart = go.Scatter(x=predictDates, y=inv_y, name= 'Actual Price')
multi_predict_chart = go.Scatter(x=predictDates, y=inv_yhat, name= 'Multi Predict Price')
predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name= 'Predict Price')


def determine_level(pred, actual):
    true_val = 0
    false_val = 0
    
    for pm_pred, pm_act in zip(pred, actual):
        level = 0
        levelA = 0
        if pm_pred >= 16 and pm_pred <=35:
            level = 1
        elif pm_pred >= 36 and pm_pred <=75:
            level = 2
        else:
            level = 3
        
        if pm_act >= 16 and pm_act <=35:
            levelA = 1
        elif pm_act >= 36 and pm_act <=75:
            levelA = 2
        else:
            levelA = 3
        if level == levelA:
            true_val+=1
        else:
            false_val+=1
    acc = float(true_val)/float(true_val+false_val)
    return acc

print ("Multi-predict accuracy: %.2f" % determine_level(inv_y, inv_yhat))
print ("Predict accuracy: %.2f" % determine_level(inv_y, yhat_reshape))

