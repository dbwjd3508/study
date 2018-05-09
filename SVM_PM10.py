
# coding: utf-8

# In[1]:


import pandas as pd
from math import sqrt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import plotly.offline as py
import plotly.graph_objs as go
import time


data = pd.read_csv(filepath_or_buffer="../DL/PM10_SVM.csv", index_col="date")

start_time = time.time()


x = data.iloc[:, :-1]
y = data.iloc[:,-1]

scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.017, random_state = 0)


svc = SVC(kernel='rbf', C=100.0, gamma=0.1, random_state=0)

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)


predictDates = data.tail(len(x_test)).index
testY_reshape = y_test.reshape(len(y_test))
yhat_reshape = y_pred.reshape(len(y_pred))
actual_chart = go.Scatter(x=predictDates, y=testY_reshape, name= 'Actual PM10')
predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name= 'Predict PM10')

rmse = sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE: %.3f' % rmse)

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
    #print ("%d" % (true_val+false_val))
    acc = float(true_val)/float(true_val+false_val)
    return acc

print ("SVM accuracy: %.3f" % determine_level(yhat_reshape, testY_reshape))
print("--- %s seconds ---" % (time.time() - start_time))
