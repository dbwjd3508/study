
# coding: utf-8

# In[30]:


import pandas as pd
import plotly.offline as py
from matplotlib import pyplot
from matplotlib.pylab import rcParams

#py.init_notebook_mode(connected=True)
#%matplotlib inline

rcParams['figure.figsize'] = (15, 6)
rcParams['axes.grid']=True

data = pd.read_csv(filepath_or_buffer="../DL/pm10_col.csv", index_col="date")
data['PM10'] = data.PM10.astype(float)
data.info()
df = data.ix['2015-01-01 0:00':'2017-12-31 23:00',["PM10", "temp"]]
df.plot()


# In[31]:


from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

m1 = OLS(df.PM10, add_constant(df.temp))
r1 = m1.fit()
print(r1.summary())


# In[32]:


r1.resid.plot()


# In[33]:


import statsmodels.api as sm

sm.tsa.graphics.plot_acf(r1.resid)


# In[34]:


temp = sm.tsa.ARMA(df.PM10, (1,1), exog=df.temp)
r2 = temp.fit()
print(r2.summary())


# In[35]:


r2.resid.plot()


# In[36]:


sm.tsa.graphics.plot_acf(r2.resid);

