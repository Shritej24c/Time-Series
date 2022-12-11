#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install --upgrade pip')


# In[178]:





# In[2]:


get_ipython().system('pip install matplotlib')


# In[5]:


get_ipython().system('pip install statsmodels')


# In[52]:


get_ipython().system('pip install scikit-learn')


# In[66]:


get_ipython().system('pip install scipy')


# In[81]:


#import necessary libraries
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from patsy import dmatrices
from pandas import Series
from matplotlib import pyplot
from datetime import datetime
from matplotlib.pyplot import figure

import statsmodels.api as sm

figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

import sys
import warnings
import itertools
warnings.filterwarnings("ignore")
from scipy.signal import periodogram, welch

import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from statsmodels.tsa.stattools import adfuller


from statsmodels.tsa.arima_model import ARIMA


# In[17]:


custom_date_parser = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

df = pd.read_csv("AirPassengers.csv")


# In[31]:


airpax_data = df.copy()
airpax_data['Month'] = pd.to_datetime(df['Month'],infer_datetime_format=True) #convert from string to datetime
airpax_data = airpax_data.set_index(['Month'])
df['Month'] = pd.to_datetime(df['Month'])


# In[32]:


df.shape


# In[33]:


plt.xlabel('Month')
plt.ylabel('Number of air passengers')
plt.plot(airpax_data)


# In[ ]:


ts = df['#Passengers']


# In[117]:


ts.describe()


# In[57]:


#Test whether Timeseries is Stationary or not
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[58]:


test_stationarity(ts)


# In[61]:


ts_df1 = ts.diff(periods = 1)
ts_df1.dropna(inplace = True)
test_stationarity(ts_df1)


# In[63]:


ts_log = np.log10(ts)
ts_logdf1 = ts_log.diff(periods = 1)
ts_logdf1.dropna(inplace = True)
test_stationarity(ts_logdf1)


# In[104]:


ts_log_df12 = ts_logdf1.diff(periods = 12)
ts_log_df12.dropna(inplace =True )
test_stationarity(ts_log_df12)


# In[177]:


(S, f) = plt.psd(ts, Fs =100)

plt.semilogy(f, S)
plt.xlim([0, 100])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


# In[105]:


(f, prdgm) = periodogram(ts_log_df12)
plt.semilogy(f, prdgm)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


# In[106]:


(f, S) = welch(ts_log_df12)
plt.semilogy(f, S)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


# In[64]:


train = ts[:60]
test = ts[61:84]

trainlog = np.log10(train)


# In[ ]:





# In[86]:


airpax_decompose = sm.tsa.seasonal_decompose(ts, model="multiplicative", period=12)
airpax_decompose.plot()
plt.show()


# In[110]:


airpax_decompose = sm.tsa.seasonal_decompose(ts_log_df12, period = 12)
airpax_decompose.plot()
plt.show()


# In[111]:


trend = airpax_decompose.trend
seasonal = airpax_decompose.seasonal
residual = airpax_decompose.resid


# In[ ]:





# In[100]:


smt.graphics.plot_acf(ts_log_df1, lags=50, ax=axes[0])


# In[102]:


smt.graphics.plot_pacf(ts_logdf1, lags=50, ax=axes[1])


# In[90]:


fig, axes = plt.subplots(1, 2)
fig.set_figwidth(12)
fig.set_figheight(4)
smt.graphics.plot_acf(ts_logdf1, lags=50, ax=axes[0])
smt.graphics.plot_pacf(ts_logdf1, lags=50, ax=axes[1])
plt.tight_layout()


# In[115]:


smt.graphics.plot_acf(ts_log_df12, lags=50, ax=axes[0])


# In[116]:


smt.graphics.plot_pacf(ts_log_df12, lags=50, ax=axes[0])


# In[119]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX 

model = SARIMAX(ts, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts, label='original')
plt.plot(results_ARIMA.fittedvalues, color='red', label='fitted')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts)**2))
plt.legend()


# In[132]:


model1 = SARIMAX(log(, order=(0,1,1), seasonal_order=(0,1,1,12))
results_ARIMA1 = model1.fit(disp=-1)  
plt.plot(ts_log_df12, label='original')
plt.plot(results_ARIMA1.fittedvalues, color='red', label='fitted')
plt.title('RSS: %.4f'% sum((results_ARIMA1.fittedvalues-ts)**2))
plt.legend()


# In[153]:


mod = sm.tsa.statespace.SARIMAX(np.log10(train),
                                order=(0,1,1),
                                seasonal_order=(0,1,1,12),
                                enforce_stationarity=True)

best_results = mod.fit()

print(best_results.summary().tables[1])


# In[154]:


mod = sm.tsa.statespace.SARIMAX(np.log10(train),
                                order=(0,1,1),
                                seasonal_order=(1,0,1,12),
                                enforce_stationarity=True)

best_results101 = mod.fit()

print(best_results101.summary().tables[1])


# In[156]:


pred99 = best_results.get_forecast(steps=24, alpha=0.1)
pred99_101 = best_results101.get_forecast(steps=24, alpha=0.1)


# In[157]:


df_test =pd.DataFrame(test)
df_test['Passengers_Forecast_11'] = np.power(10, pred99.predicted_mean)
df_test['Passengers_Forecast_101'] = np.power(10, pred99_101.predicted_mean)


# In[159]:


df_test['ape11'] = abs(df_test['Passengers_Forecast_11'] - df_test['#Passengers'])/df_test['#Passengers']
df_test['ape101'] = abs(df_test['Passengers_Forecast_101'] - df_test['#Passengers'])/df_test['#Passengers']


# In[160]:


df_test


# In[161]:


print(df_test['ape11'].mean())
print(df_test['ape101'].mean())


# In[162]:


best_results.plot_diagnostics(lags=30, figsize=(16,12))
plt.show()


# In[ ]:





# In[163]:


best_results101.plot_diagnostics(lags=30, figsize=(16,12))
plt.show()

