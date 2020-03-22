## Import Libraries
import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#%matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plt.rcParams['figure.figsize'] = [15, 5]
from IPython import display
from ipywidgets import interact, widgets

import datetime
from datetime import timedelta

from typing import List
import os

## Read Data for Cases, Deaths and Recoveries
ConfirmedCases_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
Deaths_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
Recoveries_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')

def cleandata(df_raw):
    df_cleaned=df_raw.melt(id_vars=['Province/State','Country/Region','Lat','Long'],value_name='Cases',var_name='Date')
    df_cleaned=df_cleaned.set_index(['Country/Region','Province/State','Date'])
    return df_cleaned 

# Clean all datasets
ConfirmedCases=cleandata(ConfirmedCases_raw)
Deaths=cleandata(Deaths_raw)
Recoveries=cleandata(Recoveries_raw)

# Log function
dataset = "Confirmed"
url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-{dataset}.csv"
df = pd.read_csv(url, error_bad_lines=False)
dfPL = df.loc[df["Country/Region"]=="Poland"]
dfPL2 = dfPL.drop(columns=["Province/State","Country/Region","Lat","Long"])
PL_list = dfPL2.values.tolist()
flat_list = [item for sublist in PL_list for item in sublist]
readyPL = list(filter(lambda a: a != 0, flat_list))

x_data = np.arange(start=1, stop=(len(readyPL)+1), step=1)
y_data = np.array(readyPL)
log_x_data = np.log(x_data)
log_y_data = np.log(y_data)

curve_fit = np.polyfit(x_data, log_y_data, 1)
y = np.exp(0.01076242) * np.exp(0.40488364*x_data)
plt.plot(x_data, y_data, "o")
plt.plot(x_data, y)

curve_fit2 = np.polyfit(x_data, np.log(y_data), 1, w=np.sqrt(y_data))
y = np.exp(0.9091747) * np.exp(0.32353551*x_data)
plt.plot(x_data, y_data, "o")
plt.plot(x_data, y)

curve_fit4 = sp.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x_data,  y_data,  p0=(4, 0.1))
y = 3.63824612 * np.exp(0.29429372*x_data)
plt.plot(x_data, y_data, "o")
plt.plot(x_data, y)

y = 6.58828455 * np.exp(0.24566981*x_data)
plt.plot(x_data, y_data, "o")
plt.plot(x_data, y)
