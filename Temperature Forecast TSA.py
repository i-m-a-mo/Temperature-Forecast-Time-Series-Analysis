#%%# 0) Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score

from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from sktime.utils.plotting import plot_series

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14,6)

#%%# 1) Define the Business Goal

''' Build a model that can predict tomorrow's temperature as precisely as possible.'''

#%%# 2) Get the Data

### 2.1) Load the Data

df = pd.read_csv('./_data/TG_STAID000186.csv', index_col=1, parse_dates=True)



df.describe()

df.value_counts([df['TG']==-9999])/len(df)

df['TG'] = df['TG'].replace(-9999,np.NaN)

df.describe()

# extract year
df['year'] = df.index.year   
#df['month'] = df.index.month
#df['day'] = df.index.day
#df.head()

sns.heatmap(df.isna(), cbar=False);

#sns.heatmap(df[df['year']== 1945].isna(), cbar=False);

### 2.2) Clean the Data

df['TG'] = df['TG']/10

# only take year 1957+ due to missings in the periods before
df_clean = df[df['year']>=1957][['TG']]
df_clean

#%%# 3) Train-Test-Split

df_train = df_clean[:-365]

df_test = df_clean[-365:]

#%%# 4) Visualize the Data

df_train.groupby(df_train.index.year)['TG'].mean().plot()

df_train.groupby(df_train.index.month)['TG'].mean().plot()

#%%# 5) Feature Engineer

#%%# 6) Train a model

### Model Trend

df_train['timestep'] = range(len(df_train))

X = df_train[['timestep']]
y = df_train['TG']

m = LinearRegression()
m.fit(X, y)
m.coef_, m.intercept_

df_train['trend'] = m.predict(X)

df_train.resample('1M').mean()[['TG', 'trend']].plot();
df_train[['trend']].plot();

## Seasonality

df_train.groupby(df_train.index.month)['TG'].mean().plot();

month_dummies = pd.get_dummies(df_train.index.month, prefix='month').set_index(df_train.index)

#day_dummies = pd.get_dummies(df_train.index.dayofyear, prefix='day').set_index(df_train.index)

df_train2 = df_train.join(month_dummies)
#df_train

X = df_train2.drop(['TG','trend'], axis=1)

m2 = LinearRegression()
m2.fit(X, y)

df_train2['trend_seasonal'] = m2.predict(X)

df_train2[['TG', 'trend_seasonal']].plot(xlim=('2019','2020'));



#month_dummies = pd.get_dummies(df_train.index.month, prefix='month').set_index(df_train.index)

day_dummies = pd.get_dummies(df_train.index.dayofyear, prefix='day').set_index(df_train.index)
df_train = df_train.join(day_dummies)
#df_train

X = df_train.drop(['TG','trend'], axis=1)

m2 = LinearRegression()
m2.fit(X, y)

df_train['trend_seasonal'] = m2.predict(X)

df_train[['TG', 'trend_seasonal']].plot(xlim=('2019','2020'));

#df_train[df_train.index.year >2019][['TG', 'trend_seasonal']].plot();
df_train[['TG', 'trend_seasonal']].plot();

df_train['remainder'] = df_train['TG'] - df_train['trend_seasonal']

#df_train[df_train.index.year >2019]['remainder'].plot();
df_train['remainder'].plot();



## statsmodels

#sd = seasonal_decompose(df_train[df_train.index.year >2019]['TG'])
sd = seasonal_decompose(df_train['TG'])

sd.plot();

### Add lag feature — for modeling remainder

df_train['lag1'] = df_train['remainder'].shift(1)

df_train.head(3)

plt.scatter(x='remainder', y='lag1', data=df_train);

df_train[['remainder','lag1']].corr()

plot_pacf(df_train['remainder']);

# in shaded area is 95% CI, show stat independence

plot_acf(df_train['remainder']);



selected_order = ar_select_order(df_train['remainder'], maxlag=20)

selected_order.ar_lags

#selected_order.bic # first row and then the drop



# Define X and y for full model
df_train.dropna(inplace=True)
X_full = df_train.drop(['TG','trend','trend_seasonal','remainder'], axis=1)
y_full = df_train['TG']

m_full = LinearRegression()
m_full.fit(X_full, y_full)

df_train['full_model'] = m_full.predict(X_full)

df_train[['TG', 'trend_seasonal', 'full_model']].plot()

df_train['remainder_full_model'] = df_train['TG'] - df_train['full_model']

#df_train[df_train.index.year >2019][['remainder', 'remainder_full_model']].plot()
df_train[['remainder', 'remainder_full_model']].plot(xlim=('2019','2020'));

df_train.drop('remainder_full_model', axis=1, inplace=True)

#%%# 7) Cross-Validate and Optimize Hyperparameters

# Create a TimeSeriesSplit object
ts_split = TimeSeriesSplit(n_splits=5)

X_full.shape

# See how the folds work: 
for i, (train_index, validation_index) in enumerate(ts_split.split(X_full, y_full)):
    print(f'The training data for the {i+1}th iteration are the observations {train_index[0]} to {train_index[-1]}')
    print(f'The validation data for the {i+1}th iteration are the observations {validation_index[0]} to {validation_index[-1]}')
    print()

time_series_split = ts_split.split(X_full, y_full)

result = cross_val_score(estimator=m_full, X=X_full, y=y_full, cv=time_series_split)

result

round(result.mean(), 3)



#%%# 8) Test

last_train_timestep = df_train['timestep'][-1]

last_train_timestep

df_test['timestep'] = range(last_train_timestep+1, last_train_timestep+1+len(df_test))

df_test

#seasonal_dummies = pd.get_dummies(df_test.index.month, prefix='month').set_index(df_test.index)
seasonal_dummies = pd.get_dummies(df_test.index.dayofyear, prefix='day').set_index(df_test.index)
df_test = df_test.join(seasonal_dummies)

# Assign X_test
X_test = df_test.drop('TG', axis=1)

df_test['day_121'] = 0
X_test['day_121'] = 0

# Predict trend-seasonal component
df_test['trend_seasonal'] = m2.predict(X_test)

df_test[['TG', 'trend_seasonal']].plot() 

# Calculate the remainder
df_test['remainder'] = df_test['TG'] - df_test['trend_seasonal']

# Add the lag1 feature
df_test['lag1'] = df_test['remainder'].shift(1)

# Fill in the NaN in the lag1 column
df_test.loc['2020-05-01', 'lag1'] = df_train.loc['2020-04-30', 'remainder']

# Assign X_full
X_full = df_test.drop(['TG', 'trend_seasonal', 'remainder'], axis=1)

# Create predictions
df_test['full_model'] = m_full.predict(X_full)

df_full = df_train[['TG', 'trend_seasonal', 'full_model']]\
    .append(df_test[['TG', 'trend_seasonal', 'full_model']])

df_full.plot()

round(m_full.score(X_full, df_test['TG']), 3)





#%%# Predict the future

# Combine the datasets
df_combined = df_train.append(df_test)

# Re-train the model on the whole dataset
X_combined = df_combined.drop(columns=['TG','trend', 'trend_seasonal', 'remainder', 'full_model'])
y_combined = df_combined['TG']

m_combined = LinearRegression()
m_combined.fit(X_combined, y_combined)

df_combined

### Components of the future

timestep = df_combined['timestep'].max() + 1
days = [0] * 366
days[120] = 1
lag = df_combined.loc['2021-04-30', 'remainder']

# Create a future data point
X_future1 = []
X_future1.append(timestep)
X_future1.extend(days)
X_future1.append(lag)

X_future1 = pd.DataFrame([X_future1], columns = X_combined.columns)

m_combined.predict(X_future1)

# How does this look like for the next day?
timestep = df_combined['timestep'].max() + 2
days = [0] * 366
days[121] = 1
lag = 0
# This is too far in the future to calculate lag, 
# we don't have remainder for the previous data point (X_future1),
# this is now only modeling trend-seasonal component

X_future2 = pd.DataFrame([[timestep] + days + [lag]], columns = X_combined.columns)

m_combined.predict(X_future2)





## Run autoregression model

plot_pacf(df_train['TG']);

# in shaded area is 95% CI, show stat independence

plot_acf(df_train['TG']);



selected_order = ar_select_order(df_train['TG'], maxlag=20)

selected_order.ar_lags

#selected_order.bic # first row and then the drop



ar_model = AutoReg(endog=df_train['TG'], lags=1).fit()

ar_model.predict().plot()
df_train['TG'].plot(xlim=('2019','2020'))


df_train['prediction_sm'] = ar_model.predict()

ar_model.summary()

y_true = df_train.iloc[1:]['TG']
y_pred = df_train.iloc[1:]['prediction_sm']

r2_score(y_true, y_pred)



ar_model2 = AutoReg(endog=df_train['TG'], lags=16).fit()

ar_model2.predict().plot()
df_train['TG'].plot(xlim=('2019','2020'))


df_train['prediction_sm2'] = ar_model2.predict()

ar_model2.summary()



y_true = df_train.iloc[16:]['TG']
y_pred = df_train.iloc[16:]['prediction_sm2']

r2_score(y_true, y_pred)



ar_model3 = AutoReg(endog=df_train['TG'], lags=3).fit()

ar_model3.predict().plot()
df_train['TG'].plot(xlim=('2019','2020'))


df_train['prediction_sm3'] = ar_model3.predict()

ar_model3.summary()

y_true = df_train.iloc[3:]['TG']
y_pred = df_train.iloc[3:]['prediction_sm3']

r2_score(y_true, y_pred)



#%%# Predict the future II

# Combine the datasets
df_combined = df_train.append(df_test)[['TG','timestep']]

selected_order = ar_select_order(df_combined['TG'], maxlag=20)

selected_order.ar_lags

ar_model_F = AutoReg(endog=df_combined['TG'], lags=16).fit()

df_train['prediction_sm_F'] = ar_model_F.predict()



### Components of the future

timestep = df_combined['timestep'].max() + 1


# Create a future data point
X_future1 = []
X_future1.append(timestep)
#X_future1.extend(days)
#X_future1.append(lag)

X_future1 = pd.DataFrame([X_future1], columns = ['timestep'])