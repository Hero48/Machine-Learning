import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import talib
import streamlit as st
import matplotlib.pyplot as plt

symbol = 'AAPL'
interval = '5m'
LOOKBACK = 14

# Define the technical indicators to use
INDICATORS = ['SMA', 'EMA', 'MACD', 'RSI', 'ADX', 'ATR', 'OBV', 'CCI', 'ROC']

# Load historical price and indicator data from yfinance
data = yf.download(symbol, interval=interval, period='5d')
data = data.dropna()
data = data.reset_index()

# Calculate the technical indicators and append them to the dataframe
for indicator in INDICATORS:
    if indicator == 'SMA':
        data[indicator] = talib.SMA(data['Close'], timeperiod=20)
    elif indicator == 'EMA':
        data[indicator] = talib.EMA(data['Close'], timeperiod=20)
    elif indicator == 'MACD':
        macd, signal, hist = talib.MACD(data['Close'], fastperiod=15, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_signal'] = signal
        data['MACD_hist'] = hist
    elif indicator == 'RSI':
        data[indicator] = talib.RSI(data['Close'], timeperiod=2)
    elif indicator == 'ADX':
        data[indicator] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    elif indicator == 'ATR':
        data[indicator] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    elif indicator == 'OBV':
        data[indicator] = talib.OBV(data['Close'], data['Volume'])
    elif indicator == 'CCI':
        data[indicator] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
    elif indicator == 'ROC':
        data[indicator] = talib.ROC(data['Close'], timeperiod=14)

# Drop rows with missing data
data = data.dropna()

# Define the target variable as the direction of the next candle
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Define the features as the current price and the past 60 candles' price and indicator data
features = ['Close']
for i in range(1, LOOKBACK+1):
    for indicator in INDICATORS:
        features.append(f'{indicator}_{i}')
        data[f'{indicator}_{i}'] = data[indicator].shift(i)
    features.append('Close_{}'.format(i))
    data[f'Close_{i}'] = data['Close'].shift(i)

# Split the data into training and testing sets
train_size = int(len(data) * 0.5)
train_data = data[:train_size]
test_data = data[train_size:]

# Define the XGBoost parameters
params = {'max_depth': st.sidebar.slider('max_depth', 1, 10, 5),
          'learning_rate': st.sidebar.slider('learning_rate', 0.01, 1.0, 0.1),
          'n_estimators': st.sidebar.slider('n_estimators', 10, 500, 100),
          'objective': 'binary:logistic',
          'eval_metric': 'error'}

# Train the XGBoost model
X_train = train_data[features]
y_train = train_data['Target']
X_test = test_data[features]
y_test = test_data['Target']

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

model = xgb.train(params, dtrain)

# Make predictions on the test set
test_data['Predictions'] = model.predict(dtest)

# Calculate the accuracy of the model
test_data['Predictions'] = np.where(test_data['Predictions'] > 0.5, 1, 0)
accuracy = (test_data['Predictions'] == test_data['Target']).mean()

# Display the accuracy of the model
st.write(f'Test accuracy: {accuracy:.2%}')

# Display the historical price and indicator data
st.subheader('Historical Data')
st.write(data)

# Display the feature importance
st.subheader('Feature Importance')
fig, ax = plt.subplots(figsize=(10, 10))
xgb.plot_importance(model, max_num_features=15, height=0.8, ax=ax)
st.pyplot(fig)

