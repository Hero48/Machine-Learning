import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import talib
import streamlit as st



st.title('Forex Predictor')


col1, col2, col3, col4 = st.columns(4)


with col1:
    symbol = st.selectbox('Symbol', ('EURUSD=X', 'AAPL', 'USDJPY=X', 'GBPUSD=X', 'GBPJPY=X'))


with col2:
    # Define the number of candles to look back for training
    LOOKBACK = st.number_input('LOOKBACK', min_value=1, max_value=100, value=5)

with col3:
    interval = st.selectbox('TimeFrame', ('15m', '5m', '30m'))

with col4:
    period = st.number_input('Period', min_value=1, max_value=100, value=5)
    period = f'{period}d'





# symbol = 'AAPL'
# interval = '15m'
# LOOKBACK = 20

# Define the technical indicators to use
INDICATORS = ['SMA', 'EMA', 'MACD', 'RSI', 'ADX', 'ATR', 'OBV', 'CCI', 'ROC',
              'Average Price', 'Weighted Close', 'Median Price', 'Typical Price',
              'Accumulation/Distribution Line', 'Chaikin Oscillator', 'Ease of Movement', 'Money Flow Index', 'Negative Volume Index',
              'Positive Volume Index', 'Volume Price Trend', 'Commodity Channel Index',
              'Momentum', 'Rate of Change', 'Trix', 'Ultimate Oscillator',
              'Williams %R', 'Arnaud Legoux Moving Average', 'Chande Momentum Oscillator',
              'Linear Regression']

# Load historical price and indicator data from yfinance
data = yf.download(symbol, interval='15m', period=period)
data = pd.read_csv('AAPL.csv')
data = data.dropna()
data = data.reset_index()
ma = data['Close'].rolling(window=20).mean()
std = data['Close'].rolling(window=20).std()
upper_band = ma + 2 * std
lower_band = ma - 2 * std
data['Bollinger Bands Upper'] = upper_band
data['Bollinger Bands Lower'] = lower_band
# Calculate the technical indicators and append them to the dataframe

# Apply the indicators to the data
for indicator in INDICATORS:
    if indicator == 'SMA':
        data[indicator] = talib.SMA(data['Close'], timeperiod=20)
    elif indicator == 'EMA':
        data[indicator] = talib.EMA(data['Close'], timeperiod=20)
    elif indicator == 'MACD':
        macd, signal, hist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_signal'] = signal
        data['MACD_hist'] = hist
    elif indicator == 'RSI':
        data[indicator] = talib.RSI(data['Close'], timeperiod=14)
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
   
  

    elif indicator == 'Average Price':
        data[indicator] = talib.AVGPRICE(data['Open'], data['High'], data['Low'], data['Close'])
    elif indicator == 'Weighted Close':
        data[indicator] = talib.WCLPRICE(data['High'], data['Low'], data['Close'])
    elif indicator == 'Median Price':
        data[indicator] = talib.MEDPRICE(data['High'], data['Low'])
    elif indicator == 'Typical Price':
        data[indicator] = talib.TYPPRICE(data['High'], data['Low'], data['Close'])
  
    elif indicator == 'Accumulation/Distribution Line':
        data[indicator] = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])
    elif indicator == 'Chaikin Oscillator':
        data[indicator] = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=10)
    elif indicator == 'Ease of Movement':
        high, low, volume = data['High'], data['Low'], data['Volume']
        dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        br = volume / ((high - low))
        emv = dm / br
        emv_ma = emv.rolling(window=14).mean()
        data[indicator] = emv_ma

    elif indicator == 'Money Flow Index':
        data[indicator] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)
    elif indicator == 'Negative Volume Index':
        close, volume = data['Close'], data['Volume']
        price_diff = close.diff()
        price_diff[price_diff >= 0] = 0
        nvi = 1000
        nvi_list = [nvi]
        for i in range(1, len(data)):
            if volume[i] < volume[i-1]:
                nvi += (price_diff[i] / close[i-1]) * nvi
            else:
                nvi += 0
            nvi_list.append(nvi)
        data[indicator] = nvi_list

    elif indicator == 'Positive Volume Index':
        close, volume = data['Close'], data['Volume']
        price_diff = close.diff()
        price_diff[price_diff <= 0] = 0
        pvi = 1000
        pvi_list = [pvi]
        for i in range(1, len(data)):
            if volume[i] > volume[i-1]:
                pvi += (price_diff[i] / close[i-1]) * pvi
            else:
                pvi += 0
            pvi_list.append(pvi)
        data[indicator] = pvi_list

    elif indicator == 'Volume Price Trend':
        close, volume = data['Close'], data['Volume']
        vpt = ((close.diff() / close) * volume).cumsum()
        data[indicator] = vpt

    elif indicator == 'Volume Rate of Change':
        close, volume = data['Close'], data['Volume']
        roc = close.pct_change()
        vroc = roc.rolling(window=14).mean() * volume
        data[indicator] = vroc

    elif indicator == 'Commodity Channel Index':
        data[indicator] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
    

    elif indicator == 'Momentum':
        data[indicator] = talib.MOM(data['Close'], timeperiod=10)
    elif indicator == 'Rate of Change':
        data[indicator] = talib.ROC(data['Close'], timeperiod=10)
   
    elif indicator == 'Trix':
        data[indicator] = talib.TRIX(data['Close'], timeperiod=30)
    elif indicator == 'Ultimate Oscillator':
        data[indicator] = talib.ULTOSC(data['High'], data['Low'], data['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    elif indicator == 'Williams %R':
        data[indicator] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
    elif indicator == 'Arnaud Legoux Moving Average':
        close = data['Close']
        m = 9
        s = 6
        offset = 0.85
        w = np.arange(m) * 2 / (m - 1) - 1
        w = np.exp(-w ** 2 / (2 * (s / 100) ** 2))
        w /= sum(w)
        alma = np.zeros_like(close)
        for i in range(m - 1, len(close)):
            window = close[i - m + 1:i + 1]
            alma[i] = sum(w * window)
        alma = alma * (1 - offset) + alma.mean() * offset
        data[indicator] = alma

    elif indicator == 'Chande Momentum Oscillator':
        data[indicator] = talib.CMO(data['Close'], timeperiod=14)
   

    

    elif indicator == 'Linear Regression':
        data[indicator] = talib.LINEARREG(data['Close'], timeperiod=14)
    elif indicator == 'Moving Average Ribbon':
        data['MA_5'] = talib.MA(data['Close'], timeperiod=5)
        data['MA_10'] = talib.MA(data['Close'], timeperiod=10)
        data['MA_20'] = talib.MA(data['Close'], timeperiod=20)
        data['MA_50'] = talib.MA(data['Close'], timeperiod=50)
        data['MA_100'] = talib.MA(data['Close'], timeperiod=100)
        data['MA_200'] = talib.MA(data['Close'], timeperiod=200)

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
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Define the XGBoost parameters
params = {'max_depth': 5,
          'learning_rate': 0.1,
          'n_estimators': 100,
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
st.subheader(f'Test accuracy: {accuracy:.2%}')
print(f'Test accuracy: {accuracy:.2%}')

st.table((test_data['Predictions'], test_data['Target']))

def test(money):
    for i, j in zip(test_data['Predictions'], test_data['Target']):
        if i == j:
            money += ((money * 0.3) * 0.70)
        else:
            money -= (money * 0.3)
    return round(money, 2)
amount=st.number_input('Amount', min_value=1, max_value=100, value=1)
st.header(f'Returns Investment: ${test(amount)}')


