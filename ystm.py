##https://freedium.cfd/https://tradingtechai.medium.com/cryptocurrency-price-prediction-with-lstm-neural-networks-capturing-market-volatility-for-better-b3ed125993a1
import yfinance as yf

# Download Bitcoin price data
btc_data = yf.download('BTC-USD', start='2017-01-01', end='2024-01-31')

# Download Ethereum price data
eth_data = yf.download('ETH-USD', start='2017-01-01', end='2024-01-31')

from sklearn.preprocessing import MinMaxScaler

# Scale Bitcoin price data
btc_scaler = MinMaxScaler()
scaled_btc_data = btc_scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

# Scale Ethereum price data
eth_scaler = MinMaxScaler()
scaled_eth_data = eth_scaler.fit_transform(eth_data['Close'].values.reshape(-1, 1))

train_size = int(0.8 * len(scaled_btc_data))
train_btc_data = scaled_btc_data[:train_size]
test_btc_data = scaled_btc_data[train_size:]

train_eth_data = scaled_eth_data[:train_size]
test_eth_data = scaled_eth_data[train_size:]

from keras.models import Sequential
from keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    return model

# Build the Bitcoin LSTM model
btc_model = build_lstm_model((train_btc_data.shape[1], 1))

# Build the Ethereum LSTM model
eth_model = build_lstm_model((train_eth_data.shape[1], 1))

btc_model.compile(optimizer='adam', loss='mean_squared_error')
btc_model.fit(train_btc_data, train_btc_data, epochs=100, batch_size=32)

eth_model.compile(optimizer='adam', loss='mean_squared_error')
eth_model.fit(train_eth_data, train_eth_data, epochs=100, batch_size=32)

btc_loss = btc_model.evaluate(test_btc_data, test_btc_data)
eth_loss = eth_model.evaluate(test_eth_data, test_eth_data)

print(f"Bitcoin Model Loss: {btc_loss}")
print(f"Ethereum Model Loss: {eth_loss}")

print(f"test_btc_data={test_btc_data}")
print(f"test_eth_data={test_eth_data}")
print("-------------Predictions----------------")
btc_predictions = btc_model.predict(test_btc_data)
eth_predictions = eth_model.predict(test_eth_data)
print("----------------------------------------")
