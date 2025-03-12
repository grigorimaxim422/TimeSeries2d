import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from synth.price_data_provider import PriceDataProvider
from lstm2.model import LSTMModel,train_model, predict_future, calculate_metrics
def create_sequences(data, sequence_length, future_steps):
    X, y = [], []
    for i in range(len(data) - sequence_length - future_steps):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length:i+sequence_length+future_steps])
    return np.array(X), np.array(y)


def reset_seed():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
def update_model():
    
    price_data_provider = PriceDataProvider("BTC")
    start_time = "2025-02-26T00:00:00"
    data, end_time = price_data_provider.fetch_csv(start_time)
    print("----------------------------")
    print(data.shape)
    # df = df.drop(columns=["volume"]).values
    # data = df['Close'].values
    # print("----------------------------")
    # print(data)
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['price'] = scaler.fit_transform(data[['Close']])
    
    prices = data['price'].values
    print("--------------------------------")
    print(f"prices.shape={prices.shape}")
    # Define sequence length (how many past steps to use)
    sequence_length = 48  # Using last 4 hours (48 steps of 5 min each)
    forecast_horizon = 288  # Predict next 24 hours (288 steps of 5 min)
    X,y = create_sequences(prices, sequence_length, forecast_horizon)
        
    # DataLoader
    batch_size = 64
    train_dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32).unsqueeze(-1), 
        torch.tensor(y, dtype=torch.float32).unsqueeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # network = BiLSTMModel(input_dim=X_train.shape[2], hidden_dim=224, num_layers=1, dropout=0.14063490978424847)
    network = LSTMModel(1, 64, 2, 288, 0.14063490978424847)
    learning_rate = 0.00191084309168345
    model = train_model(network, train_loader, 20, learning_rate)
    
    last_sequence = prices[-sequence_length:]
    predicted_prices = predict_future(model, last_sequence, forecast_horizon, scaler)
    
            
    future_timestamps = pd.date_range(data.index[-1], periods=forecast_horizon+1, freq='5T')[1:]
    forecast_df = pd.DataFrame({'timestamp': future_timestamps, 'predicted_price': predicted_prices})
    print(forecast_df.head())
    
    real_prices, trans = price_data_provider.fetch_afterward(end_time,300)
    print(trans)

    metrics = calculate_metrics(predicted_prices, real_prices)
    print(f"metrics={metrics}")
    # print({'timestamp': future_timestamps, 'predicted_price': predicted_prices})
    
reset_seed()
update_model()