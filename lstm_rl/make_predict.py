import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from lstm_rl.predict import train_model, evaluate_model, calculate_metrics, create_sequences,BiLSTMModel, predict_future, GRUModel
from synth.price_data_provider import PriceDataProvider
def reset_before():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
def update_model():
    price_data_provider = PriceDataProvider("BTC")
    start_time = "2025-02-26T00:00:00"
    data = price_data_provider.fetch_csv(start_time)
    data = data.drop(columns=["volume"])
    print("=-======================")
    print(f"normalized_data={data}")
    numeric_data = data.drop(columns=["timestamp"])
    numeric_data = numeric_data[['Close'] + [col for col in numeric_data.columns if col != 'Close']]

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(numeric_data)

    data[numeric_data.columns] = normalized_data
    print("-------------------\n")
    print(f"numeric_data.columns={numeric_data.columns}")
    print("--------------------")
    print(f"data={data}")
    train_data = data.drop(columns=["timestamp"]).values
    test_data = data[-288:].drop(columns=["timestamp"]).values
    
    # print(f"train_data={train_data}")
    print(f"train_data shape={train_data.shape}")
    print(f"test_data shape={test_data.shape}")
    sequence_length = 288 #means one day
    X_train, y_train = create_sequences(train_data, sequence_length)
    
    print("--------------------")
    # print(f"X_train={X_train}")
    # print(X_train)
    print("X_train=")
    print(len(X_train))
    print(X_train.shape)
    print("--------------------")
    # print(y_train)
    print(f"Y_train=")

    print(len(y_train))
    print(y_train.shape)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    results = {}
    print(f"X_train.shape[2]={X_train.shape[2]}")
    # network = BiLSTMModel(input_dim=X_train.shape[2],hidden_dim=2224, num_layers=1, dropout=0.14063490978424847)
    # learning_rate = 0.00191084309168345
    network = GRUModel(input_dim=X_train.shape[2], hidden_dim=32, num_layers=1, dropout=0.30883575663821067)
    learning_rate = 0.016030491965989343
    
    
    print(f"Training ...")
    
    print("--------------------")
    # last_sequence = price_data_provider.fetch_last_day(start_time)    
    # print(last_sequence)
    # print(f"last_sequence={len(last_sequence)}")
    # print(last_sequence.shape)
    
    trained_model = train_model(network, train_loader, num_epochs=2, learning_rate=learning_rate)
    torch.save(trained_model, 'bi.pt')
    
    
    
    predictions = predict_future(trained_model, test_data)
    print("-------==================================")
    print(f"predictions={predictions}")
    
    
reset_before()    
update_model()
    # future_timestamps = pd.date_range(df.index[-1], )
    
# numeric_data = df.drop(columns=["Date"])
# numeric_data = numeric_data[['Close'] + [col for col in numeric_data.columns if col != 'Close']]

# #df = df.resample('1T').mean().fillna(method="ffill")
# scaler = MinMaxScaler(feature_range=(0,1))
# normalized_data = scaler.fit_transform(numeric_data)
# print("=-======================")
# print(f"normalized_data={normalized_data}")

