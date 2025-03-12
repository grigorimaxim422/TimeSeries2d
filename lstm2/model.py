import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=288, dropout=0.14063490978424847):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use last LSTM output
        return output.unsqueeze(-1)  # Match y_train shape

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size, dropout):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_size)

    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)
        output = self.fc(bilstm_out[:, -1, :])
        return output.unsqueeze(-1)  # Match y_train shape

def train_model(model, train_loader, num_epochs=20, learning_rate=0.1):
    # Initialize model
    # model = LSTMModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print(f"train epoch={epoch}...")
        model.train()
        for X_batch, y_batch in train_loader:
            # print("single batch... ")
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()            
            predictions = model(X_batch)#.squeeze()                        
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
    return model

def predict_future(model, last_sequence, forecast_horizon, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        input_seq = input_seq.to(device)
        prediction = model(input_seq).squeeze(-1)
        prediction = prediction.cpu().numpy()
        predictions.extend(prediction)                
    # return np.array(predictions)
    return scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()


from sklearn.metrics import mean_absolute_error, r2_score
def calculate_metrics(predictions, targets):
    mse = np.mean((predictions - targets) ** 2)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100
    r2 = r2_score(targets, predictions)
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }    
