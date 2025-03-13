import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# تنظیمات اولیه برای تولید نتایج قابل بازتولید


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)
        return self.fc(bilstm_out[:, -1, :])

# 2. تعریف مدل‌های مختلف
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])

class GRUAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        return self.fc(context_vector)

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(context_vector)

class LSTMGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMGRUModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        return self.fc(gru_out[:, -1, :])




# 3. تابع آموزش و ارزیابی مدل‌ها
def train_model(model, train_loader, num_epochs, learning_rate):
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
            predictions = model(X_batch).squeeze()                        
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
    return model

def predict_future(model, last_sequence, forecast_horizon=288):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # last_sequence = np.array(last_sequence)
    # cv, _ = create_sequences(last_sequence, 288)
    # test_dataset = TensorDataset(torch.tensor(cv, dtype=torch.float32),torch.tensor(cv, dtype=torch.float32))
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # # input_seq = torch.tensor(last_sequence, dtype=torch.float32)
    # # print("input_seq.shape=")
    # # print(input_seq.shape)
    # # last_dataset=TensorDataset(to)    
    # predictions = []    
    
    # with torch.no_grad():
    #     for input_seq,_ in test_loader:        
    #         input_seq = input_seq.to(device)
    #         prediction = model(input_seq).squeeze()
    #         predictions.extend(prediction.cpu().numpy())                
    # return np.array(predictions)
    # input_seq = np.array(last_sequence)
    # print(f"input_seq={input_seq.shape}")
    # with torch.no_grad():
    
        
        
def evaluate_model(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            print(f"eval: X_batch.shape={X_batch.shape}")
            preds = model(X_batch).squeeze()
            predictions.extend(preds.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())

    return np.array(predictions), np.array(targets)

# محاسبه معیارهای مختلف
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

# تعریف تابع برای ایجاد توالی‌های زمانی
def create_sequences(data, sequence_length=30):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length, 0]  # ستون "Close" به عنوان هدف
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

