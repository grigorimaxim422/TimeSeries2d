import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from gym import spaces, Env
import matplotlib.pyplot as plt
import random
from predict import train_model, evaluate_model, calculate_metrics, create_sequences,BiLSTMModel
from optim import TradingEnvBiLSTM, evaluate_trading_strategy

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load and preprocess data
data = pd.read_csv("Book1.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(by="Date")

# Remove non-numeric columns
numeric_data = data.drop(columns=["Date"])

# Move 'Close' column to the first position
numeric_data = numeric_data[['Close'] + [col for col in numeric_data.columns if col != 'Close']]

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_data)

# Add normalized data back to the original dataframe for splitting by date
data[numeric_data.columns] = normalized_data

# Split data based on years
train_data = data[(data["Date"].dt.year >= 2017) & (data["Date"].dt.year <= 2022)].drop(columns=["Date"]).values
validation_data = data[data["Date"].dt.year == 2023].drop(columns=["Date"]).values
test_data = data[data["Date"].dt.year == 2024].drop(columns=["Date"]).values

# Convert data to sequences for time series
sequence_length = 30
X_train, y_train = create_sequences(train_data, sequence_length)
X_validation, y_validation = create_sequences(validation_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)
# Hyperparameters for Bi-LSTM
input_dim = X_train.shape[2]
hidden_dim = 224
num_layers = 1
dropout = 0.14063490978424847
learning_rate = 0.00191084309168345
num_epochs = 20
batch_size = 32
results = {}
name = "Bi-LSTM"
print("----------------------\n")
print(f"X_train={X_train}")
print("----------------------\n")
print(f"X_test={X_test}")
print("----------------------\n")

# # Prepare data for PyTorch
# train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
# validation_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32), torch.tensor(y_validation, dtype=torch.float32))
# test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# network = BiLSTMModel(input_dim, hidden_dim, num_layers, dropout).to(device)

# print(f"Training {name}...")
# trained_model = train_model(network, train_loader, validation_loader, num_epochs=20, learning_rate=learning_rate)
# predictions, targets = evaluate_model(trained_model, test_loader)
# metrics = calculate_metrics(predictions, targets)
# results[name] = metrics
# print(f"{name} Metrics: {metrics}")

# # Split data for training and testing PPO
# train_env_data = data[(data["Date"].dt.year >= 2017) & (data["Date"].dt.year <= 2023)].drop(columns=["Date"]).values
# test_env_data = data[data["Date"].dt.year == 2024].drop(columns=["Date"]).values

# # Create separate environments for training and testing
# train_env = TradingEnvBiLSTM(train_env_data, network, device)
# test_env = TradingEnvBiLSTM(test_env_data, network, device)

# # Train the PPO model
# ppo_model = PPO("MlpPolicy", train_env, learning_rate=0.001400281687989954, n_steps=1280, batch_size=160, verbose=1, gamma=0.9005911776350177)
# ppo_model.learn(total_timesteps=50000)

# cumulative_return, portfolio_values = evaluate_trading_strategy(test_env, ppo_model)

# # Calculate portfolio cumulative return using cumprod
# portfolio_daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
# portfolio_cumulative_return = np.cumprod(1 + portfolio_daily_returns)

# # Final cumulative return based on cumprod
# final_cumulative_return = portfolio_cumulative_return[-1]

# print("Cumulative Return of PPO Model (via cumprod):", final_cumulative_return)

# #Compare with 
# data['returns'] = data['Close'].pct_change()
# data['cumulative_return'] = (1+data['returns']).cumprod()

# #Filter data for 2024
# btc_data_2024 = data[data['Date'].dt.year == 2024]
# # Align data lengths
# dates = btc_data_2024['Date'][sequence_length:sequence_length + len(portfolio_values)]
# btc_cumulative_return = btc_data_2024['BTC_Cumulative_Return'][sequence_length:sequence_length + len(portfolio_values)]

# if len(dates) > len(portfolio_values):
#     dates = dates[:len(portfolio_values)]
# elif len(dates) < len(portfolio_values):
#     portfolio_values = portfolio_values[:len(dates)]

# # Ensure BTC Cumulative Return is aligned
# if len(btc_cumulative_return) > len(portfolio_values):
#     btc_cumulative_return = btc_cumulative_return[:len(portfolio_values)]

# # Calculate additional metrics
# sharpe_ratio = np.mean(portfolio_daily_returns) / np.std(portfolio_daily_returns)
# max_drawdown = np.min(portfolio_cumulative_return / np.maximum.accumulate(portfolio_cumulative_return) - 1)

# print("Cumulative Return of PPO Model:", final_cumulative_return)
# print("Sharpe Ratio:", sharpe_ratio)
# print("Max Drawdown:", max_drawdown)






