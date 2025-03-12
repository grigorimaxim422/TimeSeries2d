import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from lstm_rl.predict import train_model, evaluate_model, calculate_metrics, create_sequences,BiLSTMModel

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
# 1. بارگذاری و پیش‌پردازش داده‌ها
data = pd.read_csv("Book1.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(by="Date")

# حذف ستون‌های غیرعددی و نرمال‌سازی داده‌ها
numeric_data = data.drop(columns=["Date"])
# numeric_data = numeric_data[['Close'] + [col for col in numeric_data.columns if col != 'Close']]

print("--------------------")
print(f"numeric_data={numeric_data}")
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_data)

data[numeric_data.columns] = normalized_data
print("-------------------\n")
print(f"numeric_data.columns={numeric_data.columns}")
print("--------------------")
print(f"data={data}")

# تقسیم داده‌ها به سه بخش: آموزش، ولیدیشن و تست
train_data = data[(data["Date"].dt.year >= 2017) & (data["Date"].dt.year <= 2022)].drop(columns=["Date"]).values
val_data = data[data["Date"].dt.year == 2023].drop(columns=["Date"]).values
test_data = data[data["Date"].dt.year == 2024].drop(columns=["Date"]).values


sequence_length = 30
X_train, y_train = create_sequences(train_data, sequence_length)
X_val, y_val = create_sequences(val_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

print("--------------------")
print(f"X_train={X_train}")
print("--------------------")
print(f"Y_train={y_train}")

# آماده‌سازی داده‌ها برای PyTorch
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

results = {}
name = "Bi-LSTM"
network = BiLSTMModel(input_dim=X_train.shape[2], hidden_dim=224, num_layers=1, dropout=0.14063490978424847)
learning_rate = 0.00191084309168345

print(f"Training {name}...")
trained_model = train_model(network, train_loader, num_epochs=20, learning_rate=learning_rate)
predictions, targets = evaluate_model(trained_model, test_loader)

print("-----------------------------\n")
print(f"{name} Predictions: {predictions}")
print("-----------------------------\n")
metrics = calculate_metrics(predictions, targets)
results[name] = metrics
print(f"{name} Metrics: {metrics}")


# 5. نمایش نتایج
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Results:")
print(results_df)


import matplotlib.pyplot as plt

# رسم نمودارهای جداگانه برای هر معیار
metrics = ["MSE", "MAE", "RMSE", "MAPE"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    results_df[[metric]].plot(kind="bar", legend=False)
    plt.title(f"Model Comparison on {metric}")
    plt.ylabel(f"{metric} Value")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
