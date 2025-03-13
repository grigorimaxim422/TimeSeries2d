CryptoRL-HybridFramework
This repository contains the code and dataset for the research paper:
"A Hybrid Framework for Cryptocurrency Trading Using Bi-LSTM and Reinforcement Learning".

Overview
This framework combines advanced predictive deep learning models with reinforcement learning (RL) algorithms to optimize cryptocurrency trading strategies. The primary goal is to leverage accurate price forecasting using Bi-LSTM and enhance decision-making using RL algorithms such as SAC, PPO, and A2C.

The framework is evaluated across different market conditions (bullish and bearish) and cryptocurrencies (Bitcoin, Ripple, and Dogecoin). The results demonstrate superior performance of the hybrid model compared to traditional strategies such as Buy-and-Hold and standalone RL models.

Key Features
Predictive Models: Six predictive models are implemented and evaluated:

LSTM
GRU
Bi-LSTM
GRU-Attention
LSTM-Attention
GRU-LSTM
The Bi-LSTM model is selected as the primary predictive model due to its superior accuracy.
Reinforcement Learning Algorithms:

Advantage Actor-Critic (A2C)
Proximal Policy Optimization (PPO)
Soft Actor-Critic (SAC)
Hyperparameter Optimization: All models are fine-tuned using the Optuna framework for optimal performance.

Dataset
The dataset used in this study includes historical price data and technical indicators for three cryptocurrencies:

Bitcoin (BTC): Book1.csv
Ripple (XRP): Book2.csv
Dogecoin (DOGE): Book3.csv
Each dataset contains the following features:

Open, High, Low, Close prices
Volume
Technical indicators (e.g., SMA, EMA, RSI, MACD, etc.)
