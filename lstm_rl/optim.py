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

class TradingEnvBiLSTM(Env):
    def __init__(self, data, model, device, sequence_length=30):
        super(TradingEnvBiLSTM, self).__init__()
        self.data = data
        self.model = model
        self.device = device
        self.sequence_length = sequence_length
        self.current_step = 0
        self.total_steps = len(data) - sequence_length
        self.portfolio_value = 1.0

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(sequence_length, data.shape[1] + 1), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1.0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data[self.current_step:self.current_step + self.sequence_length]
        current_sequence = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        predicted_price = self.model(current_sequence).item()

        # Add the predicted price to the observation
        obs_with_prediction = np.hstack((obs, np.full((obs.shape[0], 1), predicted_price)))
        return obs_with_prediction

    def step(self, action):
        self.current_step += 1

        # Map continuous actions to discrete
        if action < -0.33:
            discrete_action = 2  # Sell
        elif action > 0.33:
            discrete_action = 1  # Buy
        else:
            discrete_action = 0  # Hold

        reward = 0
        done = self.current_step >= self.total_steps

        # Calculate reward
        if discrete_action == 1:  # Buy
            reward = self.data[self.current_step, 0] - self.data[self.current_step - 1, 0]
        elif discrete_action == 2:  # Sell
            reward = self.data[self.current_step - 1, 0] - self.data[self.current_step, 0]

        # Update portfolio value
        self.portfolio_value += reward

        obs = self._next_observation() if not done else None
        return obs, reward, done, {"portfolio_value": self.portfolio_value}


        
def evaluate_trading_strategy(env, model):
    obs = env.reset()
    done = False
    portfolio = [1.0]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    cumulative_return = portfolio[-1] - portfolio[0]
    return cumulative_return, portfolio
