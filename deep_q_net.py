import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch as T

from typing import Tuple

class DenseDeepQNetwork(nn.Module):
    def __init__(self, lr: float, n_actions: int, input_dims: Tuple[int, ...], dir:str, name:str) -> None:
        super(DenseDeepQNetwork, self).__init__()

        self.checkpoint_file = os.path.join(dir, name)
        self.n_actions = n_actions
        self.dense = nn.Sequential(
            nn.Linear(*input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        actions = self.dense(data)
        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), f=f"{self.checkpoint_file}.pt")

    def load_checkpoint(self):
        self.load_state_dict(T.load(f"{self.checkpoint_file}.pt"))

class ConvDeepQNetwork(nn.Module):
    def __init__(self, lr: float, n_actions: int, input_dims: Tuple[int, ...], dir:str, name:str) -> None:
        super(ConvDeepQNetwork, self).__init__()

        self.checkpoint_file = os.path.join(dir, name)
        self.n_actions = n_actions
        in_channel = input_dims[0]

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        fc_input_dims = self.calculate_fc_input_dims(input_dims)
        
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(*fc_input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_fc_input_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv(state)
        return int(np.prod(dims.size()))

    def forward(self, data):
        conv_data = self.conv(data)
        actions = self.dense(conv_data)
        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), f=f"{self.checkpoint_file}.pt")

    def load_checkpoint(self):
        self.load_state_dict(T.load(f"{self.checkpoint_file}.pt"))