import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size, output_size: int, learning_rate: float):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # Initialize loss function and optimizer
        self.loss_fn = torch.nn.functional.mse_loss
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.net(torch.tensor(x).float())

    def fit(self, states: np.ndarray, q_values: np.ndarray):
        q_value_pred = self.net(torch.tensor(states))
        loss = self.loss_fn(q_value_pred, q_values)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_model(self, other_model: nn.Module):
        self.load_state_dict(other_model.state_dict())

    def load_model(self, path: str):
        self.net.load_state_dict(torch.load(path))

    def save_model(self, path: str):
        torch.save(self.net.state_dict(), path)

