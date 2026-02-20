"""
NN for delivery_days regression: 2 hidden layers, 10 neurons each.
"""
import torch
import torch.nn as nn


class DeliveryDateNN(nn.Module):
    """Predict delivery_days from 17 features. Two hidden layers of 10 units each."""

    def __init__(self, input_dim=17, hidden_dim=10, num_hidden=2, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        in_d = input_dim
        for _ in range(num_hidden):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(input_dim=17, hidden_dim=10, num_hidden=2, dropout=0.0, device=None):
    device = device or _get_device()
    model = DeliveryDateNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        dropout=dropout,
    ).to(device)
    return model
