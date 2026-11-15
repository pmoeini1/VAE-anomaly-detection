from datasets import load_dataset
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# import 1 building's hourly electricity consumption
ds = load_dataset("LeoTungAnh/electricity_hourly", split="train", streaming=True)
row = next(iter(ds))
series = np.array(row["target"], dtype=np.float32)

WINDOW = 48
PERIOD = 168

# last 20% reserved for testing
split = int(0.8 * len(series))
train_raw, test_raw = series[:split].copy(), series[split:].copy()

profile = np.array([np.median(train_raw[i::PERIOD]) for i in range(PERIOD)])
# get difference between raw data point and median for that hour of the week
residual = train_raw - profile[np.arange(len(train_raw)) % PERIOD]

q1, q3 = np.percentile(residual, [25, 75])
iqr = q3-q1
outlier = (residual < q1 - 1.5 * iqr) | (residual > q3 + 1.5 * iqr)

# get 48 hour sliding windows and remove outliers from training
training_windows = sliding_window_view(train_raw, WINDOW)
outlier_window = sliding_window_view(outlier, WINDOW).any(axis=1)
x_train = training_windows[~outlier_window]

class VAE(nn.Module):
    
    def __init__(self, window=48, hidden=128, latent=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(window, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.x_mu = nn.Linear(hidden, window)
        self.x_logvar = nn.Linear(hidden, window)
    
    def encode(self, x):
        h = self.encode(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std*torch.randn_like(std)

    def decode(self, z):
        h = self.dec(z)
        return self.x_mu(h), self.x_logvar(h).clamp(-8, 4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_mu, x_logvar = self.decode(z)
        return x_mu, x_logvar, mu, logvar
