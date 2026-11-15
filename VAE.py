import torch
import torch.nn as nn
import torch.nn.functional as F

# encoder makes 48-hr window into a distribution in 8-dim latent space
# decoder reconstructs from 8-dim latent space
class VAE(nn.Module):

    def __init__(self, window=48, hidden=128, latent=8):
        super().__init__()
        # small MLP for encoding
        self.enc = nn.Sequential(
            nn.Linear(window, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # output heads for mean and log of variance
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        # small MLP for decoding
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # mean and log of variance of output
        self.x_mu = nn.Linear(hidden, window)
        self.x_logvar = nn.Linear(hidden, window)
    
    # returns mean and log of variance of latent distribution
    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    # draws point from Gaussian distribution encoder described
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std*torch.randn_like(std)

    # takes latent code and returns mean and clamped log variance of output
    def decode(self, z):
        h = self.dec(z)
        return self.x_mu(h), self.x_logvar(h).clamp(-8, 4)

    # forward pass of encoder and decoder
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_mu, x_logvar = self.decode(z)
        return x_mu, x_logvar, mu, logvar


    
