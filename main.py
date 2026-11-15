from datasets import load_dataset
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset
import torch
from VAE import VAE
from loss_functions import gaussian_log_prob, neg_elbo

# import 1 client's hourly electricity consumption
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

# split into training and validation sets
X = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
n_val = int(0.1*len(X))
X_tr, X_val = X[: len(X) - n_val - WINDOW], X[len(X) - n_val:]

loader = DataLoader(TensorDataset(X_tr), batch_size=256, shuffle=True)
model = VAE(window=WINDOW)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# training
for epoch in range(30):
    model.train()
    # go over each minibatch
    for (xb,) in loader:
        # clear out old gradients
        opt.zero_grad()
        # forward pass + calculate loss
        loss = neg_elbo(xb, *model(xb))
        # backpropagate + step
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = neg_elbo(X_val, *model(X_val)).item()
        print(f"epoch {epoch+1:02d}  train {loss.item():.2f}  val {val_loss:.2f}")

@torch.no_grad()
def anomaly_score(model, x, L=20):
    model.eval()
    mu, logvar = model.encode(x)
    log_probs = []
    for _ in range(L):
        z = model.reparameterize(mu, logvar)
        x_mu, x_logvar = model.decode(z)
        log_probs.append(gaussian_log_prob(x, x_mu, x_logvar))
    return -torch.stack(log_probs).mean(dim=0)

removed = torch.from_numpy(np.ascontiguousarray(training_windows[outlier_window], dtype=np.float32))
print("Clean val mean score: ", anomaly_score(model, X_val).mean().item())
print("Removed window mean score: ", anomaly_score(model, removed).mean().item())
