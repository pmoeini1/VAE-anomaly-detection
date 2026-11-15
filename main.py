from datasets import load_dataset
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset
from VAE import VAE
from loss_functions import neg_elbo
import torch
from anomaly_score import anomaly_score
import matplotlib.pyplot as plt

def main():
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
    X_tr, X_val = X[:len(X) - n_val - WINDOW], X[len(X) - n_val:]

    loader = DataLoader(TensorDataset(X_tr), batch_size=256, shuffle=True)
    model = VAE(window=WINDOW)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training
    for epoch in range(30):
        model.train()
        total, n = 0.0, 0
        # go over each minibatch
        for (xb,) in loader:
            # clear out old gradients
            opt.zero_grad()
            # forward pass + calculate loss
            loss = neg_elbo(xb, *model(xb))
            # backpropagate + step
            loss.backward()
            opt.step()

            total += loss.item() * len(xb)
            n += len(xb)
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            val_loss = neg_elbo(X_val, *model(X_val)).item()
        print(f"epoch {epoch+1:02d}  train {train_loss.item():.2f}  val {val_loss:.2f}")

    removed = torch.from_numpy(np.ascontiguousarray(training_windows[outlier_window], dtype=np.float32))
    clean = anomaly_score(model, X_val)
    removed = anomaly_score(model, removed)
    print("Clean val mean score: ", clean.mean().item())
    print("Removed window mean score: ", removed.mean().item())

    clean = clean.numpy()
    removed = removed.numpy()
    fig, ax = plt.subplots(figsize=(8,4))
    bins = np.linspace(min(clean.min(), removed.min()), max(clean.max(), removed.max()), 80)

    ax.hist(clean, bins=bins, alpha=0.6, label="Clean Validation")
    ax.hist(removed, bins=bins, alpha=0.6, label="Removed (Outlier) Windows")
    ax.set_xscale("symlog")
    ax.set_yscale("log")
    ax.set_xlabel("Anomaly score (negative log-likelihood)")
    ax.set_ylabel("Number of windows")
    ax.legend()
    plt.show()



if __name__=="__main__":
    main()