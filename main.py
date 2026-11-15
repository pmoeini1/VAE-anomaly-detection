from datasets import load_dataset
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset
from VAE import VAE
from loss_functions import neg_elbo
import torch
from anomaly_score import anomaly_score
from sklearn.metrics import precision_recall_fscore_support

def main():
    torch.manual_seed(42)
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

    test_slots = (np.arange(len(test_raw)) + split) % PERIOD
    test_residuals = test_raw - profile[test_slots]
    test_outliers = (test_residuals < q1 - 1.5*iqr) | (test_residuals > q3 + 1.5*iqr)
    test_windows = sliding_window_view(test_raw, WINDOW)
    y_test = sliding_window_view(test_outliers, WINDOW).any(axis=1)

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
        print(f"epoch {epoch+1:02d}  train {train_loss:.2f}  val {val_loss:.2f}")

    # sanity check that outliers have higher anomaly scores than non-outliers
    removed = torch.from_numpy(np.ascontiguousarray(training_windows[outlier_window], dtype=np.float32))
    clean = anomaly_score(model, X_val)
    removed = anomaly_score(model, removed)
    print("Clean val mean score: ", clean.mean().item())
    print("Removed window mean score: ", removed.mean().item())

    # get scores for test windows
    X_test = torch.from_numpy(np.ascontiguousarray(test_windows, dtype=np.float32))
    test_scores = anomaly_score(model, X_test).numpy()

    if y_test.sum()==0: 
        print("no anomalous test windows, skipping metrics")
        return

    # investigate test scores w/ anomaly score greater than 99th percentile
    val_scores = clean.numpy()
    thr = np.percentile(val_scores, 99)
    preds = test_scores > thr
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    print(f"threshold {thr:.2f}  precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}")

if __name__=="__main__":
    main()