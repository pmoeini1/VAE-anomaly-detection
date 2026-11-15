from datasets import load_dataset
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
